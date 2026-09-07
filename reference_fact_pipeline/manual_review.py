"""Apply a completed Excel human review to a frozen Reference Fact collection."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from infobudget.quality_router.io import file_sha256, iter_jsonl, write_jsonl
from infobudget.rl_router.ledger import atomic_write_json
from reference_fact_pipeline.audit import FACT_TYPES, STATE_STATUSES
from reference_fact_pipeline.io import export_reference_jsonl


REVIEW_SHEET = "复核任务"
REQUIRED_REVIEW_COLUMNS = {
    "复核状态",
    "复核决定",
    "复核 ID",
    "审核备注",
    "修正 Facts（JSON）",
}
DEFAULT_PASS_DECISIONS = {
    "zero_reference_set": "VALID_EMPTY",
    "fact_cap_reached": "COMPLETE",
    "json_repair_used": "REPAIR_SAFE",
    "near_duplicate_pair": "DISTINCT",
    "stratified_quality_sample": "ACCEPT",
}
ALLOWED_DECISIONS = {
    "zero_reference_set": {"VALID_EMPTY", "MISSING_FACTS", "ESCALATE"},
    "fact_cap_reached": {
        "COMPLETE",
        "TRUNCATION_LOSS",
        "REPLACE_LOW_VALUE",
        "ESCALATE",
    },
    "json_repair_used": {
        "REPAIR_SAFE",
        "CONTENT_CHANGED",
        "NEEDS_REGENERATION",
        "ESCALATE",
    },
    "near_duplicate_pair": {"DISTINCT", "DUPLICATE", "MERGE", "ESCALATE"},
    "stratified_quality_sample": {"ACCEPT", "REVISE", "REJECT", "ESCALATE"},
}
BLOCKING_DECISIONS = {"ESCALATE", "CONTENT_CHANGED", "NEEDS_REGENERATION"}
NO_CHANGE_DECISIONS = {
    "VALID_EMPTY",
    "COMPLETE",
    "REPAIR_SAFE",
    "DISTINCT",
    "ACCEPT",
}
_REVISION_RE = re.compile(
    r"应该是\s*(?P<fact_type>[a-z_]+)(?:\s*/\s*(?P<state_status>[a-z_]+))?",
    re.IGNORECASE,
)
_PLAIN_FACT_RE = re.compile(
    r"(?m)^\s*(?:\d+\.\s*)?"
    r"\[(?P<fact_type>[a-z_]+)/(?P<state_status>[a-z_]+)\]\s*"
    r"(?P<fact_text>[^\r\n]+)\r?\n"
    r"来源\s*Turn\s*[:：]?\s*(?P<source_turn_ids>[0-9,，\s]+)\s*$",
    re.IGNORECASE,
)


def apply_completed_review(
    *,
    references_path: str | Path,
    manifest_path: str | Path,
    review_queue_path: str | Path,
    review_workbook_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    review_version: str = "manual_review_v1",
    blank_means_pass: bool = False,
) -> dict[str, Any]:
    """Create a reviewed derivative without modifying the frozen parent artifacts."""

    references_path = Path(references_path)
    manifest_path = Path(manifest_path)
    review_queue_path = Path(review_queue_path)
    review_workbook_path = Path(review_workbook_path)
    output_dir = Path(output_dir)

    reference_rows = list(iter_jsonl(references_path))
    reference_by_segment = {
        str(row["segment_id"]): deepcopy(row) for row in reference_rows
    }
    if len(reference_by_segment) != len(reference_rows):
        raise ValueError("reference collection contains duplicate segment_id values")

    queue_rows = list(iter_jsonl(review_queue_path))
    queue_by_id = {str(row["review_id"]): row for row in queue_rows}
    if len(queue_by_id) != len(queue_rows):
        raise ValueError("manual review queue contains duplicate review_id values")

    workbook_rows = read_xlsx_table(review_workbook_path, REVIEW_SHEET)
    workbook_by_id: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(workbook_rows, start=2):
        review_id = str(row.get("复核 ID") or "").strip()
        if not review_id:
            raise ValueError(f"review workbook row {row_number} lacks 复核 ID")
        if review_id in workbook_by_id:
            raise ValueError(f"duplicate 复核 ID in workbook: {review_id}")
        workbook_by_id[review_id] = {**row, "_workbook_row": row_number}
    missing = sorted(set(queue_by_id) - set(workbook_by_id))
    extra = sorted(set(workbook_by_id) - set(queue_by_id))
    if missing or extra:
        raise ValueError(
            f"review workbook/queue mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    decisions: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    blank_default_count = 0
    ignored_note_on_pass_count = 0
    for review_id, queue_row in queue_by_id.items():
        workbook_row = workbook_by_id[review_id]
        status = str(workbook_row.get("复核状态") or "").strip().lower()
        if status != "reviewed":
            raise ValueError(
                f"review {review_id} is not completed: 复核状态={status!r}"
            )
        category = str(queue_row.get("category") or "")
        raw_decision = str(workbook_row.get("复核决定") or "").strip().upper()
        blank_defaulted = not raw_decision
        if blank_defaulted:
            if not blank_means_pass:
                raise ValueError(f"review {review_id} has an empty decision")
            try:
                effective_decision = DEFAULT_PASS_DECISIONS[category]
            except KeyError as exc:
                raise ValueError(
                    f"review {review_id} has no default pass decision for {category}"
                ) from exc
            blank_default_count += 1
        else:
            effective_decision = raw_decision
        if effective_decision not in ALLOWED_DECISIONS.get(category, set()):
            raise ValueError(
                f"review {review_id} decision {effective_decision} is invalid for {category}"
            )
        if effective_decision in BLOCKING_DECISIONS:
            raise ValueError(
                f"review {review_id} remains unresolved: {effective_decision}"
            )
        notes = str(workbook_row.get("审核备注") or "").strip()
        corrected_facts = parse_corrected_facts(
            workbook_row.get("修正 Facts（JSON）"), review_id=review_id
        )
        if effective_decision in NO_CHANGE_DECISIONS and notes:
            ignored_note_on_pass_count += 1
        decision_counts[effective_decision] += 1
        decisions.append(
            {
                "schema_version": "reference_fact_review_decision_v1",
                "review_id": review_id,
                "category": category,
                "segment_id": str(queue_row["segment_id"]),
                "review_status": "reviewed",
                "raw_review_decision": raw_decision,
                "effective_review_decision": effective_decision,
                "blank_defaulted_to_pass": blank_defaulted,
                "reviewer_notes": notes,
                "corrected_facts": corrected_facts,
                "reviewed_fact_ids": [
                    str(fact.get("reference_fact_id") or "")
                    for fact in queue_row.get("facts", [])
                ],
                "workbook_row": int(workbook_row["_workbook_row"]),
            }
        )

    decisions.sort(key=lambda item: item["workbook_row"])
    canonical_decisions = json.dumps(
        decisions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    decision_hash = hashlib.sha256(canonical_decisions.encode("utf-8")).hexdigest()
    parent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_config_hash = str(parent_manifest.get("config_hash") or "")
    reviewed_config_hash = hashlib.sha256(
        json.dumps(
            [parent_config_hash, review_version, decision_hash],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    decisions_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        decisions_by_segment[decision["segment_id"]].append(decision)

    changed_segment_ids: list[str] = []
    action_counts: Counter[str] = Counter()
    no_op_revision_count = 0
    output_rows: list[dict[str, Any]] = []
    for original_row in reference_rows:
        segment_id = str(original_row["segment_id"])
        reviewed_row, row_actions, no_ops = _apply_segment_decisions(
            original_row, decisions_by_segment.get(segment_id, []), queue_by_id
        )
        if row_actions:
            changed_segment_ids.append(segment_id)
            action_counts.update(row_actions)
        no_op_revision_count += no_ops
        reviewed_row["parent_run_id"] = str(original_row.get("run_id") or "")
        reviewed_row["parent_config_hash"] = str(original_row.get("config_hash") or "")
        reviewed_row["parent_reference_set_hash"] = str(
            original_row.get("reference_set_hash") or ""
        )
        reviewed_row["manual_review_version"] = review_version
        reviewed_row["manual_review_decision_hash"] = decision_hash
        reviewed_row["manual_review_applied"] = True
        reviewed_row["run_id"] = run_id
        reviewed_row["config_hash"] = reviewed_config_hash
        reviewed_row["prompt_version"] = (
            str(original_row.get("prompt_version") or "") + f"+{review_version}"
        )
        output_rows.append(reviewed_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    references_output = output_dir / "reference_facts.jsonl"
    manifest_output = output_dir / "manifest.json"
    decisions_output = output_dir / "manual_review_decisions.jsonl"
    summary_output = output_dir / "manual_review_application.json"
    ledger_output = output_dir / "reference_facts.sqlite3"
    export_reference_jsonl(references_output, output_rows)
    write_jsonl(decisions_output, decisions)
    _write_reviewed_ledger(ledger_output, output_rows)

    source_fact_count = sum(len(row.get("reference_facts", [])) for row in reference_rows)
    reviewed_fact_count = sum(len(row.get("reference_facts", [])) for row in output_rows)
    summary = {
        "schema_version": "reference_fact_review_application_v1",
        "complete": True,
        "review_version": review_version,
        "source_references": str(references_path.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_review_queue": str(review_queue_path.resolve()),
        "source_review_workbook": str(review_workbook_path.resolve()),
        "source_review_workbook_sha256": file_sha256(review_workbook_path),
        "manual_review_decision_hash": decision_hash,
        "review_item_count": len(decisions),
        "blank_defaulted_to_pass_count": blank_default_count,
        "ignored_note_on_pass_count": ignored_note_on_pass_count,
        "decision_counts": dict(sorted(decision_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "no_op_revision_count": no_op_revision_count,
        "changed_segment_count": len(changed_segment_ids),
        "changed_segment_ids": sorted(changed_segment_ids),
        "source_fact_count": source_fact_count,
        "reviewed_fact_count": reviewed_fact_count,
        "fact_count_delta": reviewed_fact_count - source_fact_count,
        "run_id": run_id,
        "parent_run_id": str(parent_manifest.get("run_id") or ""),
        "config_hash": reviewed_config_hash,
        "parent_config_hash": parent_config_hash,
        "references_output": str(references_output.resolve()),
        "manifest_output": str(manifest_output.resolve()),
        "decisions_output": str(decisions_output.resolve()),
        "ledger_output": str(ledger_output.resolve()),
    }
    atomic_write_json(summary_output, summary)

    manifest = deepcopy(parent_manifest)
    manifest.update(
        {
            "run_id": run_id,
            "config_hash": reviewed_config_hash,
            "segment_count": len(output_rows),
            "fact_count": reviewed_fact_count,
            "reference_facts_output": str(references_output.resolve()),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "built_this_invocation": 0,
            "skipped_as_completed": 0,
            "failed_this_invocation": 0,
            "run_complete": True,
            "run_paused": False,
            "pause_reason": "",
            "remaining_segment_count": 0,
            "unresolved_failure_count": 0,
            "manual_review": {
                "complete": True,
                "review_version": review_version,
                "parent_run_id": str(parent_manifest.get("run_id") or ""),
                "parent_config_hash": parent_config_hash,
                "source_workbook_sha256": summary[
                    "source_review_workbook_sha256"
                ],
                "decision_hash": decision_hash,
                "review_item_count": len(decisions),
                "blank_defaulted_to_pass_count": blank_default_count,
                "changed_segment_count": len(changed_segment_ids),
                "fact_count_delta": reviewed_fact_count - source_fact_count,
                "application_summary": str(summary_output.resolve()),
                "decisions_output": str(decisions_output.resolve()),
            },
        }
    )
    atomic_write_json(manifest_output, manifest)
    return summary


def finalize_completed_review_audit(
    *,
    automated_audit_path: str | Path,
    post_review_queue_path: str | Path,
    original_review_queue_path: str | Path,
    decisions_path: str | Path,
    application_summary_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Reconcile post-review audit triggers with the completed original review."""

    automated_audit_path = Path(automated_audit_path)
    post_review_queue_path = Path(post_review_queue_path)
    original_review_queue_path = Path(original_review_queue_path)
    decisions_path = Path(decisions_path)
    application_summary_path = Path(application_summary_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    automated = json.loads(automated_audit_path.read_text(encoding="utf-8"))
    application = json.loads(application_summary_path.read_text(encoding="utf-8"))
    original_queue = list(iter_jsonl(original_review_queue_path))
    post_review_queue = list(iter_jsonl(post_review_queue_path))
    decisions = list(iter_jsonl(decisions_path))
    decisions_by_id = {str(item["review_id"]): item for item in decisions}
    if len(decisions_by_id) != len(original_queue):
        raise ValueError("completed decision count does not match original review queue")

    segment_coverage: set[tuple[str, str]] = set()
    distinct_pair_coverage: set[frozenset[str]] = set()
    accepted_by_category = {
        "zero_reference_set": {"VALID_EMPTY"},
        "fact_cap_reached": {"COMPLETE", "REPLACE_LOW_VALUE"},
        "json_repair_used": {"REPAIR_SAFE"},
    }
    for item in original_queue:
        decision = decisions_by_id.get(str(item["review_id"]))
        if decision is None:
            raise ValueError(f"completed decision is missing: {item['review_id']}")
        effective = str(decision.get("effective_review_decision") or "")
        category = str(item.get("category") or "")
        if effective in accepted_by_category.get(category, set()):
            segment_coverage.add((category, str(item["segment_id"])))
        if category == "near_duplicate_pair" and effective == "DISTINCT":
            distinct_pair_coverage.add(
                frozenset(
                    str(fact.get("reference_fact_id") or "")
                    for fact in item.get("facts", [])
                )
            )

    uncovered: list[dict[str, Any]] = []
    carried_forward_counts: Counter[str] = Counter()
    for item in post_review_queue:
        category = str(item.get("category") or "")
        if category == "near_duplicate_pair":
            pair = frozenset(
                str(fact.get("reference_fact_id") or "")
                for fact in item.get("facts", [])
            )
            covered = pair in distinct_pair_coverage
        else:
            covered = (category, str(item.get("segment_id") or "")) in segment_coverage
        if covered:
            carried_forward_counts[category] += 1
        else:
            uncovered.append(
                {
                    "review_id": str(item.get("review_id") or ""),
                    "category": category,
                    "segment_id": str(item.get("segment_id") or ""),
                    "fact_ids": [
                        str(fact.get("reference_fact_id") or "")
                        for fact in item.get("facts", [])
                    ],
                }
            )

    automated_ok = int(automated.get("error_count", -1)) == 0
    application_ok = bool(application.get("complete"))
    complete = automated_ok and application_ok and not uncovered
    status = (
        "AUTOMATED_PASS_MANUAL_COMPLETE"
        if complete
        else "AUTOMATED_PASS_MANUAL_FOLLOWUP_REQUIRED"
        if automated_ok
        else "AUTOMATED_FAIL"
    )
    result = {
        "schema_version": "reference_fact_final_audit_v1",
        "status": status,
        "complete": complete,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(application.get("run_id") or ""),
        "config_hash": str(application.get("config_hash") or ""),
        "reference_row_count": int(automated.get("reference_row_count") or 0),
        "fact_count": int(automated.get("fact_count") or 0),
        "automated_error_count": int(automated.get("error_count") or 0),
        "automated_warning_count": int(automated.get("warning_count") or 0),
        "automated_warnings": automated.get("warnings", []),
        "near_duplicates": automated.get("near_duplicates", {}),
        "original_review_item_count": len(original_queue),
        "completed_decision_count": len(decisions),
        "post_review_trigger_count": len(post_review_queue),
        "carried_forward_review_count": sum(carried_forward_counts.values()),
        "carried_forward_counts": dict(sorted(carried_forward_counts.items())),
        "uncovered_review_count": len(uncovered),
        "uncovered_reviews": uncovered,
        "manual_review_application": str(application_summary_path.resolve()),
        "automated_audit": str(automated_audit_path.resolve()),
        "completed_decisions": str(decisions_path.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    final_json = output_dir / "final_audit.json"
    final_report = output_dir / "final_audit_report.md"
    atomic_write_json(final_json, result)
    final_report.write_text(_render_final_audit_report(result), encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manual_review = dict(manifest.get("manual_review") or {})
    manual_review.update(
        {
            "final_status": status,
            "final_audit_complete": complete,
            "post_review_trigger_count": len(post_review_queue),
            "carried_forward_review_count": sum(carried_forward_counts.values()),
            "uncovered_review_count": len(uncovered),
            "final_audit": str(final_json.resolve()),
            "final_audit_report": str(final_report.resolve()),
        }
    )
    manifest["manual_review"] = manual_review
    atomic_write_json(manifest_path, manifest)
    return result


def parse_corrected_facts(value: Any, *, review_id: str = "") -> list[dict[str, Any]]:
    """Accept the documented JSON array or the reviewer-friendly plain Fact format."""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        payload = list(value)
    else:
        text = str(value).strip()
        if not text or text == "[]":
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [
                {
                    "fact_text": match.group("fact_text").strip(),
                    "source_turn_ids": [
                        int(item)
                        for item in re.split(r"[,，\s]+", match.group("source_turn_ids"))
                        if item
                    ],
                    "fact_type": match.group("fact_type").lower(),
                    "state_status": match.group("state_status").lower(),
                }
                for match in _PLAIN_FACT_RE.finditer(text)
            ]
            remainder = _PLAIN_FACT_RE.sub("", text).strip()
            if not payload or remainder:
                label = f" for {review_id}" if review_id else ""
                raise ValueError(f"corrected Facts are neither JSON nor recognized text{label}")
    if not isinstance(payload, list):
        raise ValueError(f"corrected Facts must be a JSON array for {review_id}")
    return [_normalize_corrected_fact(item, review_id=review_id) for item in payload]


def read_xlsx_table(path: str | Path, sheet_name: str) -> list[dict[str, Any]]:
    """Read a simple worksheet table using only the Python standard library."""

    path = Path(path)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {
        "r": "http://schemas.openxmlformats.org/package/2006/relationships"
    }
    office_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared_strings.append(
                    "".join(node.text or "" for node in item.findall(".//m:t", ns))
                )
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_id = None
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib.get(f"{{{office_rel}}}id")
                break
        if relationship_id is None:
            raise ValueError(f"worksheet is missing: {sheet_name}")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relationship in relationships.findall("r:Relationship", rel_ns):
            if relationship.attrib.get("Id") == relationship_id:
                target = relationship.attrib.get("Target")
                break
        if not target:
            raise ValueError(f"worksheet relationship is missing: {sheet_name}")
        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        else:
            sheet_path = str(PurePosixPath("xl") / target)
        worksheet = ET.fromstring(archive.read(sheet_path))
        rows: list[dict[int, Any]] = []
        for row in worksheet.findall("m:sheetData/m:row", ns):
            values: dict[int, Any] = {}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "")
                column = _column_number(ref)
                cell_type = cell.attrib.get("t", "")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.findall(".//m:t", ns)
                    )
                else:
                    node = cell.find("m:v", ns)
                    raw = node.text if node is not None else None
                    if raw is None:
                        value = None
                    elif cell_type == "s":
                        value = shared_strings[int(raw)]
                    elif cell_type == "b":
                        value = raw == "1"
                    elif cell_type in {"str", "e"}:
                        value = raw
                    else:
                        value = _numeric_value(raw)
                values[column] = value
            rows.append(values)
    if not rows:
        return []
    header_row = rows[0]
    max_column = max(header_row, default=0)
    headers = [str(header_row.get(index) or "").strip() for index in range(1, max_column + 1)]
    missing_columns = sorted(REQUIRED_REVIEW_COLUMNS - set(headers))
    if missing_columns:
        raise ValueError(f"review worksheet lacks columns: {missing_columns}")
    result: list[dict[str, Any]] = []
    for values in rows[1:]:
        item = {header: values.get(index) for index, header in enumerate(headers, start=1) if header}
        if any(value not in (None, "") for value in item.values()):
            result.append(item)
    return result


def _apply_segment_decisions(
    original_row: dict[str, Any],
    decisions: list[dict[str, Any]],
    queue_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str], int]:
    row = deepcopy(original_row)
    segment_id = str(row["segment_id"])
    original_facts = [deepcopy(item) for item in row.get("reference_facts", [])]
    facts_by_id = {str(item["reference_fact_id"]): item for item in original_facts}
    original_rank = {
        str(item["reference_fact_id"]): int(item.get("selection_rank") or 0)
        for item in original_facts
    }
    full_replacements = [
        item for item in decisions if item["effective_review_decision"] == "REPLACE_LOW_VALUE"
    ]
    if len(full_replacements) > 1:
        raise ValueError(f"multiple full replacements for {segment_id}")
    if full_replacements:
        decision = full_replacements[0]
        if not decision["corrected_facts"]:
            raise ValueError(f"REPLACE_LOW_VALUE lacks corrected Facts: {decision['review_id']}")
        conflicting = [
            item
            for item in decisions
            if item is not decision
            and item["effective_review_decision"] not in NO_CHANGE_DECISIONS
        ]
        if conflicting:
            raise ValueError(
                f"full replacement conflicts with other edits in {segment_id}: "
                f"{[item['review_id'] for item in conflicting]}"
            )
        facts = [
            _manual_fact(item, decision, rank=index)
            for index, item in enumerate(decision["corrected_facts"], start=1)
        ]
        actions = ["replace_reference_set"]
        no_ops = 0
    else:
        removed: set[str] = set()
        appended: list[dict[str, Any]] = []
        actions = []
        no_ops = 0
        for decision in decisions:
            effective = decision["effective_review_decision"]
            queue_row = queue_by_id[decision["review_id"]]
            target_ids = [
                str(item.get("reference_fact_id") or "")
                for item in queue_row.get("facts", [])
            ]
            if effective in NO_CHANGE_DECISIONS:
                continue
            if effective == "REVISE":
                if len(target_ids) != 1:
                    raise ValueError(f"REVISE must target one Fact: {decision['review_id']}")
                target_id = target_ids[0]
                if target_id not in facts_by_id:
                    raise ValueError(f"REVISE target is missing: {target_id}")
                before_type = str(facts_by_id[target_id].get("fact_type") or "")
                before_status = str(facts_by_id[target_id].get("state_status") or "")
                if decision["corrected_facts"]:
                    removed.add(target_id)
                    for corrected in decision["corrected_facts"]:
                        appended.append(
                            _manual_fact(
                                corrected,
                                decision,
                                rank=original_rank[target_id],
                            )
                        )
                else:
                    match = _REVISION_RE.search(decision["reviewer_notes"])
                    if match is None:
                        raise ValueError(
                            f"REVISE lacks a parseable correction: {decision['review_id']}"
                        )
                    fact_type = match.group("fact_type").lower()
                    state_status = match.group("state_status")
                    revised_status = state_status.lower() if state_status else before_status
                    if (fact_type, revised_status) == (before_type, before_status):
                        no_ops += 1
                    else:
                        facts_by_id[target_id]["fact_type"] = fact_type
                        facts_by_id[target_id]["state_status"] = revised_status
                        facts_by_id[target_id]["origin"] = "manual_review"
                        facts_by_id[target_id]["grounding_reason"] = _manual_reason(decision)
                        actions.append("revise_metadata")
                continue
            if effective in {"MERGE", "DUPLICATE"}:
                if len(target_ids) < 2:
                    raise ValueError(f"{effective} requires at least two Facts: {decision['review_id']}")
                if any(target_id not in facts_by_id for target_id in target_ids):
                    raise ValueError(f"{effective} target is missing: {decision['review_id']}")
                if decision["corrected_facts"]:
                    merged = [
                        _manual_fact(
                            item,
                            decision,
                            rank=min(original_rank[target_id] for target_id in target_ids),
                        )
                        for item in decision["corrected_facts"]
                    ]
                else:
                    candidates = [facts_by_id[target_id] for target_id in target_ids]
                    selected = max(
                        candidates,
                        key=lambda fact: (
                            len(str(fact.get("fact_text") or "")),
                            -int(fact.get("selection_rank") or 0),
                        ),
                    )
                    merged_fact = deepcopy(selected)
                    merged_fact["source_turn_ids"] = sorted(
                        {
                            int(source_id)
                            for fact in candidates
                            for source_id in fact.get("source_turn_ids", [])
                        }
                    )
                    merged_fact["origin"] = "manual_review"
                    merged_fact["grounding_reason"] = _manual_reason(decision)
                    merged_fact["selection_rank"] = min(
                        int(fact.get("selection_rank") or 0) for fact in candidates
                    )
                    merged = [merged_fact]
                removed.update(target_ids)
                appended.extend(merged)
                actions.append("merge_facts")
                continue
            if effective == "REJECT":
                removed.update(target_ids)
                actions.append("reject_fact")
                continue
            if effective == "MISSING_FACTS":
                if not decision["corrected_facts"]:
                    raise ValueError(f"MISSING_FACTS lacks additions: {decision['review_id']}")
                appended.extend(
                    _manual_fact(item, decision, rank=len(original_facts) + index)
                    for index, item in enumerate(decision["corrected_facts"], start=1)
                )
                actions.append("add_missing_facts")
                continue
            if effective == "TRUNCATION_LOSS":
                raise ValueError(
                    f"TRUNCATION_LOSS needs an explicit replacement decision: {decision['review_id']}"
                )
            raise ValueError(f"unsupported review decision: {effective}")
        facts = [
            fact for fact_id, fact in facts_by_id.items() if fact_id not in removed
        ] + appended

    normalized_facts = _finalize_facts(segment_id, facts, row["segment_turn_ids"])
    row["reference_facts"] = normalized_facts
    row["frozen_fact_count"] = len(normalized_facts)
    row["grounded_accept_count"] = max(
        int(row.get("grounded_accept_count") or 0), len(normalized_facts)
    )
    row["raw_proposal_count"] = max(
        int(row.get("raw_proposal_count") or 0), row["grounded_accept_count"]
    )
    if len(normalized_facts) < 15 and full_replacements:
        row["truncated_to_k"] = False
    row["reference_set_hash"] = _reference_set_hash(normalized_facts)
    return row, actions, no_ops


def _normalize_corrected_fact(value: Any, *, review_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"corrected Fact must be an object for {review_id}")
    fact_text = str(value.get("fact_text") or value.get("text") or "").strip()
    source_turn_ids = sorted({int(item) for item in value.get("source_turn_ids", [])})
    fact_type = str(value.get("fact_type") or "").strip().lower()
    state_status = str(value.get("state_status") or "").strip().lower()
    if not fact_text or not source_turn_ids:
        raise ValueError(f"corrected Fact lacks text/source Turn for {review_id}")
    if fact_type not in FACT_TYPES:
        raise ValueError(f"invalid corrected fact_type {fact_type!r} for {review_id}")
    if state_status not in STATE_STATUSES:
        raise ValueError(f"invalid corrected state_status {state_status!r} for {review_id}")
    return {
        "fact_text": fact_text,
        "source_turn_ids": source_turn_ids,
        "fact_type": fact_type,
        "state_status": state_status,
    }


def _manual_fact(
    value: dict[str, Any], decision: dict[str, Any], *, rank: int
) -> dict[str, Any]:
    return {
        **value,
        "text": value["fact_text"],
        "origin": "manual_review",
        "grounding_reason": _manual_reason(decision),
        "selection_rank": rank,
    }


def _manual_reason(decision: dict[str, Any]) -> str:
    note = str(decision.get("reviewer_notes") or "").strip()
    suffix = f": {note}" if note else ""
    return (
        f"Human review {decision['review_id']} "
        f"({decision['effective_review_decision']}){suffix}"
    )


def _finalize_facts(
    segment_id: str, facts: Iterable[dict[str, Any]], valid_source_ids: Iterable[int]
) -> list[dict[str, Any]]:
    valid_ids = {int(item) for item in valid_source_ids}
    ordered = sorted(
        (deepcopy(item) for item in facts),
        key=lambda item: (
            int(item.get("selection_rank") or 10**9),
            str(item.get("fact_text") or ""),
        ),
    )
    if len(ordered) > 15:
        raise ValueError(f"reviewed reference set exceeds 15 Facts: {segment_id}")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for rank, fact in enumerate(ordered, start=1):
        text = str(fact.get("fact_text") or "").strip()
        source_ids = tuple(sorted({int(item) for item in fact.get("source_turn_ids", [])}))
        if not text or not source_ids or not set(source_ids).issubset(valid_ids):
            raise ValueError(f"invalid reviewed Fact in {segment_id}: {fact}")
        fact_type = str(fact.get("fact_type") or "")
        state_status = str(fact.get("state_status") or "")
        if fact_type not in FACT_TYPES or state_status not in STATE_STATUSES:
            raise ValueError(f"invalid reviewed metadata in {segment_id}: {fact}")
        key = (_normalize_fact(text), source_ids)
        if key in seen:
            raise ValueError(f"duplicate reviewed Fact in {segment_id}: {text}")
        seen.add(key)
        fact["reference_fact_id"] = _reference_fact_id(segment_id, text, source_ids)
        fact["fact_text"] = text
        fact["text"] = text
        fact["source_turn_ids"] = list(source_ids)
        fact["selection_rank"] = rank
        result.append(fact)
    return result


def _reference_fact_id(segment_id: str, text: str, source_ids: tuple[int, ...]) -> str:
    payload = json.dumps(
        [segment_id, _normalize_fact(text), list(source_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "rf_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _reference_set_hash(facts: list[dict[str, Any]]) -> str:
    payload = [
        [fact["reference_fact_id"], fact["fact_text"], fact["source_turn_ids"]]
        for fact in facts
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize_fact(text: str) -> str:
    # Keep byte-for-byte compatibility with ReferenceFactPipeline and audit.py.
    return " ".join(str(text).casefold().strip().rstrip("。.!！?").split())


def _write_reviewed_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"reviewed ledger already exists: {path}")
    with sqlite3.connect(path) as connection:
        # This derivative is written once and then consumed read-only, so a
        # self-contained DELETE-journal database is preferable to WAL sidecars.
        connection.execute("PRAGMA journal_mode=DELETE")
        for table in ("reference_fact_sets", "reference_fact_failures"):
            connection.execute(
                f'CREATE TABLE "{table}" ('
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                "dedupe_key TEXT NOT NULL UNIQUE, "
                "row_json TEXT NOT NULL)"
            )
        for row in rows:
            dedupe_key = json.dumps(
                [
                    row["run_id"],
                    row["segment_id"],
                    row["source_content_hash"],
                    row["config_hash"],
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            connection.execute(
                'INSERT INTO "reference_fact_sets"(dedupe_key, row_json) VALUES (?, ?)',
                (dedupe_key, json.dumps(row, ensure_ascii=False, sort_keys=True)),
            )


def _column_number(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if letters is None:
        raise ValueError(f"invalid cell reference: {cell_reference}")
    result = 0
    for character in letters.group(0):
        result = result * 26 + ord(character) - ord("A") + 1
    return result


def _numeric_value(raw: str) -> int | float | str:
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def _render_final_audit_report(result: dict[str, Any]) -> str:
    near_duplicates = result.get("near_duplicates", {})
    lines = [
        "# Reference Fact 人工审核完成报告",
        "",
        f"- 最终状态：`{result['status']}`",
        f"- Reference 行数：{result['reference_row_count']}",
        f"- Fact 数量：{result['fact_count']}",
        f"- 自动审计错误：{result['automated_error_count']}",
        f"- 自动审计警告：{result['automated_warning_count']}",
        f"- 原始人工复核项：{result['original_review_item_count']}",
        f"- 已完成决定：{result['completed_decision_count']}",
        f"- 复审触发项：{result['post_review_trigger_count']}",
        f"- 已由原审核覆盖：{result['carried_forward_review_count']}",
        f"- 未覆盖项：{result['uncovered_review_count']}",
        "",
        "## 近重复复审",
        "",
        f"- 是否执行：{near_duplicates.get('checked', False)}",
        f"- 阈值：{near_duplicates.get('threshold')}",
        f"- 候选对数：{near_duplicates.get('pair_count', 0)}",
        "",
    ]
    if result["complete"]:
        lines.extend(
            [
                "人工审核决定已全部应用，复审产生的触发项均可追溯到原审核决定。",
                "该 Reference Fact 版本可进入候选匹配、标签生成与监督训练阶段。",
            ]
        )
    else:
        lines.extend(
            [
                "当前版本尚未满足最终完成条件。",
                "请处理 `final_audit.json` 中的 `uncovered_reviews` 后再次运行终审。",
            ]
        )
    lines.append("")
    return "\n".join(lines)
