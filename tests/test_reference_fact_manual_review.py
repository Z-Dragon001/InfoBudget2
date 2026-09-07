from __future__ import annotations

import json
import sqlite3
import zipfile
from html import escape
from pathlib import Path

from reference_fact_pipeline.manual_review import (
    _reference_fact_id,
    _reference_set_hash,
    apply_completed_review,
    finalize_completed_review_audit,
    read_xlsx_table,
)


def _fact(segment_id: str, text: str, source_ids: list[int], rank: int) -> dict:
    return {
        "reference_fact_id": _reference_fact_id(segment_id, text, tuple(source_ids)),
        "fact_text": text,
        "text": text,
        "source_turn_ids": source_ids,
        "fact_type": "state",
        "state_status": "current",
        "origin": "initial",
        "grounding_reason": "Supported by the source.",
        "selection_rank": rank,
    }


def _reference_row(segment_id: str, facts: list[dict], turn_ids: list[int]) -> dict:
    return {
        "schema_version": "frozen_reference_fact_v1",
        "dataset_name": "locomo",
        "dataset": "locomo",
        "split": "full",
        "sample_id": "conv-1",
        "session_id": "session-1",
        "segment_id": segment_id,
        "segment_order": int(segment_id[-1]),
        "segmentation_method": "nsp_text_tiling",
        "segmentation_version": "nsp_text_tiling_v1",
        "source_content_hash": f"source-{segment_id}",
        "segment_turn_ids": turn_ids,
        "reference_set_hash": _reference_set_hash(facts),
        "reference_facts": facts,
        "rejected_facts": [],
        "raw_proposal_count": len(facts),
        "grounded_accept_count": len(facts),
        "frozen_fact_count": len(facts),
        "truncated_to_k": False,
        "run_id": "parent-run",
        "prompt_version": "prompt-v1",
        "config_hash": "parent-config",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_review_xlsx(path: Path, rows: list[list[str]]) -> None:
    all_rows = [
        ["复核状态", "复核决定", "复核 ID", "审核备注", "修正 Facts（JSON）"],
        *rows,
    ]
    xml_rows = []
    for row_number, row in enumerate(all_rows, start=1):
        cells = []
        for column_number, value in enumerate(row, start=1):
            column = chr(ord("A") + column_number - 1)
            cells.append(
                f'<c r="{column}{row_number}" t="inlineStr"><is><t xml:space="preserve">'
                f"{escape(value)}"
                "</t></is></c>"
            )
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="复核任务" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_read_xlsx_table_and_apply_completed_review(tmp_path: Path) -> None:
    seg1 = "seg-1"
    accepted = _fact(seg1, "Alice likes tea.", [1], 1)
    revised = _fact(seg1, "Alice visited Rome.", [2], 2)
    merge_short = _fact(seg1, "Alice likes swimming.", [3], 3)
    merge_long = _fact(seg1, "Swimming is Alice's favorite activity.", [4], 4)
    seg2 = "seg-2"
    old_a = _fact(seg2, "A low-value image detail.", [5], 1)
    old_b = _fact(seg2, "Another low-value image detail.", [6], 2)
    reference_rows = [
        _reference_row(seg1, [accepted, revised, merge_short, merge_long], [1, 2, 3, 4]),
        _reference_row(seg2, [old_a, old_b], [5, 6]),
    ]
    queue = [
        {
            "review_id": "review-pass",
            "category": "stratified_quality_sample",
            "segment_id": seg1,
            "facts": [accepted],
        },
        {
            "review_id": "review-revise",
            "category": "stratified_quality_sample",
            "segment_id": seg1,
            "facts": [revised],
        },
        {
            "review_id": "review-merge",
            "category": "near_duplicate_pair",
            "segment_id": seg1,
            "facts": [merge_short, merge_long],
        },
        {
            "review_id": "review-replace",
            "category": "fact_cap_reached",
            "segment_id": seg2,
            "facts": [old_a, old_b],
        },
    ]
    references_path = tmp_path / "reference_facts.jsonl"
    queue_path = tmp_path / "manual_review_queue.jsonl"
    manifest_path = tmp_path / "manifest.json"
    workbook_path = tmp_path / "review.xlsx"
    output_dir = tmp_path / "reviewed"
    _write_jsonl(references_path, reference_rows)
    _write_jsonl(queue_path, queue)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "parent-run",
                "config_hash": "parent-config",
                "run_complete": True,
                "unresolved_failure_count": 0,
                "segment_count": 2,
                "fact_count": 6,
                "cost_complete": True,
            }
        ),
        encoding="utf-8",
    )
    corrected = "[event/historical] Alice completed a marathon.\n来源 Turn: 5"
    _write_review_xlsx(
        workbook_path,
        [
            ["reviewed", "", "review-pass", "", "[]"],
            ["reviewed", "REVISE", "review-revise", "应该是event/historical", "[]"],
            ["reviewed", "MERGE", "review-merge", "", "[]"],
            ["reviewed", "REPLACE_LOW_VALUE", "review-replace", "", corrected],
        ],
    )

    assert len(read_xlsx_table(workbook_path, "复核任务")) == 4
    summary = apply_completed_review(
        references_path=references_path,
        manifest_path=manifest_path,
        review_queue_path=queue_path,
        review_workbook_path=workbook_path,
        output_dir=output_dir,
        run_id="reviewed-run",
        blank_means_pass=True,
    )

    assert summary["complete"] is True
    assert summary["blank_defaulted_to_pass_count"] == 1
    assert summary["action_counts"] == {
        "merge_facts": 1,
        "replace_reference_set": 1,
        "revise_metadata": 1,
    }
    assert summary["fact_count_delta"] == -2
    output_rows = [
        json.loads(line)
        for line in (output_dir / "reference_facts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    first = output_rows[0]
    assert first["frozen_fact_count"] == 3
    accepted_output = next(
        fact for fact in first["reference_facts"] if fact["fact_text"] == "Alice likes tea."
    )
    assert accepted_output["reference_fact_id"] == accepted["reference_fact_id"]
    revised_output = next(
        fact for fact in first["reference_facts"] if fact["fact_text"] == "Alice visited Rome."
    )
    assert (revised_output["fact_type"], revised_output["state_status"]) == (
        "event",
        "historical",
    )
    merged = next(
        fact
        for fact in first["reference_facts"]
        if fact["fact_text"] == "Swimming is Alice's favorite activity."
    )
    assert merged["source_turn_ids"] == [3, 4]
    second = output_rows[1]
    assert [fact["fact_text"] for fact in second["reference_facts"]] == [
        "Alice completed a marathon."
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manual_review"]["complete"] is True
    assert manifest["fact_count"] == 4
    with sqlite3.connect(output_dir / "reference_facts.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM reference_fact_sets"
        ).fetchone()[0] == 2


def test_finalize_completed_review_audit_carries_forward_review(tmp_path: Path) -> None:
    original_queue = [
        {
            "review_id": "review-zero",
            "category": "zero_reference_set",
            "segment_id": "seg-1",
            "facts": [],
        },
        {
            "review_id": "review-pair",
            "category": "near_duplicate_pair",
            "segment_id": "seg-2",
            "facts": [
                {"reference_fact_id": "rf-a"},
                {"reference_fact_id": "rf-b"},
            ],
        },
    ]
    decisions = [
        {"review_id": "review-zero", "effective_review_decision": "VALID_EMPTY"},
        {"review_id": "review-pair", "effective_review_decision": "DISTINCT"},
    ]
    original_queue_path = tmp_path / "original.jsonl"
    post_queue_path = tmp_path / "post.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(original_queue_path, original_queue)
    _write_jsonl(post_queue_path, original_queue)
    _write_jsonl(decisions_path, decisions)
    automated_path = tmp_path / "automated.json"
    automated_path.write_text(
        json.dumps(
            {
                "reference_row_count": 2,
                "fact_count": 2,
                "error_count": 0,
                "warning_count": 0,
                "warnings": [],
                "near_duplicates": {"checked": True, "threshold": 0.9, "pair_count": 1},
            }
        ),
        encoding="utf-8",
    )
    application_path = tmp_path / "application.json"
    application_path.write_text(
        json.dumps({"complete": True, "run_id": "reviewed", "config_hash": "hash"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"manual_review": {"complete": True}}), encoding="utf-8")

    result = finalize_completed_review_audit(
        automated_audit_path=automated_path,
        post_review_queue_path=post_queue_path,
        original_review_queue_path=original_queue_path,
        decisions_path=decisions_path,
        application_summary_path=application_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "final",
    )

    assert result["status"] == "AUTOMATED_PASS_MANUAL_COMPLETE"
    assert result["carried_forward_review_count"] == 2
    assert result["uncovered_review_count"] == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manual_review"]["final_audit_complete"] is True
