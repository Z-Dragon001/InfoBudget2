"""Assembly staging plus append-only route accounting."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from infobudget.rl_router.ledger import SqliteLedger
from infobudget.rl_router.qdrant_store import AssemblyResult, FactQdrantStore
from infobudget.rl_router.schemas import Tier, TopicSegment


class AssemblyManager:
    def __init__(self, store: FactQdrantStore, ledger_path: str | Path):
        self.store = store
        path = Path(ledger_path)
        legacy = path if path.suffix.lower() == ".jsonl" else None
        database = path.with_suffix(".sqlite3") if legacy else path
        self.ledger = SqliteLedger(
            database,
            "assemblies",
            ("assembly_id", "segment_id"),
            legacy_jsonl_path=legacy,
        )

    def create(
        self,
        *,
        dataset_name: str,
        split: str,
        sample_id: str,
        segments: list[TopicSegment],
        actions: list[Tier],
        probabilities: list[float] | None,
        episode_id: str,
        policy_version: str,
        router_type: str,
        candidate_extraction_run_id: str | None = None,
        route_metadata: list[dict] | None = None,
    ) -> AssemblyResult:
        result = self.store.assemble(
            dataset_name=dataset_name,
            split=split,
            sample_id=sample_id,
            segments=segments,
            actions=actions,
            episode_id=episode_id,
            policy_version=policy_version,
            extraction_run_id=candidate_extraction_run_id,
        )
        created_at = datetime.now(timezone.utc).isoformat()
        probabilities = probabilities or [1.0] * len(segments)
        route_metadata = route_metadata or [{} for _ in segments]
        if not (
            len(segments) == len(actions) == len(probabilities) == len(route_metadata)
        ):
            raise ValueError(
                "segments, actions, probabilities, and route_metadata must have equal lengths"
            )
        allowed_metadata = {
            "selected_model_id",
            "selected_profile_id",
            "predicted_quality",
            "selected_cost",
            "route_decision_id",
            "quality_checkpoint_hash",
            "budget_run_id",
            "sample_budget",
            "sample_total_selected_cost",
        }
        for order, (segment, tier, probability, metadata) in enumerate(
            zip(segments, actions, probabilities, route_metadata), start=1
        ):
            unknown = sorted(set(metadata) - allowed_metadata)
            if unknown:
                raise ValueError(f"unsupported route metadata fields: {unknown}")
            self.ledger.append(
                {
                    "assembly_id": result.assembly_id,
                    "episode_id": episode_id,
                    "sample_id": sample_id,
                    "segment_id": segment.segment_id,
                    "selected_tier": tier,
                    "action_probability": probability,
                    "router_type": router_type,
                    "policy_version": policy_version,
                    "candidate_extraction_run_id": candidate_extraction_run_id,
                    "route_order": order,
                    "status": result.status,
                    "point_count": result.point_count,
                    "created_at": created_at,
                    "cleaned_at": None,
                    **metadata,
                }
            )
        return result

    def cleanup(self, result: AssemblyResult, *, dataset_name: str, split: str) -> None:
        self.store.delete_assembly(
            dataset_name=dataset_name,
            split=split,
            sample_id=result.sample_id,
            assembly_id=result.assembly_id,
        )
