# Deferred memory-update design

## Status and boundary

This document specifies the future memory-update module. It is intentionally not part of
the current implementation. Candidate extraction, router training, and evaluation must not
interpret this document as authorization to mutate the frozen L/M/H candidate collections.

The existing collection roles remain:

- L/M/H: immutable candidate snapshots generated for router training;
- S: immutable-per-assembly rollout/evaluation copies, removable after an episode;
- D (future): canonical deployed memory revisions after the trained router selects an
  extractor tier.

Update processing belongs only to D. Training data must remain immutable so the three
extractor actions stay comparable and experiments remain reproducible.

## Incoming event and candidate discovery

Every extracted deployment fact first becomes an append-only incoming event. The event is
not an active memory yet. It contains an idempotency key, user/sample scope, event time,
router and extractor versions, fact text, embedding, and complete multi-source provenance.

For each event, search only earlier active D revisions under the same user/sample scope.
Similarity produces `update_candidates`; it never decides the mutation by itself. Each
candidate link records point ID, stable memory ID, revision, similarity score, and retrieval
configuration. A decision model then emits exactly one action:

- `add`: the event represents a new durable memory;
- `update`: the event corrects or enriches one or more existing memories;
- `delete`: newer evidence retracts an existing memory;
- `ignore`: duplicate, irrelevant, or insufficient evidence.

The decision, prompt/model version, candidate list, and raw response belong in an
append-only update ledger even when no Qdrant mutation occurs.

## Revision schema

Use a stable logical `memory_id` across revisions and a distinct Qdrant Point ID for every
revision. A future D payload should include at least:

```json
{
  "memory_id": "stable logical ID",
  "revision": 3,
  "operation": "add or update",
  "status": "pending, active, superseded, or deleted",
  "supersedes_point_ids": ["older point ID"],
  "superseded_by_point_id": "newer point ID",
  "valid_from": "event time",
  "valid_to": null,
  "created_at": "write time",
  "updated_at": "write time",
  "fact_text": "canonical current memory",
  "source_ids": [3, 7],
  "source_turn_ids": [4, 8],
  "source_provenance": [],
  "source_point_ids": [],
  "update_candidates": [],
  "update_decision_run_id": "auditable decision ID",
  "router_version": "...",
  "extractor_model": "...",
  "embedding_model_hash": "..."
}
```

Changing `fact_text` always requires recomputing the vector. Updating only the payload text
would leave retrieval semantics inconsistent with the stored vector.

## Operation semantics

### Add

Create a new `memory_id`, revision 1, and an active D Point. The idempotency key prevents a
retried extraction event from creating a duplicate logical memory.

### Update

Create a new Point with the same `memory_id`, incremented revision, merged provenance, and
a freshly computed vector. Do not overwrite the old revision. Mark the old revision as
superseded and link both directions. Multiple old points may be superseded when consolidation
merges memories.

### Delete

Prefer a tombstone revision or `status=deleted` over immediate physical deletion. Retrieval
filters out non-active revisions, while audits and historical evaluation remain possible.
Physical garbage collection is a separate maintenance operation.

### Ignore

Write only the update-decision ledger entry. Do not create or mutate a D Point.

## Crash consistency

Qdrant does not provide a transaction spanning the new revision and every superseded Point.
Use the existing project pattern of a SQLite state machine plus append-only JSONL audit:

1. create an update transaction in `planned` state;
2. write the new revision as `pending`;
3. mark old revisions `superseded`;
4. activate the new revision;
5. mark the transaction `committed`;
6. reconcile unfinished transactions on restart.

Serialize updates per user/sample. Retrieval must filter `status=active` and pin a
`memory_snapshot_id` or validity interval for reproducible evaluation.

## Evaluation requirements

Knowledge-update evaluation must query a frozen D snapshot and must not retrieve superseded
or deleted revisions. Reports should separate extraction cost, update-decision cost, and QA
reader/judge cost. The initial implementation should add offline parity tests against fixed
ADD/UPDATE/DELETE/IGNORE fixtures before enabling the module in any benchmark run.
