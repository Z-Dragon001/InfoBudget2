# RL Router MVP implementation

## Current-to-target mapping

| Pipeline stage | Current implementation |
|---|---|
| YAML loader and model registry | `infobudget.config` plus strict `rl_router.config` validation for five roles, price snapshots, prompts, and local embeddings |
| LoCoMo/LongMemEval preprocessors | processed-v3 samples, sessions, turns, questions, hashes, synthetic timestamps, one-time image captions |
| Leakage-safe split manifests | LoCoMo conversation CV plus LongMemEval grouped stratification, evidence-component isolation, fixed/CV manifests, source hashes, and partition audits |
| Lite/BERT segmenters | sample-isolated `segments.jsonl`, trace, manifest, NSP and fine-tuned BERT-MLP TextTiling |
| Fact extraction | one shared factual prompt, strict multi-source JSON parser, independently runnable extraction tiers, candidate generator, and cost ledgers |
| Vector storage | four namespaced L/M/H/S collections with mandatory sample and assembly filters |
| Routing | leakage-safe structural features, frozen embedding + MLP policy/value heads, baselines, and constrained actor-critic |
| QA evaluation | external reader/judge prompts and S-only `AssemblyEvaluator` |

Fact extraction uses separate audited prompt contracts for LoCoMo and LongMemEval. Each
campaign pins the selected dataset prompt role, semantic version, and SHA-256 so changes to
one dataset prompt do not invalidate campaigns for the other dataset.

The repository contains only the fact-only RL pipeline. Relation, preference, constraint,
episode, consolidation, fixed-percentile routing, and legacy memory paths have been removed.

## Enforced invariants

- A segment is the routing and enqueue unit; buffers never mix samples.
- A batch is the provider usage/cost unit. Largest-remainder allocation conserves all
  integer input and output tokens.
- The model returns JSON with `processed_segment_ids` and `data[]` entries containing only
  `segment_id`, `source_ids`, and `fact`. Legacy single `source_id` responses remain readable.
  IDs, timestamps, remaining payload fields, costs,
  and vectors are code-generated.
- Small, medium, and large use the same prompt, JSON protocol, maximum of 15 facts per
  segment, output reservation, and buffer limits. The main experiment therefore changes
  model and provider price without confounding them with prompt density or batch size.
- Buffer admission is token-first. `max_segments=6` is a secondary guard for many short
  segments; every candidate addition is still checked against input and total token limits.
- `max_input_tokens` is the ordinary multi-segment batching threshold. With
  `allow_oversize_singleton=true`, a segment above that threshold first flushes the current
  buffer and runs alone when its rendered input plus reserved output still fits
  `max_total_context_tokens`.
- A logical segment that cannot fit the safe singleton total-context budget is
  deterministically tail-truncated to the largest prefix that fits every selected tier.
  It retains its original segment ID, source-content hash, turn IDs/range, timestamps, and
  fact ownership and still uses one model call. Only a temporary `visible_source_ids`
  allow-list is derived from the truncated text and accepted by the parser. Visible/dropped
  source IDs, retained/dropped lengths, and before/after token counts are audited in the
  manifest; truncation metadata is also written to the segment-cost ledger and fact payload.
  Cost remains the actual provider usage of this single request.
- L/M/H candidates are generated once. Rollouts copy the selected points into a physical S
  assembly; they never call extraction models again.
- Every candidate operation requires dataset, split, and sample. Every S operation also
  requires `assembly_id`.
- QA queries only S. Human-readable exports contain no vectors and are never an input.
- Full-data extraction is grouped by an immutable campaign manifest. Training accepts only
  a complete campaign whose sample runs share one scope hash and whose aggregate empty-fact,
  15-fact saturation, schema-repair, and failed-batch rates pass configured thresholds.
- Qdrant collection namespaces include the configured model family and the first 12
  hexadecimal characters of the local memory-embedding directory hash. Existing collection
  vector size and distance are checked before any paid extraction request.
- Formal embedding/tokenizer adapters use local paths and fail if files are missing; there
  is no hashing fallback in the RL path.
- Router features accept only segment text and structural counts. Questions, answers,
  evidence, and judge labels cannot enter the feature builder API.
- Checkpoints contain MLP weights and the fitted training-split scaler together.

## Outputs

Candidate generation writes Qdrant points plus cross-process-safe `batches`,
`segment_costs`, and `failures` tables in `candidate_ledger.sqlite3`. Attempt audit rows
live in each run's `run_ledger.sqlite3`. Assembly, QA, validation, and training ledgers also
use SQLite WAL databases. Legacy JSONL extraction ledgers are imported read-only on first
resume. Every processed segment has one `segment_costs` row per extracted tier, including
`fact_count=0`/`status=no_fact` rows that cannot be represented by a Qdrant Fact point.
Training writes best/final checkpoints alongside its SQLite ledger.

Candidate Qdrant persistence is a batch-scoped replace: delete every Point matching
dataset/split/sample/run/batch in the selected tier, upsert the current response, and audit
the resulting count before committing SQLite state. This removes stale Points after a
crash/retry. `scripts/reconcile_extraction_run.py` performs a read-only manifest/SQLite/
Qdrant reconciliation. Router training runs the same check for every campaign-pinned run
before loading the router models or issuing any paid QA call.
Candidate and selected final memories can be exported under `human_readable/` for manual
inspection only.

## What requires external resources

The repository does not commit datasets, model weights, provider tokenizers, API keys, or
Qdrant experiment data. Local model directories are ignored by Git. Consequently tests use
injected encoders and an in-memory Qdrant instance. Full LoCoMo/LongMemEval accuracy,
multi-budget Pareto curves, and provider cost totals require the frozen local resources and
audited price snapshots described in the requirements.

The imported legacy working tree contained plaintext provider credentials. They were
removed from all project configuration. Those credentials should be revoked and replaced
through environment variables before any external run.
