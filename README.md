# InfoBudget

InfoBudget is a Python 3.12 research implementation for budget-aware long-term-memory
extraction. The current MVP standardizes LoCoMo/LongMemEval conversations, performs local
BERT topic segmentation, generates immutable small/medium/large fact candidates, builds
real Qdrant strategy assemblies, and trains an Embedding + MLP router against QA quality
and virtual deployment cost.

## Setup

```powershell
uv python install 3.12
uv sync --frozen --group dev --python 3.12
uv run pytest
```

Create the local credential file once. It is ignored by Git and loaded automatically by
all project configuration entry points; variables already exported by the operating system
take precedence:

```powershell
Copy-Item .env.example .env
```

On Linux:

```bash
cp .env.example .env
chmod 600 .env
```

Fill only the API keys required by the selected command. Keep the `api_key_env` values in
`configs/models.yaml` unchanged because they are environment-variable names, not secrets.

Models and datasets are deliberately never downloaded by training or evaluation. Place
them under the paths documented in `docs/hardware_and_environment_setup.md`. Configure API
keys only through the environment variables named in `configs/models.yaml`.

Formal LoCoMo and LongMemEval runs share one Qdrant server at
`http://127.0.0.1:6333`, with model-family- and dataset-specific collection namespaces.
Set `model_family` in the selected config directory to match its Qwen or Llama extraction
roles. Start the server before the
pipeline; the Linux Docker Compose deployment and lifecycle commands are documented in
`docs/qdrant_server_deployment.md`. Candidate and S memories are still exported as local,
per-sample JSON for human inspection, but training and evaluation read only Qdrant.

## Pipeline

```powershell
uv run python scripts/preprocess_datasets.py --datasets locomo
uv run python scripts/preprocess_datasets.py --datasets longmemeval
uv run python scripts/segment_datasets.py locomo full --method nsp_text_tiling --alpha 0.5
uv run python scripts/segment_datasets.py longmemeval full --method nsp_text_tiling --alpha 0.5
& .\scripts\build_locomo_rl_candidates.ps1 -Method nsp_text_tiling -Alpha 0.5
& .\scripts\build_longmemeval_rl_candidates.ps1 -Method nsp_text_tiling -Alpha 0.5
uv run python scripts/run_routed_cv_experiment.py locomo full --method nsp_text_tiling_alpha_0p5 --campaign-id locomo_full_nsp_text_tiling_alpha_0p5 --split-manifest datasets/splits/locomo/cv5_seed42.json --epochs 3 --steps-per-sample 10
```

Parameterized segmentation artifacts are isolated under names such as
`nsp_text_tiling_alpha_0p5`; training and full-experiment outputs add an `epochs_<N>`
directory. Routed test extraction and QA/judging can be run independently with
`evaluate_routed_deployment.py --stage extract` and `--stage qa`. See
`docs/locomo_alpha_epoch_experiment_guide.md` for the five-stage commands and audit rules.

For a resumable candidate run, assign a stable run ID and reuse the exact same segments
file when recovering:

```powershell
uv run python scripts/build_rl_candidates.py <segments.jsonl> --extraction-run-id <run_id>
uv run python scripts/build_rl_candidates.py <segments.jsonl> --resume <run_id>
uv run python scripts/build_rl_candidates.py <segments.jsonl> --resume <run_id> --retry-terminal
uv run python scripts/build_rl_candidates.py <segments.jsonl> --resume <run_id> --tier medium --recover-terminal-with-singletons
uv run python scripts/reconcile_extraction_run.py <run_id>
```

Candidate tiers can also run as independent processes while sharing one immutable run ID.
Create the run with the first tier, then resume it for the remaining tiers:

```powershell
uv run python scripts/build_rl_candidates.py <segments.jsonl> --extraction-run-id <run_id> --tier small
uv run python scripts/build_rl_candidates.py <segments.jsonl> --resume <run_id> --tier medium
uv run python scripts/build_rl_candidates.py <segments.jsonl> --resume <run_id> --tier large
```

The run manifest remains `partial` until all three tiers are committed. Each invocation
loads and validates only its selected tier credentials and tokenizer. Repeat `--tier` to
run a subset together; omitting it preserves the original all-tier behavior. API keys are
validated before segment loading, local model loading, Qdrant initialization, or run-file
creation.

The LongMemEval PowerShell scheduler processes all 500 samples with deterministic
per-sample run IDs and resumes existing manifests automatically. Qdrant server holds the
global L/M/H/S collections and indexed sample/run filters. Useful controls include `-Tier small`, `-StartIndex`,
`-Limit`, `-RetryTerminal`, `-ContinueOnError`, and `-DryRun`.

The scheduler also creates and refreshes an immutable extraction campaign manifest. Router
training requires `--campaign-id` and refuses campaigns with incomplete sample runs,
mixed scope hashes, or aggregate empty/saturated/repair/failure rates above the configured
quality gates.

Candidate buffers use the same limits for all three tiers: at most six topic segments per
batch and at most fifteen facts per logical segment. `max_input_tokens` is the normal
multi-segment batching threshold, not a hard single-segment cutoff. A segment above that
threshold runs alone when its input plus reserved output still fits
`max_total_context_tokens`. A segment that cannot fit this singleton budget is
deterministically tail-truncated to the largest prefix that fits every selected tier. It
retains the original `segment_id`, source hash, turn range, timestamps, and fact ownership,
and still uses one model call. Only the temporary `visible_source_ids` parser allow-list is
derived from the truncated text. The run manifest, segment-cost ledger, and fact payload
record visible and dropped source IDs for audit. Cost uses only that call's provider usage.

LoCoMo and LongMemEval use separate, dataset-specific fact-extraction prompts. LoCoMo
prioritizes directly supported personal background, relationships, experiences, plans,
preferences, and image-grounded context. LongMemEval prioritizes cross-session entities,
temporal evidence, knowledge updates, preferences, and evidence-preserving abstention. A
campaign freezes only the prompt role, version, and SHA-256 selected for its own dataset;
changing that prompt requires a new campaign and new extraction runs for that dataset.

`--resume` skips committed batches and retries only recoverable work. Terminal schema
failures remain skipped unless an explicit recovery mode is selected. `--retry-terminal`
repeats the failed parent request. `--recover-terminal-with-singletons` instead preserves
the cached parent response, directly extracts every Topic in that failed parent as a
singleton with one read-only preceding turn, and commits only after all children validate.
The parent cost is allocated by each Topic's content-token weight; every singleton's actual
provider usage is added to its Topic. Existing singleton children are never overwritten;
use `--recover-cached-singleton` only for the separately constrained, zero-call cached-ID
normalization path. State, raw request/response archives, attempt-level cost ledgers,
run-scoped exports, and `manifest.json` are stored under
`outputs/rl_router/runs/<run_id>/`. No authorization header or API key is archived.

For LoCoMo, `full` is the frozen ten-conversation preprocessing/candidate source, not a
router-training partition. The five-fold manifest under `datasets/splits/locomo` selects
exactly eight training conversations for each run; the other two are held out. Existing
processed, question, session, turn, segmentation, and candidate files remain under `full`
and must not be copied or rewritten per fold.

LongMemEval follows the same physical-data rule for all 500 question/history samples. Its
committed manifests provide a fixed 400/50/50 split and a grouped, nested five-fold
360/40/100 protocol. They stratify by question type, abstention status, and frozen NSP
segment-count quartile while grouping all occurrences of evidence-bearing sessions and
answerable/abstention counterparts. Background-only distractor sessions may cross
partitions and their observed overlaps are disclosed in each fold's audit block.

Router training requires only the `QA_READER_API_KEY` and `JUDGE_MODEL_API_KEY` model
credentials because L/M/H extraction has already finished. It discovers every sample under
the manifest's training partition, fits one leakage-safe feature scaler over those training
conversations only, uses the campaign-pinned candidate extraction run per sample, and trains one shared
router. Both datasets require `--split-manifest` and `--fold`; manual `--sample-id`
selection is rejected when a manifest is active. For LongMemEval, validation routing is
deterministic and update-free after each epoch; `best.pt` is selected by validation QA
score under the configured budget, with `--early-stopping-patience` controlling stopping.

The configured budget applies to normalized extraction cost, where All-Small is 0 and
All-Large is 1. Raw virtual USD cost and normalized cost are both retained in
the `episodes` table in `training_ledger.sqlite3`. Checkpoints are written to the training run's `checkpoints/best.pt` and
`checkpoints/final.pt`.

QA evaluation uses the external LightMEM prompts verbatim. LoCoMo uses its JSON
`CORRECT`/`WRONG` judge, while LongMemEval routes single-session, temporal reasoning,
knowledge update, preference, abstention, and fallback questions to distinct Yes/No judges.
Per-question retrieval, answer, judge response, token usage, latency, and reader/judge costs
are written under the sample's SQLite QA ledger.

`infobudget.rl_router.experiment.RLExperimentTrainer` is the complete training API. It
creates an immutable `assembly_id` for every rollout, evaluates questions only against S,
replays deployment buffers for virtual cost, updates the constrained actor-critic, and
saves best/final checkpoints with the paired feature scaler.

See `docs/rl_router_implementation.md` for module mapping, invariants, and operational
limits.

## Capability-conditioned quality router (primary research path)

The primary method is now a supervised scalar scorer rather than a fixed three-class
actor-critic. It concatenates a 384-dimensional
`sentence-transformers/all-MiniLM-L6-v2` segment embedding, six leakage-safe structural
features, and the frozen seven-dimensional MemoryPrint. The resulting 397-dimensional
input predicts one `silver_strict_fact_f1` value for each `(segment, model)` pair. A
deterministic multiple-choice budget optimizer then selects exactly one model per segment.
The previous RL implementation remains available as an experimental baseline.

MiniLM accepts at most 256 wordpieces per call. Router segment text therefore uses a fixed
non-overlapping chunking rule, mean-pools the normalized chunk vectors, and normalizes the
result again. Memory Fact and retrieval-query encoding retain the normal truncate behavior;
all paths still produce exactly 384 dimensions.

Download the pinned local embedding model once:

```powershell
uv run python scripts/download_local_models.py --role router
```

Existing candidate Fact text can be reused without repeating small/medium/large extraction.
Copy it from an old 1024-dimensional namespace and re-embed it into the new namespace:

```powershell
uv run python scripts/reembed_candidate_collections.py <one-sample-segments.jsonl> `
  --source-namespace <old-qdrant-namespace> `
  --source-vector-size 1024 `
  --extraction-run-id <candidate-extraction-run-id> `
  --manifest <reembedding-manifest.json>
```

The quality-router artifact sequence is:

```powershell
# Requires frozen reference facts, candidate Fact exports, MemoryPrint profiles,
# and pairwise equivalent/non-equivalent decisions from the fixed Judge.
uv run python scripts/build_fact_quality_labels.py `
  --segments <segmented-root> `
  --references <reference_facts.jsonl> `
  --candidates <candidate_facts.jsonl> `
  --capabilities <model_capabilities.json> `
  --judge-decisions <fact_equivalence_judgments.jsonl> `
  --output <fact_quality_labels.jsonl>

uv run python scripts/train_quality_router.py `
  --segments <segmented-root> `
  --train-labels <train_labels.jsonl> `
  --validation-labels <validation_labels.jsonl> `
  --capabilities <model_capabilities.json> `
  --output-dir <quality-training-output>

uv run python scripts/route_with_quality_budget.py `
  --segments <test-segment-root> `
  --capabilities <model_capabilities.json> `
  --costs <segment_model_costs.jsonl> `
  --checkpoint <quality-training-output/quality_scorer.pt> `
  --budget <absolute-usd-budget> `
  --output <routing_decisions.jsonl>

uv run python scripts/assemble_quality_routes.py <one-sample-segments.jsonl> `
  --decisions <routing_decisions.jsonl> `
  --extraction-run-id <candidate-extraction-run-id>
```

`qa_retrieval_trace.jsonl` can be aggregated into segment usage artifact B with
`scripts/aggregate_quality_validation.py usage`. Prejoined counterfactual rows in C are
evaluated with the `consistency` command using sign agreement and Spearman correlation;
the local-quality and QA-delta scales are never directly subtracted.

### Local quality-gap routing without a sample-level budget

The independent `infobudget.quality_gap_router` package reuses the trained scalar quality
scorer but replaces the multiple-choice budget optimizer with an epsilon-noninferiority
decision. For each segment it selects the cheapest candidate whose predicted Strict
Fact-F1 is within the validation-selected `epsilon` of the predicted best candidate.
QA/Reader/Judge signals are not used by this decision path.

Calibrate `epsilon` and the optional conservative pairwise-gap residual bound on validation
artifacts only:

```powershell
uv run python scripts/calibrate_quality_gap.py `
  --predictions <validation_predictions.jsonl> `
  --labels <validation_fact_quality_labels.jsonl> `
  --costs <validation_segment_model_costs.jsonl> `
  --output <quality_gap_calibration.json> `
  --sweep-output <quality_gap_validation_sweep.jsonl>
```

Route held-out or deployment segments with the frozen scorer and calibration artifact:

```powershell
uv run python scripts/route_with_quality_gap.py `
  --segments <test-segment-root> `
  --capabilities <model_capabilities.json> `
  --costs <test_segment_model_costs.jsonl> `
  --checkpoint <quality-training-output/quality_scorer.pt> `
  --calibration <quality_gap_calibration.json> `
  --output <quality_gap_routing_decisions.jsonl>
```

Evaluate realized held-out Fact quality, regret, noninferiority violations, model selection
counts, and cost saving against the highest-cost candidate:

```powershell
uv run python scripts/evaluate_quality_gap_router.py `
  --predictions <test_predictions.jsonl> `
  --labels <test_fact_quality_labels.jsonl> `
  --costs <test_segment_model_costs.jsonl> `
  --calibration <quality_gap_calibration.json> `
  --output <quality_gap_test_metrics.json> `
  --rows-output <quality_gap_test_rows.jsonl>
```

The full method, validation discipline, uncertainty rule, safeguards, and experimental
boundaries are documented in `docs/基于质量差距的局部模型路由完整方案.md`.

The 384-dimensional embedding changes the Qdrant vector schema. Existing 1024-dimensional
BGE-M3 collections remain historical artifacts and must not be reused. Collection names
include the embedding directory hash, so freshly extracted MiniLM candidates receive a
separate namespace automatically.
