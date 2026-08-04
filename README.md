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

Models and datasets are deliberately never downloaded by training or evaluation. Place
them under the paths documented in `docs/hardware_and_environment_setup.md`. Configure API
keys only through the environment variables named in `configs/models.yaml`.

Formal LoCoMo and LongMemEval runs share one Qdrant server at
`http://127.0.0.1:6333`, with dataset-specific collection namespaces. Start it before the
pipeline; the Linux Docker Compose deployment and lifecycle commands are documented in
`docs/qdrant_server_deployment.md`. Candidate and S memories are still exported as local,
per-sample JSON for human inspection, but training and evaluation read only Qdrant.

## Pipeline

```powershell
uv run python scripts/preprocess_datasets.py --datasets locomo longmemeval
uv run python scripts/segment_datasets.py locomo full --method nsp_text_tiling
uv run python scripts/segment_datasets.py locomo full --method bert_mlp_text_tiling
uv run python scripts/build_longmemeval_splits.py
uv run python scripts/manage_extraction_campaign.py init --campaign-id locomo_full_nsp_text_tiling --dataset locomo --split full --method nsp_text_tiling --run-prefix locomo_full
Get-ChildItem datasets/segmented/locomo/full/nsp_text_tiling/samples/*/segments.jsonl | ForEach-Object { $runId = "locomo_full_nsp_text_tiling_$($_.Directory.Name)"; uv run python scripts/build_rl_candidates.py $_.FullName --extraction-run-id $runId --campaign-id locomo_full_nsp_text_tiling }
pwsh -File scripts/build_longmemeval_rl_candidates.ps1 -Method nsp_text_tiling
uv run python scripts/assemble_rl_baseline.py datasets/segmented/locomo/full/nsp_text_tiling/samples/<sample_id>/segments.jsonl --policy all-small
uv run python scripts/train_rl_router.py locomo full --method nsp_text_tiling --campaign-id locomo_full_nsp_text_tiling --split-manifest datasets/splits/locomo/cv5_seed42.json --fold 0 --epochs 3 --steps-per-sample 10 --device auto
uv run python scripts/evaluate_rl_assembly.py datasets/segmented/locomo/full/nsp_text_tiling/samples/<test_sample_id>/segments.jsonl --assembly-id <assembly_id> --split-manifest datasets/splits/locomo/cv5_seed42.json --fold 0 --partition test
uv run python scripts/train_rl_router.py longmemeval full --method nsp_text_tiling --campaign-id longmemeval_full_nsp_text_tiling --split-manifest datasets/splits/longmemeval/fixed_80_10_10_seed42_nsp_text_tiling.json --fold 0 --epochs 10 --early-stopping-patience 3
```

For a resumable candidate run, assign a stable run ID and reuse the exact same segments
file when recovering:

```powershell
uv run python scripts/build_rl_candidates.py <segments.jsonl> --extraction-run-id <run_id>
uv run python scripts/build_rl_candidates.py <segments.jsonl> --resume <run_id>
uv run python scripts/build_rl_candidates.py <segments.jsonl> --resume <run_id> --retry-terminal
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

`--resume` skips committed batches and retries only recoverable work. Terminal schema
failures remain skipped unless `--retry-terminal` is explicit. State, raw request/response
archives, attempt-level cost ledgers, run-scoped exports, and `manifest.json` are stored
under `outputs/rl_router/runs/<run_id>/`. No authorization header or API key is archived.

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
