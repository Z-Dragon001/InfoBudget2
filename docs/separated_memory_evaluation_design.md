# Separated Memory Build and Evaluation Design

InfoBudget follows a LightMem-style two-stage experiment flow:

```text
Step 1: Build memory collections only.
Step 2: Evaluate QA using existing memory collections only.
```

The build stage never runs QA. The evaluation stage never calls the memory
extractor. This lets one persisted memory collection be reused for different
retrieval, top-k, judge, or QA settings.

## Memory Storage

Memory collections are stored by dataset, split, scoring mode, extraction mode,
and sample:

```text
outputs/memory/{dataset}/{split}/{scoring_mode}/{extraction_mode}/{sample_id}/
|-- memory_jsonl/
|   |-- memory_entries.jsonl
|   |-- segments.jsonl
|   `-- cost_logs.jsonl
`-- qdrant/
    `-- local Qdrant collection files
```

`memory_jsonl/` is kept as a human-readable audit mirror. Retrieval uses the
local Qdrant collections under `qdrant/`.

Each scoring mode has its own memory folder. The 9 supported routing-score modes
are:

```text
entropy_only
lexical_density_only
entity_density_only
concept_density_only
information_gain_only
actionability_only
intrinsic_only
utility_only
full
```

The memory build stage also writes:

```text
outputs/memory/{dataset}/{split}/{scoring_mode}/{extraction_mode}/build_manifest.json
```

`extraction_mode` is either:

```text
flat   # default; one LLM call, factual memories only
event  # two LLM calls, factual + relational memories, no summary layer
```

## Evaluation Storage

Evaluation artifacts are separated by dataset, split, scoring mode, and
extraction mode:

```text
outputs/evaluation/{dataset}/{split}/{scoring_mode}/{extraction_mode}/
|-- metrics.json
|-- predictions.jsonl
|-- retrieval_traces.jsonl
`-- run_manifest.json
```

## Commands

Build one LoCoMo sample:

```bash
python scripts/build_dataset_memory.py --datasets locomo --splits full --scoring-modes full --extraction-mode flat --limit 1
```

Evaluate the previously built memory store:

```bash
python scripts/evaluate_dataset_memory.py --datasets locomo --splits full --scoring-modes full --extraction-mode flat --limit 1
```

Build or evaluate all 9 routing-score variants:

```bash
python scripts/build_dataset_memory.py --datasets locomo --splits full --all-scoring-modes
python scripts/evaluate_dataset_memory.py --datasets locomo --splits full --all-scoring-modes
```

Run event extraction mode:

```bash
python scripts/build_dataset_memory.py --datasets locomo --splits full --scoring-modes full --extraction-mode event
python scripts/evaluate_dataset_memory.py --datasets locomo --splits full --scoring-modes full --extraction-mode event
```

The legacy command remains available for quick smoke tests:

```bash
python scripts/run_dataset_evaluation.py --datasets locomo --splits full --scoring-mode full --limit 1
```

That command internally builds memories and then evaluates them. For formal
experiments, use the two separated scripts above.
