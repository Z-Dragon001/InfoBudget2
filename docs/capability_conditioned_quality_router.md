# Capability-conditioned quality router

This document is the operational contract for the primary research method. The legacy
actor-critic router is retained only as a baseline.

## Frozen formulation

For segment `d` and candidate model `m`, the scorer receives:

- a 384-dimensional normalized `all-MiniLM-L6-v2` segment embedding;
- six structural segment features fitted only on the training split;
- a seven-dimensional frozen MemoryPrint for model `m`.

The concatenated 397-dimensional feature predicts one scalar
`silver_strict_fact_f1`. It does not predict seven quality heads, use model identity as an
action class, consume QA correctness during training, or optimize an RL reward.

Because the observed theme segments are frequently longer than MiniLM's 256-wordpiece
limit, router input is split into consecutive non-overlapping token chunks. Each chunk is
encoded and normalized independently; the segment representation is their normalized mean.
This is a fixed preprocessing rule, not a tuned routing parameter. Fact and query vectors
continue to use direct encoding because they are retrieval units rather than router segment
representations.

The budget layer maximizes summed predicted quality subject to an absolute cost budget and
selects exactly one model per segment. Costs are external to the scorer and may be replaced
without retraining it.

## Required artifacts

1. `model_capabilities.json`: validated by `configs/model_capabilities.schema.json`.
2. `reference_facts.jsonl`: one frozen silver reference Fact set per segment.
3. candidate Fact JSONL or a Qdrant human-inspection export.
4. `fact_equivalence_judgments.jsonl`: fixed-Judge binary decisions for candidate/reference
   Fact pairs.
5. `fact_quality_labels.jsonl`: scalar labels built by
   `scripts/build_fact_quality_labels.py`.
6. separate train and validation label files whose sample IDs do not overlap.
7. `segment_model_costs.jsonl`: one non-negative absolute cost for each segment/model pair.

The training CLI refuses sample-level train/validation overlap. A checkpoint freezes the
embedding name/dimension, structural scaler, MemoryPrint dimension order, label hashes and
capability-file hash.

`scripts/route_with_quality_budget.py` creates D. For an offline replay,
`scripts/assemble_quality_routes.py` consumes D and copies the selected candidate tier into
the existing physical S collection. Its assembly ledger persists model/profile IDs,
predicted quality, selected cost, budget-run ID and quality-checkpoint hash. The existing
`scripts/evaluate_rl_assembly.py` then runs the unchanged Retriever, Reader and Judge.

## A/B/C/D roles

- A, `qa_retrieval_trace.jsonl`, is the question-level retrieval/Reader/Judge trace.
- B, `segment_question_usage.jsonl`, aggregates which questions used each segment.
- C, `counterfactual_segment_effects.jsonl`, stores controlled QA effects and is external
  validation only.
- D, `routing_decisions.jsonl`, stores all candidate scores, the selected model/tier, cost,
  budget, checkpoint hash and route decision ID.

C never enters the Huber training loss. Its QA delta is compared with predicted local
quality delta only through sign agreement and rank correlation.

## Qdrant migration rule

The previous BGE-M3 collections use 1024-dimensional vectors. MiniLM uses 384 dimensions,
so candidates must be re-embedded into new collections. Do not mutate an old collection's
vector size. The namespace already contains an embedding hash and therefore isolates the
new collections while preserving historical reproducibility.

`scripts/reembed_candidate_collections.py` performs this migration one sample at a time.
It reconstructs the immutable Fact payload, preserves the original extraction model,
prompt, source IDs, token allocation and extraction cost, replaces only the embedding
fields, and records the source namespace. Consequently, changing the embedding model does
not require paying for candidate Fact extraction again.

Reference facts, silver labels, MemoryPrint profiles and counterfactual results stay in
versioned JSON/JSONL artifacts. Qdrant remains the immutable retrieval store for candidate
and assembled facts.
