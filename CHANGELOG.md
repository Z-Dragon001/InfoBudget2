# Changelog

## 2026-08-02

- Implemented the fact-only reinforcement-learning memory router described by the current
  design documents: processed-v3 datasets, two formal BERT/TextTiling segmenters, tiered
  extraction buffers, strict fact parsing, cost allocation, Qdrant L/M/H/S assemblies,
  BGE-M3 embeddings, constrained actor-critic routing, baselines, manifests, and tests.
- Added the explicit local-model downloader and pinned BAAI/bge-m3 revision. Training and
  evaluation remain local-only and never download models implicitly.
- Removed the superseded joint-memory extractor, relation/event schemas, fixed-percentile
  router, legacy scoring, memory/retrieval/evaluation stacks, hashing/MiniLM fallback,
  non-formal segmentation methods, obsolete scripts/prompts/tests, and their configuration.
- Reduced the public configuration to preprocessing, formal segmentation, five API model
  roles, price snapshots, local embeddings, and the RL-router experiment configuration.
- Verified all retained modules import successfully and the focused suite passes.
