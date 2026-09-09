# Segment 级 Gold Fact 集合评价流程（2026-09-09）

正式协议直接比较审核后的原始 Gold Fact 集合与匿名 Candidate Fact 集合，不再拆分
Gold。来源 Turn ID 和冗余不参与评价；Gold 含准确时间时启用时间硬门槛。

## 固定路径

```bash
cd /home/zxl/VScode/InfoBudget2

SEGMENTS=datasets/segmented/locomo/full/nsp_text_tiling
GOLD_DIR=output/reference_facts/locomo_nsp_full_v2_reviewed_v2
GOLD="$GOLD_DIR/reference_facts.jsonl"
GOLD_MANIFEST="$GOLD_DIR/manifest.json"
ROOT=outputs/quality_router/locomo_qwen25_v11
CANDIDATES="$ROOT/candidate_facts.jsonl"
CANDIDATE_INVENTORY="$ROOT/candidate_inventory.json"
```

旧 `gold_evaluation_units_v1*` 仅为已停止的 Pilot，不进入正式 Judge 或训练标签。

## 1. 只读规划

```bash
uv run python scripts/judge_segment_fact_sets.py \
  --segments "$SEGMENTS" \
  --references "$GOLD" \
  --reference-manifest "$GOLD_MANIFEST" \
  --candidates "$CANDIDATES" \
  --candidate-inventory "$CANDIDATE_INVENTORY" \
  --output-dir "$ROOT/segment_fact_set_judge_v2" \
  --output "$ROOT/segment_fact_set_judgments_v2.jsonl" \
  --anonymization-seed 42 \
  --plan-only
```

## 2. 50 Segment Pilot

```bash
uv run python scripts/judge_segment_fact_sets.py \
  --segments "$SEGMENTS" \
  --references "$GOLD" \
  --reference-manifest "$GOLD_MANIFEST" \
  --candidates "$CANDIDATES" \
  --candidate-inventory "$CANDIDATE_INVENTORY" \
  --output-dir "$ROOT/segment_fact_set_judge_v2" \
  --output "$ROOT/segment_fact_set_judgments_v2.jsonl" \
  --anonymization-seed 42 \
  --max-segments 50
```

## 3. 导出 Pilot 审核表

```bash
uv run python scripts/export_segment_set_pilot_review.py \
  --segments "$SEGMENTS" \
  --references "$GOLD" \
  --candidates "$CANDIDATES" \
  --judgments "$ROOT/segment_fact_set_judgments_v2.jsonl" \
  --jsonl-output "$ROOT/segment_set_pilot_review_v2.jsonl" \
  --csv-output "$ROOT/segment_set_pilot_review_v2.csv" \
  --manifest-output "$ROOT/segment_set_pilot_review_manifest_v2.json"
```

审核范围只有语义正确性、Gold 覆盖等级和准确时间。`FULL=1`、`PARTIAL=0.5`、
`NONE/CONTRADICTED=0`；准确时间缺失或冲突时该 Gold 最终得分为 0。

## 4. 全量续跑

Pilot 通过后重复第 2 步命令并删除 `--max-segments 50`。完整 manifest 必须满足
`run_complete=true`、`completed_segment_count=1054`。

## 5. 构建训练标签

```bash
uv run python scripts/build_fact_quality_labels.py \
  --judge-decisions "$ROOT/segment_fact_set_judgments_v2.jsonl" \
  --judge-manifest "$ROOT/segment_fact_set_judge_v2/manifest.json" \
  --references "$GOLD" \
  --candidates "$CANDIDATES" \
  --capabilities <model_capabilities.json> \
  --output "$ROOT/fact_quality_labels_v2.jsonl" \
  --details-output "$ROOT/fact_quality_label_details_v2.jsonl"
```

全量预期 `1054 × 3 = 3162` 条标签。主标签 `set_quality_f2` 使用部分正确率和
部分覆盖率计算，召回权重为 2；时间硬门槛已包含在覆盖率中。
