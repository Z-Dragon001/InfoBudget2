# Segment 级 Fact 集合评价与训练标签流程（2026-09-08）

本流程取代 Candidate×Gold 严格 Pair 枚举。评价单位是一个 Segment 内某模型的
完整 Candidate Fact 集合。协议不评价来源 Turn ID，也不评价冗余；Gold 中的明确
时间是严格覆盖的硬门槛。

## 固定输入

```bash
cd /home/zxl/VScode/InfoBudget2

SEGMENTS=datasets/segmented/locomo/full/nsp_text_tiling
GOLD=output/reference_facts/locomo_nsp_full_v2_reviewed_v2/reference_facts.jsonl
CANDIDATES=outputs/quality_router/locomo_qwen25_v11/candidate_facts.jsonl
CANDIDATE_INVENTORY=outputs/quality_router/locomo_qwen25_v11/candidate_inventory.json
ROOT=outputs/quality_router/locomo_qwen25_v11
```

不要再使用以下旧产物作为标签输入：

- `fact_equivalence_pairs*.jsonl`
- `fact_relation_pairs*.jsonl`
- `fact_relation_judgments_v2*.jsonl`
- `fact_relation_judgments_v3*.jsonl`
- `excluded_pair_review_queue.jsonl`

它们可以保留为历史实验记录，但不属于 v1 Segment 集合评价协议。

## 1. 只读成本规划

```bash
uv run python scripts/build_gold_evaluation_units.py \
  --references "$GOLD" \
  --output-dir "$ROOT/gold_evaluation_units_v1" \
  --output "$ROOT/gold_evaluation_units_v1.jsonl" \
  --plan-only
```

`paid_api_called` 必须是 `false`。该步骤不会调用 API。

## 2. Gold 评价单元 Pilot

```bash
uv run python scripts/build_gold_evaluation_units.py \
  --references "$GOLD" \
  --output-dir "$ROOT/gold_evaluation_units_v1" \
  --output "$ROOT/gold_evaluation_units_v1.jsonl" \
  --max-segments 50
```

`--max-segments 50` 使用按 conversation 轮询的确定性抽样，而不是只取第一个
conversation 的前 50 个片段。

重点人工检查：复合事实拆分是否遗漏限定词、Gold 没有明确时间时是否错误增加时间、
Gold 有明确日期时 `required_time` 是否正确。确认后重复同一命令并去掉
`--max-segments`，程序从 SQLite 断点继续，而不是重跑已完成 Segment。

全量完成后检查：

```bash
python -m json.tool "$ROOT/gold_evaluation_units_v1/manifest.json"
wc -l "$ROOT/gold_evaluation_units_v1.jsonl"
```

必须满足 `status=complete`、`run_complete=true`、`completed_segment_count=1054`。

Pilot 阶段先导出 Gold claim/time 审核表：

```bash
uv run python scripts/export_gold_evaluation_unit_review.py \
  --references "$GOLD" \
  --gold-units "$ROOT/gold_evaluation_units_v1.jsonl" \
  --jsonl-output "$ROOT/gold_evaluation_units_review_v1.jsonl" \
  --csv-output "$ROOT/gold_evaluation_units_review_v1.csv" \
  --manifest-output "$ROOT/gold_evaluation_units_review_manifest_v1.json"
```

审核 `claim_units_readable` 和 `required_times_readable`。空白
`review_status` 表示尚未审核，不表示自动通过。

## 3. Segment 集合 Judge 只读规划

```bash
uv run python scripts/judge_segment_fact_sets.py \
  --segments "$SEGMENTS" \
  --gold-units "$ROOT/gold_evaluation_units_v1.jsonl" \
  --gold-units-manifest "$ROOT/gold_evaluation_units_v1/manifest.json" \
  --candidates "$CANDIDATES" \
  --candidate-inventory "$CANDIDATE_INVENTORY" \
  --output-dir "$ROOT/segment_fact_set_judge_v1" \
  --output "$ROOT/segment_fact_set_judgments_v1.jsonl" \
  --anonymization-seed 42 \
  --plan-only
```

每次调用只包含一个 Segment，但同时包含匿名且打乱顺序的三个 Candidate 集合。
模型名称不会进入提示词，真实 Set→模型映射保存在判断结果中。

## 4. Segment 集合 Judge Pilot

```bash
uv run python scripts/judge_segment_fact_sets.py \
  --segments "$SEGMENTS" \
  --gold-units "$ROOT/gold_evaluation_units_v1.jsonl" \
  --gold-units-manifest "$ROOT/gold_evaluation_units_v1/manifest.json" \
  --candidates "$CANDIDATES" \
  --candidate-inventory "$CANDIDATE_INVENTORY" \
  --output-dir "$ROOT/segment_fact_set_judge_v1" \
  --output "$ROOT/segment_fact_set_judgments_v1.jsonl" \
  --anonymization-seed 42 \
  --max-segments 50
```

Pilot 人工审核至少覆盖：所有时间失败项，以及随机抽取的语义覆盖项。不得使用来源
Turn ID 或重复数量推翻模型判断。若修改提示词，必须换新输出目录，禁止在旧 SQLite
上续跑。

导出方便审核的 JSONL 和 Excel 可直接打开的 UTF-8 CSV：

```bash
uv run python scripts/export_segment_set_pilot_review.py \
  --segments "$SEGMENTS" \
  --gold-units "$ROOT/gold_evaluation_units_v1.jsonl" \
  --candidates "$CANDIDATES" \
  --judgments "$ROOT/segment_fact_set_judgments_v1.jsonl" \
  --jsonl-output "$ROOT/segment_set_pilot_review_v1.jsonl" \
  --csv-output "$ROOT/segment_set_pilot_review_v1.csv" \
  --manifest-output "$ROOT/segment_set_pilot_review_manifest_v1.json"
```

Pilot 通过后去掉 `--max-segments`，使用完全相同的输入、提示词和 seed 续跑全量。

## 5. 生成训练前标签

只有集合 Judge manifest 完整时，标签构建器才允许运行：

```bash
uv run python scripts/build_fact_quality_labels.py \
  --judge-decisions "$ROOT/segment_fact_set_judgments_v1.jsonl" \
  --judge-manifest "$ROOT/segment_fact_set_judge_v1/manifest.json" \
  --gold-units "$ROOT/gold_evaluation_units_v1.jsonl" \
  --candidates "$CANDIDATES" \
  --capabilities <model_capabilities.json> \
  --output "$ROOT/fact_quality_labels_v1.jsonl" \
  --details-output "$ROOT/fact_quality_label_details_v1.jsonl"
```

主要训练标签是 `set_quality_f2`。它由严格 Candidate 正确率和严格 Gold claim
召回率计算，召回权重为 2。时间缺失已经在 claim 召回中作为硬失败处理，不再重复
扣分。标签同时保留以下诊断字段：

- `strict_candidate_precision`
- `strict_claim_recall`
- `strict_gold_fact_recall`
- `temporal_recall`
- `soft_candidate_precision`
- `soft_claim_recall`

预期全量标签数量为 `1054 × 3 = 3162`。最终训练前还必须按 conversation 划分
训练、验证和测试，禁止同一 `sample_id` 跨分区。

## 不变量

- 更换 Gold、Candidate、提示词、Judge 模型或匿名 seed 时必须创建新运行目录。
- `manifest.json` 中的输入 SHA-256 与实际文件不一致时，程序拒绝续跑或构建标签。
- Candidate 缺少 Gold 要求的准确时间时，相关 Gold claim 不计覆盖，该 Candidate
  不能标为 `SUPPORTED`。
- 无时间 Gold 的 Candidate 不因缺少日期而失败。
- Pair 级旧输出永远不能传给新的标签构建器。
