# Frozen Reference Fact Pipeline

该目录独立实现 LoCoMo 与 LongMemEval 的强参考 Fact 构建，不修改候选模型的 Fact 提取逻辑。最终的 `reference_facts.jsonl` 可直接作为现有 `scripts/build_fact_quality_labels.py --references` 的输入。

## 设计边界

每个主题段依次经过：

1. 非候选模型的高召回初始提取；
2. 固定 Grounding Judge 逐条检查蕴含性、原子性、来源充分性、外部推断和重复；
3. 仅根据“原主题段 + 已接受参考 Fact”进行一次覆盖补全；
4. 新 Fact 再次经过同一个 Grounding Judge；
5. 确定性去重和排序，最多冻结 K 条（默认 K=15，与候选 Fact 上限一致）。

参考提取器看不到候选 Fact、问题、Gold answer、QA 正误或下游路由结果，避免参考标签向任一候选模型或任务问题泄漏。默认模型角色来自项目的 `configs/models.yaml`：提取使用 `qa_reader`，Grounding 使用 `judge_llm`；程序会验证这些角色的实际模型不等于 `small/medium/large` 的候选模型。若论文实验改用更强参考模型，只需在项目模型配置中增加/修改角色，然后编辑本目录的 `config.yaml`。

## 构建冻结参考 Fact

LoCoMo 示例：

```powershell
.\.venv\Scripts\python.exe -m reference_fact_pipeline.cli `
  --dataset locomo `
  --segments datasets\segmented\locomo\full\nsp_text_tiling `
  --project-config-dir configs `
  --pipeline-config reference_fact_pipeline\config.yaml `
  --output-dir output\reference_facts\locomo `
  --run-id locomo_reference_v1
```

LongMemEval 只需把 `--dataset` 和 `--segments` 改成对应数据集。小规模连通性验证可添加 `--limit 2`；中断后使用相同 `--run-id --resume`，SQLite ledger 会跳过已完成且内容哈希、配置哈希一致的主题段。

需要的 API key 仍从项目 `.env` 读取。默认角色对应 `QA_READER_API_KEY` 和 `JUDGE_MODEL_API_KEY`。

输出目录包括：

- `reference_facts.jsonl`：冻结参考集合，兼容现有质量标签脚本；
- `reference_facts.sqlite3`：幂等、可恢复的逐主题段 ledger；
- `raw_responses/<run-id>/`：每阶段完整 prompt、模型原始响应、token 与重试审计；
- `manifest.json`：数据集、模型角色、配置哈希、Fact 数量和总成本。

每条冻结 Fact 保留 `reference_fact_id`、文本、`source_turn_ids`、类型、时态状态、初始/补全来源、Grounding 理由和选择次序。Fact ID 由主题段 ID、规范化文本与证据 ID 生成；集合哈希由最终冻结列表生成，因此相同输入与相同被接受内容可跨运行稳定比较。

## 候选 Fact 对比与完整指标

正式实验应提供冻结的语义等价 Judge 结果：

```powershell
.\.venv\Scripts\python.exe -m reference_fact_pipeline.compare_cli `
  --references output\reference_facts\locomo\reference_facts.jsonl `
  --candidates <candidate-facts.jsonl-or-memory.json> `
  --judge-decisions <candidate-reference-pairs.jsonl> `
  --beta 1.0 `
  --output output\reference_facts\locomo\fact_metrics.jsonl
```

仅用于程序连通性测试时，可用 `--allow-exact-baseline` 替代 Judge；论文主结果不应把字符串精确匹配当作语义等价判断。

匹配采用最大一对一二分图匹配，一个候选 Fact 或参考 Fact 最多贡献一次 TP。来源为空或越出主题段 `segment_turn_ids` 的候选 Fact 不参与匹配并计为 FP。指标实现位于 `metrics.py`：

- `precision = TP / (TP + FP)`；
- `recall = TP / (TP + FN)`；
- `F1 = 2TP / (2TP + FP + FN)`；
- `Fβ = (1 + β²)TP / ((1 + β²)TP + β²FN + FP)`；
- `Jaccard = TP / (TP + FP + FN)`；
- `false_discovery_rate = FP / (TP + FP)`；
- `false_negative_rate = FN / (TP + FN)`；
- `exact_set_match = 1` 当且仅当 `FP=FN=0`；
- `source_validity_rate = 来源合法候选数 / 候选总数`。

开放式 Fact 提取没有可枚举的 true-negative 全集，因此不报告 accuracy 或 specificity。空集约定是：两侧均为空时集合匹配为 1；仅参考集为空时 recall 为 1（没有遗漏），但所有预测均为 FP，因此 precision、F1 与 Fβ 为 0。

## 与现有训练链路衔接

构建参考 Fact 后，现有标量监督标签仍可这样生成：

```powershell
.\.venv\Scripts\python.exe scripts\build_fact_quality_labels.py `
  --segments <segments-root> `
  --references output\reference_facts\locomo\reference_facts.jsonl `
  --candidates <candidate-facts> `
  --capabilities <memoryprint.json> `
  --judge-decisions <candidate-reference-pairs.jsonl> `
  --output <quality-labels.jsonl>
```

路由训练的主监督量仍是单一 `silver_strict_fact_f1`；precision、recall 和本目录输出的其他指标用于数据审计、消融和错误分析，不作为额外优化目标，从而避免增加路由训练复杂度。

## 数据集特定规则

- LoCoMo：关注具名说话人的身份、状态、事件、偏好、目标和关系；保留时间更新；图片描述只有在主题段文本中出现时才可作为证据；跳过寒暄、未回答的问题和空泛赞美。
- LongMemEval：除用户长期信息外，可保留具有后续复用价值的核心 assistant answer、精确实体/数字和知识更新；跳过礼貌语、重复通用建议和无依据结论；模型拒答本身不能推出“真实答案未知”。

Gold QA evidence 不参与参考 Fact 生成。如需论文审计，可在冻结之后单独统计 QA evidence turn 是否被至少一条参考 Fact 覆盖；该统计只能用于发现遗漏并触发人工审计，不能反向修改同一版冻结标签。
