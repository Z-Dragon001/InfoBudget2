# Frozen Reference Fact Pipeline

该目录独立实现 LoCoMo 与 LongMemEval 的强参考 Fact 构建，不修改候选模型的 Fact 提取逻辑。最终的 `reference_facts.jsonl` 可直接作为现有 `scripts/build_fact_quality_labels.py --references` 的输入。

## 设计边界

每个主题段依次经过：

1. 非候选模型使用与小/中/大候选提取器相同的语义 Fact Policy，进行高召回初始提取；
2. 固定 Grounding Judge 逐条检查蕴含性、原子性、来源充分性、外部推断和重复；
3. 仅根据“原主题段 + 已接受参考 Fact”进行一次覆盖补全；
4. 新 Fact 再次经过同一个 Grounding Judge；
5. 确定性去重和排序，最多冻结 K 条（默认 K=15，与候选 Fact 上限一致）。

模型输入只暴露一基的规范 `SOURCE_TURN_ID`。分段文件 `text` 中原有的零基编号仅用于加载时验证 `legacy_source_id + 1 == SOURCE_TURN_ID`，渲染提示词时会被移除，避免同一行出现两套编号。模型响应若不是合法 JSON，流水线会先归档原始响应，再执行一次只修复 JSON 结构、不新增或改写 Fact 的修复请求。

参考提取器看不到候选 Fact、问题、Gold answer、QA 正误或下游路由结果，避免参考标签向任一候选模型或任务问题泄漏。参考流水线会直接读取 `configs/prompts/locomo_memory_extraction.txt` 或 `longmemeval_memory_extraction.txt` 中从 `Mandatory processing procedure` 到 `Output contract` 之前的完整语义规则，并注入初始提取、覆盖补全和 Grounding 三类提示词。由此 Gold 与候选共享 Fact 范围、粒度、时间更新、证据规则和15条最终优先级，而不共享候选的输出格式和路由元数据。该共享 Policy 也进入有效配置哈希，修改候选语义提示词后旧 Gold 结果不会被断点续跑误用。

默认模型角色来自项目的 `configs/models.yaml`：初始提取和覆盖补全使用 Cloudflare 上的 `gold_fact_extractor=openai/gpt-5.6-luna`，Grounding 使用 `judge_llm`。后端路由会让 Gold 提取走 `/ai/v1/responses`，同时让 Judge 继续走 `/chat/completions`。程序会验证这些角色的实际模型不等于 `small/medium/large` 的候选模型。

Cloudflare 调用前需要在项目 `.env` 中配置：

```dotenv
CLOUDFLARE_ACCOUNT_ID=你的Cloudflare账户ID
CLOUDFLARE_API_TOKEN=具有Workers AI权限的Token
JUDGE_MODEL_API_KEY=固定Grounding Judge的API Key
```

`gold_fact_extractor.api_base_url` 使用 `{account_id}` 占位符，客户端只在运行时从 `CLOUDFLARE_ACCOUNT_ID` 替换，不会把账户信息写进配置或产物。Cloudflare 模型目录没有公开当前账户的实际价格快照，因此项目没有伪造 `prices.yaml` 条目：token 会正常记录，但相关阶段写入 `cost_status=unknown_missing_price_snapshot`，manifest 标记 `cost_complete=false`。从 Cloudflare Dashboard 获得实际价格后，可在 `configs/prices.yaml` 为 `openai/gpt-5.6-luna` 增加价格快照，费用将自动恢复为完整统计。

Gold 专用 Cloudflare 客户端默认在每次 HTTP 请求前保持至少 3 秒间隔。只有当 HTTP 402 响应正文明确包含 `Wholesale rate limit exceeded` 时，才按 60、120、300 秒依次长退避并自动重试当前请求；普通的付款类 402 不会被误当成限流。三次退避后仍持续受限时，CLI 进入 300 秒熔断等待并自动重试同一个主题段，默认最多自动恢复两轮；恢复成功后继续后续未完成段。两轮后仍受限才暂停本次 campaign，不再让后续上千段快速失败。manifest 会记录 `circuit_pause_count`、`run_paused`、`pause_reason` 和 `remaining_segment_count`。之后使用相同 `--run-id --resume`，成功段会被跳过，失败段和未尝试段会继续处理。

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

运行时进度写入 stderr，最终 manifest JSON 单独写入 stdout，因此仍可安全重定向机器可读结果。交互终端显示单行实时进度条；非交互日志按事件逐行输出。进度包含当前主题段和阶段、已构建/跳过/失败数、API 调用数、累计输入/输出 token，以及 Luna 请求间隔、402长退避和 campaign 熔断等待时长。例如：

```text
Gold Fact: |####--------------------| 180/1054 segments elapsed=... eta=... item=... stage=initial_grounding built=175 skipped=5 failed=0 calls=594 input_tok=... output_tok=...
```

需要的 API key 仍从项目 `.env` 读取。默认角色对应 `CLOUDFLARE_ACCOUNT_ID`、`CLOUDFLARE_API_TOKEN` 和 `JUDGE_MODEL_API_KEY`。

输出目录包括：

- `reference_facts.jsonl`：冻结参考集合，兼容现有质量标签脚本；
- `reference_facts.sqlite3`：幂等、可恢复的逐主题段 ledger；
- `raw_responses/<run-id>/`：每阶段完整 prompt、模型原始响应、token 与重试审计；
- `manifest.json`：数据集、模型角色、配置哈希、Fact 数量、全运行 input/output/total token、按模型角色聚合的调用/token 数量和总成本。

同一个 SQLite 文件还包含 `reference_fact_failures` 表。单个主题段在 JSON 修复后仍失败或 API 调用最终失败时，会记录异常类型、错误信息、传输尝试和尝试次数，然后继续处理后续主题段。`manifest.json` 通过 `unresolved_failure_count`、`failed_segment_ids` 和 `run_complete` 明确标记产物是否完整；使用相同 `--run-id --resume` 可重试失败段。

每个主题段的 `stage_usage` 分别记录初始提取、初始 Grounding、覆盖补全和覆盖 Grounding 的 `input_tokens`、`output_tokens`、`total_tokens` 与 `usage_source`；主题段顶层和运行 manifest 还提供聚合值。Cloudflare 返回官方 usage 时标记为 `provider`，缺失时使用本地 tokenizer 估算并计入 `estimated_usage_stage_count`，不会把估算值伪装成供应商计费值。

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
