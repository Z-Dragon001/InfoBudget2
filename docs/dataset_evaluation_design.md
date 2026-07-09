# InfoBudget Dataset & Evaluation Design

## 1. Design Scope

本设计只吸收 LightMem / BudgetMem 在以下方面的工程经验：

- 加载与解析
- 检索与评估
- 数据与结果工件的存储方式

不继承其记忆方法论、训练策略或路由算法。

## 2. Unified Dataset Model

InfoBudget 对 `LoCoMo` 和 `LongMemEval` 统一采用三层数据视图：

1. `sample`
   - 一个完整评测单元
   - 包含 `sessions`、扁平化 `dialogue`、`qa_pairs`
2. `session`
   - 一个时间片或上下文块
   - 保留 `session_id`、原始时间戳、解析后时间戳、turns
3. `question`
   - 一个评测问题
   - 保留 `question_type/category`、evidence、judge profile、是否 abstention

## 3. Dataset-Specific Normalization

### 3.1 LoCoMo

- 输入组织：一个 `sample_id` 绑定整段长期对话和多个 QA
- 处理方式：
  - 解析 `conversation.session_n` 与 `session_n_date_time`
  - 解析 `qa` 中的 `question / answer / category / evidence`
  - 将 `D1:3` 这类证据引用映射为 `session_1`
  - 保留 `event_summary / observation / session_summary`

### 3.2 LongMemEval

- 输入组织：一个 `question_id` 对应一组 `haystack_sessions`
- 处理方式：
  - 每条原始记录就是一个 `sample`
  - `sample_id = question_id`
  - `sessions = haystack_sessions`
  - `qa_pairs` 默认只有一个问题
  - 保留 `question_type / answer_session_ids / question_date`
  - `_abs` 后缀问题标记为 `is_unanswerable=true`

## 4. Processed Storage Design

每个 `dataset/split` 目录下输出四类工件：

```text
datasets/processed/{dataset}/{split}/
├── manifest.json
├── samples.jsonl
├── questions.jsonl
└── sessions.jsonl
```

说明：

- `samples.jsonl`
  - 主工件
  - 一个 sample 同时保存 dialogue 与问题
- `questions.jsonl`
  - 扁平问题视图
  - 便于做 QA 级统计、抽样、错误分析
- `sessions.jsonl`
  - 扁平 session 视图
  - 便于做 chunking、session-level retrieval、evidence alignment
- `manifest.json`
  - 数据版本、原始来源文件、样本数、问题数、session 数

## 5. Memory Build & Retrieval Design

### 5.1 Build Stage

- 输入：`DatasetDialogueExample.dialogue`
- 流程：
  - `LiteTopicSeg` 分段
  - `InformationScorer` 打分
  - `BudgetAwareRouter` 路由
  - `JointMemoryExtractor` 压缩为 memory entries
  - `MemoryStore` 写入 JSONL 审计文件 + local Qdrant collections

### 5.2 Retrieval Scope

- 默认检索范围只在当前 sample 内
- 不跨 `sample_id`
- 与 LoCoMo 的“每个 sample 独立评估”以及 LongMemEval 的“每个 question 对应一组 haystack sessions”一致

### 5.3 Retrieval Trace

每个问题都记录：

- 检索到的 `memory_id`
- 对应 `segment_id`
- summary
- evidence hit
- evidence recall@k

## 6. Evaluation Design

### 6.1 Judge Profiles

当前设计按问题类型使用可扩展 judge profile：

- `locomo_qa`
- `longmemeval_single_session`
- `longmemeval_temporal_reasoning`
- `longmemeval_knowledge_update`
- `longmemeval_preference`
- `longmemeval_abstention`
- `generic`

### 6.2 Metrics

统一输出：

- `accuracy`
- `precision`
- `recall`
- `total_cost_usd`
- `avg_cost_per_query`
- `avg_cost_per_memory`
- `total_tokens`
- `api_calls`
- `local_calls`
- `build_latency_ms`
- `qa_latency_ms`
- `router_distribution`

新增 retrieval/evidence 维度：

- `evidence_hit_rate`
- `evidence_recall_at_k`
- `avg_retrieved_memories`
- `abstention_accuracy`
- `group_accuracy`

## 7. Evaluation Result Storage

每个 `dataset/split` 输出：

```text
outputs/evaluation/{dataset}/{split}/
├── metrics.json
├── predictions.jsonl
├── retrieval_traces.jsonl
└── run_manifest.json
```

说明：

- `metrics.json`
  - 聚合指标
- `predictions.jsonl`
  - 每个问题的预测答案、judge 结果、匹配原因
- `retrieval_traces.jsonl`
  - 每个问题的检索轨迹与证据命中情况
- `run_manifest.json`
  - 本次评估配置、top-k、样本数等

## 8. Extensibility

扩展新数据集时，只需新增：

1. 一个 `Preprocessor`
2. 一个可选 `JudgeProfile`
3. 必要时新增 `category/question_type` 映射

原有：

- processed 工件布局
- loader 接口
- evaluation runner
- artifact 存储

均无需修改。
