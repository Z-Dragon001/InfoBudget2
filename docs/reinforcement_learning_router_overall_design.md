# 强化学习路由整体方案

# 长期记忆 Fact 提取、三级缓冲、Qdrant 候选记忆与强化学习路由设计说明

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Mode：实验与系统设计
- Status：DESIGN SPEC / UNVERIFIED
- Version：rl_router_fact_qdrant_mlp_v3
- Date：2026-08-02
- Code Changes：无，本文件仅作为后续代码设计与实现提示词

## 一、文档用途与实现边界

本文档用于指导后续对 InfoBudget 项目进行代码分析、架构设计和代码修改。目标是在长期记忆写入流程中实现一个三级模型路由器，根据主题段的内容和复杂度，将其分配给 small、medium 或 large 记忆提取模型，并通过端到端 QA 质量和记忆提取成本训练路由策略。

本版本采用严格的 MVP 范围：

1. 记忆类型只实现 `fact`；
2. 不实现 relation、preference、constraint、episode、consolidation 和知识图谱；
3. small、medium、large 模型只负责返回主要事实；
4. 完整 payload、时间字段、来源字段、成本字段、向量和 Qdrant Point 全部由代码生成；
5. 训练前在 Qdrant 中建立三个候选 Collection：L、M、H；
6. 每次路由 rollout 在 Qdrant 的 S Collection 中实际组装一套带 `assembly_id` 的记忆；
7. Qdrant 是 fact、向量和检索 payload 的正式数据源；批次、成本与路由账本使用 SQLite WAL 持久化；
8. 同步导出的 JSON 只供人工查看，实验和检索不得读取 JSON；
9. 预处理和分割不在本文档中重复展开，只依赖既有详细设计文档。

预处理和 BERT 主题分割的完整规范参见：

```text
docs/data_preprocessing_and_bert_segmentation_design.md
```

后续实现前，必须先检查当前项目的配置加载、模型注册、提示词加载、记忆存储、成本日志、检索、QA reader、judge 和训练入口，并输出“当前实现 → 目标模块”的对应关系。不要进行与本方案无关的大规模重构。

## 二、系统总目标

完整流程：

```text
原始会话
  ↓
预处理与主题分割
  ↓
按 sample 得到稳定 Topic Segments
  ↓
训练候选库生成阶段：
每个 segment 同时进入 small / medium / large 三个 buffer
  ↓
三个 buffer 分别批量调用对应模型
  ↓
将每个 sample 的候选 facts 写入 Qdrant 的 L / M / H Collections
  ↓
强化学习路由器为该 sample 的每个 segment 选择一个 tier
  ↓
从 L/M/H 复制所选 Points，实际组装带 assembly_id 的 S Collection 数据
  ↓
只从 S Collection 按 sample_id + assembly_id 过滤并检索 Top-K
  ↓
固定 QA reader 生成回答
  ↓
固定 judge LLM 判定正确性
  ↓
计算 QA 质量与模拟部署提取成本
  ↓
更新路由策略
```

训练完成后的部署流程：

```text
新 Topic Segment
  ↓
本地 embedding + 结构特征
  ↓
训练好的路由器选择 small / medium / large
  ↓
进入对应 tier buffer
  ↓
buffer 满足 flush 条件后批量提取 facts
  ↓
代码拼装 Point 并写入正式 Qdrant Collection
```

优化目标：

$$
\max_\theta E[Q(S)]
$$

满足成本约束：

$$
E[C(S)]\le B
$$

其中：

- $\theta$：路由器参数；
- $Q(S)$：从实际 Qdrant S Collection 检索后得到的端到端 QA 质量；
- $C(S)$：按照当前路由动作和 buffer 规则模拟出的部署提取成本；
- $B$：给定预算。

## 三、上游输入约定

本文档不重新定义预处理和分割算法。强化学习流程只读取已经冻结的分割结果：

```text
datasets/segmented/{dataset}/{split}/{segmentation_method}/samples/{sample_id}/segments.jsonl
```

每条 segment 至少包含：

```text
dataset_name
split
sample_id
session_id
segment_id
segmentation_method
segmentation_version
start_turn
end_turn
turn_ids
start_timestamp
end_timestamp
text
token_count
source_content_hash
```

其中 `text` 已使用标准格式：

```text
[2023-05-08T13:56:00.000, Mon] 0.Caroline: 对话文本
[2023-05-08T13:56:01.000, Mon] 1.Melanie: 对话文本
```

LoCoMo 图片描述已经在预处理阶段追加到对话末尾，记忆提取阶段不得重复追加。

候选库生成、路由训练、S 库组装和 QA 评估都必须保持 sample 隔离，不允许跨 sample 读取 segment、memory 或 question。

## 四、推荐目录结构

```text
InfoBudget/
├── configs/
│   ├── config.yaml
│   ├── models.yaml
│   ├── prices.yaml
│   ├── embeddings.yaml
│   ├── rl_router.yaml
│   └── prompts/
│       ├── locomo_memory_extraction.txt
│       └── longmemeval_memory_extraction.txt
│       ├── locomo_answer.txt
│       ├── longmemeval_answer.txt
│       ├── judge_locomo.txt
│       └── judge_longmemeval.txt
│
├── models/
│   ├── embeddings/
│   │   └── bge-m3/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       └── ...
│   ├── tokenizers/
│   │   ├── small/
│   │   ├── medium/
│   │   ├── large/
│   │   ├── qa_reader/
│   │   └── judge_llm/
│   └── router/
│       ├── checkpoints/
│       └── future_adapters/       # 预留，MVP 不启用
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── segmented/
│
└── outputs/
    └── rl_router/
        ├── runs/                   # run manifest 与按 sample 的人工查看导出
        └── {dataset}/
            └── {split}/
                └── {segmentation_method}/
                    ├── experiment_manifest.json
                    ├── qdrant_collections.json
                    ├── training/
                    │   ├── checkpoints/
                    │   ├── training_ledger.sqlite3
                    │   ├── metrics.jsonl
                    │   └── best_policy.json
                    └── samples/
                        └── {sample_id}/
                            ├── human_readable/
                            │   ├── L_memories.json
                            │   ├── M_memories.json
                            │   ├── H_memories.json
                            │   └── S_memories.json
                            ├── extraction/
                            │   └── candidate_ledger.sqlite3
                            └── routing/
                                ├── current_route.json
                                ├── best_route.json
                                └── ledger.sqlite3
```

目录职责：

- `configs`：所有可修改配置和外部提示词；
- `models/embeddings`：正式实验使用的本地 embedding 模型；
- `models/tokenizers`：成本估计和请求构造使用的本地 tokenizer；
- `models/router/checkpoints`：当前 Embedding + MLP 路由器权重；
- `datasets/segmented`：冻结的按 sample 分割结果；
- `deploy/qdrant/storage`：本机 Qdrant server 的持久化目录，由 Docker Compose 挂载；
- `qdrant_collections.json`：Collection 名称、向量配置、payload index 和版本清单；
- `human_readable`：仅供人工检查的 JSON，不参与实验；
- `extraction`：批次、成本与失败日志；
- `routing`：当前和最佳路由决策。

## 五、模型配置

### 5.1 配置原则

small、medium、large、QA reader 和 judge LLM 必须全部在 `configs/models.yaml` 中配置。业务代码不得硬编码模型名称、API 地址、上下文长度或 tokenizer。

价格全部配置在 `configs/prices.yaml`。模型角色与提示词映射、buffer、Qdrant、训练和评估参数配置在 `configs/config.yaml` 或独立的 `configs/rl_router.yaml`。

API 密钥禁止以明文写入版本控制中的 YAML。配置文件只保存环境变量名：

```yaml
api_key_env: "SMALL_MODEL_API_KEY"
```

运行时只从环境变量读取。日志、manifest 和错误信息不得输出 API key。

### 5.2 `configs/models.yaml` 建议结构

为了兼容现有扁平模型注册方式，第一版建议使用以下模型键：

```yaml
models:
  small:
    deploy: "api"
    backend: "openai_compatible"
    model_name: "configured-small-model"
    request_model_name: "provider-small-model-name"
    tokenizer_name: "configured-small-tokenizer"
    tokenizer_local_path: "./models/tokenizers/small"
    api_base_url: "https://provider.example/v1"
    api_key_env: "SMALL_MODEL_API_KEY"
    max_context_tokens: 32768
    max_output_tokens: 8192
    tensor_parallel_size: 1
    dtype: "n/a"

  medium:
    deploy: "api"
    backend: "openai_compatible"
    model_name: "configured-medium-model"
    request_model_name: "provider-medium-model-name"
    tokenizer_name: "configured-medium-tokenizer"
    tokenizer_local_path: "./models/tokenizers/medium"
    api_base_url: "https://provider.example/v1"
    api_key_env: "MEDIUM_MODEL_API_KEY"
    max_context_tokens: 262144
    max_output_tokens: 16384
    tensor_parallel_size: 1
    dtype: "n/a"

  large:
    deploy: "api"
    backend: "openai_compatible"
    model_name: "configured-large-model"
    request_model_name: "provider-large-model-name"
    tokenizer_name: "configured-large-tokenizer"
    tokenizer_local_path: "./models/tokenizers/large"
    api_base_url: "https://provider.example/v1"
    api_key_env: "LARGE_MODEL_API_KEY"
    max_context_tokens: 131072
    max_output_tokens: 32000
    tensor_parallel_size: 1
    dtype: "n/a"

  qa_reader:
    deploy: "api"
    backend: "openai_compatible"
    model_name: "configured-reader-model"
    request_model_name: "provider-reader-model-name"
    tokenizer_name: "configured-reader-tokenizer"
    tokenizer_local_path: "./models/tokenizers/qa_reader"
    api_base_url: "https://provider.example/v1"
    api_key_env: "QA_READER_API_KEY"
    max_context_tokens: 128000
    max_output_tokens: 16384
    tensor_parallel_size: 1
    dtype: "n/a"

  judge_llm:
    deploy: "api"
    backend: "openai_compatible"
    model_name: "configured-judge-model"
    request_model_name: "provider-judge-model-name"
    tokenizer_name: "configured-judge-tokenizer"
    tokenizer_local_path: "./models/tokenizers/judge_llm"
    api_base_url: "https://provider.example/v1"
    api_key_env: "JUDGE_MODEL_API_KEY"
    max_context_tokens: 128000
    max_output_tokens: 16384
    tensor_parallel_size: 1
    dtype: "n/a"
```

字段名应根据现有 `ModelSpec` 做最小兼容扩展，例如新增可选的 `tokenizer_local_path`。不应另行创建一套完全独立的模型加载体系。

### 5.3 `configs/prices.yaml`

small、medium、large、QA reader 和 judge LLM 都应有价格快照：

```yaml
prices:
  configured-small-model:
    official_price_in_per_1m: 0.0
    official_price_out_per_1m: 0.0
    currency: "USD"
    price_effective_date: "YYYY-MM-DD"

  configured-medium-model:
    official_price_in_per_1m: 0.0
    official_price_out_per_1m: 0.0
    currency: "USD"
    price_effective_date: "YYYY-MM-DD"

  configured-large-model:
    official_price_in_per_1m: 0.0
    official_price_out_per_1m: 0.0
    currency: "USD"
    price_effective_date: "YYYY-MM-DD"

  configured-reader-model:
    official_price_in_per_1m: 0.0
    official_price_out_per_1m: 0.0
    currency: "USD"
    price_effective_date: "YYYY-MM-DD"

  configured-judge-model:
    official_price_in_per_1m: 0.0
    official_price_out_per_1m: 0.0
    currency: "USD"
    price_effective_date: "YYYY-MM-DD"
```

第一版只使用输入价格和输出价格，不启用、配置或计算 prompt cache/cache hit/cache write 价格。即使模型供应商的响应中附带缓存相关字段，本实验也不得在没有明确缓存价格配置和独立实验设计的情况下自行引入缓存计费。

正式实验必须在 experiment manifest 中保存完整价格快照，防止供应商后续调价导致结果无法复现。

QA reader 和 judge 的成本应单独记录。第一版强化学习奖励默认只优化 `memory_extraction_cost`，不把 QA reader 和 judge 的评估开销混入记忆写入成本。

## 六、Embedding 模型本地化

### 6.1 MVP 推荐

第一版推荐让“路由特征”和“记忆检索”共同使用同一个本地 embedding 模型：

```text
BAAI/bge-m3
```

本地目录固定为：

```text
models/embeddings/bge-m3/
```

这样可以减少模型管理复杂度。后续如果实验表明路由 embedding 和检索 embedding 需要分离，再分别配置：

```text
router_embedding
memory_embedding
```

### 6.2 `configs/embeddings.yaml`

```yaml
embeddings:
  router:
    model_name: "BAAI/bge-m3"
    local_path: "./models/embeddings/bge-m3"
    dimension: 1024
    max_length: 8192
    normalize: true
    local_files_only: true

  memory:
    model_name: "BAAI/bge-m3"
    local_path: "./models/embeddings/bge-m3"
    dimension: 1024
    max_length: 8192
    normalize: true
    local_files_only: true
```

### 6.3 正式实验要求

正式实验中：

1. embedding 模型必须提前下载到指定目录；
2. 运行时必须使用 `local_files_only=true`；
3. 模型缺失时立即失败；
4. 不允许自动联网下载；
5. 不允许静默回退到 hashing encoder；
6. manifest 必须记录模型名称、revision、文件哈希、维度和最大长度；
7. L、M、H、S 四个 Qdrant Collections 必须使用同一个 memory embedding 模型；
8. embedding 归一化方式必须一致。

建议后续实现专用下载入口：

```text
scripts/download_local_models.py
```

下载脚本只负责将明确配置的模型保存到目标目录，不应在训练流程中隐式下载。

## 七、提示词外部化

### 7.1 存储方式

沿用项目当前提示词存储方式，全部放在：

```text
configs/prompts/
```

通过现有 `prompt_loader` 按文件名读取。禁止把长提示词写在 Python 常量中。

第一版需要：

```text
configs/prompts/joint_memory_extraction_{small,medium,large}.txt
configs/prompts/locomo_answer.txt
configs/prompts/longmemeval_answer.txt
configs/prompts/judge_locomo.txt
configs/prompts/judge_longmemeval.txt
```

现有 answer prompt 可以保留和适配。当前写在 judge 代码中的 LoCoMo/LongMemEval judge prompt 应迁移到上述文本文件。

### 7.2 提取提示词公平性

为了让 L/M/H 的差异主要来自模型能力，而不是提示词差异，默认应让三个模型使用同一个：

```text
joint_memory_extraction_{small,medium,large}.txt
```

如果未来确实需要 tier-specific prompt，配置层应允许覆盖，但实验报告必须明确提示词不同，不能将全部效果归因于模型规模。

### 7.3 提取模型输出边界

提取模型只负责输出主要 facts 的最小 JSON，不负责计算向量、成本、时间戳、speaker、Point ID 或其他 payload。

由于一次调用包含多个 segment，模型至少必须回显 `segment_id`，否则代码无法判断 fact 属于哪个主题段。`segment_id` 只是批量解析标记，不属于最终记忆内容。

推荐输入格式：

```text
--- Topic conv-26:nsp_text_tiling:seg_000001 ---
[2023-05-08T13:56:00.000, Mon] 0.Caroline: ...
[2023-05-08T13:56:01.000, Mon] 1.Melanie: ...

--- Topic conv-26:nsp_text_tiling:seg_000002 ---
...
```

推荐输出格式：

```json
{
  "processed_segment_ids": [
    "conv-26:nsp_text_tiling:seg_000001",
    "conv-26:nsp_text_tiling:seg_000002"
  ],
  "data": [
    {
      "segment_id": "conv-26:nsp_text_tiling:seg_000001",
      "source_id": 0,
      "fact": "Caroline attended an LGBTQ support group."
    },
    {
      "segment_id": "conv-26:nsp_text_tiling:seg_000001",
      "source_id": 0,
      "fact": "The support group made Caroline feel accepted."
    }
  ]
}
```

第二个 segment 已出现在 `processed_segment_ids` 且没有对应 `data[]` 条目，表示无事实。

禁止模型输出：

```text
完整 Qdrant payload
vector
cost
token_count
Point ID
category
subcategory
relation
episode
preference
constraint
consolidation result
```

解析器只接受顶层 `processed_segment_ids` 和 `data`；`data[]` 每项只允许
`segment_id`、`source_ids` 和 `fact`。`source_ids` 是非空整数数组，用于保存支持同一 fact 的全部消息来源。

每个 fact 必须：

1. 是独立、完整、可理解的陈述；
2. 保留准确的人名、日期、数字、地点和实体；
3. 不包含推测；
4. 不输出纯寒暄、临时表达或无长期价值内容；
5. 不合并互相无关的事实；
6. 数量受 `max_facts_per_segment` 配置约束。

### 7.4 Judge 提示词

judge prompt 必须外部化，并按数据集分别维护：

```text
judge_locomo.txt
judge_longmemeval.txt
```

judge 输入至少包括：

```text
question
gold_answer
predicted_answer
question_type/category
is_unanswerable
```

judge 输出采用最小协议：

```text
CORRECT
```

或：

```text
INCORRECT
```

如需理由，可以允许第二行简短解释，但训练标签只解析第一行。judge 的模型、prompt version、temperature 和最大输出必须固定并记录。

## 八、三级 Extraction Buffer

### 8.1 为什么必须使用 buffer

如果每个主题段单独调用一次 LLM，会在每次请求中重复支付：

- system prompt；
- 任务说明；
- 输出格式约束；
- API 请求固定开销。

因此必须设置三个相互独立的 buffer：

```text
SmallExtractionBuffer
MediumExtractionBuffer
LargeExtractionBuffer
```

每个 buffer 只接收对应 tier 的 segment，并批量调用对应模型。

### 8.2 训练候选库生成阶段

为了生成 L/M/H 三个候选库，同一个 segment 必须分别进入三个 buffer：

```text
segment_i → small buffer
segment_i → medium buffer
segment_i → large buffer
```

三个 buffer 分别 flush，最终生成：

```text
small buffer  → Qdrant L Collection
medium buffer → Qdrant M Collection
large buffer  → Qdrant H Collection
```

这一步只在候选库生成阶段执行一次。后续强化学习 rollout 不重新调用提取模型。

### 8.3 训练完成后的部署阶段

部署时每个 segment 只进入路由器选中的 buffer：

```text
route(segment_i) = small  → small buffer
route(segment_i) = medium → medium buffer
route(segment_i) = large  → large buffer
```

buffer 满足 flush 条件后才调用模型。

### 8.4 Buffer 配置

buffer 不应只有“segment 数量”一个限制，还必须同时设置 token 限制，防止少数超长 segment 超过模型上下文。

建议配置：

```yaml
extraction:
  prompt_files:
    locomo: "locomo_memory_extraction.txt"
    longmemeval: "longmemeval_memory_extraction.txt"
  max_facts_per_segment: 15
  reserve_output_tokens_per_segment: 1024
  require_provider_usage: true
  flush_at_sample_end: true
  allow_cross_sample_batch: false
  allow_oversize_singleton: true
  truncate_over_total_context: true
  quality_gates:
    max_empty_fact_segment_rate: 0.25
    max_saturated_segment_rate: 0.10
    max_repair_batch_rate: 0.20
    max_failed_batch_rate: 0.0

  buffers:
    small:
      max_segments: 6
      max_input_tokens: 16384
      max_total_context_tokens: 24576

    medium:
      max_segments: 6
      max_input_tokens: 16384
      max_total_context_tokens: 24576

    large:
      max_segments: 6
      max_input_tokens: 16384
      max_total_context_tokens: 24576
```

主实验故意让三个 buffer 使用相同参数，并受最小的 32K 模型约束，以隔离模型能力与价格变量。普通批次最多包含 6 个主题段；请求始终还要满足 `max_total_context_tokens=24576`，该上限低于小模型 32768 总窗口并保留 8192 token 安全余量。

token 限制是主要 flush 条件；`max_segments=6` 只是大量短 segment 的二级安全阀。请求的 `max_tokens` 必须按当前批次实际 segment 数计算，而不是始终按 6 段计算，并且不得超过模型的 `max_output_tokens`。

### 8.5 Flush 条件

添加新 segment 前，按照实际模型 tokenizer 预测请求长度。如果满足任一条件，则先 flush 当前 buffer：

1. 当前 segment 数达到 `max_segments`；
2. 加入新 segment 后预计输入超过 `max_input_tokens`；
3. 加入新 segment后，输入加预留输出超过 `max_total_context_tokens`；
4. 到达当前 sample 末尾；
5. 调用方显式执行 `finalize()`；
6. 离线任务结束或异常关闭前执行安全 flush。

buffer 禁止跨 sample 混批：

```yaml
allow_cross_sample_batch: false
```

原因：

- 每个 Point 都有 `sample_id`，所有读取和删除操作强制使用 sample filter；
- 成本按 sample 统计；
- 跨 sample 混批会使提示词成本和失败恢复变复杂；
- 容易产生跨 sample 写错和信息污染。

### 8.6 超长主题段与 singleton batch

`max_input_tokens=16384` 是普通多 segment 合批的输入阈值，不等同于模型的物理上下文上限。启用 `allow_oversize_singleton=true` 后，单个主题段按以下顺序处理：

1. 若渲染后的输入不超过 `max_input_tokens`，按普通规则进入 buffer；
2. 若输入超过 `max_input_tokens`，但“输入 token + 该段预留输出 token”不超过 `max_total_context_tokens`，先 flush 当前 buffer，再把该主题段作为唯一元素立即发送，即 singleton batch；
3. singleton batch 保持原始 `segment_id`、文本、来源、成本归属与事实归属，不修改上游主题分割，也不做截断；
4. 若完整单段连 `max_total_context_tokens` 都无法满足，采用确定性的尾部截断：保留能同时满足所有已选 tier 总上下文限制的最长文本前缀，并追加显式截断标记；
5. 截断后仍保持原始 `segment_id`、`source_content_hash`、`turn_ids`、起止 turn、时间范围和事实归属，不创建 extraction chunk；只从本次截断文本临时计算 `visible_source_ids`，parser 只允许模型引用这些来源；manifest、segment cost ledger 和事实 payload 记录 visible/dropped source IDs 及截断前后 token/字符数；本次费用只采用这一次实际 API 调用返回的 provider usage；
6. 只有连最短非空前缀和截断标记都无法容纳时才明确报错，该情况意味着 prompt 与输出预留本身已经超过配置窗口。

LongMemEval 当前数据审计只发现一个底层超长主题段：sample `852ce960`（NSP 为 `seg_000041`，BERT-MLP 为 `seg_000036`）。三个模型 tokenizer 下，其渲染输入约为 19.5K tokens；加 1024 输出预留后约为 20.6K，仍小于 24,576。因此该段应完整地以 singleton batch 各发送给 small、medium、large 模型一次，不拆分、不截断，也不需要从数据集中移除。

### 8.7 Buffer 状态

每个 buffer 至少维护：

```text
tier
sample_id
ordered_segments
estimated_input_tokens
estimated_reserved_output_tokens
buffer_sequence
```

segment 在 buffer 中的顺序必须与分割文件顺序一致。相同输入、配置和 tokenizer 下，批次划分必须可复现。

## 九、批量提取解析与代码端 JSON 拼装

### 9.1 解析流程

```text
LLM 原始文本响应
  ↓
解析 JSON 根对象
  ↓
验证 processed_segment_ids 与当前 batch 完全一致且顺序相同
  ↓
验证每条 segment_id、全部 source_ids 和 fact
  ↓
去除完全重复的 fact
  ↓
代码读取原 segment 元数据
  ↓
代码生成 fact_id、时间、来源、成本和向量
  ↓
显式调用本地 Embedding 模型向量化 fact_text
  ↓
将 id + vector + payload upsert 到 Qdrant
```

如果模型返回未知 `segment_id`、遗漏 `processed_segment_ids`、非法 `source_ids` 或 JSON 无法解析，该 batch 应记录为失败，不得把无法归属的 fact 写入 Qdrant。

### 9.2 Fact 记录

第一版只保留必要字段：

```json
{
  "fact_id": "uuid-or-deterministic-id",
  "dataset_name": "locomo",
  "split": "train",
  "sample_id": "conv-26",
  "session_id": "session_1",
  "segment_id": "conv-26:nsp_text_tiling:seg_000001",
  "segment_hash": "sha256...",
  "source_turn_ids": [1],
  "source_id": 0,

  "fact_text": "Caroline attended an LGBTQ support group.",
  "fact_index": 0,
  "fact_count_in_segment": 2,

  "memory_tier": "small",
  "extractor_model": "configured-small-model",
  "prompt_version": "joint_memory_extraction_batch_json_v3",
  "batch_id": "batch-id",
  "extraction_run_id": "run-id",

  "segment_start_timestamp": "2023-05-08T13:56:00.000",
  "segment_end_timestamp": "2023-05-08T13:56:01.000",

  "allocated_input_tokens": 0,
  "allocated_output_tokens": 0,
  "allocated_total_tokens": 0,
  "allocated_input_cost": 0.0,
  "allocated_output_cost": 0.0,
  "allocated_total_cost": 0.0,

  "embedding_model": "BAAI/bge-m3",
  "embedding_dimension": 1024,
  "embedding": "stored-as-qdrant-vector"
}
```

第一版不保存或不使用以下字段：

```text
relation
preference
constraint
episode
memory_class
category
subcategory
topic_summary
compressed_memory
consolidated
update_queue
```

## 十、批次成本和 segment/fact 成本分配

### 10.1 实际批次成本

每次 LLM 请求完成后，优先读取模型 API 响应中的 `usage`，将模型返回的输入和输出 token 数作为该次请求的真实消耗。不同 OpenAI-compatible 接口可能使用 `prompt_tokens/completion_tokens` 或 `input_tokens/output_tokens`，适配层必须统一映射为：

```text
input_tokens
output_tokens
total_tokens
model_name
price_snapshot
retry_count
latency_ms
usage_source = "provider"
```

其中：

$$
T_b^{total}=T_b^{input}+T_b^{output}
$$

价格配置采用“每 100 万 token 的输入/输出价格”。批次费用直接根据模型返回的 token 数计算：

$$
C_b^{input}=\frac{T_b^{input}}{10^6}P^{input}
$$

$$
C_b^{output}=\frac{T_b^{output}}{10^6}P^{output}
$$

$$
C_b=C_b^{input}+C_b^{output}
$$

第一版不使用缓存价格，也不设置 `cached_prompt_tokens`、`cached_input_cost` 等成本字段。

模型 API 返回的 usage 是批次总消耗的权威来源，本地 tokenizer 不应取代 provider usage 来计算真实账单。只有以下情况允许本地计算：

1. 请求发送前，为了判断 buffer 是否超过上下文上限而估算输入 token；
2. provider 明确不返回 usage；
3. 本地推理后端没有 usage 字段，但可以从实际 tokenizer 编码结果和生成长度精确取得 token 数；
4. 为了把批次总 token 按比例归属到各 segment。

如果 provider 不返回 usage，记录必须设置：

```text
usage_source = "tokenizer_estimate"
```

不能把估算值伪装成 provider 返回值。正式 API 实验建议配置 `require_provider_usage=true`；启用后，如果响应缺少 usage，应使该调用失败并记录原因，避免真实成本不可审计。

如果发生重试，实际消耗的所有请求都应计入实验成本。失败调用也必须记录，不得只记录最终成功请求。

### 10.2 输入成本分配

设批次包含主题段集合 $I_b$：

- $T_b^{input}$：provider 返回的批次总输入 token；
- $x_i$：使用该模型对应 tokenizer 计算的主题段 $i$ 在实际序列化后的独占输入 token；
- $P^{input}$：输入 token 的每 100 万 token 单价。

API 一般只返回整个 batch 的输入 token，不会返回每个 segment 的 token。因此，本地 tokenizer 仅用于计算 segment 之间的分配权重：

$$
w_i=\frac{x_i}{\sum_{j\in I_b}x_j}
$$

批次中的 system prompt、任务说明和公共结构已经包含在 provider 返回的 $T_b^{input}$ 中，它们随总输入 token 一起按上述权重分摊，不再单独重复计费：

$$
T_i^{input}=w_iT_b^{input}
$$

$$
C_i^{input}=\frac{T_i^{input}}{10^6}P^{input}
$$

实现时可使用最大余数法把 segment 级 token 分配为整数，保证：

$$
\sum_{i\in I_b}T_i^{input}=T_b^{input}
$$

### 10.3 输出成本分配

设 provider 返回批次总输出 token 为 $T_b^{output}$。解析器按 `segment_id` 将 `data[]` 重组为每个 segment 的最小 JSON，并用对应模型 tokenizer 计算相对 token 长度 $y_i$：

$$
v_i=\frac{y_i}{\sum_{j\in I_b}y_j}
$$

然后将 provider 返回的总输出 token 按比例归属：

$$
T_i^{output}=v_iT_b^{output}
$$

$$
C_i^{output}=\frac{T_i^{output}}{10^6}P^{output}
$$

其中 $y_i$ 包括该 segment 的 marker、`FACT` 行和结束标记。无法归属的公共输出 token 也按 $v_i$ 分配。实现时同样使用最大余数法，保证 segment 级输出 token 之和严格等于 provider 返回的批次总输出 token。

主题段总成本：

$$
C_i=C_i^{input}+C_i^{output}
$$

若主题段产生 $n_i$ 个 facts，并要求同一主题 facts 使用相同平均成本：

$$
c_{ij}=\frac{C_i}{n_i}
$$

token 数也按相同规则记录到 fact：

$$
t_{ij}^{input}=\frac{T_i^{input}}{n_i},\qquad
t_{ij}^{output}=\frac{T_i^{output}}{n_i},\qquad
t_{ij}^{total}=t_{ij}^{input}+t_{ij}^{output}
$$

由于平均后可能出现小数，fact 级平均 token 字段使用 `REAL`；批次和 segment 级 token 字段仍使用 `INTEGER`。同一 segment 产生的所有 facts 具有相同的平均 token 数和平均费用。

如果 $n_i=0$，成本仍保存在 `segment_extraction_costs` 和 `extraction_batches` 中，但没有 fact 行承载该成本。

必须同时满足 token 守恒和成本守恒：

$$
\sum_{i\in I_b}T_i^{input}=T_b^{input},\qquad
\sum_{i\in I_b}T_i^{output}=T_b^{output}
$$

$$
\sum_{i\in I_b}C_i=C_b
$$

### 10.4 Token 账本与指标用途

token 数必须与费用同时持久化，不能只保存最终金额。金额是由 token 数和价格快照计算出的派生数据，而 token 是后续比较模型效率的基础指标。

必须分别保存三种口径：

```text
batch actual tokens：模型 API usage 返回的批次真实输入、输出和总 token
segment allocated tokens：从 batch 真实 token 分摊到主题段的 token
fact average tokens：segment token 除以 fact 数得到的平均 token
```

同时区分：

```text
candidate_generation_actual_tokens：离线生成 L/M/H 候选库实际消耗的 token
policy_virtual_deployment_tokens：当前路由策略按虚拟 buffer replay 估算的部署 token
reader_judge_actual_tokens：QA reader 和 judge 实际消耗的 token
```

三类 token 不得混为一个指标。强化学习默认使用 `policy_virtual_deployment_tokens/cost` 描述策略效率；候选库生成和 reader/judge token 单独报告。

### 10.5 代码生成 JSON 不计 LLM 输出成本

完整 payload、Qdrant Point 字段和 embedding 均由本地代码生成，它们不属于 LLM completion token，因此不能计入模型输出成本。

## 十一、Qdrant 中的 L/M/H/S 实际存储

### 11.1 Collection 组织原则

Qdrant 是本方案的正式 fact、向量和 payload 数据源。MVP 不为每个 sample 创建四个 Collection，因为 LoCoMo 尚可承受，但 LongMemEval 会产生大量小 Collection，增加初始化、索引和管理开销。

每个 dataset、split、分段版本和实验命名空间建立四个实际 Collections：

```text
{namespace}_L：small 模型生成的候选 facts
{namespace}_M：medium 模型生成的候选 facts
{namespace}_H：large 模型生成的候选 facts
{namespace}_S：当前路由 rollout 实际组装的 facts
```

其中 `namespace` 至少编码：

```text
dataset_name
split
segmentation_version
memory_embedding_version
schema_version
```

sample 隔离通过 Point payload 中的 `sample_id` 实现。所有 scroll、search、delete 和 count 操作必须显式携带 `dataset_name + split + sample_id` filter；对 S 的操作还必须携带 `assembly_id`。禁止无 filter 查询或删除。

### 11.2 Collection 向量配置

四个 Collections 使用完全相同的向量配置：

```yaml
vectors:
  size: 1024
  distance: "Cosine"
  on_disk: false
```

MVP 只有一个 dense vector，使用 BAAI/bge-m3 对 `fact_text` 编码。L/M/H/S 的向量维度、距离函数、归一化方式和 embedding 版本必须一致，否则禁止复制或联合比较。

建议为以下 payload 字段建立 keyword index：

```text
dataset_name
split
sample_id
session_id
segment_id
memory_tier
extraction_run_id
assembly_id
policy_version
```

### 11.3 L/M/H 生成

候选库生成阶段：

1. 读取一个 sample 的全部 segments；
2. 按固定顺序分别填充 small、medium、large buffer；
3. 模型返回 fact 文本后，由代码拼装 payload；
4. 代码显式调用本地 memory embedding 模型编码 `fact_text`；
5. 将 `id + vector + payload` 分别 upsert 到 L、M、H Collection；
6. sample 结束时 flush 三个 buffer；
7. 使用带 `sample_id` 的 count/scroll 检查候选 facts；
8. 无事实 segment、失败、批次 token 和成本写入该 sample 的 SQLite 账本。

L/M/H 构建完成后冻结。强化学习期间只能带 filter 读取，禁止覆盖或删除候选 Points。

### 11.4 S 实际组装

对于路由动作：

```text
route(segment_1) = small
route(segment_2) = large
route(segment_3) = medium
```

必须实际执行：

```text
从 L Collection 读取 segment_1 的 Points
从 H Collection 读取 segment_2 的 Points
从 M Collection 读取 segment_3 的 Points
复制 vector 和 payload
写入 S Collection，并增加当前 assembly_id
```

S 不是内存列表或逻辑视图。QA 必须只查询 S Collection，并同时过滤当前 `sample_id` 和 `assembly_id`。

### 11.5 使用 assembly_id 实现一致性

Qdrant 不提供 SQLite 式的跨多 Point 事务，因此不得采用“先删除旧 S，再逐条写入新 S”的方式。每次 rollout 使用新的不可变 `assembly_id`：

```text
生成 assembly_id
→ 按路由动作从 L/M/H 读取候选 Points
→ 以新的 S point_id 写入 S，并附加 assembly_id
→ 校验每个 segment 只对应一个 selected_tier
→ 校验 Point 数量、来源和 sample_id
→ 将 assembly 标记为 ready
→ QA 显式使用该 assembly_id 检索
→ 评估完成后按保留策略清理旧 assembly
```

未通过完整性校验的 assembly 标记为 `failed`，不得进入 QA。训练 SQLite ledger 的 `assemblies` 表保存 `assembly_id`、episode、policy、状态、Point 数量和清理状态。

### 11.6 Point ID 与来源

候选 fact Point ID 必须稳定且唯一，建议使用 UUIDv5 或内容哈希生成，并能够关联：

```text
sample_id
segment_id
tier
extraction_run_id
fact_index
```

S Point 使用新的 ID，例如：

```text
uuid5(assembly_id + source_collection + source_point_id)
```

S payload 必须保留：

```text
source_collection_tier
source_point_id
source_extraction_run_id
assembly_id
episode_id
policy_version
```

## 十二、Qdrant Point 与账本 Schema

### 12.1 Qdrant 不自动向量化

Qdrant 核心存储层接收的是已经计算好的 vector，不会因为 payload 中存在 `fact_text` 就自动向量化。即使客户端支持 FastEmbed 或其他推理集成，正式实验也必须使用项目代码显式执行：

```text
fact_text
→ 本地 BAAI/bge-m3
→ normalize 后的 1024维 float32 vector
→ Qdrant Point(vector, payload)
```

这样才能固定模型目录、revision、维度、归一化方式和哈希。模型缺失时必须失败，不得由 Qdrant 客户端隐式下载或切换 embedding 模型。

向量化内容的 MVP 规则：

- 记忆入库只向量化 `fact_text`；
- payload 元数据、费用、时间戳和 ID 不拼接进 embedding 文本；
- QA query 使用同一个 memory embedding 模型编码；
- 路由器另行对完整 `segment_text` 编码，得到 router embedding；
- fact vector 和 segment router vector 用途不同，即使当前共享 BGE-M3，也不得混用缓存键。

### 12.2 Point 样例

逻辑结构：

```json
{
  "id": "7ee9c9d0-9e4b-5eab-96a7-b62f77162c1a",
  "vector": [0.023, -0.145, 0.678, 0.034],
  "payload": {
    "schema_version": "qdrant_fact_v2",
    "dataset_name": "locomo",
    "split": "train",
    "sample_id": "conv-26",
    "session_id": "session_3",
    "segment_id": "conv-26:nsp_text_tiling:seg_000012",
    "segment_order": 12,
    "source_turn_ids": [21],
    "source_id": 20,
    "source_content_hash": "sha256...",
    "fact_text": "Caroline attended an LGBTQ support group.",
    "fact_index": 0,
    "fact_count_in_segment": 2,
    "memory_tier": "medium",
    "extractor_model": "configured-medium-model",
    "prompt_version": "joint_memory_extraction_batch_json_v3",
    "batch_id": "batch-medium-008",
    "extraction_run_id": "run-001",
    "allocated_input_tokens": 356.5,
    "allocated_output_tokens": 24.0,
    "allocated_total_tokens": 380.5,
    "allocated_input_cost": 0.0000713,
    "allocated_output_cost": 0.000024,
    "allocated_total_cost": 0.0000953,
    "embedding_model": "BAAI/bge-m3",
    "embedding_dimension": 1024,
    "embedding_normalized": true,
    "created_at": "2026-08-02T10:00:00Z"
  }
}
```

示例中的 vector 仅为缩写，实际必须正好包含 1024 个 float 值。vector 不重复写入 payload。

### 12.3 Batch 与 segment 成本账本

Qdrant 只存储实际 fact Points。由于无事实或失败 segment 没有可存的向量 Point，批次和 segment 状态不能只依赖 Qdrant，必须分别写入：

```text
samples/{sample_id}/extraction/candidate_ledger.sqlite3
```

SQLite `batches` 表每行至少包含：

```text
batch_id, sample_id, tier, model_name, segment_ids
input_tokens, output_tokens, total_tokens, usage_source
input_cost, output_cost, total_cost
input_price_per_1m, output_price_per_1m, price_effective_date, currency
retry_count, latency_ms, status, raw_response_path, created_at
```

SQLite `segment_costs` 表每行至少包含：

```text
segment_id, batch_id, tier
allocated_input_tokens, allocated_output_tokens, allocated_total_tokens
allocated_input_cost, allocated_output_cost, allocated_total_cost
fact_count, status
```

attempt 与 failure 审计采用 append-only；batch/segment cost 是可恢复的最终投影，通过 SQLite 唯一键原子 upsert。Qdrant payload 中的 fact 平均 token/成本必须能与 SQLite 账本相互核对。

### 12.4 路由与 assembly 账本

每次 rollout 的路由决策写入 SQLite `assemblies` 表，至少包含：

```text
assembly_id, episode_id, sample_id
segment_id, selected_tier, action_probability
router_type, router_checkpoint, route_order
status, point_count, created_at, cleaned_at
```

Qdrant S payload 同时保存 `assembly_id` 和来源字段，使检索结果能够回溯到路由决策和候选 Point。

## 十三、人工查看 JSON

### 13.1 定位

除 Qdrant 正式存储外，每个 Collection 的 sample 子集都可以导出一份 JSON：

```text
human_readable/L_memories.json
human_readable/M_memories.json
human_readable/H_memories.json
human_readable/S_memories.json
```

JSON 只用于：

- 人工抽查 fact；
- 比较 L/M/H 提取差异；
- 检查 segment 来源和成本；
- 调试格式错误。

实验检索、QA 和训练不得读取这些人工 JSON。Qdrant 是唯一正式 fact/vector 检索数据源，SQLite 仅作为批次、成本、状态和路由账本。

### 13.2 导出策略

L/M/H 在候选库构建完成后各导出一次。

S 在强化学习期间不建议每个 rollout 都导出，以免产生大量无意义 I/O。默认：

```yaml
human_readable_export:
  export_candidates: true
  export_s_every_episode: false
  export_s_on_best_checkpoint: true
  export_s_on_final_policy: true
```

### 13.3 JSON 结构

```json
{
  "metadata": {
    "dataset_name": "locomo",
    "sample_id": "conv-26",
    "collection_tier": "L",
    "embedding_model": "BAAI/bge-m3",
    "generated_at": "...",
    "warning": "Human inspection only. Experiments must query Qdrant."
  },
  "memories": [
    {
      "fact_id": "...",
      "segment_id": "...",
      "source_turn_ids": [1, 2],
      "fact_text": "Caroline attended an LGBTQ support group.",
      "memory_tier": "small",
      "allocated_total_cost": 0.0
    }
  ]
}
```

JSON 默认不导出 vector 数组，避免文件过大。只保存 embedding 模型名、维度和可选向量哈希。

## 十四、强化学习期间的成本模拟

### 14.1 为什么不能直接相加候选库历史成本

L/M/H 候选库是在“所有 segments 都送到同一个 tier”的条件下生成的。混合路由后，各 tier 收到的 segment 子集不同，buffer 的批次边界和 system prompt 重复次数也会变化。

因此，训练奖励不能简单把 L/M/H 中保存的 fact 平均成本相加作为策略成本。

### 14.2 确定性 buffer replay

每次 rollout 应使用路由动作重放部署 buffer：

```text
按 segment 原始顺序遍历
→ 根据 action 放入 small/medium/large 虚拟 buffer
→ 使用与部署相同的 max_segments/max_input_tokens 规则
→ 确定虚拟 batch 边界
→ 根据该 tier 的输入价格、输出价格、共享 prompt token、segment token 和历史输出 token 估计成本
```

候选库中必须保存每个 segment 在对应 tier 下的：

```text
serialized_input_tokens
attributed_output_tokens
fact_count
```

虚拟 buffer replay 不调用 LLM，只重新计算在该路由策略下预计形成的批次、输入/输出 token 和成本。这里的 token 是策略部署成本估计，不得覆盖候选库生成阶段由 provider usage 返回的真实 token 账本。

### 14.3 训练成本与实验实际支出

必须区分：

```text
candidate_generation_actual_cost：生成 L/M/H 三套候选库的真实实验支出
policy_virtual_deployment_cost：某个路由策略在部署时预计发生的写入成本
qa_evaluation_actual_cost：reader 和 judge 的真实评估支出
```

强化学习奖励默认使用：

```text
policy_virtual_deployment_cost
```

不得使用 L+M+H 候选生成总支出作为路由策略成本。

## 十五、当前路由器：Embedding + MLP

### 15.1 当前实现边界

MVP 只实现 `EmbeddingMLPRouter`。BERT 分类器、1B–3B + LoRA/QLoRA 和其他生成式路由器仅保留统一接口与配置扩展位，本阶段不下载、不训练、不参与实验主结果。

当前路由器输入：

```text
本地 BGE-M3 segment embedding
+ log_token_count
+ turn_count
+ speaker_count
+ segment_position
+ time_span
+ has_temporal_expression
+ has_update_expression
+ entity_count
+ number_count
+ date_count
```

其中完整 `segment_text` 使用本地 BGE-M3 得到 1024 维 router embedding；结构特征需要进行训练集统计量标准化，并将 scaler 参数随 checkpoint 保存。

路由器不得看到：

```text
当前 QA question
gold answer
evidence IDs
question_type
judge label
测试集统计信息
```

训练时 QA 和 evidence 可以用于奖励或训练分析，但不允许出现在策略状态中。

### 15.2 MLP 网络

建议默认结构：

```text
1024维 segment embedding
+ 标准化后的结构特征
→ Linear(输入维度, 512)
→ GELU + Dropout
→ Linear(512, 128)
→ GELU + Dropout
→ policy_head: Linear(128, 3)
→ softmax: P(small), P(medium), P(large)
```

如使用 Actor-Critic，可从同一个 128 维隐藏表示增加 `value_head: Linear(128, 1)`。BGE-M3 在 MVP 中冻结，只训练 MLP policy/value head，避免小数据集上破坏通用语义表示。

输出：

$$
\pi_\theta(a_i|z_i),\quad a_i\in\{small,medium,large\}
$$

策略使用 categorical distribution，保留 logits、三个动作概率、采样动作和 log probability，供 policy gradient、复现和审计。

### 15.3 可替换路由接口

存储、buffer、S assembly 和 QA 评估不得依赖 MLP 的内部结构。所有路由器统一实现：

```text
RouterInput:
  segment_id
  segment_text
  segment_embedding
  numeric_features

RouterOutput:
  action
  action_probabilities
  selected_probability
  log_probability
  value（可选）
  router_type
  router_version
```

当前注册：

```text
router.type = embedding_mlp
```

后续可以新增 `bert_classifier`、`llm_lora_1b`、`llm_lora_3b` 等实现，只要遵守相同输入输出契约，就不需要修改 L/M/H/S、成本模拟和 QA 评估流程。扩展实现必须作为新的实验分支和配置项加入，不能改变当前 Embedding + MLP 基线的行为。

## 十六、强化学习算法与奖励

### 16.1 问题类型

该任务优先建模为带预算约束的组合上下文 Bandit：

```text
Budget-Constrained Contextual Combinatorial Bandit
```

多个 segment 动作共同决定 Qdrant S assembly 和最终 QA 奖励。

### 16.2 推荐算法

MVP 推荐：

```text
Categorical Policy
+ Actor-Critic 或 REINFORCE with learned baseline
+ Lagrange multiplier
```

如果训练明显不稳定，再使用 PPO-Lagrangian。不要使用将整个 $3^N$ 联合选择视为单一动作的 DQN。

### 16.3 奖励

受约束目标：

$$
\max_\theta E[Q(S)]
$$

$$
E[C_{virtual}(S)]\le B
$$

拉格朗日形式：

$$
R=Q(S)-\lambda(C_{norm}(S)-B)
$$

成本归一化：

$$
C_{norm}(S)=
\frac{C(S)-C_{all-small}}
{C_{all-large}-C_{all-small}+\epsilon}
$$

必须同时保存原始值：

```text
qa_score
virtual_extraction_cost
normalized_cost
reward
lambda
```

不能只保存最终加权 reward。

### 16.4 可选 warm start

监督预训练可以作为可选 warm start，但不是 MVP 的硬性要求。如果使用，只能基于训练 split 构造标签，测试 evidence 不得参与。

## 十七、训练 Episode

### 17.1 通用 sample episode

对于一个 sample：

1. 加载该 sample 的冻结 segments；
2. 加载只读 L/M/H；
3. 路由器为每个 segment 采样动作；
4. 使用新的 `assembly_id` 在 Qdrant S Collection 实际组装 Points；
5. 从 S Collection 按 `sample_id + assembly_id` 检索；
6. QA reader 生成回答；
7. judge LLM 评分；
8. 虚拟 buffer replay 计算部署成本；
9. 聚合 QA 和成本得到 reward；
10. 更新策略；
11. 保存 episode 日志。

训练 rollout 不重新执行 memory extraction。

### 17.2 LoCoMo

LoCoMo 的独立单元是 conversation/sample：

```text
一个 sample
→ Qdrant L/M/H/S Collections 中的一组 sample 数据
→ 多个 QA
```

一次 rollout 对该 sample 的全部 segments 路由一次，然后使用同一个 S `assembly_id` 评估一个分层 QA minibatch。不要每回答一个问题就重新提取记忆。

建议：

```text
QA minibatch：16～64
按 single-hop / multi-hop / temporal / open-domain / adversarial 分层
```

数据划分必须按 `sample_id`，不能把同一 conversation 的问题拆到不同 split。

#### 17.2.1 全量候选工件与实验划分分离

LoCoMo 的预处理、主题分割和 L/M/H 候选记忆统一对全部 10 个 conversation 执行，
并保存在 `split=full` 下。这里的 `full` 是冻结的候选来源，不表示路由器可以使用
全部 conversation 训练。已有文件按以下规则处理：

```text
datasets/raw/locomo/locomo10.json                     保留，不修改
datasets/processed/locomo/full/{samples,questions,
  sessions,turns}.jsonl                               保留全部 10 个 sample，不拆分
datasets/segmented/locomo/full/{method}/samples/...   保留全部 10 个 sample，不拆分
outputs/rl_router/locomo/full/{method}/samples/...    提前生成全部 10 个 sample 的 L/M/H
datasets/splits/locomo/cv5_seed42.json                新增，只保存每折 sample_id
outputs/rl_router/training/locomo/cv5/fold_{k}/...    每折独立的训练日志与 checkpoint
outputs/rl_router/evaluation/locomo/cv5/fold_{k}/...  每折独立的正式评估账本
```

不得按 fold 复制或改写 `samples.jsonl`、`questions.jsonl`、`sessions.jsonl`、
`turns.jsonl` 和 `segments.jsonl`。物理数据只保存一份，逻辑划分由 split manifest
完成，这样候选生成可以复用，同时避免十套近似文件发生版本漂移。

正式五折 manifest 使用 conversation 为原子单位，每折必须满足：

1. `train` 恰好 8 个，`test` 恰好 2 个，二者不相交且覆盖全部 10 个 sample；
2. 五折中每个 sample 恰好进入一次 `test`；
3. 候选生成只读 segment，不读取 QA answer/evidence，因此可在划分前完成；
4. 路由训练、策略更新、reward 计算和 scaler 拟合只接收当前折 `train` ID；
5. `test` 只能在该折 checkpoint、scaler 和超参数冻结后用于路由评估；
6. 五折没有 validation，超参数必须预先固定；需要调参时使用独立开发划分或嵌套交叉验证；
7. 每次训练的实验 manifest 必须记录 split manifest 的路径、SHA-256、fold 和三类 ID。

split manifest 示例：

```json
{
  "schema_version": "conversation_split_v1",
  "dataset_name": "locomo",
  "source_split": "full",
  "protocol": "cv5",
  "seed": 42,
  "folds": [
    {
      "fold": 0,
      "train": ["conv-41", "conv-49", "...共 8 个"],
      "validation": [],
      "test": ["conv-48", "conv-42"]
    }
  ]
}
```

`source_split=full` 决定从哪里读取预处理、分割和候选工件；`train/test` 决定某次
实验允许哪些 sample 参与哪种操作。这两个概念不得合并。

### 17.3 LongMemEval

LongMemEval 一个 question/history instance 视为一个 sample，每个 sample 都有独立 L/M/H/S。

由于单个 sample 通常只有一个 QA，策略更新应聚合多个 sample 的 episode 结果，例如 16～64 个 history instances，再计算 batch 平均 QA 和成本。

LongMemEval 与 LoCoMo 一样使用 `source_split=full` 保存全部基础工件和 L/M/H 候选，
不复制 `processed`、`questions`、`sessions`、`turns` 或 `segments` 文件。实验角色由
以下独立 manifest 控制：

```text
datasets/splits/longmemeval/fixed_80_10_10_seed42_nsp_text_tiling.json
datasets/splits/longmemeval/cv5_360_40_100_seed42_nsp_text_tiling.json
```

固定方案为 400/50/50；正式五折方案为每折 360/40/100，其中 40 validation 只能
用于确定性路由评估、早停和 checkpoint 选择，不执行梯度更新、scaler 重拟合或
策略更新。test 只能在该折最佳 checkpoint 冻结后使用。

划分不是普通随机抽样。生成器同时使用：

1. `question_type`；
2. `is_unanswerable`；
3. 固定 NSP 分段结果的 segment-count quartile；
4. evidence-session connected component；
5. answerable/abstention counterpart ID。

“evidence 隔离”定义为：先收集全局所有 evidence session ID，再把这些 session 在
任何 sample haystack 中的所有出现位置合并为同一 connected component。这样即使
某个 session 在另一个样本中只是 distractor，也不会跨 partition 暴露。当前 500 个
样本形成 466 个组，432 个单样本组、34 个双样本组，最大组大小为 2，因此可以满足
精确集合大小。每份 manifest 必须保存：

```text
processed manifest SHA-256
用于 segment-count 分层的 segmentation manifest SHA-256
seed、分层字段和 quartile 阈值
每个 partition 的类型/不可回答/段数区间统计
跨 partition evidence session overlap（必须为 0）
跨 partition 纯背景 distractor session overlap（披露，不强制为 0）
```

允许从未作为任何问题 evidence 的纯背景 distractor session 跨集合出现。这保留了
原始 LongMemEval 的 500 个 benchmark sample。若强制所有 haystack session 完全
隔离，大规模共享背景会使样本图连成不可用的大组件；此时必须重新构造、去重并版本化
数据集，结果不得继续声称是原始 LongMemEval benchmark。

## 十八、QA Reader 与 Judge

### 18.1 Reader

QA reader：

1. 只从 S Collection 检索，并强制过滤当前 `sample_id + assembly_id`；
2. 使用固定本地 embedding 模型编码 query；
3. 固定 Top-K、相似度、时间排序和 prompt；
4. 不允许访问 L/M/H；
5. 不允许访问原始完整 conversation；
6. 输出回答和调用 usage。

LoCoMo 和 LongMemEval 使用各自外部 answer prompt。

### 18.2 Judge

judge：

1. 使用 `models.yaml` 的 `judge_llm`；
2. 使用外部 `judge_locomo.txt` 或 `judge_longmemeval.txt`；
3. temperature 固定为 0 或供应商允许的最低确定性值；
4. 只解析第一行 `CORRECT/INCORRECT`；
5. 保存原始 judge 响应和 prompt version；
6. judge 成本单独记录，不计入默认路由成本奖励。

## 十九、配置汇总示例

`configs/rl_router.yaml`：

```yaml
experiment:
  seed: 42
  schema_version: "rl_router_fact_qdrant_mlp_v3"
  segmentation_document: "docs/data_preprocessing_and_bert_segmentation_design.md"

models:
  small: "small"
  medium: "medium"
  large: "large"
  qa_reader: "qa_reader"
  judge: "judge_llm"

prompts:
  fact_extraction_locomo: "locomo_memory_extraction.txt"
  fact_extraction_longmemeval: "longmemeval_memory_extraction.txt"
  locomo_answer: "locomo_answer.txt"
  longmemeval_answer: "longmemeval_answer.txt"
  locomo_judge: "judge_locomo.txt"
  longmemeval_judge: "judge_longmemeval.txt"

extraction:
  fact_only: true
  max_facts_per_segment: 15
  reserve_output_tokens_per_segment: 1024
  require_provider_usage: true
  flush_at_sample_end: true
  allow_cross_sample_batch: false
  allow_oversize_singleton: true
  truncate_over_total_context: true

  buffers:
    small:
      max_segments: 6
      max_input_tokens: 16384
      max_total_context_tokens: 24576
    medium:
      max_segments: 6
      max_input_tokens: 16384
      max_total_context_tokens: 24576
    large:
      max_segments: 6
      max_input_tokens: 16384
      max_total_context_tokens: 24576

storage:
  backend: "qdrant"
  mode: "server"
  url: "http://127.0.0.1:6333"
  grpc_port: 6334
  prefer_grpc: false
  timeout_seconds: 30
  api_key_env: ""
  collection_namespace: "{dataset}_{split}_{segmentation_version}_{embedding_hash}_fact_v2"
  collections:
    small: "{namespace}_L"
    medium: "{namespace}_M"
    large: "{namespace}_H"
    assembled: "{namespace}_S"
  vector_size: 1024
  distance: "Cosine"
  explicit_embedding: true
  auto_embedding: false
  require_sample_filter: true
  require_assembly_filter_for_s: true
  payload_indexes:
    - dataset_name
    - split
    - sample_id
    - session_id
    - segment_id
    - memory_tier
    - extraction_run_id
    - assembly_id
    - policy_version
  assembly_retention:
    keep_failed: false
    keep_latest_per_sample: 1
    keep_best_and_final: true

human_readable_export:
  enabled: true
  export_candidates: true
  export_s_every_episode: false
  export_s_on_best_checkpoint: true
  export_s_on_final_policy: true
  include_embedding_values: false

router:
  type: "embedding_mlp"
  algorithm: "actor_critic_lagrangian"
  actions: ["small", "medium", "large"]
  embedding_role: "router"
  freeze_embedding_model: true
  numeric_feature_normalization: "train_split_standard_scaler"
  mlp:
    hidden_dimensions: [512, 128]
    activation: "gelu"
    dropout: 0.1
    policy_output_dimensions: 3
    value_head: true
  budget: 0.5
  cost_normalization: "all_small_all_large"
  checkpoint_dir: "./models/router/checkpoints"
  future_implementations:
    enabled: false
    allowed_types: ["bert_classifier", "llm_lora_1b", "llm_lora_3b"]

evaluation:
  objective_cost_scope: "memory_extraction_only"
  judge_temperature: 0.0
  save_predictions: true
  save_retrieval_traces: true
```

## 二十、实验日志与可复现性

每次实验必须记录：

```text
experiment_id
dataset_name
dataset_version
split_manifest
sample_ids
random_seed
segmentation_method
segmentation_version
processed_manifest_hash
segmented_manifest_hash
small_model_config
medium_model_config
large_model_config
qa_reader_config
judge_config
price_snapshot
embedding_model
embedding_model_hash
tokenizer_paths
prompt_files
prompt_hashes
buffer_config
qdrant_point_schema_version
qdrant_collection_config
qdrant_collection_names
router_algorithm
router_checkpoint
budget
lambda
training_steps
git_commit
created_at
```

每个 episode 至少记录：

```text
episode_id
sample_id
route decisions
tier counts
S fact count
QA ids
QA score
virtual extraction cost
reward
lambda
reader cost
judge cost
duration
```

## 二十一、必须实现的测试

### 21.1 配置与提示词

1. 五个模型角色都能从配置加载；
2. 价格配置完整；
3. 所有 prompt 文件存在并可加载；
4. judge prompt 不再硬编码在 Python；
5. 配置和日志中不存在明文 API key；
6. 模型和 prompt 缺失时明确失败。

### 21.2 本地 embedding

1. 只从配置的本地路径加载；
2. 正式模式不联网；
3. 模型缺失不回退 hashing；
4. embedding 维度固定；
5. L/M/H/S 使用同一模型；
6. Qdrant Collection vector size 与 embedding 输出维度一致；
7. 相同文本产生可复现向量；
8. Qdrant 不触发隐式下载或自动向量化；
9. fact 只向量化 `fact_text`，query 使用同一 memory embedding 模型。

### 21.3 Buffer

1. 三个 buffer 相互独立；
2. 训练候选阶段每个 segment 进入三个 buffer；
3. 部署阶段每个 segment 只进入所选 buffer；
4. 达到 segment 数限制时 flush；
5. 达到 token 限制时 flush；
6. sample 结束时 flush；
7. 不跨 sample 混批；
8. 同配置批次划分可复现；
9. oversize segment 不被静默截断；
10. 失败 batch 不写入无法归属的 facts。

### 21.4 Fact 解析

1. 正确解析包含多个 segment 的 JSON；
2. 支持 `processed_segment_ids` 中存在但 `data[]` 中无事实的 segment；
3. 拒绝未知 segment_id；
4. 拒绝非法 source_id 和不完整的 processed_segment_ids；
5. 限制每个 segment 最大 fact 数；
6. 代码正确拼装 Qdrant Point、vector 和 payload；
7. LLM 返回只包含最小事实 JSON，不包含完整 Qdrant payload。

### 21.5 成本

1. API 返回的 `input_tokens + output_tokens = total_tokens`；
2. 根据输入/输出单价计算出的 batch 实际成本正确；
3. segment 分摊 token 之和分别等于 batch 输入、输出 token；
4. segment 分摊成本之和等于 batch 成本；
5. 同一 segment 的 fact 平均 token 和平均成本一致；
6. 无事实 segment 的 token 和成本仍被记录在 segment/batch 表；
7. provider 不返回 usage 时必须标记 `usage_source=tokenizer_estimate`；
8. retry 和失败调用的 token 与成本被记录；
9. JSON 拼装不计 LLM token 和成本；
10. 第一版不存在缓存价格或缓存成本计算；
11. policy virtual cost 使用 buffer replay，而非简单累加候选库平均成本。

### 21.6 Qdrant L/M/H/S Collections

1. 当前 namespace 创建四个实际 Collections；
2. 所有操作强制带 dataset、split 和 sample filter；
3. S 查询额外强制携带 `assembly_id`；
4. L/M/H 训练期间只读；
5. S 根据路由动作实际复制 Points；
6. QA 只查询 S；
7. 未 ready 的 assembly 不会进入 QA；
8. S 中每个 segment 只来自一个 tier；
9. S 保留候选 Point 来源；
10. Point ID、payload index 和向量维度约束有效；
11. 无 filter 的 search、scroll 和 delete 被封装层拒绝；
12. assembly 清理不会删除其他 sample 或其他 assembly 的 Points。

### 21.7 JSON 导出

1. L/M/H 构建后生成 JSON；
2. 最佳和最终 S 生成 JSON；
3. JSON 不包含完整 vector；
4. 删除 JSON 后实验仍能完整运行；
5. 修改 JSON 不会影响 Qdrant 检索结果。

### 21.8 强化学习与数据泄漏

1. 路由器输入不包含 question、answer、evidence 或 judge label；
2. LoCoMo 按 sample 划分；
3. LongMemEval 相关实例不跨 split；
4. 训练 rollout 不重新提取记忆；
5. 相同 seed 可复现动作采样；
6. checkpoint 加载后 Embedding + MLP 策略一致；
7. feature scaler 与 MLP checkpoint 成对加载；
8. reward 同时保存原始 QA 和成本；
9. 当前实验 manifest 明确记录 `router_type=embedding_mlp`。

## 二十二、基线与实验报告

至少实现：

```text
All-Small：S 全部从 L 复制
All-Medium：S 全部从 M 复制
All-Large：S 全部从 H 复制
Random Router
Length-Heuristic Router
Supervised Embedding + MLP（如启用 warm start）
Budget-Constrained RL Embedding + MLP（当前主方法）
```

所有基线都必须实际组装 Qdrant S assembly，并使用相同的 `sample_id + assembly_id` filter 从 S 评估，不能采用不同检索路径。BERT 和 1B–3B LoRA 路由不属于当前 MVP 基线，留待后续扩展实验。

报告：

```text
总体 QA accuracy
各 question type/category accuracy
Recall@K
virtual memory extraction cost
candidate generation actual cost
reader/judge actual evaluation cost
candidate generation input/output/total tokens
virtual deployment input/output/total tokens
reader/judge input/output/total tokens
各 tier 的输入/输出/总 token
平均每 segment 和每 fact token
每个正确回答消耗的 memory extraction tokens
small/medium/large 选择比例
buffer 平均利用率
平均每 batch segment 数
平均每 batch 输入/输出/总 token
平均每 segment fact 数
无事实 segment 比例
L/M/H/S fact 数
相对 All-Large 的成本节省
相对 All-Small 的质量提升
多预算 QA-Cost Pareto 曲线
```

## 二十三、实现阶段

### 阶段 1：配置与提示词外部化

- 扩展 `models.yaml` 的五个模型角色；
- 扩展 `prices.yaml`；
- 新增 `embeddings.yaml` 和 `rl_router.yaml`；
- 新增 fact batch extraction prompt；
- 将 judge prompt 从代码迁移到 `configs/prompts`；
- 移除配置中的明文 API key，统一使用环境变量。

### 阶段 2：本地 embedding 与 tokenizer

- 规定本地目录；
- 实现显式下载脚本；
- 正式模式强制 local-only；
- 保存模型和 tokenizer 哈希；
- 禁用 hashing fallback。

### 阶段 3：Fact-only 数据结构和 Qdrant

- 删除新流程对 relation/episode 等结构的依赖；
- 实现 Qdrant Collection 创建、版本和 payload index；
- 实现显式 fact embedding、Point upsert 和 filter 查询；
- 实现 batch/segment SQLite WAL 账本；
- 实现人工 JSON 导出。

### 阶段 4：三级 buffer 与批量解析

- 实现 small/medium/large buffer；
- 实现双重容量约束；
- 实现 segment marker 协议；
- 实现 batch parser；
- 实现失败和部分失败处理。

### 阶段 5：L/M/H 候选库生成

- 每个 sample 独立生成；
- 全部 segments 同时进入三个 tier 流程；
- 保存 batch 和成本账本；
- 冻结 L/M/H；
- 导出人工 JSON。

### 阶段 6：S 实际组装

- 实现基于 `assembly_id` 的 staging、校验和 ready 状态；
- 实现 Qdrant L/M/H Point 到 S Point 的实际复制；
- 保存 route decisions；
- 验证每个 segment 只来自一个 tier；
- 确保 QA 只读 S。

### 阶段 7：强化学习与成本模拟

- 实现 BGE-M3 segment embedding 和结构特征标准化；
- 实现当前 `EmbeddingMLPRouter` categorical policy/value head；
- 实现统一 `RouterPolicy` 接口，为后续 BERT/1B–3B LoRA 预留注册点但不启用；
- 实现 buffer replay cost simulator；
- 实现受约束 Actor-Critic/REINFORCE；
- 保存 checkpoint 和 episode 日志。

### 阶段 8：QA/Judge 与完整实验

- 固定 reader、judge 和 prompt；
- 运行所有基线；
- 运行多预算训练；
- 输出 Pareto 曲线和错误分析；
- 导出最佳/最终 S 的人工 JSON。

## 二十四、明确禁止事项

1. 不得让一个 segment 单独调用一次 LLM，除非它自身已达到 buffer token 上限；
2. 不得跨 sample 混合 extraction batch；
3. 不得在每个强化学习 rollout 中重新调用 L/M/H 提取模型；
4. 不得让提取模型生成完整 memory JSON；
5. 不得让模型生成 vector、cost 或 Point ID；
6. 不得在 MVP 中实现 relation、episode、preference、constraint 或 consolidation；
7. 不得把 L/M/H 候选生成总成本当作某个路由策略的部署成本；
8. 不得把 S 实现成仅内存列表或逻辑视图；
9. 不得让 QA 直接读取 L/M/H；
10. 不得让实验读取人工 JSON；
11. 不得执行缺少 `sample_id` filter 的 Qdrant 查询、遍历或删除；
12. 不得静默截断 oversize segment；
13. 不得在正式实验中静默回退 hashing embedding；
14. 不得把 API key 明文保存在配置、日志或 manifest；
15. 不得把 judge prompt 继续硬编码在 Python；
16. 不得将 QA question、answer 或 evidence 输入路由器；
17. 不得把同一 LoCoMo sample 的问题拆到不同 split；
18. 不得把当前重复生成的 LongMemEval `full/val` 直接视为独立训练和验证集。
19. 不得依赖 Qdrant/FastEmbed 隐式自动向量化或运行时下载 embedding；
20. 不得在当前 MVP 主实验中混入 BERT 或 1B–3B LoRA 路由结果；这些实现必须使用新的配置和实验标识。

## 二十五、最终交付内容

完成代码实现后，应提供：

1. 当前架构到新模块的映射；
2. 修改文件清单；
3. 配置文件说明；
4. 五个模型角色和价格快照；
5. prompt 文件清单和版本；
6. 本地 embedding/tokenizer 目录与哈希；
7. buffer 行为和 flush 规则；
8. fact-only 输出协议；
9. 成本分配和 virtual cost simulator；
10. Qdrant Collection、Point/payload schema 与 SQLite 账本；
11. L/M/H/S 生成和组装流程；
12. 人工 JSON 导出说明；
13. 路由算法、奖励和 checkpoint；
14. LoCoMo/LongMemEval split manifest；
15. 单元测试和集成测试结果；
16. All-Small/Medium/Large 等基线；
17. 多预算 QA-Cost Pareto 结果；
18. 已知限制和下一阶段建议。

## 二十六、核心结论

本方案的四个核心数据边界是：

```text
Segment：路由决策和 buffer 入队的最小单位。

Batch：LLM 实际调用和真实成本计量的最小单位。

Fact：模型唯一需要产生的记忆内容，其他字段由代码构造。

Sample：Qdrant payload 过滤、训练 episode 和 QA 评估的逻辑隔离单位。
```

候选库生成时，同一 sample 的全部 segments 分别通过三个 buffer 生成 Qdrant L/M/H Points；强化学习时不再提取，只根据 Embedding + MLP 的动作创建新的 S assembly；训练完成后，新 segment 只进入路由器选中的 buffer。Qdrant 是正式 fact/vector 检索数据源，JSON 只用于人工查看，SQLite 保存无 fact 的批次、成本、状态和路由账本。后续路由器可以替换为 BERT 或 1B–3B LoRA，但必须保持统一 RouterPolicy 接口和相同的 Qdrant/评估流程。
