# 数据预处理与 BERT 主题分割设计

## 1. 文档目标

本文档定义长期记忆路由系统上游的数据预处理和主题分割流程。整个流程拆分为两个相互独立、可重复执行的阶段：

```text
阶段一：原始数据预处理
原始数据
→ session/turn 结构统一
→ 生成递增 turn 时间戳
→ LoCoMo 图片描述追加
→ 保存标准化数据

阶段二：主题分割
读取标准化数据
→ NSP-BERT + TextTiling
或 BERT-MLP + TextTiling
→ 按 sample 保存两套独立分割结果
```

这一设计保证两种分割算法读取完全相同的输入数据，使分割差异只来自算法，而不是预处理差异。

分割结果必须从一开始就按 `sample_id` 隔离存储，因为后续记忆提取、路由、向量库组装和 QA 评估都以 sample 为独立单元。

## 2. 推荐目录结构

```text
InfoBudget/
├── configs/
│   ├── config.yaml
│   └── segmentation/
│       ├── nsp_text_tiling.yaml
│       └── bert_mlp_text_tiling.yaml
│
├── datasets/
│   ├── raw/
│   │   ├── locomo/
│   │   │   └── locomo10.json
│   │   └── longmemeval/
│   │       └── longmemeval_s_cleaned.json
│   │
│   ├── processed/
│   │   ├── locomo/
│   │   │   └── full/
│   │   │       ├── samples.jsonl
│   │   │       ├── sessions.jsonl
│   │   │       ├── turns.jsonl
│   │   │       ├── questions.jsonl
│   │   │       └── manifest.json
│   │   │
│   │   └── longmemeval/
│   │       └── full/
│   │           ├── samples.jsonl
│   │           ├── sessions.jsonl
│   │           ├── turns.jsonl
│   │           ├── questions.jsonl
│   │           └── manifest.json
│   │
│   ├── splits/
│   │   ├── locomo/
│   │   │   └── cv5_seed42.json
│   │   └── longmemeval/
│   │       ├── fixed_80_10_10_seed42_nsp_text_tiling.json
│   │       └── cv5_360_40_100_seed42_nsp_text_tiling.json
│   │
│   └── segmented/
│       ├── locomo/
│       │   └── full/
│       │       ├── nsp_text_tiling/
│       │       │   ├── manifest.json
│       │       │   └── samples/
│       │       │       ├── conv-26/
│       │       │       │   ├── segments.jsonl
│       │       │       │   └── segmentation_trace.json
│       │       │       └── ...
│       │       │
│       │       └── bert_mlp_text_tiling/
│       │           ├── manifest.json
│       │           └── samples/
│       │               ├── conv-26/
│       │               │   ├── segments.jsonl
│       │               │   └── segmentation_trace.json
│       │               └── ...
│       │
│       └── longmemeval/
│           └── full/
│               ├── nsp_text_tiling/
│               │   ├── manifest.json
│               │   └── samples/
│               │       ├── question-or-history-id/
│               │       │   ├── segments.jsonl
│               │       │   └── segmentation_trace.json
│               │       └── ...
│               │
│               └── bert_mlp_text_tiling/
│                   ├── manifest.json
│                   └── samples/
│                       └── ...
│
├── seg_models/
│   ├── bert-base-uncased/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   │
│   └── bert_mlp/
│       └── best.pt
│
└── outputs/
    ├── memory/
    ├── evaluation/
    └── training/
```

目录职责：

- `datasets/raw`：原始数据，只读，不修改；
- `datasets/processed`：完成时间戳、图片描述和结构归一化后的标准数据；
- `datasets/splits`：按 sample_id 定义实验角色，不复制基础数据；
- `datasets/segmented`：两种算法按 sample 产生的分割结果；
- `seg_models`：分割模型和 checkpoint；
- `outputs/memory`：后续 L/M/H 记忆提取结果；
- `outputs/evaluation`：检索与 QA 评估结果；
- `outputs/training`：路由器训练日志与 checkpoint。

主题分割结果不应只保存在 `outputs/memory/.../segments.jsonl` 中，否则主题分割会和记忆提取耦合，难以单独复用和比较。

LoCoMo 的全部 10 个 conversation 统一保存在 `processed/locomo/full` 和
`segmented/locomo/full`。后续 8/2 或 8/1/1 实验划分不修改这些文件，只在
`datasets/splits/locomo` 新增版本化 split manifest。`full` 是基础工件版本，
训练/验证/测试是 manifest 中的逻辑角色，两者含义不同。

LongMemEval 同样把全部 500 个 question/history sample 保存在 `full`。其 split
manifest 在预处理和两种分段完成后生成，因为分层需要读取冻结的 segment 数量；
manifest 只保存 sample ID、数据/分段 manifest 哈希、分层统计和共享会话审计，
不会改写任何预处理或分段 JSONL。两种分段方法共用以 NSP segment-count quartile
构造的固定划分，以保证方法比较时 sample 集合一致。

## 3. 阶段一：原始数据预处理

预处理阶段只负责数据标准化，不执行主题分割，不进行信息评分，也不调用 small、medium 或 large 记忆提取模型。

### 3.1 输入数据

LoCoMo：

```text
datasets/raw/locomo/locomo10.json
```

LongMemEval：

```text
datasets/raw/longmemeval/longmemeval_s_cleaned.json
```

原始文件必须保持不变。所有新增字段和标准化结果写入 `datasets/processed`。

### 3.2 统一样本结构

两个数据集统一转换成：

```text
DatasetDialogueExample
├── sample_id
├── dataset_name
├── split
├── sessions
├── dialogue
├── qa_pairs
└── metadata
```

其中：

- LoCoMo：一个 conversation 是一个 sample；
- LongMemEval：一个 `question_id + haystack history` 是一个 sample；
- `sessions` 保留原始 session 边界；
- `dialogue` 是所有 turns 的扁平视图；
- `qa_pairs` 保留问题、答案和 evidence；
- `metadata` 保留数据集特有字段。

### 3.3 Session 排序

LoCoMo 按 session 数字顺序排列：

```text
session_1
session_2
session_3
...
```

不能按照字符串排序，否则可能出现：

```text
session_1
session_10
session_2
```

LongMemEval 按以下三个列表的原始对应顺序排列：

```text
haystack_session_ids
haystack_dates
haystack_sessions
```

预处理时必须检查三个列表长度一致。

### 3.4 Turn 编号

每个 sample 内部，所有 turn 使用连续的全局编号：

```text
turn_id = 1, 2, 3, ...
```

编号跨 session 连续。例如：

```text
session_1：turn_id 1～20
session_2：turn_id 21～45
```

最终文本展示时使用：

```text
display_turn_index = turn_id - 1
```

因此第一轮显示为 `0.Caroline`，而不是 `1.Caroline`。

建议同时保存 session 内局部编号：

```text
session_turn_index = 0, 1, 2, ...
```

但 segment 展示继续使用 sample 全局的零基编号，以便与 evidence 和 source turn 对齐。

### 3.5 递增时间戳

每个 session 有一个基础时间：

```text
session_base_timestamp
```

然后为 session 内每个 turn 生成严格递增时间戳：

$$
t_i=t_{session}+i\times\Delta t
$$

可以延续当前项目的步长配置：

```yaml
timestamp_policy:
  locomo:
    turn_timestamp_step_ms: 1000
  longmemeval:
    turn_timestamp_step_ms: 500
```

LoCoMo 示例：

```text
Session 原始时间：2023-05-08 13:56:00

turn 0：2023-05-08T13:56:00.000
turn 1：2023-05-08T13:56:01.000
turn 2：2023-05-08T13:56:02.000
```

LongMemEval 示例：

```text
turn 0：2023-05-20T02:21:00.000
turn 1：2023-05-20T02:21:00.500
turn 2：2023-05-20T02:21:01.000
```

每个 turn 应保存：

```json
{
  "timestamp": "2023-05-08T13:56:00.000",
  "weekday": "Mon",
  "timestamp_source": "synthetic_turn_from_session",
  "session_timestamp": "2023-05-08T13:56:00.000",
  "session_raw_timestamp": "01:56 PM on 08 May, 2023",
  "timestamp_offset_ms": 0
}
```

时间戳必须满足：

- 同一 session 内严格递增；
- 不同 session 保留各自原始时间关系。

如果缺失 session 时间戳，不应静默生成虚假的绝对日期。应将该 session 标记为 `missing_timestamp`，写入预处理质量报告，并根据配置决定终止或保留无时间格式。

对于 LoCoMo 和 LongMemEval 正式 benchmark，建议使用严格模式：缺失关键时间戳时预处理失败。

### 3.6 LoCoMo 图片描述处理

如果 LoCoMo turn 中存在非空 `blip_caption`，则在预处理阶段把图片描述追加到对话文本末尾。

统一格式保持与当前项目一致：

```text
原始对话文本 (image description: 图片描述)
```

例如原始文本：

```text
The transgender stories were so inspiring!
```

处理后：

```text
The transgender stories were so inspiring! (image description: a mural with rainbow colors)
```

同时保留原始内容：

```json
{
  "raw_text": "The transgender stories were so inspiring!",
  "text": "The transgender stories were so inspiring! (image description: a mural with rainbow colors)",
  "blip_caption": "a mural with rainbow colors",
  "image_description_appended": true
}
```

如果没有图片描述：

```json
{
  "raw_text": "普通文本",
  "text": "普通文本",
  "blip_caption": "",
  "image_description_appended": false
}
```

必须保证图片描述只追加一次。后续修改当前 `Turn.memory_text()` 时，需要防止发生：

```text
文本 (image description: ...) (image description: ...)
```

推荐规范：

- 预处理阶段完成追加；
- 分割器直接读取处理后的 `text`；
- `memory_text()` 不再重复追加，或检查 `image_description_appended`。

图片 URL、caption 和其他原始图片元数据仍保存在 metadata 中。不下载图片，也不执行新的视觉模型推理。

### 3.7 区分算法输入和输出展示文本

每个 turn 建议保存两种文本形式。

分割算法输入：

```text
segmentation_text
```

只包含：

```text
对话文本 + 可选图片描述
```

不要把时间戳和 turn 编号输入 BERT，否则 BERT 可能学习到无关的日期和编号模式。

最终展示文本：

```text
rendered_line
```

格式为：

```text
[2023-05-08T13:56:00.000, Mon] 0.Caroline: 对话文本
```

因此：

- BERT 使用 `segmentation_text`；
- `segments.jsonl` 中的 `text` 使用多个 `rendered_line` 拼接。

### 3.8 推荐 Turn 结构

```json
{
  "dataset_name": "locomo",
  "split": "full",
  "sample_id": "conv-26",
  "session_id": "session_1",

  "turn_id": 1,
  "display_turn_index": 0,
  "session_turn_index": 0,

  "role": "Caroline",
  "raw_text": "The story was inspiring.",
  "text": "The story was inspiring. (image description: a colorful mural)",
  "segmentation_text": "The story was inspiring. (image description: a colorful mural)",
  "rendered_line": "[2023-05-08T13:56:00.000, Mon] 0.Caroline: The story was inspiring. (image description: a colorful mural)",

  "token_count": 18,

  "timestamp": "2023-05-08T13:56:00.000",
  "weekday": "Mon",
  "timestamp_source": "synthetic_turn_from_session",
  "timestamp_offset_ms": 0,

  "dia_id": "D1:1",
  "blip_caption": "a colorful mural",
  "image_description_appended": true,

  "metadata": {}
}
```

### 3.9 预处理输出文件

#### `samples.jsonl`

保存完整嵌套结构：

```text
sample
├── sessions
├── dialogue
└── qa_pairs
```

#### `sessions.jsonl`

一行一个 session，方便按 session 执行主题分割。

#### `turns.jsonl`

一行一个标准化 turn，是分割阶段最直接的输入。该文件建议在当前项目中新增。

#### `questions.jsonl`

一行一个 QA，保留 evidence 信息。

#### `manifest.json`

建议包含：

```json
{
  "dataset_name": "locomo",
  "split": "full",
  "schema_version": "processed_v3",
  "source_files": ["datasets/raw/locomo/locomo10.json"],
  "source_file_hashes": {
    "locomo10.json": "sha256..."
  },

  "timestamp_policy": {
    "mode": "synthetic_from_session",
    "turn_timestamp_step_ms": 1000
  },

  "image_policy": {
    "append_blip_caption": true,
    "format": " (image description: {caption})"
  },

  "num_samples": 10,
  "num_sessions": 272,
  "num_turns": 0,
  "num_questions": 1986,

  "missing_timestamp_sessions": 0,
  "image_caption_turns": 0,

  "created_at": "...",
  "code_version": "..."
}
```

预处理完成后，`datasets/processed` 应视为不可变输入。两种分割算法都只能读取这份结果，不能各自重新预处理。

## 4. 阶段二：主题分割

第二阶段从以下位置读取已经处理好的 turns：

```text
datasets/processed/{dataset}/{split}
```

然后分别运行：

```text
nsp_text_tiling
bert_mlp_text_tiling
```

两个算法的结果必须写入不同目录，并继续按照 `sample_id` 隔离。

推荐输出位置：

```text
datasets/segmented/{dataset}/{split}/{method}/samples/{sample_id}/
├── segments.jsonl
└── segmentation_trace.json
```

每条 segment 仍必须保存 `sample_id` 和 `session_id`，不能只依赖目录名称。

## 5. 两种算法的共同流程

两个算法的差别只在于如何计算相邻 turn 的连贯性，后面的 TextTiling 边界选择和长短段处理完全一致。

```text
读取一个 sample
    ↓
按 session_id 分组
    ↓
每个 session 独立分割
    ↓
取相邻 turn 对
    ↓
BERT 计算连贯性分数
    ↓
TextTiling 计算深度分数
    ↓
自适应阈值选择边界
    ↓
合并过短段
    ↓
强制拆分过长段
    ↓
按 sample 输出标准 Segment
```

### 5.1 不跨 session 分割

配置保持：

```yaml
preserve_session_boundaries: true
```

分割器必须先按 `session_id` 分组。禁止生成包含两个 session turns 的 segment。

新的 session 必须形成新 segment 起点，其首段可以记录：

```text
boundary_reason = session_boundary
```

### 5.2 构造相邻 turn 对

对于包含 $N$ 个 turns 的 session，构造：

```text
(turn_1, turn_2)
(turn_2, turn_3)
...
(turn_{N-1}, turn_N)
```

总共 $N-1$ 个相邻对。

BERT 输入使用：

```text
turn_i.segmentation_text
turn_{i+1}.segmentation_text
```

不使用 `rendered_line`。

当前配置：

```yaml
bert_max_length: 128
bert_batch_size: 16
```

因此相邻文本对经过 BERT tokenizer 后最多保留 128 token，超出部分截断。正式实验中必须在 trace 或 manifest 中记录这些配置。

### 5.3 连贯性曲线

对每个相邻 turn 对得到一个连贯性分数：

$$
c_i=P(\text{turn}_i\text{ 与 }\text{turn}_{i+1}\text{ 连贯})
$$

一个 session 得到：

$$
C=[c_1,c_2,\ldots,c_{N-1}]
$$

分数越高，表示相邻 turns 越可能属于同一主题；分数越低，越可能存在主题边界。

### 5.4 TextTiling 深度分数

对每个连贯性位置计算局部谷底深度：

$$
d_i=\frac{p_i^{left}+p_i^{right}-2c_i}{2}
$$

其中：

- $p_i^{left}$ 是左侧局部峰值；
- $p_i^{right}$ 是右侧局部峰值；
- $c_i$ 是当前位置连贯性分数。

如果当前位置明显低于左右两边，$d_i$ 会较大，说明这里可能发生主题切换。

### 5.5 自适应阈值

当前阈值公式：

$$
\tau=\operatorname{mean}(D)+\alpha\operatorname{std}(D)
$$

当前配置：

```yaml
adaptive_alpha: 0.5
```

即：

$$
\tau=\operatorname{mean}(D)+0.5\operatorname{std}(D)
$$

如果 $d_i>\tau$，则在 `turn_i` 和 `turn_{i+1}` 之间生成候选边界。

### 5.6 密集边界处理

当前配置：

```yaml
min_boundary_gap: 1
```

如果多个边界距离小于最小间隔，则保留深度分数更大的边界。

### 5.7 合并过短段

当前配置：

```yaml
min_segment_turns: 2
min_segment_tokens: 20
merge_short_segment: true
```

如果候选段满足任一条件：

```text
turn 数 < 2
token 数 < 20
```

则尝试与左侧或右侧段合并。与哪一侧的连贯性更高，就合并到哪一侧。

合并后记录：

```text
boundary_reason = merged_short
```

### 5.8 拆分过长段

当前配置：

```yaml
max_segment_turns: 12
max_segment_tokens: 768
```

如果满足任一条件：

```text
turn 数 > 12
token 数 > 768
```

则必须继续拆分。拆分点选择规则：

1. 优先选择内部连贯性最低的位置；
2. 连贯性相同时，选择左右 token 更均衡的位置；
3. 仍相同时，选择位置更靠前的点。

记录：

```text
boundary_reason = forced_split
```

如果单个 turn 已经超过 768 token，turn 级分割无法继续拆解，预处理质量报告应单独标记这种情况。

## 6. 方法一：NSP-BERT + TextTiling

方法名：

```text
nsp_text_tiling
```

当前实现位置：

```text
infobudget/segmentation/nsp_text_tiling.py
```

### 6.1 模型目录

```text
seg_models/bert-base-uncased/
```

当前实际文件包括：

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.txt
```

配置：

```yaml
bert_model_dir: "./seg_models/bert-base-uncased"
```

模型只从本地加载：

```text
local_files_only = true
```

运行分割时不应自动访问网络下载模型。

### 6.2 连贯性计算

对相邻 turns 构造 BERT sentence pair：

```text
[CLS] turn_i [SEP] turn_{i+1} [SEP]
```

使用：

```text
AutoModelForNextSentencePrediction
```

得到 NSP logits，经过 softmax 后取类别 0 的概率：

```text
probabilities[:, 0]
```

将其视为 `P(IsNext)`，也就是两个 turns 属于连贯后续内容的概率。然后把整条 NSP 连贯性曲线交给共同的 TextTiling 流程。

### 6.3 输出边界原因

由 TextTiling 检测出的主题边界记录为：

```text
nsp_texttiling_depth
```

此外仍可能出现：

```text
start
session_boundary
single_turn
merged_short
forced_split
```

## 7. 方法二：BERT-MLP + TextTiling

方法名：

```text
bert_mlp_text_tiling
```

当前实现位置：

```text
infobudget/segmentation/bert_mlp_text_tiling.py
```

### 7.1 模型目录

基础 BERT：

```text
seg_models/bert-base-uncased/
```

微调 checkpoint：

```text
seg_models/bert_mlp/best.pt
```

配置：

```yaml
bert_model_dir: "./seg_models/bert-base-uncased"
bert_mlp_checkpoint: "./seg_models/bert_mlp/best.pt"
bert_mlp_activation: "relu"
```

checkpoint 需要同时包含：

```text
bert.*
coherence_decoder.*
```

因此它不只是一个单独 MLP，而是包含微调后的 BERT 参数和连贯性分类器参数。

### 7.2 连贯性计算

对相邻 turns 同样构造 sentence pair：

```text
[CLS] turn_i [SEP] turn_{i+1} [SEP]
```

然后：

1. 使用 checkpoint 中的微调 BERT；
2. 读取最后一层 `[CLS]` 表示；
3. 将 `[CLS]` 输入 `coherence_decoder`；
4. 对 logits 执行 softmax；
5. 取类别 0 的概率作为连贯性分数。

公式：

$$
h_i=\operatorname{BERT}_{\theta}(u_i,u_{i+1})_{CLS}
$$

$$
c_i=\operatorname{softmax}(\operatorname{MLP}(h_i))_0
$$

两种方法的主要差别：

```text
NSP：
使用预训练 BERT 自带的 Next Sentence Prediction head。

BERT-MLP：
使用专门训练的 BERT 参数和 coherence_decoder。
```

### 7.3 输出边界原因

由该方法产生的 TextTiling 主题边界记录为：

```text
bert_mlp_texttiling_depth
```

## 8. 分割后的文本格式

每个 segment 由一个或多个标准 `rendered_line` 拼接：

```text
[2023-05-08T13:56:00.000, Mon] 0.Caroline: 对话文本
[2023-05-08T13:56:01.000, Mon] 1.Melanie: 对话文本
[2023-05-08T13:56:02.000, Mon] 2.Caroline: 对话文本
```

如果 LoCoMo turn 包含图片描述：

```text
[2023-05-08T13:56:02.000, Mon] 2.Caroline: The story was inspiring. (image description: a mural with rainbow colors)
```

segment 的 `text` 字段就是这些行使用换行符连接后的完整字符串。

## 9. Segment 数据结构

推荐保存：

```json
{
  "dataset_name": "locomo",
  "split": "full",
  "sample_id": "conv-26",
  "session_id": "session_1",

  "segmentation_method": "nsp_text_tiling",
  "segmentation_version": "nsp_text_tiling_v1",
  "preprocessing_version": "processed_v3",

  "segment_id": "conv-26:nsp_text_tiling:seg_000001",
  "segment_index": 1,

  "start_turn": 1,
  "end_turn": 2,
  "turn_ids": [1, 2],

  "start_timestamp": "2023-05-08T13:56:00.000",
  "end_timestamp": "2023-05-08T13:56:01.000",

  "text": "[2023-05-08T13:56:00.000, Mon] 0.Caroline: ...\n[2023-05-08T13:56:01.000, Mon] 1.Melanie: ...",

  "token_count": 30,
  "mean_coherence_score": 0.73,
  "boundary_reason": "merged_short",

  "source_turn_count": 2,
  "source_content_hash": "sha256...",

  "model_name": "google-bert/bert-base-uncased",
  "model_path": "seg_models/bert-base-uncased",
  "checkpoint_path": null,

  "bert_max_length": 128,
  "adaptive_alpha": 0.5
}
```

BERT-MLP 结果中：

```json
{
  "segmentation_method": "bert_mlp_text_tiling",
  "model_path": "seg_models/bert-base-uncased",
  "checkpoint_path": "seg_models/bert_mlp/best.pt"
}
```

两种方法的 `segment_id` 必须包含方法名，因为相同 turns 在两种方法下可能形成不同 segment。

## 10. 按 sample 存储要求

分割结果必须按 sample 独立存储：

```text
datasets/segmented/{dataset}/{split}/{method}/samples/{sample_id}/
├── segments.jsonl
└── segmentation_trace.json
```

原因：

1. 后续每个 sample 独立执行记忆提取；
2. 每个 sample 独立生成 L/M/H 候选记忆；
3. 每个 sample 独立组装策略数据库 S；
4. QA 评估以 sample/history 为隔离边界；
5. 可以防止向量检索跨 sample 污染；
6. 可以单独重跑失败 sample，而不必重跑整个数据集；
7. 可以独立统计每个 sample 的分割数量、成本和 QA 表现。

即使已经按照目录隔离，每条 segment 仍必须保存：

```text
dataset_name
split
sample_id
session_id
segmentation_method
```

不能只依赖父目录推断归属。

## 11. Segmentation trace

每个 sample、每种方法保存独立 trace：

```json
{
  "dataset_name": "locomo",
  "sample_id": "conv-26",
  "segmentation_method": "nsp_text_tiling",

  "preprocessing_manifest_hash": "sha256...",
  "model_hash": "sha256...",
  "checkpoint_hash": null,

  "sessions": [
    {
      "session_id": "session_1",
      "turn_ids": [1, 2, 3, 4],
      "coherence_scores": [0.92, 0.18, 0.88],
      "depth_scores": [0.0, 0.72, 0.0],
      "threshold": 0.41,
      "candidate_boundaries": [3],
      "final_boundaries": [1, 3]
    }
  ],

  "num_input_turns": 419,
  "num_segments": 0,
  "created_at": "..."
}
```

trace 用于分析：

- 两种模型在哪些位置产生了不同连贯性；
- 某个边界为什么被选中；
- 边界是否因短段合并而消失；
- 是否因为长度约束被强制切分。

## 12. 分割输出 manifest

每种算法目录保存独立 `manifest.json`：

```json
{
  "dataset_name": "locomo",
  "split": "full",
  "segmentation_method": "nsp_text_tiling",
  "segmentation_version": "nsp_text_tiling_v1",

  "processed_data_dir": "datasets/processed/locomo/full",
  "processed_manifest_hash": "sha256...",

  "model_dir": "seg_models/bert-base-uncased",
  "model_hash": "sha256...",
  "checkpoint_path": null,

  "parameters": {
    "bert_max_length": 128,
    "bert_batch_size": 16,
    "adaptive_alpha": 0.5,
    "min_boundary_gap": 1,
    "min_segment_turns": 2,
    "min_segment_tokens": 20,
    "max_segment_turns": 12,
    "max_segment_tokens": 768,
    "preserve_session_boundaries": true
  },

  "num_samples": 10,
  "num_sessions": 272,
  "num_turns": 0,
  "num_segments": 0,

  "created_at": "...",
  "code_version": "..."
}
```

BERT-MLP manifest 额外记录：

```text
checkpoint_path
checkpoint_hash
bert_mlp_activation
```

## 13. 配置建议

公共配置：

```yaml
preprocessing:
  append_locomo_blip_caption: true
  image_description_template: " (image description: {caption})"

  timestamp:
    strict_missing_timestamp: true
    locomo_turn_step_ms: 1000
    longmemeval_turn_step_ms: 500

  save_flat_turns: true
  schema_version: "processed_v3"

segmentation:
  preserve_session_boundaries: true

  bert_model_dir: "./seg_models/bert-base-uncased"
  bert_mlp_checkpoint: "./seg_models/bert_mlp/best.pt"

  bert_max_length: 128
  bert_batch_size: 16
  adaptive_alpha: 0.5
  bert_mlp_activation: "relu"

  min_boundary_gap: 1
  min_segment_turns: 2
  min_segment_tokens: 20
  max_segment_turns: 12
  max_segment_tokens: 768
  merge_short_segment: true
```

NSP 方法：

```yaml
method: "nsp_text_tiling"
output_dir: "./datasets/segmented/{dataset}/{split}/nsp_text_tiling"
```

BERT-MLP 方法：

```yaml
method: "bert_mlp_text_tiling"
output_dir: "./datasets/segmented/{dataset}/{split}/bert_mlp_text_tiling"
```

## 14. 数据校验要求

### 14.1 预处理校验

必须验证：

1. 每个 sample 的 `turn_id` 连续且唯一；
2. 每个 session 内时间戳严格递增；
3. `weekday` 与时间戳一致；
4. LoCoMo 图片描述只追加一次；
5. 无图片的 turn 文本不发生变化；
6. 原始文本仍可追溯；
7. session 数、turn 数和 QA 数与原始数据一致；
8. evidence 引用仍能映射到预处理后的 turns；
9. LongMemEval 三个 haystack 列表长度一致。

### 14.2 分割校验

每种方法分别验证：

1. 所有输入 turn 恰好被一个 segment 覆盖；
2. 不丢 turn；
3. 不重复 turn；
4. 不跨 session；
5. segment 内 turn 顺序严格递增；
6. `segment.text` 与对应 turns 的 `rendered_line` 完全一致；
7. 两种方法读取相同的 processed manifest；
8. NSP 使用正确的本地 BERT 模型；
9. BERT-MLP 使用正确的 checkpoint；
10. 模型或 checkpoint 缺失时立即失败，不回退到其他分割算法；
11. 相同配置和模型下重复运行结果一致；
12. 不同 sample 的 segment 不写入彼此目录；
13. 每个 segment 的 `sample_id` 与目录中的 sample 一致。

## 15. 最终流程定义

```text
1. 原始数据只读保存于 datasets/raw。

2. 运行数据预处理：
   - 统一 sample/session/turn/QA 结构；
   - 为 session 内 turns 生成递增时间戳；
   - 计算 weekday；
   - LoCoMo 图片 caption 追加到文本末尾；
   - 保存 raw_text、processed text、segmentation_text 和 rendered_line；
   - 结果保存到 datasets/processed。

3. 冻结 processed 数据：
   - 生成 manifest 和内容哈希；
   - 后续两种算法读取完全相同的 processed 数据。

4. 运行 NSP-BERT + TextTiling：
   - 使用 seg_models/bert-base-uncased；
   - 计算相邻 turns 的 NSP 连贯性；
   - 使用 TextTiling 深度分数选择边界；
   - 按 sample 保存到
     datasets/segmented/{dataset}/{split}/nsp_text_tiling/samples/{sample_id}。

5. 运行 BERT-MLP + TextTiling：
   - 使用 seg_models/bert-base-uncased；
   - 加载 seg_models/bert_mlp/best.pt；
   - 使用微调 BERT 和 coherence_decoder 计算连贯性；
   - 使用相同 TextTiling 后处理；
   - 按 sample 保存到
     datasets/segmented/{dataset}/{split}/bert_mlp_text_tiling/samples/{sample_id}。

6. 分割结果统一保存为：
   [timestamp, weekday] zero_based_turn.role: processed_text

7. 后续记忆提取、L/M/H 候选库和路由器训练只能读取
   datasets/segmented 中已经冻结且按 sample 隔离的分割结果。
```

预处理负责生成唯一、稳定、可复用的标准 turns；两个 BERT 分割器只负责决定这些 turns 应如何组合成 topic segments。按 sample 隔离是从主题分割一直延续到记忆提取、数据库组装和 QA 评估的基本数据边界。
