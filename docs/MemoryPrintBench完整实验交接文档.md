# MemoryPrintBench 完整实验交接文档

> 交接日期：2026-08-13  
> 目标工作目录：`S:\Workfile\MemoryPrintBench`  
> 任务目标：从 HaluMem 与 MemOps 构造冻结探针，使用统一的 `Fact + source_ids` 提取协议评价候选模型，最终为每个候选模型生成 7 维 MemoryPrint。  
> 本文是切换工作目录后的主操作文档；环境部署细节仍可参考 InfoBudget2 中的 `docs/MemoryPrint外部评测环境部署与运行说明.md`。

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan/handoff`
- Origin Date: `2026-08-13`
- Verification Status: `ANALYZED`
- Version Label: `memoryprintbench-handoff-v1`
- Upstream Dependencies:
  - HaluMem commit `c29025f43b347f68fc36a06bee8ed29b4dc6c3fb`
  - MemOps commit `312af65e2c7b6d1b70f062ffa8b4cde32aaf6f35`
  - HaluMem-Medium SHA-256 `486FBC130A5C8781A2AF27FFA508A1D7855245137AA449C193AC4D29C45634E7`
  - MemoryPrint 7 维指标定义 `memoryprint-v1`

## 1. 先明确本文件夹要完成什么

本文件夹的最终职责不是复现两个数据集的所有排行榜指标，而是完成以下闭环：

```text
官方原始数据
  → 无 Gold 泄漏的标准化探针
  → 每个候选模型使用同一提示提取 Fact + source_ids
  → HaluMem 三项官方兼容提取评分
  → MemOps 四项项目适配评分
  → 每个模型的 7 维 MemoryPrint + mask + 置信区间
  → 导出给 InfoBudget2
```

最终核心向量为：

\[
\Psi_7(m)=
[R_H, P_H^{target}, A_H, S_M^{current}, B_M, R_M^{old}, F1_M^E]
\]

明确排除：

- HaluMem QA Correct/Hallucination/Omission；
- MemOps 最终 Answer Accuracy；
- Retriever 和 Reader 的能力；
- 价格、延迟和 API 稳定性；
- Mem0、Zep 等完整记忆系统的数据库维护能力。

这些可以进入诊断或下游效用验证，但不能进入 7 维模型提取指纹。

## 2. 一个关键边界：核心指纹不应直接使用 Mem0 的最终输出

HaluMem 官方 `eval_memzero.py` 调用的是 Mem0 云端 `MemoryClient`。如果直接运行它，测到的是“Mem0 服务 + Mem0 内部配置 + Reader/Judge”的综合表现，不一定是指定候选 LLM 的直接记忆提取能力。

因此分成两条流水线：

| 流水线 | 用途 | 是否生成核心 MemoryPrint |
|---|---|---:|
| HaluMem + Mem0 官方兼容运行 | 验证官方工程和环境，或报告完整记忆系统结果 | 否 |
| 固定候选 LLM 直接提取 `Fact + source_ids` | 比较 Qwen、Llama 等候选模型本身的写入能力 | 是 |

核心实验应编写自己的 direct-extractor adapter，但复用 HaluMem 官方的 Memory Integrity 与 Memory Accuracy Judge prompt/函数。不要直接修改 `upstream/HaluMem` 和 `upstream/MemOps`；适配代码放在本项目自己的 `src/` 下。

只有当 Mem0 能显式固定并确认使用待测候选模型，而且所有候选模型使用完全相同的 Mem0 配置时，Mem0 输出才可以作为另一组“系统级指纹”，但它仍应与本研究的模型级 MemoryPrint 分开命名和报告。

## 3. 当前已有内容

```text
S:\Workfile\MemoryPrintBench\
├── datasets\
│   └── halumem\
│       ├── HaluMem-Medium.jsonl
│       └── HaluMem-Long.jsonl
├── outputs\
└── upstream\
    ├── HaluMem\
    │   ├── .venv\
    │   ├── data\
    │   └── eval\
    └── MemOps\
        ├── .venv\
        └── generated_result\
```

当前环境状态：

- `uv 0.10.12`；
- 两个上游仓库均使用独立 Python 3.11 环境；
- HaluMem 环境已安装 `mem0ai==0.1.118` 和评分依赖；
- MemOps 环境已安装官方 `requirements.txt`；
- 两个环境均已通过 `uv pip check`；
- HaluMem 的 `.env` 已从官方模板创建，但仍是占位符；
- 尚未运行需要真实 API key 的在线实验。

## 4. 建议补齐的项目目录

切换到本文件夹后，建议把自研代码和实验产物组织为：

```text
MemoryPrintBench\
├── MEMORYPRINT_HANDOFF.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── configs\
│   ├── fingerprint_v1.yaml
│   ├── models.yaml
│   └── prompts\
│       ├── extract_facts_v1.txt
│       ├── halumem_judge_lock.json
│       └── memops_alignment_judge_v1.txt
├── schemas\
│   ├── fact_output_v1.schema.json
│   ├── probe_input_v1.schema.json
│   └── memoryprint_v1.schema.json
├── src\memoryprintbench\
│   ├── preprocess_halumem.py
│   ├── preprocess_memops.py
│   ├── freeze_probe.py
│   ├── run_extractor.py
│   ├── score_halumem_extraction.py
│   ├── derive_memops_state.py
│   ├── score_memops_core.py
│   ├── aggregate_memoryprint.py
│   └── validate_artifacts.py
├── tests\
├── data\processed\
│   ├── halumem\probe_v1\
│   └── memops\probe_v1\
├── manifests\
├── runs\
├── exports\
├── datasets\
├── outputs\
└── upstream\
```

约束：`datasets/`、`runs/`、`outputs/`、`.env` 和所有 `.venv/` 必须被 `.gitignore` 排除。可以提交脚本、schema、配置、冻结 ID 清单和不含第三方文本的聚合结果；不要把 HaluMem 修改或重打包后提交到 Git。

## 5. 根目录 `uv` 编排环境

现有两个 `.venv` 属于上游项目。建议再建立一个根目录编排环境，用于预处理、调用候选模型、对齐、聚合和测试：

```powershell
Set-Location S:\Workfile\MemoryPrintBench
uv init --bare --python 3.11
uv venv --python 3.11
uv add pydantic pyyaml jsonlines pandas pyarrow numpy scipy scikit-learn openai tenacity tqdm
uv add --dev pytest ruff
```

不要删除或复用以下环境：

```text
upstream\HaluMem\.venv
upstream\MemOps\.venv
```

三套环境的职责分别是：根环境运行自研流水线；HaluMem 环境运行官方 Judge 兼容代码；MemOps 环境运行官方脚本或核对官方输出。

## 6. 数据预处理总原则

### 6.1 输入与 Gold 必须物理分离

每个数据集都生成两类文件：

- `inputs.jsonl`：可以发送给候选模型，只含 probe ID、对话、source ID 和非答案元数据；
- `gold.jsonl`：只供评分器读取，包含 memory points、operation trace、target、old/new value 和 evidence。

候选模型调用代码不得加载 `gold.jsonl`。运行前增加自动扫描，确认 candidate-facing payload 中不存在下列字段：

```text
memory_points
memory_content
operations
target_fact
expected_answer
gold_memory_state
gold_provenance
old_value
new_value
judge_rubric
```

### 6.2 所有模型使用同一输入

必须在调用第一个候选模型之前冻结：

- probe ID；
- `fingerprint` / `audit` 划分；
- 消息排序和 source ID；
- 提取 prompt；
- JSON schema；
- 温度、最大输出 token 和重试规则；
- Judge 模型及 Judge prompt；
- 指标公式和缺失值规则。

冻结之后不能根据某个模型的结果改变样本或评分口径。

### 6.3 统一 source ID

候选模型看到的 `source_ids` 一律是一个 probe 内从 0 开始的局部整数。每个 probe 另存映射表：

```json
{
  "source_id": 0,
  "original_segment_index": 1,
  "original_turn_index": 1,
  "role": "user",
  "content_sha256": "..."
}
```

HaluMem 原始 `dialogue_turn` 已接近 session 内局部编号；MemOps 的 Gold 使用 `(segment_index, turn_index)`，因此必须通过上述映射还原，不能直接把模型输出的整数当作 MemOps turn index。

## 7. HaluMem 数据预处理

### 7.1 使用的数据

核心指纹只使用：

```text
datasets\halumem\HaluMem-Medium.jsonl
```

实际核验规模：

| 统计项 | 数量 |
|---|---:|
| 用户 | 20 |
| session | 1,387 |
| dialogue turn | 60,146 |
| question | 3,467 |
| memory point | 14,948 |
| update memory point | 3,122 |
| Persona / Event / Relationship | 9,116 / 4,550 / 1,282 |
| system / secondary / interference | 1,849 / 10,451 / 2,648 |

Long 版本只用于后续长历史压力测试，不混入第一版核心指纹。

### 7.2 标准化单位

一个 HaluMem session 对应一个 probe。生成稳定 ID：

```text
halumem-medium:<user_uuid>:s<四位session序号>
```

输入行建议结构：

```json
{
  "schema_version": "probe-input-v1",
  "dataset": "halumem-medium",
  "probe_id": "halumem-medium:<uuid>:s0001",
  "group_id": "<user_uuid>",
  "messages": [
    {
      "source_id": 0,
      "role": "user",
      "content": "...",
      "timestamp": "..."
    }
  ],
  "source_map": [
    {
      "source_id": 0,
      "original_dialogue_turn": 0,
      "content_sha256": "..."
    }
  ]
}
```

Gold 行保留：

- `memory_content`；
- `memory_type`；
- `memory_source`；
- `is_update`；
- `original_memories`；
- `event_source`；
- `importance`；
- 原始用户 UUID 和 session 序号。

QA 的 `question`、`answer` 和 `evidence` 可以保存在 `gold_full.jsonl` 供诊断，但不能进入候选输入，也不能进入 7 维聚合。

### 7.3 处理 update 与 interference

核心 HaluMem 三维属于 Memory Extraction：

- 非 `interference`、非 update 的 Gold 用于 Recall；
- `interference` Gold 用于 FMR 诊断，不进入核心 Recall 分母；
- `is_update=True` 的 Gold 单独保留为更新诊断，不直接混入 HaluMem 核心三维；
- 所有候选事实都进入 Memory Accuracy Judge，以检查是否受到对话或 Gold 支持。

不能把自研 direct-extractor 的输出不加处理地塞进官方 `evaluation.py`。官方代码依靠 `memories_from_system` 决定是否把 update memory 转入更新评分；直接提取流程没有该字段时，update 可能错误进入完整性评分。正确做法是编写 `score_halumem_extraction.py`：复用官方 `eval_tools.py` 的两个提取 Judge，并在自研编排层明确过滤 update 和 QA。

### 7.4 冻结 600 个 session

固定随机种子建议设为 `20260813`，抽取 600 个 session：

- 480 个 `fingerprint`；
- 120 个 `audit`。

执行多标签分层，至少覆盖：memory type、update、interference、Gold 数量、对话长度、importance 区间和 UUID。优先按用户 UUID 做 group-aware 划分，避免同一用户同时进入 fingerprint 与 audit；若精确 480/120 与严格 group split 冲突，优先保持 group 不重叠，并在 manifest 记录实际数量。

输出：

```text
data\processed\halumem\probe_v1\all_sessions.jsonl
data\processed\halumem\probe_v1\inputs.jsonl
data\processed\halumem\probe_v1\gold.jsonl
data\processed\halumem\probe_v1\source_map.jsonl
manifests\halumem_probe_v1_split.jsonl
manifests\halumem_probe_v1_stats.json
```

## 8. MemOps 数据预处理

### 8.1 主数据源

主指纹使用 adjacent evidence：

```text
upstream\MemOps\generated_result\2-evidence_conversation\*.json
```

第 4 阶段的 50-session 注入数据只用于 longitudinal 压力测试：

```text
upstream\MemOps\generated_result\4-inject_evidence_with_distractors\*.json
```

不要把 longitudinal 分数与 adjacent 核心分数平均。

### 8.2 实际文件构成及 200 个样本的修正方案

`2-evidence_conversation` 实际有 403 个已生成文件：

| operation type | 文件数 |
|---|---:|
| Remember | 68 |
| Update | 77 |
| TrajectoryOps | 82 |
| Forget | 80 |
| Reflect | 96 |

由于当前输出只有 `Fact + source_ids`，不能可靠表达 Forget/Delete；Reflect 又包含受约束推断。因此核心 200 个 probe 使用：

- 全部 68 个 Remember；
- 全部 77 个 Update；
- 从 82 个 TrajectoryOps 中分层抽取 55 个，只评分其中可由显式 Remember/Update trace 定义的当前状态；
- Forget 与 Reflect 不进入核心 200，保留为诊断。

这比早期文档中“约 80 个 Remember”更可执行，因为官方已生成的验证文件实际只有 68 个 Remember。

### 8.3 标准化与状态重放

一个 JSON 文件对应一个 MemOps probe。建议 ID：

```text
memops-adjacent:<文件stem>
```

对 `conversations[]` 按 `segment_index` 排序，对每个 segment 内 dialogue 按原顺序展开，并为每条消息分配局部 `source_id`。`source_map` 必须保存：

- 局部 `source_id`；
- 原始 `segment_index`；
- 原始 1-based `turn_index`；
- role；
- content hash。

Gold 状态通过 `operations[]` 重放得到。第一版规则冻结为：

1. 按 `trigger_span.segment_index`、`trigger_span.turn_index`、`chain_step` 排序；
2. 只用 `remember` 和 `update` 构造核心当前状态；
3. `validity=confirmed` 才能成为当前有效值；
4. tentative 不覆盖已确认值；
5. retracted/forget 的目标不进入第一版核心状态分母；
6. 每个 `target_id` 的最后一个已确认有效值作为 `current_value`；
7. 更早的已确认值进入 `stale_values`；
8. 当前值的决定性证据来自对应 operation 的 `evidence_spans`；
9. `answer.expected_answer` 和最终 QA 结果不参与核心状态推导。

派生 Gold 行建议包含：

```json
{
  "probe_id": "memops-adjacent:A01_update",
  "states": [
    {
      "target_id": "job_title",
      "target_name": "Current job title at Bridgemark Solutions",
      "current_value": "Senior Data Analyst",
      "stale_values": ["Junior Data Analyst", "Data Analyst"],
      "decisive_evidence": [
        {"segment_index": 2, "turn_index": 3}
      ],
      "eligible_current_state": true,
      "eligible_stale_rejection": true
    }
  ]
}
```

### 8.4 冻结 160/40 划分

200 个样本分成：

- 160 个 `fingerprint`；
- 40 个 `audit`。

文件名中的基础案例 ID，例如 `A01`，作为 `group_id`。同一基础案例的 Remember/Update/TrajectoryOps 变体不能跨 fingerprint 与 audit，以防模板和人物信息泄漏。分层变量至少包含 operation type、recency trap、update chain、multi-target、multi-hop、negative seed 和 adversarial injection。

输出：

```text
data\processed\memops\probe_v1\all_adjacent.jsonl
data\processed\memops\probe_v1\inputs.jsonl
data\processed\memops\probe_v1\gold.jsonl
data\processed\memops\probe_v1\source_map.jsonl
manifests\memops_probe_v1_split.jsonl
manifests\memops_probe_v1_stats.json
```

## 9. 候选模型统一提取协议

### 9.1 输出 schema

所有候选模型必须返回：

```json
{
  "probe_id": "...",
  "facts": [
    {
      "fact": "Atomic, durable memory fact.",
      "source_ids": [0, 2]
    }
  ]
}
```

规则：

- `fact` 必须是原子、可独立理解的长期记忆事实；
- `source_ids` 是非空、去重、升序的整数数组；
- source ID 必须属于该 probe 的可见输入；
- 不允许输出数据库操作、解释、Markdown 或额外字段；
- 没有可写入事实时返回空 `facts`，这不是解析失败；
- 一个 probe 一次调用，避免跨 probe 的 source ID 混淆。

### 9.2 解码与失败策略

建议固定：

- temperature `0`；
- top-p `1` 或 provider 默认值，但所有模型一致；
- 固定最大输出 token；
- provider 支持 seed 时固定 seed；
- 网络失败可重试；
- JSON/schema 失败最多进行一次固定 repair prompt；
- repair 后仍失败则记录失败，不允许手工补答案。

每次请求都保存 raw response、parsed output、usage、request hash、prompt hash、模型实际返回名、时间戳、重试次数和错误类别。

## 10. 七个最终指标及精确来源

所有维度都在 `[0,1]`，且高值更优。

| # | 输出键 | 含义 | 数据集 | 核心计算 |
|---:|---|---|---|---|
| 1 | `halumem_recall` | 事实覆盖能力 | HaluMem | 官方严格 Recall；Gold Judge 得分为 2 才算覆盖 |
| 2 | `halumem_target_precision` | 目标记忆精确率 | HaluMem | 对齐官方 `target_accuracy(all)` |
| 3 | `halumem_memory_accuracy` | 事实忠实度 | HaluMem | 对齐官方 `weighted_accuracy(all)` |
| 4 | `memops_current_state` | 当前状态识别 | MemOps | 正确 target 且 current value 正确 |
| 5 | `memops_target_binding` | 目标绑定能力 | MemOps | 主体/属性 target 匹配 |
| 6 | `memops_stale_rejection` | 旧值拒绝能力 | MemOps | `1 - stale_as_current_rate` |
| 7 | `memops_evidence_f1` | 证据定位能力 | MemOps | 预测 source 映射到原始 span 后的 micro F1 |

### 10.1 HaluMem 三维

核心 JSON path 固定为：

```text
R_H              = overall_score.memory_integrity.recall(all)
P_H_target       = overall_score.memory_accuracy.target_accuracy(all)
A_H              = overall_score.memory_accuracy.weighted_accuracy(all)
```

Judge 无效响应不能悄悄从分母中删除。核心保存 `all` 口径，同时报告 valid ratio；若某一核心 Judge 的 valid ratio 小于 0.98，该模型—数据集运行标记为无效并重跑 Judge，不直接发布分数。

### 10.2 MemOps 当前状态

对每个 eligible Gold state：

\[
S_i=\mathbf 1[\widehat{target}_i=target_i]\cdot
\mathbf 1[\widehat{value}_i=current\_value_i]
\]

\[
S_M^{current}=\frac{1}{N}\sum_i S_i
\]

未提取该状态计 0。时间、数字、否定和专有名词必须保留；普通同义改写可以由固定 Judge 接受。

### 10.3 MemOps Target Binding

对每个 eligible Gold state，固定 Judge 判断候选事实是否绑定到正确 `target_id/target_name`：

\[
B_M=\frac{1}{N}\sum_i\mathbf 1[\widehat{target}_i=target_i]
\]

使用一对一最大匹配；一条宽泛候选 Fact 不能重复覆盖多个 Gold target。缺失计 0，主体绑定错误计 0。

### 10.4 MemOps Stale Rejection

只在存在至少一个 `stale_value` 的 update chain 上计算：

\[
R_M^{old}=1-\frac{\#\{\text{旧值仍被断言为当前值}\}}{N_{update\_chain}}
\]

明确带过去时间限定的历史事实不算错误；只有把旧值继续写成当前有效状态才算 stale error。若没有 eligible update chain，该维 mask 为 0，不得填 1。

### 10.5 MemOps Evidence F1

模型局部 `source_ids` 先通过 `source_map` 转成 `(segment_index, turn_index)`。对已经完成语义匹配的事实，预测证据集合与 decisive Gold evidence 集合计算 micro precision/recall/F1：

\[
P_E=\frac{\sum TP_i}{\sum(TP_i+FP_i)},\quad
R_E=\frac{\sum TP_i}{\sum(TP_i+FN_i)}
\]

\[
F1_M^E=\frac{2P_ER_E}{P_E+R_E}
\]

第一版把 Evidence F1 定义为“已语义匹配事实上的证据定位能力”，避免再次编码事实召回。必须同时报告 `evidence_eligible_fact_count` 和 `evidence_coverage`；如果 eligible matched fact 数低于预注册门槛，则该维 mask 为 0，而不是给出不稳定高分。

## 11. 语义 Judge 与匹配规则

HaluMem 使用官方 Judge prompt。MemOps 自研 Judge 只做结构化对齐，输出至少包含：

```json
{
  "prediction_index": 0,
  "gold_target_id": "job_title",
  "target_match": true,
  "current_value_match": true,
  "asserts_stale_as_current": false,
  "historical_stale_only": false,
  "reason_code": "exact_current_state"
}
```

固定以下约束：

- Judge 不看到候选模型名称；
- 所有候选模型使用同一 Judge 模型和 prompt hash；
- 只允许一个预测事实匹配一个 Gold state；
- 同一预测事实不能重复得分；
- Judge 温度为 0；
- 至少抽取 10% 样本进行第二次独立 Judge 或人工复核，并报告一致率；
- Judge 失败记为 invalid，达到有效率门槛后才聚合。

## 12. 每个运行必须生成的文件

每次正式运行使用不可变 `run_id`，建议：

```text
mp-v1_<model-slug>_<YYYYMMDD-HHMMSS>_<git-short-sha>
```

目录和必需产物：

```text
runs\<run_id>\
├── manifest.json
├── logs\run.log
├── requests\request_manifest.jsonl
├── raw_responses\
│   ├── halumem.jsonl
│   └── memops.jsonl
├── predictions\
│   ├── halumem_facts.jsonl
│   └── memops_facts.jsonl
├── judge_records\
│   ├── halumem_integrity.jsonl
│   ├── halumem_accuracy.jsonl
│   └── memops_alignment.jsonl
├── metrics\
│   ├── halumem_core.json
│   ├── memops_core.json
│   ├── diagnostics.json
│   ├── memoryprint_7d.json
│   └── memoryprint_7d.csv
├── failures\failures.jsonl
└── usage\usage.jsonl
```

`manifest.json` 至少记录：

- run ID、开始/结束时间和状态；
- candidate model 请求名与实际返回名；
- endpoint 类型，但不记录 secret；
- HaluMem/MemOps commit；
- 原始数据 SHA-256；
- probe manifest hash；
- prompt/schema/config hash；
- Judge model；
- temperature、max tokens、seed、并发和 retry；
- 编排代码 Git commit；
- Python、uv 和关键依赖版本。

## 13. 最终导出文件

完成全部候选模型后生成：

```text
exports\memoryprint_v1\
├── memoryprint_7d.jsonl
├── memoryprint_7d.csv
├── memoryprint_7d_with_ci.csv
├── diagnostic_21d.csv
├── probe_manifest.json
├── run_index.jsonl
├── model_registry.json
└── README.md
```

核心 `memoryprint_7d.jsonl` 每个模型一行：

```json
{
  "schema_version": "memoryprint-v1",
  "model_id": "provider/model-name",
  "probe_version": "probe-v1",
  "dimensions": {
    "halumem_recall": 0.0,
    "halumem_target_precision": 0.0,
    "halumem_memory_accuracy": 0.0,
    "memops_current_state": 0.0,
    "memops_target_binding": 0.0,
    "memops_stale_rejection": 0.0,
    "memops_evidence_f1": 0.0
  },
  "dimension_mask": {
    "halumem_recall": 1,
    "halumem_target_precision": 1,
    "halumem_memory_accuracy": 1,
    "memops_current_state": 1,
    "memops_target_binding": 1,
    "memops_stale_rejection": 1,
    "memops_evidence_f1": 1
  },
  "effective_counts": {},
  "ci95": {},
  "run_ids": [],
  "manifest_sha256": "..."
}
```

`diagnostic_21d.csv` 保存被筛掉的官方兼容指标和切片，但不得自动拼回核心 7 维。QA 指标也只能位于 diagnostics 中。

## 14. 完整执行顺序

### 阶段 A：建立自研编排项目

1. 切换到 `S:\Workfile\MemoryPrintBench`；
2. 阅读本文件；
3. 建立根 `uv` 项目和目录；
4. 配置 `.gitignore`；
5. 新建 `models.yaml`，只写 endpoint 环境变量名，不写密钥；
6. 固定 `fingerprint_v1.yaml`、prompt 和 JSON schema；
7. 为 schema validator、状态重放和聚合公式编写单元测试。

阶段门槛：`uv run pytest` 通过；配置和 schema 都有 SHA-256。

### 阶段 B：标准化全部原始数据

1. 校验 HaluMem SHA-256 和两个上游 commit；
2. 运行 `preprocess_halumem.py`，标准化 1,387 个 session；
3. 运行 `preprocess_memops.py`，标准化 403 个 adjacent 文件；
4. 生成 inputs、gold、source_map 和 stats；
5. 验证每个 source ID 可逆映射；
6. 扫描 inputs，确认没有 Gold 字段；
7. 验证 probe ID 全局唯一。

阶段门槛：输入/Gold 数量一致；JSON schema 全通过；零 Gold 泄漏；零悬空 source ID。

### 阶段 C：冻结探针

1. HaluMem 选择约 600 session，按用户 group-aware 划分 fingerprint/audit；
2. MemOps 选择 68 Remember + 77 Update + 55 TrajectoryOps；
3. MemOps 按基础案例 ID 做 group-aware 160/40 划分；
4. 生成 split manifests 和统计报告；
5. 计算 manifest hash；
6. 将 `probe_version` 固定为 `probe-v1`。

阶段门槛：fingerprint 与 audit 的 probe ID、group ID 无交集；分层分布可接受；manifest 冻结后不再修改。

### 阶段 D：离线 scorer 测试

1. 构造 perfect prediction，确认七维都接近或等于 1；
2. 构造 empty prediction，确认 Recall/State/Binding 为 0，缺失 mask 规则正确；
3. 构造 stale prediction，确认 Stale Rejection 降低；
4. 构造错误 source ID，确认 Evidence F1 降低或 schema 拒绝；
5. 构造主体错绑，确认 Target Binding 为 0；
6. 验证一条预测不能重复覆盖两个 Gold。

阶段门槛：所有合成测试通过后才允许调用真实模型。

### 阶段 E：在线 smoke test

每个候选模型先运行：

- 5 个 HaluMem probe；
- 5 个 MemOps probe；
- 仅 fingerprint split；
- 同一固定 Judge。

检查 raw response、JSON parse、source ID、usage、Judge 有效率和费用。禁止直接从 smoke 跳到全部模型全量运行。

阶段门槛：提取 parse success ≥ 0.99；Judge valid ratio ≥ 0.98；模型实际返回名与配置一致；没有 Gold 泄漏。

### 阶段 F：正式 fingerprint 运行

对每个候选模型依次：

1. 运行 HaluMem fingerprint 480；
2. 运行 MemOps fingerprint 160；
3. 保存 raw response 和 parsed facts；
4. 执行 HaluMem 两类官方 Judge；
5. 执行 MemOps alignment Judge；
6. 生成该模型的 7 维向量；
7. 按 user UUID（HaluMem）和 base case ID（MemOps）做 cluster bootstrap，建议 1,000 次；
8. 生成 95% CI、有效样本数和 mask；
9. 锁定本模型 run manifest。

不要并发启动全部模型。先完成一个模型的完整闭环并检查产物，再批量扩展到其余模型。

### 阶段 G：隐藏 audit

核心指标、prompt、Judge 和聚合代码冻结后，再运行：

- HaluMem audit 约 120；
- MemOps audit 40。

检查 fingerprint 排名是否能预测 audit 排名，并保存 Spearman/Kendall、每维偏差和置信区间。Audit 结果不得用于回头修改 `probe-v1` 的维度或权重；如发现设计缺陷，必须创建 `probe-v2` 并完整记录变更。

### 阶段 H：导出给 InfoBudget2

只导出：

- `memoryprint_7d.jsonl/csv`；
- dimension mask；
- CI 和有效样本数；
- probe/run manifest；
- 必要的诊断指标；
- 模型 registry。

不导出第三方原始数据、API key、大体积 raw responses 或完整 Judge prompt 中可能含有的数据内容。

## 15. 推荐命令接口

实现脚本后，统一提供以下 CLI；具体脚本尚未创建，下面是需要实现的目标接口：

```powershell
# 数据标准化
uv run python -m memoryprintbench.preprocess_halumem --config configs\fingerprint_v1.yaml
uv run python -m memoryprintbench.preprocess_memops --config configs\fingerprint_v1.yaml

# 冻结探针
uv run python -m memoryprintbench.freeze_probe --config configs\fingerprint_v1.yaml

# 校验
uv run python -m memoryprintbench.validate_artifacts --stage pre-run

# 候选模型提取
uv run python -m memoryprintbench.run_extractor --model <model-id> --dataset halumem --split fingerprint
uv run python -m memoryprintbench.run_extractor --model <model-id> --dataset memops --split fingerprint

# 评分
uv run python -m memoryprintbench.score_halumem_extraction --run-id <run-id>
uv run python -m memoryprintbench.score_memops_core --run-id <run-id>

# 聚合与导出
uv run python -m memoryprintbench.aggregate_memoryprint --run-id <run-id>
uv run python -m memoryprintbench.validate_artifacts --stage final --run-id <run-id>
```

这些命令应是幂等的：只有当 probe ID、request hash、模型实际 ID、prompt hash 和 config hash 全部一致时才允许跳过已有结果。不能仅因为某个输出文件存在就盲目跳过。

## 16. 最终验收清单

- [ ] HaluMem/MemOps commit 与数据 SHA-256 已写入 manifest；
- [ ] 根 `uv` 环境、`pyproject.toml` 和 `uv.lock` 已生成；
- [ ] 原始输入未被修改；
- [ ] candidate inputs 中不存在 Gold 字段；
- [ ] probe、group 和 source ID 均唯一且可逆；
- [ ] fingerprint/audit 在 group 层面无泄漏；
- [ ] 所有模型使用同一 prompt/schema/decoding；
- [ ] raw response、parsed prediction、Judge record 和 failure 均可追溯；
- [ ] HaluMem 三维来自固定官方 Judge 语义；
- [ ] MemOps 四维只读 operation trace，不读最终 QA Accuracy；
- [ ] QA 指标未进入 `memoryprint_7d`；
- [ ] 七维都在 `[0,1]`，高值更优；
- [ ] 缺失维使用 mask，不以 0 代替；
- [ ] Judge valid ratio 和 parse success 达到门槛；
- [ ] CI 使用用户/案例级 cluster bootstrap；
- [ ] `memoryprint_7d.jsonl`、CSV、diagnostics 和 manifests 齐全；
- [ ] 导入 InfoBudget2 的文件不含密钥和第三方原始数据。

## 17. 切换目录后的第一项任务

进入 `S:\Workfile\MemoryPrintBench` 后，不要立即运行全量 API 实验。第一项实现任务应是：

1. 建立根 `uv` 项目；
2. 创建上述目录；
3. 实现两个预处理器和 source-map 校验；
4. 生成冻结 probe manifest；
5. 用 perfect/empty/stale/wrong-source 合成预测验证评分器。

只有这些离线步骤全部通过，才配置真实候选模型与 Judge 密钥并开始在线 smoke test。

