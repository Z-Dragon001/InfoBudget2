# HaluMem 与 MemOps 的 MemoryPrint 评分及跨数据集泛化方案

> 文档用途：为当前长期记忆写入路由构造候选模型能力指纹 \(\Psi(m)\)，并定义其在 LoCoMo 与 LongMemEval 上的泛化验证方法。
>
> 当前状态：实验设计文档，不包含代码实现。
>
> 核心约束：不重新人工标注大规模数据；候选大模型保持冻结；价格不进入能力指纹和质量预测器，只在最终预算求解层使用。

> 实际仓库版本、数据校验值、独立 `uv` 环境及运行命令见 [MemoryPrint 外部评测环境部署与运行说明](./MemoryPrint外部评测环境部署与运行说明.md)。

## 1. 结论先行

当前项目可以直接使用 HaluMem 和 MemOps 的现成 Gold 数据构造模型能力指纹，不需要自行标注数百或数千条长期记忆事实。

推荐分工如下：

- **HaluMem 是主探针集**：测量事实完整性、事实准确性、幻觉、干扰抵抗、Persona/Event/Relationship 记忆和事实更新能力。
- **MemOps 是补充探针集**：测量目标绑定、当前状态识别、旧值残留、无效状态拒绝和证据定位。
- **LoCoMo 用于训练和验证片段—模型质量预测器**，并评估长期记忆多次查询下的累计收益。
- **LongMemEval 完全保留为冻结后的外部测试集**，不得用于选择 MemoryPrint 维度、权重、阈值或训练超参数。

最终模型描述为：

\[
\Psi(m)=\left[\Psi_H(m);\Psi_M(m)\right]
\]

其中：

- \(\Psi_H(m)\) 来自 HaluMem；
- \(\Psi_M(m)\) 来自 MemOps；
- 所有维度都统一转换成“数值越大，能力越强”的 \([0,1]\) 指标；
- 价格、供应商延迟和 API 稳定性不进入 \(\Psi(m)\)。

必须特别说明：HaluMem 和 MemOps 的指纹**不保证天然泛化**。泛化必须通过冻结协议在 LoCoMo 和 LongMemEval 上验证。并且 LoCoMo 与 LongMemEval 本身也带有明显的生成式或受控构造属性，因此最严谨的论文表述应是：

> MemoryPrint 在不同数据构造流程、不同对话结构和不同长期记忆任务之间具有跨基准迁移能力。

不能仅凭这些实验声称：

> MemoryPrint 已经证明可以泛化到任意真实用户长期对话。

---

## 2. 为什么不能直接使用官方总分

HaluMem 和 MemOps 的官方目标主要是评估一个完整记忆系统，而当前项目需要评估的是：

> 候选 LLM 在固定长期记忆写入提示下，从一个主题片段中提取可靠事实的能力。

二者存在明显差别：

| 方面 | 数据集官方任务 | 当前项目任务 |
|---|---|---|
| 被评估对象 | 完整记忆系统或长上下文方法 | 一个候选事实提取 LLM |
| 输入 | 多轮会话、历史记忆或完整长历史 | 主题分割后的局部片段 |
| 输出 | 记忆状态、检索结果或最终答案 | Fact 列表及 `source_ids` |
| 更新操作 | 可能包含写入、覆盖、删除、遗忘 | 当前主要是事实提取和入库 |
| QA | 可能由同一系统完成 | 应固定 Retriever、Reader 和 Judge，以隔离写入质量 |

因此应同时保存两套结果：

1. **Official-Compatible Metrics**：尽量复现官方评分，便于与公开结果比较；
2. **Router-Adapted Metrics**：只保留与当前写入路由输入输出契约一致的指标，用于构造 \(\Psi(m)\)。

用于路由训练的是第二套，而不是官方排行榜中的单一总分。

---

## 3. 统一运行协议

### 3.1 候选模型输入

所有模型必须接收完全相同的内容：

\[
\text{Input}_{i,m}
=
\text{FrozenExtractionPrompt}
+
\text{TopicSegment}_i
\]

其中不得加入：

- 模型名称或参数规模；
- 模型价格；
- 标准答案；
- QA 问题；
- 供应商信息；
- 其他候选模型的结果。

使用项目部署时相同的事实提取提示和 JSON schema。这样得到的指纹反映的是“该模型在项目实际写入任务中的能力”，而不是模型在另一套提示上的能力。

### 3.2 输入粒度

HaluMem 和 MemOps 中的长历史不能直接全部输入模型，否则测试的是长上下文读取，而不是当前项目的主题片段写入能力。

推荐粒度：

- HaluMem：以一个 `session` 为基本单元；如果 session 超过项目主题段上限，再使用项目的主题分割器切分，但必须保存 session 与子片段映射。
- MemOps：主探针使用 `adjacent evidence` 或单个证据 conversation；`longitudinal` 长历史仅作为压力测试，不进入主指纹。
- 所有模型处理完全相同的片段集合。

### 3.3 解码与版本控制

推荐固定：

- `temperature = 0`；
- 相同的最大输出 Token；
- 不进行自动 Repair 后重试；
- 不使用工具调用；
- 每个模型只保留第一次有效响应；
- 解析失败单独记为格式失败，不能静默重试到成功。

每次指纹记录：

```text
model_id
provider
model_version_or_revision
quantization
probe_set_hash
prompt_hash
output_schema_version
judge_model_version
run_date
```

供应商只用于复现记录，不作为能力输入。

### 3.4 原子化输出

模型输出的每条 Fact 记为：

\[
p=(\text{text},\text{source\_ids})
\]

如果一条输出同时包含多个独立事实，应在评分前拆成原子事实。例如：

```text
Alice住在北京，并且喜欢徒步。
```

应拆为：

```text
Alice住在北京。
Alice喜欢徒步。
```

否则一个复合输出可能同时匹配多个 Gold，造成虚假的高召回。

---

## 4. HaluMem 数据使用方案

### 4.1 可直接使用的 Gold 字段

HaluMem 每个 session 包含：

- `dialogue`：多轮用户—助手对话；
- `memory_points`：该 session 对应的标准记忆点；
- `memory_type`：Persona、Event 或 Relationship；
- `memory_source`：primary、secondary、interference 或 system；
- `is_update`：是否为更新事实；
- `original_memories`：被更新或替代的旧事实；
- `importance`：事实重要性；
- `timestamp`：事实时间；
- `questions`、答案及相关证据记忆。

HaluMem 官方将评估分为 Memory Extraction、Memory Update 和 Memory QA，并使用现成评估程序计算 Recall、Weighted Recall、Target Precision、Accuracy、Interference Accuracy、F1、更新正确率、幻觉率和遗漏率。官方数据集页面给出的规模约为 14,948 个记忆点和 3,467 个 QA 对。

HaluMem 的对话和记忆由程序化生成、LLM 辅助完善与人工验证共同构成。官方数据页报告 8 名标注员人工检查了超过 50% 的 HaluMem-Medium，并报告 95.70% 的正确性。因此它比完全无人审核的合成集可靠，但仍然具有明显的生成数据分布，不能等同于真实用户日志。

来源：

- [HaluMem 论文](https://arxiv.org/abs/2511.03506)
- [HaluMem GitHub 与官方评估代码](https://github.com/MemTensor/HaluMem)
- [HaluMem Hugging Face 数据页](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem)

### 4.2 数据版本与抽样

主指纹使用 `HaluMem-Medium`，不使用整个 HaluMem-Long 作为输入，原因是当前路由器处理的是主题片段，不是百万 Token 完整历史。

推荐固定抽取 600 个 session：

- 480 个用于计算正式 \(\Psi_H(m)\)；
- 120 个作为隐藏审计集，验证指纹是否能预测未参与聚合的探针表现。

抽样单位必须是 session，而不是 memory point。一个 session 中的所有 Gold 都应同时保留，以避免只抽取容易事实。

分层维度包括：

- Persona / Event / Relationship；
- `is_update=True/False`；
- primary / secondary / interference；
- 事实数量；
- 对话长度；
- importance 区间；
- 用户 UUID。

由于这些类别会重叠，不要求每类数量简单相加等于 600。应使用多标签分层抽样，并限制单个用户对总样本的支配程度。

抽样列表和随机种子必须在运行候选模型之前冻结。

### 4.3 HaluMem 官方覆盖评分

对 session \(i\)，Gold 事实集合为 \(G_i\)，模型输出事实集合为 \(P_{i,m}\)。

HaluMem 官方 Memory Integrity Judge 会对每个 Gold 事实给出：

\[
s^{cov}_{g,m}\in\{0,1,2\}
\]

含义为：

- 2：完整覆盖或可由输出逻辑推出；
- 1：部分覆盖，但缺少关键字段或存在轻微错误；
- 0：没有覆盖或事实错误。

官方严格召回率只把 2 分视为成功：

\[
R_{strict}(m)
=
\frac{\sum_g \mathbf 1[s^{cov}_{g,m}=2]}{|G|}
\]

本项目同时保留软召回率：

\[
R_{soft}(m)
=
\frac{1}{|G|}
\sum_g\frac{s^{cov}_{g,m}}{2}
\]

软召回保留“部分提取”和“完全遗漏”的区别，更适合作为连续模型能力特征。

#### 4.3.1 项目适配的一对一匹配

HaluMem 官方完整性评分是“对每个 Gold，在全部候选 Fact 中寻找覆盖”。为避免一个宽泛的复合 Fact 重复覆盖多个 Gold，本项目的 Router-Adapted Metrics 还应执行一对一匹配。

对每个 Gold \(g\) 与候选 Fact \(p\)，由固定 Judge 给出语义事实匹配权重：

\[
w(g,p)\in\{0,0.5,1\}
\]

其中：

- 1：主体、关系、对象、时间、否定状态等关键字段均正确；
- 0.5：核心事实部分正确，但缺少非核心字段或存在轻微不完整；
- 0：不匹配、主体错误、值错误或相互矛盾。

然后使用最大权重二分匹配：

\[
M_i^*
=
\arg\max_{M\in\mathcal M(G_i,P_{i,m})}
\sum_{(g,p)\in M}w(g,p)
\]

其中每个 Gold 和每个候选 Fact 最多只能出现一次。定义：

\[
TP_{soft,i}=\sum_{(g,p)\in M_i^*}w(g,p)
\]

\[
P_{match,i}=\frac{TP_{soft,i}}{|P_{i,m}|},
\qquad
R_{match,i}=\frac{TP_{soft,i}}{|G_i|}
\]

这套匹配指标用于项目主 Fact F1；官方覆盖指标仍单独保存，用于与 HaluMem 官方结果对齐。

### 4.4 重要性加权召回

对非 interference Gold，使用 HaluMem 的 `importance`：

\[
R_{imp}(m)
=
\frac{\sum_g w_g\cdot s^{cov}_{g,m}/2}
{\sum_g w_g}
\]

其中 \(w_g\) 是 Gold 事实重要性。

该指标只用于辅助诊断，不应替代普通召回。否则模型可能通过只提取高重要性事实而掩盖大量遗漏。

### 4.5 候选事实准确性与幻觉

HaluMem 官方 Accuracy Judge 会对每个候选事实给出：

\[
s^{acc}_{p,m}\in\{0,1,2\}
\]

含义为：

- 2：候选事实中的所有信息均受到对话或 Gold 支持；
- 1：部分正确，但混入不支持或矛盾内容；
- 0：完全不支持或与对话矛盾。

定义事实忠实度：

\[
P_{faith}(m)
=
\frac{1}{|P|}
\sum_p\frac{s^{acc}_{p,m}}{2}
\]

定义幻觉率：

\[
H(m)
=
\frac{\sum_p\mathbf 1[s^{acc}_{p,m}=0]}{|P|}
\]

用于指纹时转换为高值优指标：

\[
HR(m)=1-H(m)
\]

其中 \(HR\) 表示 Hallucination Resistance。

### 4.6 目标事实精确率与过度写入

“事实受对话支持”不等于“该事实应该写入长期记忆”。因此还要判断候选事实是否与 Gold 记忆字段对应：

\[
r_{p,m}\in\{0,1\}
\]

定义：

- \(r=1\)：候选事实对应 HaluMem 标准记忆字段；
- \(r=0\)：事实即使可能受对话支持，也不属于该 session 的目标长期记忆点。

本项目目标精确率定义为：

\[
P_{target}(m)
=
\frac{1}{|P|}
\sum_p r_{p,m}\frac{s^{acc}_{p,m}}{2}
\]

过度写入率定义为：

\[
O_{store}(m)
=
\frac{\sum_p\mathbf 1[r_{p,m}=0\land s^{acc}_{p,m}>0]}{|P|}
\]

它区分了两种不同失败：

- 幻觉：写入了对话不支持的内容；
- 过度写入：写入了对话支持、但不应进入长期记忆的内容。

这对写入预算十分重要，因为过度写入会增加数据库体积、检索噪声和后续 Reader 成本。

### 4.7 事实级 F1

项目主 Fact F1 使用一对一匹配得到的 \(P_{match}\) 与 \(R_{match}\)：

\[
F1_{fact}(m)
=
\frac{2P_{match}(m)R_{match}(m)}
{P_{match}(m)+R_{match}(m)}
\]

同时报告 \(P_{target}\)、\(R_{soft}\) 和官方兼容 F1，避免与 HaluMem 公布结果不可比。论文中必须说明哪个 F1 是官方实现，哪个是项目适配实现。

### 4.8 干扰事实抵抗

对于 `memory_source=interference` 的记忆点，正确行为是不要把错误或干扰内容写入。

设干扰集合为 \(G^{int}\)，则：

\[
IR_{strict}(m)
=
\frac{\sum_{g\in G^{int}}\mathbf 1[s^{cov}_{g,m}=0]}
{|G^{int}|}
\]

软版本为：

\[
IR_{soft}(m)
=
1-
\frac{1}{|G^{int}|}
\sum_{g\in G^{int}}\frac{s^{cov}_{g,m}}{2}
\]

如果模型把干扰事实完整写入，则对应项为 0；完全拒绝则为 1。

### 4.9 分类型记忆能力

当前模型输出 schema 没有强制要求输出 `memory_type`，因此不建议另用一个分类器给候选事实补类型。更稳妥的是按照 Gold 类型计算覆盖能力：

\[
R_t(m)
=
\frac{1}{|G_t|}
\sum_{g\in G_t}\frac{s^{cov}_{g,m}}{2}
\]

其中：

\[
t\in\{Persona,Event,Relationship\}
\]

最终得到：

- \(R_{persona}\)；
- \(R_{event}\)；
- \(R_{relationship}\)。

这些指标反映模型在哪类事实上容易遗漏，而不是简单重复模型总体 F1。

### 4.10 更新事实评分

对 `is_update=True` 的 Gold，HaluMem 提供更新后的目标事实和 `original_memories`。

官方 Judge 将结果分为：

- Correct；
- Hallucination；
- Omission；
- Other。

定义：

\[
U_{correct}(m)
=
\frac{N_{Correct}}{N_{Update}}
\]

\[
U_{hall}(m)
=
\frac{N_{Hallucination}}{N_{Update}}
\]

\[
U_{omit}(m)
=
\frac{N_{Omission}}{N_{Update}}
\]

进一步单独计算旧值残留率：

\[
L_{stale}(m)
=
\frac{1}{N_{Update}}
\sum_i
\mathbf 1[\text{旧值仍作为当前事实出现在输出中}]
\]

转换为能力指标：

\[
SR(m)=1-L_{stale}(m)
\]

其中 \(SR\) 是 Stale Rejection。

重要限制：如果当前项目的候选模型只看到新片段、看不到数据库中的旧记忆，就不能要求它执行数据库删除或覆盖。此时只测：

- 是否提取新值；
- 是否错误保留同一片段中的旧值；
- 是否正确识别当前有效状态。

不能把完整记忆系统的删除能力错误归因给事实提取模型。

### 4.11 JSON 与格式遵循

格式成功率定义为：

\[
F_{schema}(m)
=
\frac{N_{一次解析成功且字段合法}}{N_{samples}}
\]

以下均视为失败：

- JSON 无法解析；
- 必需字段缺失；
- Fact 不是列表；
- `source_ids` 类型错误；
- 输出包含 schema 之外的大段解释；
- 只有 Repair 后才能解析。

格式失败样本的事实质量不应被简单删除。主结果中应将其视为该样本提取失败，另提供“仅有效解析样本”的诊断结果。

### 4.12 HaluMem QA 的正确用法

HaluMem 官方还提供 QA 指标：Correct、Hallucination 和 Omission。对于当前项目，不应让候选提取模型自己回答这些问题，否则会把“写入能力”和“回答能力”混在一起。

正确方式是：

```text
候选模型 m 提取记忆
        ↓
固定的存储与检索流程
        ↓
固定 Reader
        ↓
固定 Judge
```

得到：

\[
QA_C(m),\quad QA_H(m),\quad QA_O(m)
\]

这些指标作为 MemoryPrint 的辅助效用维度或外部验证指标，不作为 HaluMem 基础提取指纹的必要组成部分。

### 4.13 推荐的 HaluMem 指纹

第一版推荐 14 维：

\[
\Psi_H(m)=
[
F1_{fact},
R_{strict},
R_{soft},
P_{target},
P_{faith},
HR,
IR_{soft},
R_{persona},
R_{event},
R_{relationship},
U_{correct},
1-U_{hall},
SR,
F_{schema}
]
\]

不建议将这 13 维先手工加权成一个分数再输入质量预测器。保留向量可以让质量预测器学习：某类主题片段更依赖关系能力，另一类更依赖时间更新能力。

---

## 5. MemOps 数据使用方案

### 5.1 MemOps 提供的 Gold

MemOps 将长期记忆建模为操作生命周期，公开结构包括：

- `target_fact`；
- `operation_type`；
- `difficulty_knobs`；
- 对话段及段落角色；
- Gold operation trace；
- `trigger_span`；
- `target_id` 和 `target_name`；
- `old_value`；
- `new_value`；
- `validity`；
- `evidence_spans`；
- `gold_memory_state`；
- `must_include`；
- `must_not_include`；
- Gold provenance。

MemOps 官方覆盖 Remember、Forget、Update、Reflect 和组合轨迹，并报告操作检测 Precision/Recall/F1，以及 Forget 泄漏、过度遗忘、Update 旧值率、Reflect Precision/Recall、轨迹顺序、最终状态和证据支持等指标。

MemOps 于 2026 年 7 月发布，当前属于较新的 research preview。其公开仓库提供生成产物、模板和评估流程，适合补充 HaluMem 不足的生命周期诊断；但由于外部复用和独立复现积累仍有限，不应把 MemOps 作为唯一的模型能力来源。

来源：

- [MemOps 论文](https://arxiv.org/abs/2607.12893)
- [MemOps GitHub、数据结构与评估代码](https://github.com/MemTensor/MemOps)

### 5.2 哪些 MemOps 任务可以进入当前指纹

并非所有 MemOps 操作都与当前项目输出契约匹配。

| MemOps 能力 | 当前是否进入主指纹 | 原因 |
|---|---:|---|
| Remember | 是 | 与新事实提取直接一致 |
| Update | 是，但只测当前状态提取 | 当前模型未必能直接修改数据库 |
| TargetBinding | 是 | 可测主体、属性和对象绑定 |
| StateTransition | 是 | 可测新旧值与当前有效状态 |
| Evidence/Provenance | 是 | Gold 提供证据跨度，项目输出有 `source_ids` |
| Forget | 否，除非输出支持删除操作 | 普通 Fact 输出无法表达删除 |
| Reflect | 默认否 | 当前任务强调显式事实，Reflect 包含受约束推断 |
| 完整 StateTrajectory | 仅压力测试 | 当前路由输入是局部主题段，不一定看到完整轨迹 |

如果以后项目的输出 schema 新增：

```text
operation = add | update | delete | keep
```

再把 Forget、完整 Update 和 StateTrajectory 加入主指纹。当前阶段强行加入会让模型因“不被允许输出某种操作”而被错误扣分。

### 5.3 MemOps 抽样

推荐固定 200 个样本：

- 160 个用于 \(\Psi_M(m)\)；
- 40 个作为隐藏审计集。

目标构成为：

- 约 80 个 Remember；
- 约 80 个 Update；
- 约 40 个强化 TargetBinding、StateTransition、recency trap、negative seed 或 adversarial injection 的样本。

这些标签可能重叠，因此最终以操作覆盖和难度覆盖为准。

主指纹使用 adjacent evidence 设置。Longitudinal 设置只用于报告长历史压力测试，不与主分数混合。

### 5.4 Gold 当前状态

对样本 \(i\)，根据 Gold operation trace 计算：

\[
z_i^*=(target_i,value_i,status_i)
\]

其中：

- \(target_i\)：目标人物或属性；
- \(value_i\)：当前有效值；
- \(status_i\)：confirmed、tentative、retracted 或 superseded 等状态。

从候选模型的 Fact 集合中得到预测状态 \(\hat z_{i,m}\)。

### 5.5 Target Binding

目标绑定分数：

\[
B_{i,m}
=
\mathbf 1[
\widehat{target}_{i,m}=target_i
]
\]

比较时允许别名和代词消解，但主体不能错。例如“用户妹妹喜欢跑步”不能被写成“用户喜欢跑步”。

总体目标绑定能力为：

\[
B(m)=\frac{1}{N}\sum_i B_{i,m}
\]

### 5.6 当前值正确率

定义：

\[
V_{i,m}
=
\mathbf 1[
\widehat{value}_{i,m}=value_i
]
\]

时间、数值、否定和专有名词属于关键字段，不能仅依赖高语义相似度放宽。

当前状态正确率：

\[
S_{current}(m)
=
\frac{1}{N}
\sum_i B_{i,m}V_{i,m}
\]

只有目标和当前值同时正确才得分。

### 5.7 旧值残留与无效值拒绝

对 Update 样本，Gold 提供旧值 \(v_i^{old}\) 和新值 \(v_i^{new}\)。

旧值残留率：

\[
L_{old}(m)
=
\frac{1}{N_U}
\sum_i
\mathbf 1[
v_i^{old}\text{仍被输出为当前有效事实}
]
\]

旧值拒绝能力：

\[
R_{old}(m)=1-L_{old}(m)
\]

对 `must_not_include`、tentative、retracted、negative seed 和 recency trap 中的无效候选值，定义：

\[
R_{invalid}(m)
=
1-
\frac{N_{被错误写入的无效值}}
{N_{Gold无效值}}
\]

这两个指标能区分：

- 模型是否记住最新提到的信息；
- 模型是否理解“最新提到”不一定等于“当前有效”。

### 5.8 Remember 事实覆盖

对于 Remember 操作，使用与 HaluMem 相同的 0/1/2 覆盖规则：

\[
R_{remember}(m)
=
\frac{1}{N_R}
\sum_i \frac{s^{remember}_{i,m}}{2}
\]

同时计算相关错误：

\[
H_{remember}(m)
=
\frac{N_{unsupported\ facts}}
{N_{predicted\ facts}}
\]

### 5.9 证据定位

MemOps 提供 `evidence_spans` 和 Gold provenance，因此可以评价项目输出的 `source_ids`。

设预测证据轮次集合为 \(E_{i,m}\)，Gold 集合为 \(E_i^*\)：

\[
P_E=\frac{|E_{i,m}\cap E_i^*|}{|E_{i,m}|}
\]

\[
R_E=\frac{|E_{i,m}\cap E_i^*|}{|E_i^*|}
\]

\[
F1_E=\frac{2P_ER_E}{P_E+R_E}
\]

需要先将项目主题段中的局部 `source_ids` 映射回 MemOps 原始的 segment/turn 编号。没有映射成功的证据不能当作正确。

### 5.10 Update 综合得分

对当前项目，Update 不要求模型真正修改数据库，而要求它正确提取当前有效状态。单样本得分定义为：

\[
Q^{update}_{i,m}
=
B_{i,m}
\cdot V_{i,m}
\cdot \mathbf 1[v_i^{old}\text{未被输出为当前值}]
\]

总体：

\[
Q_{update}(m)
=
\frac{1}{N_U}\sum_iQ^{update}_{i,m}
\]

这是一个严格指标：主体错、新值错或旧值仍被当作当前值，都会使该样本得 0。

另外分别报告 \(B\)、\(S_{current}\) 和 \(R_{old}\)，防止一个总分隐藏具体错误原因。

### 5.11 Longitudinal 压力测试

MemOps 的 longitudinal 设置将证据分散在包含干扰项的长历史中。它适合验证完整记忆系统，但与当前局部片段路由不完全一致。

因此单独报告：

\[
\Delta_{long}(m)
=
Q_{adjacent}(m)-Q_{longitudinal}(m)
\]

\(\Delta_{long}\) 越大，说明模型或完整系统在长历史和干扰下退化越严重。该指标不进入第一版主 MemoryPrint，除非部署时确实让一个候选模型读取整段长历史。

### 5.12 推荐的 MemOps 指纹

第一版使用 7 维：

\[
\Psi_M(m)=
[
R_{remember},
1-H_{remember},
B,
S_{current},
R_{old},
R_{invalid},
F1_E
]
\]

如果当前项目的 `source_ids` 无法映射到原始轮次，则移除 \(F1_E\)，并给该维设置缺失 mask，不能用 0 代替缺失。

---

## 6. 最终 MemoryPrint 的形成

### 6.1 拼接而不是手工排序

最终得到约 20 维指纹：

\[
\Psi(m)=
[
\Psi_H(m);
\Psi_M(m)
]
\]

每个维度均为 \([0,1]\)，且越大越好。

不在输入阶段手工规定：

```text
7B < 32B < 72B
```

能力强弱由模型在固定探针集上的实际表现决定。

### 6.2 缺失维度

若某模型或某输出 schema 无法评价某项能力，使用：

\[
d_m=[\Psi(m);mask(m)]
\]

其中 `mask=0` 表示缺失，`mask=1` 表示存在。缺失项不能直接填 0，因为 0 表示该模型确实完全失败。

### 6.3 标准化

由于所有核心指标已在 \([0,1]\)，第一版可以直接输入质量预测器。

若后续进行 z-score 标准化，只能使用训练模型的固定均值和方差：

\[
\widetilde\Psi_k(m)
=
\frac{\Psi_k(m)-\mu_k^{train}}
{\sigma_k^{train}+\epsilon}
\]

新模型接入时必须沿用相同 \(\mu^{train},\sigma^{train}\)，不能根据当前候选池重新计算，否则新模型加入会改变所有旧模型的表示。

### 6.4 是否需要综合分数

路由器应使用多维向量，而不是一个综合分数。

如果论文表格需要单一 MemoryPrint Summary，只用于展示，可以使用等权能力宏平均：

\[
S_{MP}(m)
=
\frac{1}{K}\sum_{k=1}^{K}\Psi_k(m)
\]

必须同时提供完整分维结果。不能使用下游测试集选择一组使结果最好看的权重。

### 6.5 不进入 MemoryPrint 的内容

以下信息不能进入 \(\Psi(m)\)：

- API 单价；
- 本次预算；
- 网络延迟；
- 供应商重试率；
- LoCoMo 或 LongMemEval 测试答案；
- 当前候选池中其他模型的相对名次。

价格在最终决策层使用：

\[
a_i^*
=
\arg\max_m
\left[
\widehat q(x_i,\Psi(m))
-\lambda C(i,m)
\right]
\]

或通过硬预算约束求解。

---

## 7. Judge 与评分可靠性

### 7.1 固定 Judge

HaluMem 官方评估使用 LLM Judge 判断覆盖、准确性、更新和 QA。为了可比性，Official-Compatible Metrics 应使用官方 prompt 与固定 Judge 版本。

Router-Adapted Metrics 也必须使用固定 Judge，所有候选模型共用相同评分器。Judge 不得根据被评分模型改变。

### 7.2 避免 Judge 成为唯一证据

建议加入确定性校验：

- JSON schema 校验；
- 数值、日期和否定词精确检查；
- `source_ids` 集合匹配；
- `must_include` / `must_not_include` 检查；
- 重复 Fact 去重；
- 旧值与新值显式比对。

对 10% 的样本使用第二个固定 Judge 复评，报告一致率。该过程不需要新建人工 Gold，只是在现成 Gold 上验证自动评分稳定性。

如果两个 Judge 分歧很大，应把指纹视为带噪估计，而不是更换 Judge 直到得到期望排序。

### 7.3 置信区间

对每个指纹维度进行按用户或样本组的 bootstrap，报告 95% 置信区间：

\[
CI_{95\%}(\Psi_k(m))
\]

不能把同一 session 内多个 memory point 当作完全独立样本，否则置信区间会过窄。

建议进行探针规模曲线：

```text
50 / 100 / 200 / 400 / 600 HaluMem sessions
50 / 100 / 160 MemOps samples
```

观察模型排序和指纹维度何时稳定。

---

## 8. 合成数据指纹能否泛化到 LoCoMo 与 LongMemEval

### 8.1 正确答案：可能，但不能预先保证

MemoryPrint 泛化依赖它捕捉的是任务不变量，而不是合成数据的语言模板。

可能跨数据集稳定的能力包括：

- 事实遗漏倾向；
- 幻觉倾向；
- 多人物目标绑定；
- 新旧状态区分；
- 无效事实拒绝；
- 格式遵循；
- 证据定位。

可能不稳定的因素包括：

- 生成模型形成的固定语言风格；
- 模板化的更新句式；
- HaluMem 和 MemOps 的人物、事件分布；
- 合成数据中过于明确的证据边界；
- Judge 对特定模型家族语言风格的偏好；
- 当前项目主题分割方式与数据集 session 边界不一致。

因此，MemoryPrint 是一个待验证的模型条件变量，不是天然成立的“通用能力真值”。

### 8.2 LoCoMo 和 LongMemEval 也不是纯真实对话

LoCoMo 使用机器—人工流水线：LLM 代理基于 persona 和时间事件图生成长对话，再由人工核验和编辑。其论文明确描述了该构造过程。[LoCoMo 论文](https://aclanthology.org/2024.acl-long.747/)

LongMemEval 使用属性控制流程编译带时间戳的模拟聊天历史，并将用户背景事实、证据 session 和 filler session 组合成每个问题的历史。[LongMemEval 项目](https://github.com/xiaowu0162/longmemeval)

所以：

- HaluMem/MemOps → LoCoMo/LongMemEval 可以证明跨基准、跨生成管线和跨任务结构迁移；
- 它不能单独证明对真实生产用户长期对话的完全泛化。

论文中主动承认这一边界，比把 LoCoMo 和 LongMemEval 表述成纯真实世界测试更严谨。

### 8.3 为什么仍然值得做

虽然这些基准带有生成成分，但它们并非同一份模板数据：

- HaluMem 重点是提取、更新和幻觉；
- MemOps 重点是操作生命周期和状态轨迹；
- LoCoMo 重点是超长多 session 对话、QA 和事件总结；
- LongMemEval 重点是信息提取、多 session 推理、时间推理、知识更新和拒答。

如果在 HaluMem/MemOps 上形成的模型表示能够预测模型在 LoCoMo/LongMemEval 上的相对表现，这仍然是有价值的跨任务证据。

---

## 9. 跨数据集泛化实验

### 9.1 严格的数据流

```text
HaluMem + MemOps
    ↓
只生成各模型的 Ψ(m)
    ↓
LoCoMo 训练对话
    ↓
训练片段—模型质量预测器
    ↓
LoCoMo 验证对话
    ↓
选择超参数、预算策略和停止条件
    ↓
冻结全部组件
    ↓
LoCoMo 测试对话 + LongMemEval 全量外部测试
```

LongMemEval 不参与：

- 探针维度选择；
- 指纹权重选择；
- 质量预测器训练；
- 阈值选择；
- Judge prompt 调整；
- 预算系数调参。

### 9.2 LoCoMo 内部划分

LoCoMo 必须按完整 conversation 划分。不能随机拆分片段或 QA，因为同一对话的人物、事件和措辞高度相关。

每个候选模型处理同一批 LoCoMo 训练主题段，得到实际 Fact 质量：

\[
y^{fact}_{i,m}
\]

固定 Retriever、Reader 和 Judge 后，得到下游 QA 效用：

\[
y^{QA}_{q,m}
\]

质量预测器训练目标为：

\[
\widehat q_{i,m}
=
f_\theta(x_i,\Psi(m))
\]

其中价格不进入 \(f_\theta\)。

### 9.3 未见模型家族实验

如果 Qwen 和 Llama 六个模型全部作为训练模型，不能证明对未见模型家族的零样本适配。

至少做：

```text
实验A：Qwen作为训练模型，Llama作为完全未见测试模型
实验B：Llama作为训练模型，Qwen作为完全未见测试模型
```

未见家族只允许：

- 在 HaluMem/MemOps 固定探针上生成 \(\Psi(m)\)；
- 进入冻结的质量预测器；
- 在测试集上被评价。

不得用未见家族的 LoCoMo 测试结果微调质量预测器。

需要说明：只有两个模型家族时，未见家族结论仍然有限。更强的论文证据应增加第三个模型家族，或者把结论限定为 Qwen ↔ Llama 跨家族迁移。

### 9.4 LongMemEval 零样本外部测试

在 LoCoMo 完成所有训练和调参后冻结：

- MemoryPrint 定义；
- 质量预测器；
- 路由策略；
- Retriever；
- Reader；
- Judge；
- 预算档位。

然后在 LongMemEval 上：

1. 按项目方式处理每个样本的多个 session；
2. 主题分割；
3. 对每个主题段计算候选模型质量；
4. 在预算内路由；
5. 建立记忆库；
6. 固定 Top-k 检索；
7. 固定 Reader 回答该样本唯一的问题；
8. 使用官方评价程序计算 QA 正确率。

这测试的是：

\[
\text{HaluMem/MemOps指纹}
+
\text{LoCoMo训练的质量关系}
\rightarrow
\text{LongMemEval零样本性能}
\]

### 9.5 泛化指标

不能只报告最终 QA。至少包括以下四层。

#### 模型级排序

比较 \(\Psi(m)\) 预测的能力排序与模型在目标数据集上的实际排序：

- Spearman \(\rho\)；
- Kendall \(\tau\)；
- 模型两两排序准确率。

模型只有 6 个时，相关系数统计功效较弱，必须报告置信区间，不能只报告一个很高的相关系数。

#### 片段—模型质量预测

- MAE / RMSE；
- 质量概率的 Brier Score；
- Expected Calibration Error；
- 每个片段最佳模型 Top-1 选择准确率；
- 两两模型胜负预测准确率。

#### 路由决策

定义 Oracle 质量：

\[
Q_{oracle}(B)
=
\max_{\mathbf a:\,C(\mathbf a)\le B}Q(\mathbf a)
\]

路由遗憾：

\[
Regret(B)
=
Q_{oracle}(B)-Q_{router}(B)
\]

报告多个预算档位下的 Regret，而不是只报告一个 \(\lambda\)。

#### 最终系统

- QA Accuracy；
- 总写入成本；
- 预算违反率；
- 各模型路由比例；
- QA—Cost Pareto 曲线；
- 相对最便宜固定模型的质量提升；
- 相对最贵固定模型的成本节省。

### 9.6 必须比较的消融

| 输入给质量预测器的模型信息 | 目的 |
|---|---|
| 无模型信息 | 判断是否只是片段难度预测 |
| 参数规模 | 最弱能力代理 |
| 官方通用榜单分数 | 判断通用能力是否足够 |
| 仅 \(\Psi_H\) | HaluMem 单源贡献 |
| 仅 \(\Psi_M\) | MemOps 单源贡献 |
| \(\Psi_H+\Psi_M\) | 完整 MemoryPrint |
| \(\Psi_H+\Psi_M+\)静态模型属性 | 检验补充信息 |
| 模型 ID embedding | 固定模型池上限，但不能接入未见模型 |
| Oracle per-segment quality | 理论上界 |

特别重要的是比较 `参数规模` 与 `MemoryPrint`。如果 MemoryPrint 不能显著优于参数规模或固定 small/mid/large 顺序，那么复杂指纹的必要性不足。

---

## 10. 合成偏差的专门诊断

### 10.1 HaluMem 与 MemOps 的跨源一致性

分别计算：

\[
\Psi_H(m),\qquad\Psi_M(m)
\]

然后比较两者共同能力轴上的模型排序，例如：

- 新事实记住能力；
- 更新后当前状态；
- 旧值拒绝；
- 幻觉抵抗；
- 目标绑定。

如果同一个模型在 HaluMem 更新能力很强、在 MemOps 更新能力很弱，说明该维度可能依赖数据生成风格，不能直接视为稳定模型属性。

此时应保留来源分开的维度，而不是把两者先平均。

### 10.2 Leave-One-Source-Out

进行两组实验：

```text
只使用 HaluMem 指纹 → 预测 MemOps 和 LoCoMo
只使用 MemOps 指纹 → 预测 HaluMem 和 LoCoMo
```

这可以判断质量预测器是否依赖某一生成模板。

### 10.3 语言风格扰动

不改变 Gold 的前提下，对一部分探针进行：

- 对话轮次位置变化；
- 人名替换；
- 同义改写；
- 时间表达格式变化；
- 无关寒暄插入；
- 事实顺序变化。

这些变体共享原 Gold，因此不需要重新人工标注。比较原始探针和扰动探针上的能力变化：

\[
\Delta_{style,k}(m)
=
\Psi_k^{original}(m)-\Psi_k^{perturbed}(m)
\]

如果 \(\Delta_{style}\) 很大，说明指纹测到的是模板适配，不是稳定能力。

---

## 11. 什么时候可以宣称“具有泛化性”

只有同时满足以下证据时，才建议使用“generalizes to unseen models and benchmarks”一类表述：

1. HaluMem 与 MemOps 的共同能力维度具有合理的一致性；
2. MemoryPrint 比参数规模和官方通用分数更能预测 LoCoMo 片段—模型质量；
3. 在完整未见模型家族上，冻结的质量预测器仍优于固定模型和规模路由；
4. 在未参与调参的 LongMemEval 上，仍能改善 QA—Cost 前沿；
5. 结果在多个预算档位成立，而不是只在某个预算点成立；
6. 指纹规模缩减和随机子集实验显示排序稳定；
7. Judge 或提示轻微变化不会完全改变结论。

如果以下任何情况出现，应降低论文主张：

- MemoryPrint 不优于参数规模；
- HaluMem 与 MemOps 对同一能力给出相反模型排序；
- 未见家族结果接近随机或固定中型模型；
- LongMemEval 上的路由遗憾不低于基线；
- 只有在使用 LongMemEval 调权重后才能得到好结果；
- 指纹对 Judge 选择高度敏感。

此时应将 \(\Psi(m)\) 表述为“任务相关校准特征”，而不是“通用模型能力指纹”。

---

## 12. 泛化失败时的无人工标注补救方案

如果外部泛化不足，不需要回到大规模人工标注。按以下顺序处理：

### 12.1 降低指纹维度

保留最稳定的能力轴：

- 召回；
- 忠实度；
- 更新；
- 旧值拒绝；
- 目标绑定；
- 格式遵循。

删除对数据源高度敏感的细粒度维度。

### 12.2 使用多源指纹而不是混成一个分数

保留：

\[
[\Psi_H;\Psi_M]
\]

并加入来源标识或分组归一化，让质量预测器学习两类探针的互补信息。

### 12.3 使用现有目标域训练标签做轻量校准

只允许使用 LoCoMo 训练对话或 LongMemEval 官方训练/开发资源中已有标签，不触碰最终测试。

对于一个新模型，可在 20–50 个已有 Gold 的校准样本上运行，拟合一个小型线性校准层：

\[
\widehat q'_{i,m}=a_m\widehat q_{i,m}+b_m
\]

这不是重新训练整个路由器，也不需要人工新标注。

### 12.4 不确定性回退

当新模型的 \(\Psi(m)\) 明显超出训练模型能力分布，或者预测置信区间过宽时：

- 不进行激进低价路由；
- 回退到已验证的中型或大型模型；
- 或只在低风险片段上启用新模型。

这比假设所有新模型都可以零样本无损接入更可信。

---

## 13. 推荐的最小可执行实验

### 阶段一：冻结探针

1. HaluMem-Medium 抽取 600 个 session；
2. MemOps 抽取 200 个 adjacent-operation 样本；
3. 固定抽样 ID、随机种子、提示和 schema；
4. 冻结 80% 指纹集和 20% 审计集；
5. 记录许可证和数据版本。

### 阶段二：生成模型指纹

对 Qwen 和 Llama 六个候选模型：

1. 使用相同写入提示提取 Fact；
2. 运行官方兼容评分；
3. 运行 Router-Adapted 评分；
4. 计算 \(\Psi_H\)、\(\Psi_M\) 和置信区间；
5. 检查 HaluMem—MemOps 共同维度一致性；
6. 检查 80% 指纹集能否预测 20% 审计集表现。

### 阶段三：LoCoMo 训练与未见家族测试

1. 按完整 conversation 划分；
2. 所有训练模型处理同一批主题段；
3. 计算 Fact 质量和固定流水线 QA 效用；
4. 训练 \(f_\theta(x,\Psi(m))\)；
5. 执行 Qwen → Llama 和 Llama → Qwen；
6. 与参数规模、模型 ID 和固定模型路由比较。

### 阶段四：LongMemEval 冻结测试

1. 不再修改任何组件；
2. 在多个预算档位运行写入路由；
3. 使用固定 Top-k、Reader 和 Judge；
4. 报告 QA、成本、预算违反率、路由比例和 Regret；
5. 完成 \(\Psi_H\)、\(\Psi_M\)、组合指纹消融。

---

## 14. 论文中的推荐表述

可以表述为：

> We represent each candidate LLM using a price-independent, task-specific MemoryPrint obtained from two external calibration suites, HaluMem and MemOps. HaluMem characterizes extraction completeness, faithfulness, interference resistance, memory types, and update behavior, while MemOps complements it with target binding, state transition, stale-value rejection, invalid-state rejection, and provenance. The calibration suites are disjoint from downstream routing benchmarks. We train the segment–model quality predictor on LoCoMo and evaluate the frozen router on held-out model families and LongMemEval.

关于泛化边界，可以表述为：

> Since HaluMem, MemOps, LoCoMo, and LongMemEval all contain controlled or model-generated components to different degrees, our experiments establish cross-benchmark and cross-generation-pipeline transfer rather than unrestricted real-world conversational generalization.

不建议表述为：

> HaluMem and MemOps provide an objective universal representation of every LLM.

更准确的是：

> HaluMem and MemOps provide a fixed, task-conditional estimate of model behavior whose predictive validity is tested on independent memory benchmarks.

---

## 15. 数据许可与发布

HaluMem Hugging Face 页面当前标注的许可证为 `CC-BY-NC-ND-4.0`。因此论文代码仓库中不应重新发布经过修改或重打包的 HaluMem 数据。

可以发布：

- 官方下载链接；
- 选中样本的 ID 列表；
- 抽样随机种子；
- 本地转换程序；
- prompt hash；
- 指标计算程序；
- 模型级聚合指纹和置信区间。

MemOps GitHub 仓库当前标注为 MIT License，但仍应保留原作者版权与许可证说明。

在正式提交前应再次检查两个数据源的最新许可证和版本哈希。

---

## 16. 最终建议

当前项目不应继续走“大规模自建人工标注探针集”的路线。更合理的实施方式是：

1. 使用 HaluMem-Medium 作为主要事实提取能力来源；
2. 使用 MemOps 的 Remember、Update、TargetBinding、StateTransition 和 Evidence 子任务作为补充；
3. 不把当前输出 schema 无法表达的 Forget、Reflect 和完整轨迹操作强行纳入主指纹；
4. 使用多维 \(\Psi(m)\)，不把能力压成 small/mid/large 顺序或单一总分；
5. 使用 LoCoMo 训练共享质量预测器；
6. 使用未见模型家族和 LongMemEval 验证跨模型、跨数据集迁移；
7. 将泛化主张限定为跨基准和跨生成流程，除非以后增加真实用户对话验证；
8. 价格始终留在外部预算优化层。

完整证据链为：

\[
\text{HaluMem/MemOps现成Gold}
\rightarrow
\Psi(m)
\rightarrow
f_\theta(x,\Psi(m))
\rightarrow
\text{预算约束路由}
\rightarrow
\text{LoCoMo/LongMemEval独立验证}
\]

这条路线不需要新建大规模人工标注数据，同时保留了新模型接入、价格变化适应和论文可解释性。
