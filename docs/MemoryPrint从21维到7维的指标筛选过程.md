# MemoryPrint 从 21 维到 7 维的指标筛选过程

> 文档用途：记录 MemoryPrint 核心指标从初始 21 维诊断向量缩减为 7 维核心指纹的完整决策过程，作为后续论文方法部分、附录、消融实验和复现说明的依据。
>
> 文档日期：2026-08-12。
>
> 适用对象：在固定事实提取提示和固定输出 schema 下，从主题片段中输出 `Fact + source_ids` 的候选 LLM。

> 两个官方基准的实际部署版本、数据校验和运行入口见 [MemoryPrint 外部评测环境部署与运行说明](./MemoryPrint外部评测环境部署与运行说明.md)。

## 1. 研究对象与筛选目标

当前项目需要描述的不是完整长期记忆系统，而是候选 LLM 在固定写入协议下的记忆提取行为：

\[
\text{Topic Segment}
\rightarrow
\text{Candidate LLM}
\rightarrow
\{\text{Fact},\text{source\_ids}\}
\]

因此，MemoryPrint 应主要回答以下问题：

1. 模型能否覆盖应写入的事实；
2. 模型是否只写入目标记忆；
3. 写入内容是否受原对话支持；
4. 面对事实更新时，能否识别当前有效状态；
5. 能否把事实绑定到正确人物、实体或属性；
6. 能否拒绝已经失效的旧值；
7. 能否给出正确的来源证据。

MemoryPrint 不直接描述以下能力：

- Retriever 的检索能力；
- Reader 的答案生成与推理能力；
- 完整记忆系统的删除、合并或数据库维护能力；
- API 价格、延迟和稳定性；
- 下游 QA 的最终正确率。

## 2. 为什么不把 QA 正确率纳入核心指纹

HaluMem 将评估划分为 Memory Extraction、Memory Updating 和 Memory Question Answering 三个阶段。其 Memory QA 结果同时受到事实提取、记忆更新、检索和答案生成的影响，因此是端到端系统效用，而不是纯粹的记忆提取能力。[HaluMem 论文](https://arxiv.org/abs/2511.03506)和[官方仓库](https://github.com/MemTensor/HaluMem)均将三个阶段分开报告。

MemOps 也明确指出，最终答案正确率会混合多种失败来源：模型可能遗漏记忆事件、绑定错误目标、采用过期值或进行无依据推断；反过来，模型也可能在内部记忆状态错误的情况下偶然答对。因此，MemOps 的最终 Answer Accuracy 同样不进入核心 MemoryPrint。[MemOps 论文](https://arxiv.org/abs/2607.12893)

本项目采用以下分层：

| 层次 | 用途 | 是否进入核心 MemoryPrint |
|---|---|---:|
| 提取与操作级指标 | 直接评价 `Fact + source_ids` | 是 |
| QA Accuracy / QA Hallucination / QA Omission | 评价固定流水线的最终效用 | 否 |
| LoCoMo、LongMemEval QA | 验证指纹和路由的下游价值 | 否，作为预测目标或外部测试结果 |

## 3. 初始 21 维诊断向量

初始方案将 HaluMem 的 14 个诊断指标与 MemOps 的 7 个项目适配指标直接拼接：

\[
\Psi_{21}(m)=[\Psi_H^{14}(m);\Psi_M^{7}(m)]
\]

### 3.1 HaluMem 14 维

\[
\Psi_H^{14}(m)=
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

这些指标覆盖事实匹配、召回、精确性、忠实度、幻觉抵抗、干扰抵抗、分类型记忆、更新和格式遵循。

### 3.2 MemOps 7 维

\[
\Psi_M^{7}(m)=
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

这些指标覆盖 Remember、目标绑定、当前状态、旧值拒绝、无效值拒绝和证据定位。

### 3.3 初始方案的主要问题

21 维方案适合作为诊断指标全集，但不适合直接作为只有少量候选模型时的核心模型表示，原因包括：

1. **指标冗余**：Recall、Soft Recall 和 F1 共享大量信息；多个幻觉抵抗指标也高度相关。
2. **同一能力存在多个来源**：事实覆盖同时由 HaluMem 和 MemOps 测量，更新能力也在两个数据源中重复出现。
3. **派生指标重复输入**：F1 由 Precision 和 Recall 计算，再与二者共同输入会重复编码。
4. **诊断切片被当作独立能力**：Persona、Event 和 Relationship 是同一召回能力的条件切片，不一定是三个独立模型属性。
5. **操作指标与语义能力混合**：`F_schema` 更适合作为可用性门槛，而不是记忆语义能力。
6. **模型级有效样本量有限**：即使存在大量“片段—模型”样本，若仅有 6 个候选模型，实际只有 6 个不同的模型指纹向量。21 个模型侧特征会显著增加共线性和过拟合风险。

## 4. 指标筛选原则

### 4.1 任务一致性

只有能从候选模型的 `Fact + source_ids` 输出中直接评价的指标进入核心指纹。依赖完整数据库操作、Retriever、Reader 或最终 QA 的指标不进入。

### 4.2 每项能力只保留一个主来源

同一能力不对 HaluMem 和 MemOps 指标进行平均，也不手工学习一个跨数据集加权分数。主来源按数据集的专长确定：

- 静态事实提取优先使用 HaluMem；
- 生命周期状态、目标绑定和旧值控制优先使用 MemOps；
- 被替代的数据源指标保留为跨源一致性审计，不进入核心向量。

这里的“权威”指数据集的任务设计与待测能力更加直接匹配，并不表示另一个数据源无效。

### 4.3 优先原始指标，不重复输入派生指标

当 F1 已由 Recall 和 Precision 唯一确定时，核心向量保留两个原始分量，不再额外输入 F1。派生指标仍可用于表格展示和与官方结果比较。

### 4.4 宽覆盖指标优先，窄场景指标作为审计

例如，HaluMem Memory Accuracy 面向全部候选事实，而 FMR 只面向“助手提出但用户未确认”的干扰事实。核心指纹选择覆盖范围更广的 Accuracy，FMR 留作对抗性审计。

### 4.5 所有核心维度统一为高值更优

所有特征均映射到 \([0,1]\)，且数值越大表示能力越强。对于错误率指标使用互补变换，例如：

\[
R_{old}=1-L_{old}
\]

## 5. 权威数据源的分工

| 能力类别 | 选择的数据源 | 选择依据 |
|---|---|---|
| 事实覆盖 | HaluMem | Memory Extraction 直接以 Gold memory points 评价提取完整性 |
| 目标记忆精确率 | HaluMem | 官方提供 Target Memory Precision，直接对应“是否属于应写入字段” |
| 事实忠实度 | HaluMem | 官方 Memory Accuracy 对每条候选记忆进行来源支持与幻觉判断 |
| 当前状态 | MemOps | Gold operation trace 显式描述状态转换和当前有效值 |
| 目标绑定 | MemOps | Gold trace 显式提供 operation target，适合区分人物与属性归属 |
| 旧值拒绝 | MemOps | 专门提供 stale value 和更新链诊断 |
| 证据定位 | MemOps | 提供 Gold provenance、原文证据和 segment-turn 位置 |

HaluMem 负责“写入什么、写得是否正确”，MemOps 负责“写给谁、哪个值当前有效、证据在哪里”。这种分工减少了跨数据集的重复计量。

## 6. 从 21 维到 7 维的逐项映射

### 6.1 HaluMem 14 维的处理

| 初始维度 | 处理 | 最终去向 | 理由 |
|---|---|---|---|
| \(F1_{fact}\) | 删除 | 诊断/展示指标 | 由 Precision 与 Recall 派生，和原始分量共同输入会重复编码 |
| \(R_{strict}\) | 合并替换 | \(R_H\) | 核心指纹统一采用 HaluMem 官方 Memory Recall |
| \(R_{soft}\) | 合并替换 | \(R_H\) | 与严格召回测量同一覆盖能力；软评分保留为敏感性分析 |
| \(P_{target}\) | 保留并对齐官方定义 | \(P_H^{target}\) | 直接衡量候选事实是否属于目标记忆字段 |
| \(P_{faith}\) | 保留并对齐官方定义 | \(A_H\) | 与 HaluMem Memory Accuracy 的事实支持含义一致，采用官方名称和实现 |
| \(HR\) | 删除 | 幻觉审计指标 | 与 Memory Accuracy 高度相关，且只强调完全不受支持输出 |
| \(IR_{soft}\) | 删除 | 干扰压力测试 | 针对 interference 子集，覆盖面窄于总体 Accuracy |
| \(R_{persona}\) | 删除 | 分类型诊断 | 是 Recall 在 Persona 子集上的条件切片 |
| \(R_{event}\) | 删除 | 分类型诊断 | 是 Recall 在 Event 子集上的条件切片 |
| \(R_{relationship}\) | 删除 | 分类型诊断 | 是 Recall 在 Relationship 子集上的条件切片 |
| \(U_{correct}\) | 更换数据源 | \(S_M^{current}\) | 当前状态由 MemOps 的 Gold operation trace 描述得更直接 |
| \(1-U_{hall}\) | 删除 | 更新审计指标 | 与更新正确率及总体事实忠实度重叠，且条件分母可能较小 |
| \(SR\) | 更换数据源 | \(R_M^{old}\) | MemOps 显式提供 old/new value 和 stale-value 诊断 |
| \(F_{schema}\) | 删除 | 部署门槛 | 格式遵循决定输出是否可用，但不是记忆语义能力 |

### 6.2 MemOps 7 维的处理

| 初始维度 | 处理 | 最终去向 | 理由 |
|---|---|---|---|
| \(R_{remember}\) | 删除 | 跨源审计 | 与 HaluMem Recall 重复；事实覆盖以 HaluMem 为主来源 |
| \(1-H_{remember}\) | 删除 | 跨源审计 | 与 HaluMem Memory Accuracy 重复；事实忠实度以 HaluMem 为主来源 |
| \(B\) | 保留 | \(B_M\) | MemOps 对目标对象和属性具有显式 Gold 标注 |
| \(S_{current}\) | 保留 | \(S_M^{current}\) | 直接要求目标和值同时正确，契合当前状态提取 |
| \(R_{old}\) | 保留 | \(R_M^{old}\) | 直接度量过期值是否仍被当作当前事实 |
| \(R_{invalid}\) | 删除 | 安全性审计 | 混合 tentative、retracted、negative seed、recency trap 等多类无效状态，范围较杂；第一版不增加第八维 |
| \(F1_E\) | 保留 | \(F1_M^E\) | 与项目输出中的 `source_ids` 一一对应，能独立评价证据定位 |

### 6.3 数量变化

筛选后的数量变化为：

```text
初始诊断向量：HaluMem 14 维 + MemOps 7 维 = 21 维
        ↓ 删除派生指标、条件切片、重复来源和操作门槛
HaluMem 核心：3 维
MemOps 核心：4 维
        ↓
最终核心 MemoryPrint：7 维
```

被删除的 14 个位置并非全部废弃，而是分别转移到以下三类输出：

- 官方兼容结果表；
- 错误诊断与压力测试表；
- 消融和跨源一致性审计。

## 7. 最终 7 维 MemoryPrint

最终定义为：

\[
\boxed{
\Psi_7(m)=
[
R_H,
P_H^{target},
A_H,
S_M^{current},
B_M,
R_M^{old},
F1_M^E
]
}
\]

| # | 符号 | 中文名称 | 数据来源 | 指标性质 |
|---:|---|---|---|---|
| 1 | \(R_H\) | 事实覆盖能力 | HaluMem | 官方 Memory Recall |
| 2 | \(P_H^{target}\) | 目标记忆精确率 | HaluMem | 官方 Target Memory Precision |
| 3 | \(A_H\) | 事实忠实度 | HaluMem | 官方 Memory Accuracy |
| 4 | \(S_M^{current}\) | 当前状态识别 | MemOps | 基于 Gold operation trace 的项目适配指标 |
| 5 | \(B_M\) | 目标绑定能力 | MemOps | 基于 Gold target 的项目适配指标 |
| 6 | \(R_M^{old}\) | 旧值拒绝能力 | MemOps | 基于 old/new value 的项目适配指标 |
| 7 | \(F1_M^E\) | 证据定位能力 | MemOps | 基于 Gold provenance 的项目适配指标 |

需要特别说明：前三维尽量复现 HaluMem 官方 Memory Extraction 评分；后四维不直接采用 MemOps 最终 QA Accuracy，而是把候选模型输出的 `Fact + source_ids` 与 MemOps Gold operation trace 对齐后计算。这样可避免 Reader 推理能力进入提取指纹。

## 8. 七个指标的计算定义

### 8.1 HaluMem 事实覆盖能力

对 HaluMem Gold 记忆点 \(g\)，官方 Judge 给出覆盖分数。核心实现应直接复用并固定 HaluMem 官方评估代码，聚合得到：

\[
R_H(m)=\text{HaluMemOfficialRecall}(m)
\]

不选择 Weighted Recall 的原因是其混入 `importance` 权重，模型可能通过优先提取少量高重要性事实取得高分。Weighted Recall 仍作为辅助结果报告。

### 8.2 HaluMem 目标记忆精确率

\[
P_H^{target}(m)=\text{HaluMemOfficialTargetPrecision}(m)
\]

它用于判断候选输出是否对应 Gold 中应写入的记忆字段。该指标与总体事实支持度分开，避免把“对话支持但不应长期存储”的内容视为正确写入。

### 8.3 HaluMem 事实忠实度

\[
A_H(m)=\text{HaluMemOfficialMemoryAccuracy}(m)
\]

Memory Accuracy 检查候选记忆中的信息是否受到对话或 Gold 支持。相比只针对特定干扰事实的 FMR，它适用于所有候选输出，因此被选为核心忠实度指标。

### 8.4 MemOps 当前状态识别

对样本 \(i\)，Gold operation trace 给出目标 \(target_i\) 和当前有效值 \(value_i\)。定义：

\[
B_{i,m}=\mathbf 1[\widehat{target}_{i,m}=target_i]
\]

\[
V_{i,m}=\mathbf 1[\widehat{value}_{i,m}=value_i]
\]

\[
S_M^{current}(m)=\frac{1}{N}\sum_i B_{i,m}V_{i,m}
\]

只有目标与当前值同时正确才得分。时间、数值、否定和专有名词等关键字段必须严格匹配。

### 8.5 MemOps 目标绑定能力

\[
B_M(m)=\frac{1}{N}\sum_i
\mathbf 1[\widehat{target}_{i,m}=target_i]
\]

允许别名与有充分依据的代词消解，但不能把他人的属性绑定到用户本人或另一个实体。

### 8.6 MemOps 旧值拒绝能力

对 Update 样本，定义旧值残留率：

\[
L_M^{old}(m)=
\frac{1}{N_U}\sum_i
\mathbf 1[v_i^{old}\text{仍被输出为当前有效事实}]
\]

转换为高值更优的旧值拒绝能力：

\[
R_M^{old}(m)=1-L_M^{old}(m)
\]

旧值可以作为带时间限定的历史事实保留；只有在模型把它继续表述为当前有效状态时才计为残留错误。

### 8.7 MemOps 证据定位能力

设预测证据轮次集合为 \(E_{i,m}\)，Gold evidence/provenance 集合为 \(E_i^*\)。采用 micro 聚合：

\[
P_E(m)=
\frac{\sum_i|E_{i,m}\cap E_i^*|}
{\sum_i|E_{i,m}|}
\]

\[
R_E(m)=
\frac{\sum_i|E_{i,m}\cap E_i^*|}
{\sum_i|E_i^*|}
\]

\[
F1_M^E(m)=
\frac{2P_E(m)R_E(m)}{P_E(m)+R_E(m)}
\]

该指标是项目基于 MemOps Gold provenance 构造的 Router-Adapted Metric，不应误称为 MemOps 官方排行榜指标。MemOps 官方主要报告 Provenance Support；本项目使用集合 F1，是因为输出 schema 明确包含可比较的 `source_ids`。

计算前必须把主题片段中的局部 `source_ids` 映射回 MemOps 原始 segment/turn 编号。若无法建立可靠映射，该维记为缺失并配套 mask，不得填 0。

## 9. 未进入核心指纹但必须保留的指标

### 9.1 官方兼容结果

- HaluMem Weighted Recall；
- HaluMem Memory Extraction F1；
- HaluMem FMR；
- HaluMem Memory Updating C/H/O；
- HaluMem Memory QA C/H/O；
- MemOps Answer Accuracy、Operation F1、Provenance Support、Leakage Rate、Stale Value Rate 和 Reflect Precision。

这些指标用于与官方论文或公开排行榜比较，不一定适合作为路由器输入。

### 9.2 错误诊断结果

- Persona、Event、Relationship 分类型召回；
- interference resistance；
- invalid-state rejection；
- update hallucination；
- longitudinal degradation；
- JSON/schema 一次解析成功率。

### 9.3 下游效用结果

- HaluMem QA；
- MemOps 最终 Answer Accuracy；
- LoCoMo QA；
- LongMemEval QA。

它们用于检验 \(\Psi_7(m)\) 是否具有预测价值，而不是作为 \(\Psi_7(m)\) 的组成部分。

## 10. 防止信息泄漏与过拟合

1. 七个维度及其计算方式必须在运行候选模型之前冻结。
2. 不得使用 LongMemEval 选择指标、阈值或权重。
3. 不得根据 6 个候选模型在最终测试集上的排序重新删减维度。
4. 指标方向、缺失值规则和 evidence 聚合方式必须预注册。
5. 21 维诊断向量可用于附录分析，但核心预测器以 7 维为主输入。
6. 必须比较 7 维与参数规模、模型 ID、HaluMem-only、MemOps-only 和固定模型策略。

推荐消融顺序：

```text
无模型信息
参数规模
HaluMem 3维
MemOps 4维
完整7维 MemoryPrint
完整21维诊断向量
模型 ID embedding
Oracle per-segment quality
```

若 21 维未显著优于 7 维，或只在已见模型上改善而在未见模型家族上退化，应优先采用 7 维版本。

## 11. 论文附录可直接使用的表述

### 11.1 中文表述

> 初始 MemoryPrint 包含 21 个诊断维度，其中 14 个来自 HaluMem 的事实覆盖、忠实度、干扰、记忆类型、更新和格式诊断，7 个来自基于 MemOps Gold operation trace 构造的 Remember、目标绑定、状态和证据指标。为降低小规模候选模型条件下的共线性和过拟合风险，我们依据任务一致性、单一权威来源、非冗余性和输出可观测性进行预先筛选。最终保留 HaluMem Memory Recall、Target Memory Precision 和 Memory Accuracy，以及基于 MemOps Gold trace 计算的 Current State、Target Binding、Stale Rejection 和 Evidence F1，共 7 个维度。QA 正确率仅用于下游效用验证，不进入模型指纹。

### 11.2 English wording

> The initial MemoryPrint contained 21 diagnostic dimensions: 14 HaluMem-based measures of extraction, faithfulness, interference, memory type, update behavior, and schema compliance, and 7 MemOps-based measures derived from gold lifecycle-operation traces. To reduce redundancy and model-level overfitting, we preregistered a compact feature set according to task alignment, single-source authority, non-redundancy, and observability from the extractor output. The resulting seven-dimensional MemoryPrint consists of HaluMem Memory Recall, Target Memory Precision, and Memory Accuracy, together with MemOps-derived Current State Accuracy, Target Binding Accuracy, Stale-Value Rejection, and Evidence F1. End-to-end QA accuracy is reserved for downstream validation and is not included in the model fingerprint.

## 12. 复现检查清单

- [ ] 使用固定版本的 HaluMem 和 MemOps；
- [ ] 固定样本 ID、随机种子、prompt、schema 和解码参数；
- [ ] HaluMem 三项指标复用固定版本的官方评估代码；
- [ ] MemOps 四项指标只读取 Gold operation trace，不读取最终测试答案进行调参；
- [ ] 每条预测 Fact 在评分前完成原子化；
- [ ] `source_ids` 已映射到原始 segment/turn；
- [ ] 所有维度均转换为 \([0,1]\) 且高值更优；
- [ ] 缺失维度使用 mask，不以 0 代替；
- [ ] 按 session、conversation 或用户级聚类 bootstrap 报告 95% 置信区间；
- [ ] 同时保存 7 维核心指纹和 21 维诊断结果；
- [ ] LongMemEval 在所有组件冻结后只运行一次正式外部测试。

## 13. 来源与版本说明

1. Chen, D., et al. *HaluMem: Evaluating Hallucinations in Memory Systems of Agents*. arXiv:2511.03506. [论文](https://arxiv.org/abs/2511.03506)；[代码与评估说明](https://github.com/MemTensor/HaluMem)。
2. Hao, X., et al. *MemOps: Benchmarking Lifecycle Memory Operations in Long-Horizon Conversations*. arXiv:2607.12893. [论文](https://arxiv.org/abs/2607.12893)；[代码与数据结构](https://github.com/MemTensor/MemOps)。

正式实验应记录论文版本、数据版本哈希和评估代码 commit。本文档中的“官方指标”指对应论文及官方评估实现中明确报告的指标；“项目适配指标”指使用官方 Gold 字段、但按照本项目 `Fact + source_ids` 输出契约重新定义的指标。
