# 五阶段进度、S 记忆更新与提取缓冲区分析

本文记录 2026-08-09 对 InfoBudget2 与本地 LightMem 复现代码的审计结果。为避免把事实、推断和实验建议混在一起，下文分别标注“代码事实”“解释”和“建议”。

## 1. 五阶段进度与指标输出

已加入统一的 `StageProgress`，终端为 TTY 时原地刷新进度条，日志重定向或 CI 环境中按行输出快照。进度写入 stderr，最终 JSON 写入 stdout，因此不会破坏现有机器读取流程。实验 manifest、SQLite ledger 和 Qdrant 对账结果仍是统计权威来源，进度输出只是运行时可观测性。

| 阶段 | 进度单位 | 运行时指标 |
|---|---:|---|
| 信息分段 | sample | sample、segment 数、turn 数、估算 token 数、耗时、速率、ETA |
| 训练数据记忆提取 | API batch | tier、状态、segment/fact 数、逻辑调用数、输入/输出 token、费用、延迟、repair 数 |
| 路由训练 | episode | epoch、sample、QA score、归一化费用、reward、拉格朗日乘子；验证阶段另有 sample 级进度 |
| 评估数据记忆提取 | API batch | sample、tier、状态、segment/fact 数、逻辑调用数、输入/输出 token、费用 |
| QA + Judge | question | question id、累计正确数/准确率、检索 fact 数、Reader+Judge token 和费用 |

候选提取、QA 和 Judge 的 token/费用来自供应商 usage 与项目价格表；分段阶段显示的 token 是项目现有正则计数器的估算值，两者不可混作同一口径。

LoCoMo/LongMemEval 的批量候选提取 PowerShell 入口还会在每个 sample 完成后打印 `SAMPLE SUMMARY`，并在全部 sample 结束时输出成功/失败样本数、wall time、总 fact、API 调用、输入/输出/总 token 和已知费用。单 sample Python 进程仍保留 batch 级进度与最终 JSON，因而既能观察当前调用，也能得到全数据集汇总。

QA/Judge 评估完成后还会打印并记录与 LightMem LoCoMo 评估一致的总体及 Category 分类指标：问题分布、`judge_correct` mean/std/count，以及 prompt/completion/total token。标准差采用 LightMem 源码中的 `np.std` 口径，即二元正确性上的总体标准差 `ddof=0`。项目另外按 Reader/Judge 分开记录调用、token、重试和费用，并提供二者合计；这比 LightMem 示例中只累计回答模型 token 的口径更完整。LoCoMo 的 `category_5` 通过 `evaluation.excluded_categories_by_dataset` 显式排除，因此标准 10-sample 评估为 Category 1–4 共 1540 题；排除范围会写入 deployment manifest 和 aggregate，避免隐式过滤。

## 2. LightMem 离线更新的本地代码事实

本节只陈述本地仓库 `S:\Workfile\VScode\LightMem` 中的实现，不表示 InfoBudget2 已采用该逻辑。

1. `add_memory()` 在抽取和 `MemoryEntry` 转换后，根据配置选择 online/offline 写入。`offline_update(memory_entries)` 的默认参数不会构建更新队列，也不会执行 LLM 更新；LoCoMo 与 LongMemEval 的实验脚本随后分别显式调用 `construct_update_queue_all_entries()` 和 `offline_update_all_entries()`。
2. `offline_update()` 首先为记忆生成向量并插入向量库。其 payload 包含时间、主题、类别、原始/压缩记忆和 speaker 等字段。
3. 更新队列对每条新记忆检索时间不晚于它的近邻，默认检索 20 条并保留 10 条，把 `{id, score}` 写入该新记忆的 `update_queue`。
4. 离线更新反向查找“哪些较新记忆把当前目标记忆放进了队列”，达到阈值后调用 `UPDATE_PROMPT`，动作是 `update`、`delete` 或 `ignore`。
5. LightMem 已累计更新调用的 prompt/completion/total token，但当前片段未记录逐动作的价格、重试、请求 ID、prompt hash 或完整 lineage。
6. 两个实现风险需要在仿照时修正：
   - `offline_update()` 内部以 `update_sim_threshold=0.8` 调用函数，但被调函数参数名是 `score_threshold`；走这个内部触发分支会产生参数名不匹配。
   - 文本被更新后仍把旧 `entry["vector"]` 写回，未对新文本重新 embedding，可能导致向量与 payload 文本不一致。

## 3. 训练 S 是否应增加更新步骤

### 3.1 结论

不建议立即把更新作为默认训练路径。建议把它实现为一个可关闭、可复现、训练/测试完全对称的 ablation：

- 基线 A：现有 S，不更新；
- 实验 B：S 组装后、QA 前执行 consolidation；
- 冻结的 L/M/H 候选永远不原地修改；
- B 产生新的不可变 `S_consolidated` 版本，保留原 S 和完整事实 lineage。

原因是路由器的 action 本来定义了“每个 segment 选择哪一级候选记忆”。若在 S 上再做不可审计的合并/删除，reward 同时反映路由质量与更新器质量，策略学习的归因会变得含混。只有当训练 rollout 与测试部署使用完全相同的更新 prompt、模型、阈值、顺序和成本目标时，B 才是有效的端到端策略。

### 3.2 推荐处理顺序

1. 组装原始 S，冻结其 `assembly_id` 和 point 列表。
2. 只在同一 `dataset/split/sample/assembly` 内构建候选边，禁止跨 sample 合并。
3. 按时间和稳定排序键串行处理；相同输入必须得到相同候选顺序。第一版不建议并发写入。
4. 对超过阈值的候选调用更新模型，返回严格 JSON：`ignore/update/delete`、新事实、理由和被引用事实 ID。
5. `update` 后必须重新 embedding；`delete` 使用 tombstone，不物理删除原始事实。
6. 对账后生成新 `consolidated_assembly_id`，再进入 retrieval、QA 和 Judge。
7. reward 的费用项明确包含 consolidation 的真实费用；同时单独报告“不含更新费用”的诊断指标，避免成本口径含混。

### 3.3 建议的 Qdrant payload

Qdrant point 保存可检索状态和最必要的 lineage：

- 身份：`consolidation_schema_version`、`consolidation_run_id`、`consolidation_policy_version`、`source_assembly_id`、`consolidated_assembly_id`；
- 事实：`fact_id`、`root_fact_id`、`source_fact_ids`、`source_segment_ids`、`source_turn_ids`、`source_tiers`；
- 动作：`update_action`、`consolidation_status`、`content_revision`、`is_active`、`is_tombstone`、`supersedes`、`superseded_by`；
- 文本：`original_fact_text`、`consolidated_fact_text`、`conflict_type`；
- 候选：`candidate_fact_ids`、`candidate_scores`、`temporal_cutoff`、`top_k`、`keep_top_n`、`similarity_threshold`；
- 模型与 prompt：`prompt_version`、`prompt_sha256`、`updater_model`、`request_model`、`backend`、`temperature`；
- 向量：`embedding_model`、`embedding_revision`、`embedding_dimension`、`embedding_normalized`、`pre_embedding_sha256`、`post_embedding_sha256`；
- 时间：`created_at`、`updated_at`。

逐次调用细节不应全部塞进 Qdrant payload。建议另建 SQLite 表：

- `consolidation_runs`：scope、配置、状态、总计；
- `consolidation_queue`：target/candidate/score/rank/时间约束；
- `consolidation_attempts`：逻辑调用、transport attempt、usage、价格、延迟、请求 ID、错误；
- `consolidation_actions`：动作、输入/输出文本、理由、状态；
- `consolidation_lineage`：parent/child/supersede/tombstone 关系。

### 3.4 Token、费用和质量统计

每个更新调用至少记录：provider input/output/total token、input/output/total cost、currency、逻辑调用数、repair 调用数、transport attempts、retry 数、latency、usage source、provider request id、finish reason。失败与 usage 未知的尝试也必须入账，不能用成功调用均值回填。

每个 assembly 同时报告：更新前/后 fact 数、update/delete/ignore 数、压缩率、候选覆盖率、阈值命中率、空输出率、schema repair 率、失败率、孤立 provenance 数、向量-文本 hash 对账、更新 token/费用/耗时、QA 准确率变化、检索 recall 变化。最终用成对的 A/B 测试回答“更新是否有益”，而不是只看 fact 数下降。

## 4. 当前 6-segment 缓冲区分析

### 4.1 配置事实

- 所有 L/M/H 缓冲区当前均为：`max_segments=6`、`max_input_tokens=16384`、`max_total_context_tokens=24576`。
- 输出预留为每 segment 1024 token；6 个 segment 的普通 batch 预留 6144。
- 每 segment 最多输出 15 条 fact。
- small 模型上下文 32768、最大输出 8192；medium 为 262144/16384；large 为 131072/32000。
- buffer 在加入第 7 个 segment、输入超过 16384，或“输入 + 每段输出预留”超过 24576 时 flush。
- 超长单段若输入超过普通输入阈值但仍满足总上下文阈值，会作为 oversize singleton 单独处理。

### 4.2 数据证据

`token_count` 列是项目正则估算；“渲染输入”使用本地 tier tokenizer 对完整抽取 prompt 计数。

| 数据/分段 | 单段估算 token（均值 / P95 / 最大） | 每 6 段估算 token（均值 / P95 / 最大） |
|---|---:|---:|
| LoCoMo NSP，1054 段 | 149.6 / 348.4 / 582 | 876.2 / 1302.4 / 1614 |
| LoCoMo BERT-MLP，702 段 | 224.7 / 342.0 / 444 | 1292.7 / 1629.7 / 1901 |
| LongMemEval NSP，95722 段 | 414.3 / 704.0 / 12182 | 2454.9 / 3180.0 / 14118 |
| LongMemEval BERT-MLP，78424 段 | 505.7 / 729.9 / 12182 | 2986.3 / 3555.0 / 14683 |

完整渲染输入的 tokenizer 结果：

- LoCoMo NSP 的全部 180 个 6 段或尾批：空 prompt 2446 token；输入均值 4730、P95 5749、最大 6372；没有 batch 超过 16384。
- LongMemEval NSP 从 16156 个批次中以 seed 42 固定抽样 1000 个：空 prompt 2325 token；输入均值 6188、P95 7335、P99 7979、最大 9275；样本中没有 batch 超过 16384。
- 已知超长样本 `852ce960` 的 NSP `seg_000041`：单段渲染输入 20982，加入当前 1024 输出预留后为 22006，低于 24576，因此当前 oversize-singleton 路径能容纳它。
- 同一样本的 BERT-MLP `seg_000036`：单段渲染输入 20984，总计 22008，也能容纳。

### 4.3 对“仿照 LightMem 32k、512 输入缓冲、约 7800 输出”的意见

不建议直接照搬。

1. LightMem 的 512 更接近短期记忆触发的 token 阈值，而当前 `max_segments=6` 是结构化 segment 的数量上限，不是 6 token 或 6 轮；两个参数语义不同。
2. 当前普通 LoCoMo/LongMemEval batch 的真实输入远低于 16384。把输入缓冲强行降到约 512 会大幅增加 API 调用数和固定 prompt 开销，不利于当前实验的费用目标。
3. 当前每批最多 6×15 条结构化 fact，输出预留 6144；small 模型的硬上限是 8192。将“每批输出上限”设为约 7800可以作为一个候选 ablation，但不能设成“每 segment 7800”，否则总上下文规划失真。
4. 超长 singleton 的输入已到约 21k。若给它统一预留 7800，总计约 28.8k，超过当前项目人为设置的 24576，但仍低于 small 的 32768。是否提高 `max_total_context_tokens` 应单独做稳定性验证，不能和普通 batch 的 segment 数一起改。
5. 输出容量应依据真实 `finish_reason`、输出 token 分布、`fact_limit_reached` 和 schema repair 率决定。只因模型允许更长输出就放宽，可能增加冗余事实和幻觉，也会改变路由器的成本 reward。

推荐实验顺序：

- 先保持 6 / 16384 / 24576 / 1024，完成 alpha×epoch 主实验，形成基线；
- 从 ledger 汇总各 tier 的输入/输出 P50/P95/P99、truncation、fact saturation、repair、失败率；
- 若普通 batch 的固定 prompt 开销占比过高，仅改变 `max_segments` 做 6/8/10 消融，同时保持 16384 输入和 24576 总上下文；
- 若出现输出截断，再将“每批最大输出”从 6144 对比到 7800，并保持 `max_facts_per_segment` 不变；
- 对 oversize singleton 建立独立策略和指标，不用一个极端样本反向决定全部普通 batch 的缓冲区。

现有证据支持的当前判断是：6 段设置安全但偏保守，普通 batch 的输入窗口未充分利用；最值得优先测试的是 8/10 段的吞吐消融，而不是 512 输入缓冲。任何缓冲区变更都应在 alpha 与 epoch 选择之后单独实验，避免同时改变多个自变量。
