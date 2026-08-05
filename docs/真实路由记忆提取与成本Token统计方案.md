# 真实路由记忆提取与成本、Token 统计方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-06
- Verification Status: ANALYZED（方案基于当前仓库代码审查，尚未执行完整五折真实 API 实验）
- Version Label: routed_extraction_cost_plan_v1

## 1. 方案目标

本方案用于在路由器训练完成后，让外层测试集真实执行以下流程：

```text
test sample
→ 主题分段
→ 冻结路由器逐 segment 决策 small / medium / large
→ segment 只进入被选中的 tier buffer
→ 按真实 buffer 规则形成 batch
→ 调用对应记忆提取模型
→ 写入本次部署专用记忆库
→ 组装数据库 S
→ QA 检索、Reader 回答、Judge 判分
→ 汇总准确率、真实 API 调用、Token 和费用
```

主要目标如下：

1. 测量路由策略在接近真实部署条件下的记忆提取成本和 QA 正确率。
2. 获得由真实路由结果决定的 tier 分布、buffer 批次数和 API 调用次数。
3. 避免把预生成完整 L/M/H 候选库的成本误认为某个路由策略的部署成本。
4. 校验训练期 `replay_virtual_cost` 与真实部署成本之间的偏差。
5. 为论文提供可复现的准确率—成本—Token 联合证据。

## 2. 核心实验原则

### 2.1 训练期与最终测试期分离

训练期继续使用冻结的 L/M/H 候选库，不在每个强化学习 rollout 中重新调用记忆提取模型。这样才能以可控费用探索不同路由动作，并减少模型生成噪声对奖励的影响。

最终测试期使用冻结后的路由器，对 outer-test sample 进行一次真实的 route-first 提取。最终测试期不得继续训练路由器，也不得使用测试集 QA 结果选择 checkpoint、预算、提示词或 buffer 参数。

### 2.2 初次 fact 提取严格限制在单个主题段内

每个 fact 必须满足：

- 只有一个 `segment_id`；
- 全部 `source_ids` 均属于该 `segment_id`；
- 不允许一个 fact 同时引用主题段 A 和主题段 B 的来源；
- 跨主题段推理由后续 QA Reader 检索多个独立 fact 后完成。

这一约束保证当 A、B 被路由到不同 tier 和不同模型时，每个 fact 仍然可以由其实际接收的单个主题段独立生成。

### 2.3 测试样本只经过其所属外层折一次

五折实验中，每个 sample 只能在一个 fold 中作为 outer-test sample 接受正式评估。五折结果合并后，每个 sample 恰好贡献一次 out-of-fold 预测和一次真实部署成本。

### 2.4 真实部署成本与实验基础设施成本分账

必须区分：

| 成本类别 | 含义 | 是否计入主方法部署成本 |
|---|---|---:|
| 候选库生成成本 | 为训练反事实动作预生成全部 L/M/H | 否，单独报告 |
| 真实路由记忆提取成本 | test segment 只进入实际选中 tier 后产生的调用 | 是 |
| Reader 成本 | 根据 S 中检索结果回答 QA | 否，单独报告 |
| Judge 成本 | 对预测答案评分 | 否，单独报告 |
| Embedding/Qdrant 本地成本 | 向量编码、写入、检索 | 不并入 LLM Token 费用，可报告延迟 |

## 3. 五折训练与真实测试协议

以 LongMemEval 的嵌套五折 `360/40/100` 为例，每个 fold 按以下顺序执行。

### 3.1 Fold 内训练阶段

1. 使用 360 个 train sample 训练路由器。
2. 使用 40 个 validation sample 执行 early stopping、预算约束检查和 checkpoint 选择。
3. 固定以下内容：
   - 路由器 checkpoint 与 scaler；
   - 路由特征构造方法；
   - small/medium/large 模型版本；
   - 记忆提取提示词及版本；
   - buffer 容量配置；
   - 最大输出 Token；
   - schema repair 规则；
   - Reader、Judge、检索 top-k；
   - 模型价格及价格生效日期。

### 3.2 Fold 外层测试阶段

对该 fold 的 100 个 test sample，逐 sample 执行：

1. 加载冻结 checkpoint 和 scaler。
2. 仅用 segment 文本及允许的结构特征进行确定性路由，不输入 QA question、answer、evidence 或 judge label。
3. 得到每个 segment 的唯一动作：`small`、`medium` 或 `large`。
4. 为三个 tier 建立相互独立且禁止跨 sample 的 buffer。
5. 每个 segment 只加入被选择的一个 buffer。
6. 使用与真实部署完全相同的 buffer overflow 和 flush 规则形成 batch。
7. 调用对应 tier 的提取模型并记录原始 request、response、provider usage、repair 和 retry。
8. 校验每个 fact 的 `segment_id` 和 `source_ids`，写入本次 test deployment 专用 extraction run。
9. 将本次真实提取的事实组装到独立的 S assembly。
10. 对该 sample 的全部 QA 进行检索、Reader 回答和 Judge 评分。
11. 保存 sample 级路由、提取、成本、Token、QA 和失败状态。

### 3.3 五折汇总

完成 fold 0 至 fold 4 后：

- 合并所有 outer-test sample 的 out-of-fold 结果；
- 不对同一 sample 的多个 fold 结果求平均，因为每个 sample 只能属于一个 outer-test fold；
- QA 准确率以所有 outer-test QA 为总体；
- 记忆提取成本首先按 sample 汇总，再提供按 QA 摊销的辅助指标；
- fold 均值和 fold 间标准差作为稳定性指标，不替代总体 out-of-fold 指标。

这里的“五折执行五次”指五个独立的 fold 评估任务，不是五次 API 调用。真实 API 调用总数由所有 test sample 在各 tier 中形成的 batch、schema repair 和 transport retry 决定。

## 4. 真实路由提取与冻结候选评估的关系

建议保留两套结果。

### 4.1 Frozen-candidate evaluation

用途：

- 训练强化学习路由器；
- 在相同候选记忆条件下公平比较大量策略；
- 绘制多预算、多个 baseline 的受控 QA—Cost 曲线；
- 降低重复调用提取模型产生的随机差异。

成本使用 `replay_virtual_cost`，必须标记为 `virtual` 或 `estimated deployment cost`。

### 4.2 Route-first deployment evaluation

用途：

- 给出主方法最终的真实 API 调用次数；
- 给出由实际 tier 分流和 buffer batching 形成的 Token 与费用；
- 给出真实生成记忆对应的最终 QA 正确率；
- 校准虚拟成本模拟器。

这套结果应作为论文真实部署效率的主要证据。

## 5. API 调用计数口径

“API 调用次数”至少拆分成以下三项。

### 5.1 Batch 数

每次 buffer flush 形成一个 extraction batch。它反映路由结果和 buffer packing 对请求数量的直接影响。

```text
batch_count = initial extraction batches
```

### 5.2 Logical call 数

一次 initial extraction 或一次 schema repair 均视为一个 logical call：

```text
logical_call_count
= initial extraction calls
+ schema repair calls
```

### 5.3 Transport attempt 数

一个 logical call 可能由于 429、5xx、超时或网络问题产生多个 transport attempts：

```text
transport_attempt_count
= successful attempts
+ failed/retried attempts
```

论文中应同时报告 logical call 和 transport attempt。只报告成功 batch 数会隐藏 repair 与 retry 的运行成本。

## 6. 费用统计方案

### 6.1 单次成功调用费用

对于模型 `m`：

```text
input_cost
= input_tokens × input_price_per_1m / 1,000,000

output_cost
= output_tokens × output_price_per_1m / 1,000,000

total_cost
= input_cost + output_cost
```

费用记录必须同时保存：

- 模型名和 provider request model name；
- 输入、输出单价；
- 币种；
- 价格生效日期；
- provider usage 或 tokenizer estimate 的来源标记。

### 6.2 Repair 与 retry 费用

- 成功的 initial call 和 schema repair call 均计入真实提取费用。
- provider 返回 usage 的成功 retry 结果计入费用。
- 对失败 transport attempt，如果 provider 没有返回 usage，不能把费用默认为 0，应标记 `unknown_cost_attempt`。
- 汇总时同时给出 `known_cost` 和 `unknown_cost_attempts`，避免虚假精确。

### 6.3 Sample 级主指标

记忆提取发生在对话历史/sample 层面，因此主要成本指标为：

```text
mean_extraction_cost_per_sample
= total_real_extraction_cost / number_of_test_samples
```

分母必须包含所有被路由的 outer-test sample。不能因为某个 sample 提取失败或 QA 失败就从分母删除，否则会产生幸存者偏差。

建议同时报告中位数、四分位距、P90/P95 和 sample bootstrap 置信区间，因为长历史 sample 的成本分布通常偏斜。

### 6.4 QA 摊销辅助指标

```text
amortized_extraction_cost_per_QA
= total_real_extraction_cost / total_number_of_QA_questions
```

应使用“总费用除以总 QA 数”，不能先计算每个 sample 的 `cost / qa_count` 再进行无权平均。

LongMemEval 通常为一个 sample 对应一个 QA，因此两个指标接近；LoCoMo 一个 sample 可包含多个 QA，必须同时报告二者。

### 6.5 联合效率指标

可作为辅助指标报告：

```text
extraction_cost_per_correct_QA
= total_real_extraction_cost / number_of_correct_QA
```

该指标同时受准确率和成本影响，不能替代准确率或每 sample 成本；当正确答案很少时也会不稳定。

## 7. Token 统计方案

### 7.1 必须记录的原始字段

每个 logical call 至少记录：

| 字段 | 含义 |
|---|---|
| `fold` | 外层折编号 |
| `sample_id` | sample 标识 |
| `batch_id` | 提取 batch 标识 |
| `tier` | small/medium/large |
| `call_type` | initial/schema_repair |
| `logical_call_index` | logical call 序号 |
| `transport_attempt` | transport attempt 序号 |
| `model_name` | 实际提取模型 |
| `segment_ids` | 本 batch 包含的主题段 |
| `input_tokens` | provider 输入 Token |
| `output_tokens` | provider 输出 Token |
| `total_tokens` | 输入与输出之和 |
| `usage_source` | provider/tokenizer_estimate/unavailable |
| `input_cost` | 输入费用 |
| `output_cost` | 输出费用 |
| `cost_status` | known/estimated/unknown |
| `latency_ms` | 调用延迟 |
| `finish_reason` | 完成原因 |
| `status` | succeeded/failed |

### 7.2 汇总层级

Token 至少按以下层级汇总：

1. 总体：全部 outer-test sample。
2. fold：fold 0 至 fold 4。
3. tier：small、medium、large。
4. sample：一次完整真实部署。
5. batch：一次 buffer flush。
6. logical call：initial 或 schema repair。

### 7.3 主要 Token 指标

建议报告：

- extraction input/output/total Token 总量；
- 每 sample 平均和中位 input/output/total Token；
- 每 QA 摊销 input/output/total Token；
- 每 batch 平均 input/output/total Token；
- 每 tier 的 input/output/total Token；
- 每 segment 平均提取 Token；
- 每 fact 平均输出 Token和成本；
- 每个正确 QA 对应的记忆提取 Token；
- schema repair Token 占比；
- retry 已知 Token 与费用，以及 unknown attempt 数。

输入和输出 Token 必须分开报告，因为三类提取模型的输入/输出单价可能不同；只报告 total Token 无法复算费用。

### 7.4 Token 指标能够支持的论文结论

这些指标可用于说明：

1. 路由器是否真实减少了记忆提取资源消耗。
2. 成本节省来自较便宜 tier 的选择，还是来自更高的 buffer batching 效率。
3. 不同 tier 的输出长度和事实数量是否存在系统差异。
4. 真实路由是否改善了准确率—费用 Pareto trade-off。
5. 训练期虚拟成本是否能可靠预测真实部署成本。

Token 是计算量和费用的代理指标，不能单独证明能耗、碳排放或硬件计算效率；若论文提出此类结论，还需记录硬件功耗或 provider 侧计算证据。

## 8. QA、Reader 和 Judge 的统计边界

QA 正确率和记忆提取成本使用不同的自然分母：

```text
QA micro accuracy
= correct_QA / total_QA

sample macro accuracy
= mean(correct_QA_in_sample / QA_count_in_sample)

mean extraction cost per sample
= total extraction cost / test_sample_count

amortized extraction cost per QA
= total extraction cost / total_QA
```

Reader 和 Judge 的 Token、费用应分别记录，不得混入 `memory_extraction_only` 主成本。建议最终同时报告：

- memory extraction actual cost；
- Reader actual cost；
- Judge actual cost；
- end-to-end API cost，作为补充。

## 9. 虚拟成本与真实成本校准

对每个 test sample 同时计算：

- `virtual_extraction_cost`：使用冻结历史和 `replay_virtual_cost`；
- `actual_extraction_cost`：route-first 真实 API usage；
- `virtual_batch_count_by_tier`；
- `actual_batch_count_by_tier`；
- 虚拟与真实 input/output Token。

建议指标：

```text
absolute_error
= actual_cost - virtual_cost

absolute_percentage_error
= abs(actual_cost - virtual_cost) / max(actual_cost, epsilon)

signed_relative_error
= (virtual_cost - actual_cost) / max(actual_cost, epsilon)
```

其中 `epsilon = 1e-12`。报告总体、fold、tier 和 sample 分布，并检查误差是否随历史长度、segment 数和路由比例系统变化。

## 10. 失败与异常处理

### 10.1 提取失败

- 保存失败 batch、错误类型、已知费用和 unknown attempts。
- 不得因为结果质量差自动重跑整个 sample。
- 是否允许固定次数的网络 retry 和 schema repair，必须在测试前确定。
- 达到预设上限后，将 sample 标记为 extraction failure。

### 10.2 部分记忆缺失

- 不允许把部分失败 sample 当作完整成功结果静默纳入。
- 同时报告 operational success rate 和 QA accuracy。
- 正式主分析可预先规定：无法构建 ready S 的 sample 计为 QA 错误；也可以单列失败率，但不得删除失败 sample 后只报告成功子集准确率。

### 10.3 Provider usage 缺失

- 若配置要求 provider usage，则缺失 usage 应使该 batch 无法进入“真实精确费用”统计。
- 可以记录 tokenizer estimate，但必须标记 estimated，不能与 provider-known cost 混称为真实精确费用。

### 10.4 模型输出随机性

即使 temperature 为 0，远程 API 仍可能因服务实现或模型版本变化产生差异。因此必须保存请求模型名、provider request ID、时间、提示词哈希和原始响应。

## 11. 建议的输出目录和账本

建议每折使用独立目录：

```text
outputs/rl_router/deployment_evaluation/<dataset>/<protocol>/fold_<k>/<method>/<run_id>/
├── manifest.json
├── routes.sqlite3
├── samples.sqlite3
├── aggregate.json
├── extraction_runs/
│   └── <sample_id>/
│       ├── attempts.sqlite3
│       ├── batches.sqlite3
│       ├── segments.sqlite3
│       └── raw/
└── qa/
    └── evaluations.sqlite3
```

`manifest.json` 至少保存：

- split manifest 路径与 SHA-256；
- fold 与 test sample IDs；
- checkpoint 路径与 SHA-256；
- router/scaler 版本；
- prompt 路径、版本与 SHA-256；
- buffer 配置；
- 模型名、API endpoint 标识；
- tokenizer 和 embedding 模型哈希；
- 价格表与生效日期；
- seed；
- 代码 commit；
- 开始、结束时间；
- 运行状态和失败样本列表。

## 12. 建议的 sample 汇总记录

每个 sample 生成一条汇总记录：

```json
{
  "fold": 0,
  "sample_id": "sample-id",
  "segment_count": 12,
  "tier_counts": {"small": 5, "medium": 4, "large": 3},
  "batch_count_by_tier": {"small": 1, "medium": 1, "large": 1},
  "logical_call_count": 3,
  "transport_attempt_count": 3,
  "schema_repair_call_count": 0,
  "input_tokens": 12000,
  "output_tokens": 1800,
  "total_tokens": 13800,
  "known_extraction_cost": 0.0123,
  "unknown_cost_attempts": 0,
  "fact_count": 48,
  "empty_fact_segment_count": 1,
  "qa_count": 5,
  "correct_qa_count": 4,
  "qa_accuracy": 0.8,
  "virtual_extraction_cost": 0.0115,
  "actual_minus_virtual_cost": 0.0008,
  "assembly_status": "ready",
  "extraction_status": "complete"
}
```

示例数值仅用于说明字段结构，不代表实验结果。

## 13. 实现建议

建议新增独立入口脚本，例如：

```text
scripts/evaluate_routed_deployment.py
```

脚本职责：

1. 加载 split manifest、fold、test partition 和 checkpoint。
2. 校验 sample 只属于该 fold 的 test partition。
3. 加载 segment 并构建冻结路由特征。
4. 确定性路由并写 route ledger。
5. 根据 actions 构建 selected-tier-only buffers。
6. 复用当前 `CandidateGenerator` 的请求、repair、usage 和 provenance 审计逻辑，但不能要求全部三个 tier 完成。
7. 使用独立 extraction run ID 和 deployment namespace，避免误读训练候选。
8. 将真实提取结果组装到独立 S assembly。
9. 复用当前 LightMEM-compatible evaluator 完成 QA。
10. 输出 sample、fold 和五折总体统计。

真实部署评估必须复用与线上提取相同的 `ExtractionBuffer` packing 逻辑。不要用一套函数形成真实 batch、另一套近似公式预测 batch，否则 API 调用次数仍可能不一致。

## 14. 论文建议报告表

### 14.1 效果与成本主表

| Method | QA Accuracy | Cost/Sample | Cost/QA | Input Tokens/Sample | Output Tokens/Sample | Logical Calls/Sample |
|---|---:|---:|---:|---:|---:|---:|
| All-Small |  |  |  |  |  |  |
| All-Medium |  |  |  |  |  |  |
| All-Large |  |  |  |  |  |  |
| Random Router |  |  |  |  |  |  |
| Proposed Router |  |  |  |  |  |  |

### 14.2 路由和 batching 表

| Method | Small % | Medium % | Large % | Batches/Sample | Segments/Batch | Repair Rate | Failure Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed Router |  |  |  |  |  |  |  |

### 14.3 虚拟成本校准表

| Fold | Virtual Cost | Actual Cost | Absolute Error | APE | Virtual Calls | Actual Calls |
|---|---:|---:|---:|---:|---:|---:|
| 0 |  |  |  |  |  |  |
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |

## 15. 实验成功判据

一次 fold 的真实部署评估至少满足：

- 只处理 manifest 声明的 outer-test sample；
- checkpoint、prompt、模型和价格配置均已冻结并记录哈希；
- 每个 segment 恰好获得一个 tier action；
- 每个 segment 只进入一个被选中 buffer；
- buffer 不跨 sample；
- 每个 fact 的全部 `source_ids` 属于其唯一 `segment_id`；
- 全部成功 batch 具有可审计 provider usage，或明确标记 estimate；
- repair、retry、失败调用和 unknown cost 均进入账本；
- S 仅包含本次 route-first extraction 产生的事实；
- QA 只读取对应 `sample_id + assembly_id` 的 S；
- 失败 sample 未从统计分母中静默删除；
- fold 结果包含准确率、调用、Token、费用和失败率。

五折全部完成后，还需验证每个 sample 恰好作为 outer-test sample 出现一次。

## 16. 最终推荐口径

论文主结果建议使用：

1. 所有 outer-test QA 的 out-of-fold micro accuracy。
2. 真实 route-first memory extraction cost per sample。
3. 每 QA 摊销提取成本作为辅助指标。
4. 输入和输出 Token 分别报告，并提供每 sample、每 QA、每 tier 和每 batch 统计。
5. logical calls、transport attempts、repair rate 和 operational failure rate。
6. Reader/Judge 成本与记忆提取成本分账。
7. 虚拟 replay 与真实部署成本误差，用于证明训练期成本代理的有效性和局限性。

该口径既保留冻结候选训练的可行性，又通过最终 outer-test 的真实路由提取获得可信的部署成本和 API 调用数据。
