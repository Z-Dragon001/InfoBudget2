# InfoBudget2 项目代码改进交接总结

## 1. 交接范围与当前状态

本文总结本轮对 InfoBudget2 训练数据记忆提取、候选存储、路由训练和评估流程所做的代码改进，不包含 Linux 主机、Docker daemon 和网络代理的现场调试过程。

当前主要提交为：

```text
21b8cf4 feat: complete memory extraction and RL router pipeline
```

当前测试结果：

```text
56 passed
```

测试中仅出现 `.pytest_cache` 写入权限 warning，不是代码或测试失败。

## 2. 当前端到端流程

项目已经形成以下四阶段流程：

```text
数据预处理与主题分割
        ↓
训练数据记忆候选提取（L/M/H）
        ↓
组装 S 并训练强化学习路由器
        ↓
基于 S 检索、回答和评估
```

核心约束如下：

- L/M/H 分别存储小、中、大模型提取的候选事实。
- 三个 tier 使用相同的事实提取协议、buffer 限制和 1024 维 embedding。
- 路由训练不会重新调用提取模型，而是按路由动作从 L/M/H 复制候选事实组装 S。
- QA 阶段只检索当前 `assembly_id` 对应的 S。
- `sample_id`、`extraction_run_id`、`batch_id` 和 `assembly_id` 参与隔离与审计。

主要实现说明见 `docs/rl_router_implementation.md`。

## 3. 训练数据记忆提取

### 3.1 LongMemEval 全数据调度器

新增 `scripts/build_longmemeval_rl_candidates.ps1`，实现：

- 自动遍历全部 LongMemEval 分割样本；
- 支持 `StartIndex`、`Limit`、`DryRun` 和失败续跑；
- 自动创建及刷新 extraction campaign；
- 支持小、中、大 tier 分别运行；
- provider 出现不可重试错误后，对对应 tier 熔断；
- 输出全局调度结果、失败列表和已打开的 provider circuit。

### 3.2 小、中、大模型独立运行

候选提取入口支持三个 tier 使用独立命令和独立进程：

```powershell
uv run python scripts/build_rl_candidates.py <segments.jsonl> `
  --extraction-run-id <run_id> --tier small

uv run python scripts/build_rl_candidates.py <segments.jsonl> `
  --resume <run_id> --tier medium

uv run python scripts/build_rl_candidates.py <segments.jsonl> `
  --resume <run_id> --tier large
```

每次调用只加载和验证所选 tier 的 API Key、tokenizer、模型配置及 provider 客户端。三个 tier 可以独立执行，但共享同一 run 和 campaign；只有要求的 tier 全部完成，run 才能进入完整状态。

### 3.3 API Key 和模型 fail-fast

在创建新 run 产物之前执行：

- 所选 tier 的 API Key 完整性检查；
- tokenizer 与本地 embedding 模型路径检查；
- 本地模型加载和 smoke test；
- Qdrant 连通性及 Collection schema 检查。

缺少 Key、本地资源不可加载或 Qdrant schema 不兼容时，不会先遗留部分 run 产物。

### 3.4 Provider 错误分类与熔断

当前行为：

- 可重试错误按退避策略重试；
- 不可重试错误立即终止当前 tier；
- 外层 LongMemEval 调度器将对应 tier 标记为熔断；
- 后续 sample 跳过已熔断 tier，避免无效请求和费用浪费；
- 使用专门退出码区分 provider 熔断和普通 sample 失败。

## 4. Extraction campaign

新增 `infobudget/rl_router/campaign.py` 和 `scripts/manage_extraction_campaign.py`，用于管理全数据不可变 extraction campaign。

Campaign 冻结以下范围：

- 数据集、split 和分割方法；
- 全部 sample 及其 `segments.jsonl` 哈希；
- 每个 sample 对应的预期 run ID；
- prompt 和提取配置；
- 小、中、大模型配置；
- embedding 模型完整哈希；
- campaign scope hash。

训练必须显式传入 `--campaign-id`，并拒绝：

- 缺少 sample run 的 campaign；
- run 与 campaign scope 不一致；
- 分割文件发生变化；
- embedding 模型发生变化；
- 汇总质量门不通过的 campaign。

训练不再根据 JSONL 中的出现顺序选择所谓“latest complete run”，而是使用 campaign 固定的 sample 到 run 映射，避免并发执行时选择错误 run。

## 5. Buffer 与超长主题段

统一配置位于 `configs/rl_router.yaml`：

```yaml
max_facts_per_segment: 15

buffers:
  small:
    max_segments: 6
  medium:
    max_segments: 6
  large:
    max_segments: 6
```

处理策略如下：

1. 普通主题段按 token 限制组成 batch，最多六段；
2. 超过普通 batch 输入限制时，先 flush 当前 buffer；
3. 超长主题段采用 singleton batch，单独调用一次模型；
4. singleton 仍超过安全总上下文时，执行确定性尾部截断；
5. 已移除二分兜底逻辑，不生成额外 extraction chunk；
6. 一个逻辑 segment 始终只进行一次实际提取调用。

截断时保持原始：

- `segment_id`；
- `turn_ids`；
- 起止 turn；
- 时间范围；
- `source_content_hash`；
- fact 归属关系。

只根据截断后可见文本临时计算：

- `extraction_visible_source_ids`；
- `extraction_dropped_source_ids`。

Parser 只允许模型引用可见 source IDs。截断前后 token 数、字符数、可见和丢弃来源均写入 manifest、成本 ledger 和 Qdrant payload。

## 6. 事实数量与质量门

每个主题段最多允许 15 条事实，并增加以下全量质量指标：

- 空 facts segment 比例；
- 恰好达到 15 facts 的饱和比例；
- schema repair batch 比例；
- 失败 batch 比例；
- `finish_reason=length` 截断响应。

当前阈值：

```yaml
quality_gates:
  max_empty_fact_segment_rate: 0.25
  max_saturated_segment_rate: 0.10
  max_repair_batch_rate: 0.20
  max_failed_batch_rate: 0.0
```

处理行为：

- `finish_reason=length` 不再视为成功；
- 空 facts 会记录并参与质量门，不再静默通过；
- 频繁达到 15 facts 会通过 saturation 指标暴露；
- campaign 汇总质量不达标时，训练入口拒绝继续；
- schema repair 最多允许两次，只修复 JSON/schema 结构，并要求保留事实和来源；
- repair 次数进入完整统计。

## 7. 多来源 provenance

事实协议从单个 `source_id` 扩展为：

```json
{
  "segment_id": "...",
  "source_ids": [1, 3],
  "fact": "..."
}
```

当前实现：

- 一个事实可以由多个消息共同支持；
- `source_ids` 自动去重并排序；
- parser 校验所有来源必须属于对应 segment；
- 截断时只能引用 `visible_source_ids`；
- 完全相同事实重复出现时，可合并来源集合；
- 兼容读取旧版单个 `source_id` 响应。

## 8. 计费机制

计费单位是一次实际 provider batch 请求：

```text
实际输入 usage + 实际输出 usage = batch 实际费用
```

实现规则：

- 强制使用 provider 返回的实际 usage；
- 输入价格和输出价格分别计算；
- 使用最大余数法将 batch token 与费用分摊到各 segment，保证分摊总和严格等于 provider usage；
- singleton 截断段只记录该次实际调用费用；
- 被截掉的文本不估算费用；
- `segment_costs` 写入 SQLite，供训练阶段重放不同路由动作的虚拟提取成本；
- 训练成本目标只统计 memory extraction cost；
- QA reader 和 judge 的 token、延迟及费用单独记录，不混入提取预算。

## 9. SQLite 并发、恢复与 resume

原共享 JSONL ledger 只有进程内锁，现已迁移到 `infobudget/rl_router/ledger.py` 中的 SQLite ledger：

- WAL 模式；
- 数据库事务；
- 唯一键约束；
- 跨进程并发安全；
- 原子 replace/update；
- 支持从旧 JSONL 只读导入。

主要数据库包括：

- `candidate_ledger.sqlite3`：`batches`、`segment_costs`、`failures`；
- 每个 run 的 `run_ledger.sqlite3`；
- 训练、validation、assembly 和 evaluation 的独立 SQLite ledger。

Resume 根据 SQLite batch 状态跳过已经 committed 的工作，因此 API 费用不足或进程中断后，不会重复请求已经提交完成的 batch。

## 10. Qdrant 存储

### 10.1 Qdrant server 配置

项目配置已经切换为：

```yaml
mode: server
url: http://127.0.0.1:6333
grpc_port: 6334
vector_size: 1024
distance: Cosine
```

部署文件为 `deploy/qdrant/docker-compose.yml`。

Qdrant server 为常用过滤字段创建 payload index，包括：

- `dataset_name`；
- `split`；
- `sample_id`；
- `session_id`；
- `segment_id`；
- `memory_tier`；
- `extraction_run_id`；
- `batch_id`；
- `assembly_id`；
- `policy_version`。

### 10.2 Embedding hash namespace

Collection 命名格式调整为：

```text
{dataset}_{split}_{segmentation_version}_{embedding_hash}_fact_v2
```

Manifest 保存 embedding 完整哈希，namespace 使用前 12 位。即使未来替换成相同维度的 embedding，新旧模型向量也不会进入同一 Collection。

付费请求前校验已有 Collection：

- 必须使用一个 unnamed vector；
- vector size 必须为 1024；
- distance 必须为 Cosine。

### 10.3 L/M/H/S 隔离与人工查看

- L、M、H 是独立物理 Collection；
- 三者使用相同 embedding 模型，处于同一语义空间；
- 相似向量不会相互覆盖，因为 Point ID 不同；
- S 是路由选择后的独立物理 Collection；
- S 使用 `sample_id + assembly_id` 过滤，隔离不同 sample 和 rollout。

Qdrant server 使用全局 Collection，sample 通过 payload 逻辑隔离。人工查看使用按 sample 导出的 JSON：

```text
human_readable/<sample_id>/L_memories.json
human_readable/<sample_id>/M_memories.json
human_readable/<sample_id>/H_memories.json
human_readable/<sample_id>/S_memories.json
```

### 10.4 Ghost points 自动修复

候选写入改为 batch 级 replace：

```text
删除 dataset/split/sample/run/batch 范围内旧 Points
→ Upsert 当前响应生成的 facts
→ 核对实际 Point 数量
→ SQLite 标记 committed
```

响应文件损坏、崩溃或重试产生不同 facts 时，旧 Points 不会继续残留。

## 11. 四方一致性核对

新增只读命令：

```bash
uv run python scripts/reconcile_extraction_run.py <run_id>
```

核对以下四方：

1. Run manifest 的计划批次和完成批次；
2. Run SQLite state 中的 committed 批次；
3. Candidate SQLite segment-cost ledger；
4. Qdrant 中该 run/batch 的实际 Points。

比较内容包括 tier、batch 数量与 ID、segment IDs、ledger 行数、fact 数量、Qdrant Point 数量和 manifest 汇总数量。

训练入口会在加载路由模型和触发付费 QA 前，自动核对 campaign 中每个 extraction run。

## 12. 路由训练与评估

训练入口当前：

- 强制要求完整 campaign；
- 使用防泄漏 split manifest；
- 只读取 campaign 固定的 extraction run；
- 训练前执行四方 reconciliation；
- 从 L/M/H 组装 S，不重新调用提取模型；
- 使用历史 token 记录重放路由的虚拟提取成本；
- 使用 All-Small 和 All-Large 对成本归一化；
- 训练 constrained actor-critic 路由器；
- checkpoint 同时保存 MLP 权重和训练集 scaler；
- QA 评估只检索 S Collection；
- 保存每题检索、回答、judge、token、延迟和费用记录。

## 13. Update 模块状态

Update 功能尚未实现，只完成 `docs/memory_update_design.md` 设计文档。

设计边界：

- 冻结的 L/M/H 不允许更新；
- S 是一次 assembly 的不可变快照；
- 未来部署记忆使用独立 D Collection；
- 支持 `ADD / UPDATE / DELETE / IGNORE`；
- 使用稳定 `memory_id` 和递增 revision；
- 修改事实文本时必须重新计算向量；
- 支持多来源 provenance 合并；
- 使用 SQLite 状态机协调 Qdrant 多 Point 更新和崩溃恢复；
- 当前 Qdrant payload 中没有正式 `update_queue` 字段；
- 未来建议由 SQLite 保存权威更新队列，Qdrant 只保存检索所需的状态镜像。

## 14. 后续未完成事项

- 尚未使用完整 LoCoMo/LongMemEval、真实 API 和正式本地模型跑完全量实验；
- 尚未在正式 Qdrant server 上测量全量吞吐和 payload index 性能；
- 当前质量阈值需要根据首次全量 campaign 统计结果调整；
- 训练完成后的真实在线路由提取入口尚未完整实现；
- D Collection 和离线 update 模块仍处于设计阶段；
- 正式实验前需要冻结 Qdrant 镜像版本、模型目录、embedding hash、价格快照和 campaign。

## 15. 关键文件索引

| 功能 | 文件 |
|---|---|
| 主配置 | `configs/rl_router.yaml` |
| 候选提取 | `infobudget/rl_router/candidates.py` |
| Campaign | `infobudget/rl_router/campaign.py` |
| SQLite ledger | `infobudget/rl_router/ledger.py` |
| Run state | `infobudget/rl_router/run_state.py` |
| Qdrant 存储 | `infobudget/rl_router/qdrant_store.py` |
| 四方核对 | `infobudget/rl_router/reconciliation.py` |
| LongMemEval 调度器 | `scripts/build_longmemeval_rl_candidates.ps1` |
| Campaign 管理 | `scripts/manage_extraction_campaign.py` |
| 训练入口 | `scripts/train_rl_router.py` |
| Update 设计 | `docs/memory_update_design.md` |
| Qdrant 部署 | `deploy/qdrant/docker-compose.yml` |

