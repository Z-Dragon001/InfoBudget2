# 模型家族 Namespace 与 Segment 审计开发交接

## 1. 文档范围

本文记录本次会话完成的两项代码改造：

1. 为 Qwen、Llama 等不同记忆提取模型家族使用独立的 Qdrant namespace；
2. 扩充长期记忆提取阶段的 segment 级审计字段，确保生成零条 Fact 的 segment 仍然可以统计、核对和用于论文实验。

本文不包含模型最终选型、Llama 模型配置文件创建、正式数据提取或论文结果生成。

## 2. 完成状态

当前实现状态：已完成并通过回归测试。

- Qdrant Fact payload schema：`qdrant_fact_v3`；
- segment 审计 schema：`segment_extraction_audit_v1`；
- 当前默认模型家族：`qwen`；
- Qwen 与 Llama 使用不同的 Qdrant collection namespace；
- 每个已处理 segment 在每个实际提取 tier 下都有一条 `segment_costs` 记录；
- 零 Fact segment 写入 `fact_count=0` 和 `status=no_fact`；
- reconciliation 会检查新增审计字段及其一致性；
- Python 编译检查通过；
- 60 项常规测试和 1 项多进程账本测试均通过。

## 3. 核心设计

### 3.1 模型家族独立 Namespace

`configs/rl_router.yaml` 新增：

```yaml
model_family: "qwen"
```

Qdrant namespace 模板调整为：

```yaml
collection_namespace: "{model_family}_{dataset}_{split}_{segmentation_version}_{embedding_hash}_fact_v3"
```

示例：

```text
qwen_locomo_full_nsp_text_tiling_v1_a1b2c3d4e5f6_fact_v3_L
qwen_locomo_full_nsp_text_tiling_v1_a1b2c3d4e5f6_fact_v3_M
qwen_locomo_full_nsp_text_tiling_v1_a1b2c3d4e5f6_fact_v3_H

llama_locomo_full_nsp_text_tiling_v1_a1b2c3d4e5f6_fact_v3_L
llama_locomo_full_nsp_text_tiling_v1_a1b2c3d4e5f6_fact_v3_M
llama_locomo_full_nsp_text_tiling_v1_a1b2c3d4e5f6_fact_v3_H
```

因此，即使数据集、split、分段版本和 embedding 完全相同，Qwen 与 Llama 也不会进入同一组 L/M/H/S collections。

### 3.2 配置校验

`load_rl_bundle()` 现在会检查：

- `model_family` 非空；
- 只能包含小写字母、数字、下划线和连字符；
- namespace 模板必须包含以下占位符：
  - `{model_family}`
  - `{dataset}`
  - `{split}`
  - `{segmentation_version}`
  - `{embedding_hash}`
- 当 `model_family=qwen` 时，small/medium/large 三个模型名称必须包含 `qwen`；
- 当 `model_family=llama` 时，small/medium/large 三个模型名称必须包含 `llama`。

该校验用于防止切换了 `models.yaml` 却忘记切换 `model_family`，从而把 Llama 数据误写入 Qwen namespace。

### 3.3 Campaign 和 Manifest 冻结

以下运行元数据现在都会保存 `model_family`：

- extraction campaign manifest；
- candidate extraction run manifest；
- router training/evaluation manifest；
- Qdrant Fact payload；
- segment 级审计记录。

Campaign scope hash 同时包含模型家族、模型配置、Prompt、Embedding 和提取参数。模型家族改变后，旧 campaign 不能继续复用。

## 4. Segment 级审计数据

### 4.1 权威存储位置

Segment 级审计数据保存在：

```text
outputs/rl_router/<dataset>/<split>/<method>/samples/<sample_id>/extraction/candidate_ledger.sqlite3
```

表名：

```text
segment_costs
```

真实路由测试使用对应运行目录中的：

```text
deployment_ledger.sqlite3 / segment_costs
```

Qdrant 只保存实际 Fact Point。零 Fact segment 没有可写入的向量 Point，因此零 Fact 统计必须以 SQLite `segment_costs` 为准。

### 4.2 标识与运行范围字段

每条 segment 审计记录包含：

```text
audit_schema_version
dataset_name
split
sample_id
session_id
segment_id
extraction_run_id
batch_id
tier
model_family
campaign_id
campaign_scope_hash
extraction_scope_hash
qdrant_namespace
```

### 4.3 分段与来源字段

```text
segmentation_method
segmentation_version
source_content_hash
segment_order
segment_start_turn
segment_end_turn
segment_turn_ids
segment_start_timestamp
segment_end_timestamp
segment_turn_count
segment_token_count
segment_char_count
segment_line_count
```

截断相关字段继续保留：

```text
extraction_truncated
extraction_original_char_count
extraction_retained_char_count
extraction_visible_source_ids
extraction_dropped_source_ids
```

### 4.4 模型、Prompt 与 Embedding 字段

```text
extractor_configured_model
extractor_request_model
extractor_backend
extractor_api_base_url
prompt_version
prompt_sha256
embedding_model
embedding_dimension
embedding_model_hash
embedding_revision
embedding_normalized
qdrant_distance
```

API Key、Authorization Header 等敏感信息不会写入账本或 Qdrant。

### 4.5 Token、费用与价格字段

```text
allocated_input_tokens
allocated_output_tokens
allocated_total_tokens
allocated_input_cost
allocated_output_cost
allocated_total_cost
serialized_input_tokens
attributed_output_tokens
allocation_method
input_price_per_1m
output_price_per_1m
price_effective_date
currency
```

其中 segment 级 Token 和费用属于 batch usage 的分摊结果。论文中的真实总费用仍应以 attempt/batch ledger 为权威来源。

### 4.6 Fact 与调用审计字段

```text
fact_count
fact_limit
fact_limit_reached
status
batch_logical_call_count
batch_repair_call_count
batch_transport_attempt_count
batch_retry_count
batch_latency_ms
batch_unknown_cost_attempts
batch_provider_request_ids
batch_usage_source
created_at
```

状态约束：

```text
fact_count == 0  -> status == "no_fact"
fact_count > 0   -> status == "ok"
```

Batch 调用字段会在同一个 batch 的多个 segment 行中重复，仅用于回溯，不能直接跨 segment 求和。API 调用数、总延迟和总成本应从 batch/attempt ledger 聚合。

## 5. Qdrant Fact Payload 变化

Qdrant Fact schema 从 `qdrant_fact_v2` 升级为：

```text
qdrant_fact_v3
```

除了原有 Fact、来源、tier、Token、费用和 embedding 字段，Fact payload 现在还包含：

- `model_family`；
- campaign 和 extraction scope；
- Qdrant namespace；
- segmentation method/version；
- segment 长度和turn范围；
- configured/request model；
- provider backend/base URL；
- Prompt SHA-256；
- Embedding hash/revision/normalize；
- 价格快照字段。

Qdrant server 新增 `model_family` 和 `campaign_id` payload index。

## 6. Reconciliation 行为

对于 `qdrant_fact_v3` 正式运行，`reconcile_extraction_run()` 会额外检查：

1. `segment_costs` 是否包含全部必需审计字段；
2. `audit_schema_version` 是否为 `segment_extraction_audit_v1`；
3. segment 行的 `model_family` 是否与运行 manifest 一致；
4. segment 行的 `qdrant_namespace` 是否与运行 manifest 一致；
5. `fact_count` 与 `status` 是否一致；
6. SQLite计划segment、segment账本和Qdrant Point数量是否一致。

零 Fact segment 的预期行为是：

- SQLite存在对应segment行；
- `fact_count=0`；
- `status=no_fact`；
- Qdrant对应Point数量为0；
- reconciliation仍然可以通过。

## 7. 涉及的主要文件

| 文件 | 修改内容 |
|---|---|
| `configs/rl_router.yaml` | 增加`model_family`，namespace升级为family-aware fact_v3 |
| `infobudget/rl_router/config.py` | 模型家族与namespace模板校验 |
| `infobudget/rl_router/manifest.py` | manifest保存模型家族，namespace解析支持family |
| `infobudget/rl_router/campaign.py` | campaign scope冻结模型家族 |
| `infobudget/rl_router/schemas.py` | 定义Fact v3与segment审计契约 |
| `infobudget/rl_router/candidates.py` | 写入完整segment审计、零Fact行及Fact审计payload |
| `infobudget/rl_router/qdrant_store.py` | 增加模型家族和campaign索引 |
| `infobudget/rl_router/reconciliation.py` | 校验segment审计完整性和一致性 |
| `scripts/build_rl_candidates.py` | 候选提取传入family/campaign/embedding审计上下文 |
| `scripts/evaluate_routed_deployment.py` | 真实路由提取传入完整审计上下文 |
| `scripts/train_rl_router.py` | 使用模型家族namespace |
| `scripts/evaluate_rl_assembly.py` | 使用模型家族namespace |
| `scripts/assemble_rl_baseline.py` | 使用模型家族namespace |
| `tests/test_rl_router.py` | 增加零Fact和完整审计测试 |
| `tests/test_extraction_campaign.py` | 验证Qwen/Llama namespace隔离 |
| `tests/test_model_config.py` | 验证模型家族配置和namespace模板 |

## 8. Qwen 与 Llama 的使用方式

建议为两个模型家族准备独立配置目录，例如：

```text
configs/qwen/
configs/llama/
```

Qwen配置：

```yaml
model_family: "qwen"
```

Llama配置：

```yaml
model_family: "llama"
```

调用时通过现有参数选择配置目录：

```powershell
uv run python scripts/build_rl_candidates.py <segments.jsonl> `
  --config-dir configs/qwen `
  --campaign-id <qwen-campaign-id> `
  --extraction-run-id <qwen-run-id>
```

```powershell
uv run python scripts/build_rl_candidates.py <segments.jsonl> `
  --config-dir configs/llama `
  --campaign-id <llama-campaign-id> `
  --extraction-run-id <llama-run-id>
```

Campaign ID 和 run prefix 也应显式包含模型家族，便于人工审计，例如：

```text
qwen_locomo_full_nsp_v1
llama_locomo_full_nsp_v1
```

## 9. 论文统计建议

| 指标 | 推荐权威来源 |
|---|---|
| 每个segment的Fact数量 | `segment_costs.fact_count` |
| 零Fact段数量/比例 | `segment_costs.status=no_fact` |
| 每个tier的segment数量 | 按`segment_costs.tier`计数 |
| 每个sample平均提取Token | batch/attempt ledger按sample汇总 |
| 每个sample平均提取费用 | batch/attempt ledger按sample汇总 |
| 每个tier平均Fact数 | `segment_costs`按tier求`fact_count`均值 |
| API调用数与repair次数 | attempt ledger |
| 提取延迟 | batch/attempt ledger |
| 价格敏感性分析 | provider Token重新应用新价格快照 |
| Fact来源与案例分析 | Qdrant Fact payload及`source_provenance` |

不要从Qdrant中直接计算零Fact率，因为零Fact segment没有Point。也不要对每条Fact中重复的`fact_count_in_segment`求和。

## 10. 测试与验证

完成的验证包括：

```powershell
.venv\Scripts\python.exe -m compileall -q infobudget scripts
```

常规测试：

```powershell
.venv\Scripts\python.exe -m pytest -q `
  -k "not test_sqlite_ledger_is_cross_process_idempotent"
```

结果：

```text
60 passed, 1 deselected
```

多进程SQLite账本测试在限制数值库线程后独立运行：

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.venv\Scripts\python.exe -m pytest `
  tests/test_rl_router.py::test_sqlite_ledger_is_cross_process_idempotent -q
```

结果：

```text
1 passed
```

## 11. 兼容性与迁移说明

这是一次正式数据契约升级：

- 旧schema：`qdrant_fact_v2`；
- 新schema：`qdrant_fact_v3`；
- 新namespace增加`model_family`；
- 新campaign scope增加`model_family`；
- extraction scope hash增加审计上下文。

因此：

1. 不应在旧fact_v2 campaign上继续正式提取；
2. 不应让fact_v2和fact_v3数据共用collection；
3. 应为Qwen和Llama分别创建新的campaign；
4. 已存在的旧run不能直接按新scope恢复；
5. 本次没有实现旧Qdrant数据自动迁移，因为正式实验应从冻结配置创建全新的v3 campaign。

## 12. 正式付费提取前检查清单

- [ ] 创建独立的Qwen与Llama配置目录；
- [ ] `model_family`与small/medium/large模型名称一致；
- [ ] 六个提取模型均有价格快照；
- [ ] Qwen与Llama使用不同campaign ID和run prefix；
- [ ] 新campaign manifest中存在正确的`model_family`；
- [ ] namespace以`qwen_`或`llama_`开头并以`fact_v3`结尾；
- [ ] 先使用一个非正式sample运行三档提取；
- [ ] smoke test中人为包含至少一个零Fact segment；
- [ ] 检查`segment_costs`是否对所有segment都有记录；
- [ ] 检查零Fact行是否为`fact_count=0/status=no_fact`；
- [ ] 运行`reconcile_extraction_run.py`并确认通过；
- [ ] 备份Qdrant storage、outputs、配置和campaign manifest；
- [ ] 再启动正式全量付费提取。

## 13. 尚未完成的工作

本次改造没有包含以下内容：

- Llama 3B/8B/70B 的实际 `models.yaml` 与 `prices.yaml`；
- Qwen 7B/32B/72B 最终配置替换；
- Qwen与Llama配置目录自动生成；
- 实体数量、时间表达式数量等`segment_features`离线统计；
- Qdrant snapshot自动化脚本；
- 从raw selected response无API重建Qdrant的工具；
- 旧fact_v2数据迁移工具；
- 正式LoCoMo/LongMemEval全量实验。

## 14. 工作树说明

实施本任务前工作树已经存在其他未提交修改。本次开发在现有修改之上进行，没有回退、覆盖或提交用户已有改动。本交接文档也未执行Git stage或commit。
