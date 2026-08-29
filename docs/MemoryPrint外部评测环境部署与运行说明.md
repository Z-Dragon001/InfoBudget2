# MemoryPrint 外部评测环境部署与运行说明

> 更新时间：2026-08-12  
> 用途：记录 HaluMem、MemOps、HaluMem 数据集和 `uv` 环境的实际部署状态，并给出后续提取 7 维 MemoryPrint 的可复现入口。

> 后续数据预处理、冻结探针、候选模型调用、7 维评分、产物结构和完整执行顺序，统一以 [MemoryPrintBench 完整实验交接文档](./MemoryPrintBench完整实验交接文档.md) 为准。切换到外部目录后可直接阅读 `S:\Workfile\MemoryPrintBench\MEMORYPRINT_HANDOFF.md`。

## 1. 隔离原则与当前结论

HaluMem 和 MemOps 不放入 InfoBudget2 仓库。两个官方项目、数据、虚拟环境和实验输出统一放在与 InfoBudget2 同级的独立目录：

```text
S:\Workfile\MemoryPrintBench\
├── upstream\
│   ├── HaluMem\
│   │   └── .venv\
│   └── MemOps\
│       └── .venv\
├── datasets\
│   └── halumem\
└── outputs\
```

这种部署方式具有以下边界：

- InfoBudget2 只保存方法、版本、路径约定和后续适配代码，不保存第三方仓库和数据集；
- HaluMem 与 MemOps 使用两个独立的 Python 环境，避免依赖互相污染；
- 官方仓库中的运行结果留在外部评测目录，最终只把 7 维聚合指纹和必要的实验元数据导入 InfoBudget2；
- 不在 InfoBudget2 中重新发布或重打包 HaluMem 数据。

## 2. 已固定的官方版本

| 项目 | 官方地址 | 本地路径 | 当前提交 |
|---|---|---|---|
| HaluMem | `https://github.com/MemTensor/HaluMem.git` | `S:\Workfile\MemoryPrintBench\upstream\HaluMem` | `c29025f43b347f68fc36a06bee8ed29b4dc6c3fb` |
| MemOps | `https://github.com/MemTensor/MemOps.git` | `S:\Workfile\MemoryPrintBench\upstream\MemOps` | `312af65e2c7b6d1b70f062ffa8b4cde32aaf6f35` |

正式实验应继续使用以上提交。若后续升级仓库，必须重新记录提交、重新验证依赖，并把升级前后的指标差异作为版本变更处理。

## 3. 数据集部署与校验

### 3.1 HaluMem

数据来源：`IAAR-Shanghai/HaluMem` 的 Hugging Face 官方数据仓库。

本地文件如下：

| 文件 | 大小（字节） | 顶层 JSONL 记录数 | SHA-256 |
|---|---:|---:|---|
| `HaluMem-Medium.jsonl` | 33,511,525 | 20 | `486FBC130A5C8781A2AF27FFA508A1D7855245137AA449C193AC4D29C45634E7` |
| `HaluMem-Long.jsonl` | 106,535,674 | 20 | `DFDBED570B402B7B8C17E0D7808FC6F3AE7A53B6144F18FEB16BBDD3F55CB0C9` |

这里的 20 是顶层用户记录数，不是 session 数或记忆点数；每条记录内部包含多个 session、Gold memory 和 QA 项。

为了让官方脚本保持默认相对路径，同时不复制 140 MB 数据，已经在 HaluMem 的 `data` 目录建立同盘硬链接：

```text
HaluMem\data\HaluMem-long.jsonl   -> datasets\halumem\HaluMem-Long.jsonl
HaluMem\data\HaluMem-medium.jsonl -> datasets\halumem\HaluMem-Medium.jsonl
```

主指纹实验应使用 `HaluMem-Medium.jsonl`。官方 `eval_memzero.py` 当前把 `data_path` 写在脚本末尾，默认指向 Long；正式运行 Medium 前，应把该变量改为 `../data/HaluMem-medium.jsonl`，并把 `version` 设为固定实验版本。不要用文件名伪装的方式把 Medium 冒充为 Long。

### 3.2 MemOps

MemOps 当前没有独立的外部数据下载步骤。官方仓库已经携带到第 4 阶段的生成产物，因此无需重新生成背景、证据对话、干扰事实，也无需为 smoke test 下载 UltraChat。

已核验的 `generated_result`：

| 子目录 | 文件数 | 大小（字节） |
|---|---:|---:|
| `1-background` | 100 | 343,303 |
| `2-evidence_conversation` | 403 | 23,547,446 |
| `2-evidence_trace` | 493 | 10,835,869 |
| `3-distractor_facts` | 403 | 6,319,490 |
| `4-inject_evidence_with_distractors` | 403 | 179,413,783 |
| 合计 | 1,802 | 220,459,891 |

对于 7 维指纹，MemOps 主要读取第 2 阶段 evidence conversation/trace 和第 4 阶段注入结果，不需要重新执行昂贵的数据生成阶段。

## 4. `uv` 环境配置

两个环境都固定使用 Python 3.11；当前实际解释器版本为 Python 3.11.7，`uv` 版本为 0.10.12。

| 环境 | 虚拟环境路径 | 已安装包数 | 配置状态 |
|---|---|---:|---|
| HaluMem + Mem0 | `upstream\HaluMem\.venv` | 46 | 核心依赖和 `mem0ai==0.1.118` 已安装 |
| MemOps | `upstream\MemOps\.venv` | 19 | 官方 `requirements.txt` 已安装 |

两个环境都已执行 `uv pip check`，结果均为 `All installed packages are compatible`。

HaluMem 只安装了当前选择的 Mem0 云端适配器所需依赖，没有同时安装 Zep、Memobase、MemOS 和 Supermemory。若实验切换记忆系统，应在 HaluMem 自己的 `.venv` 中追加相应官方依赖，不能装入 InfoBudget2 的 `.venv`。

环境激活命令：

```powershell
# HaluMem
Set-Location S:\Workfile\MemoryPrintBench\upstream\HaluMem
.\.venv\Scripts\Activate.ps1

# MemOps
Set-Location S:\Workfile\MemoryPrintBench\upstream\MemOps
.\.venv\Scripts\Activate.ps1
```

环境一致性复查命令：

```powershell
uv pip check --python S:\Workfile\MemoryPrintBench\upstream\HaluMem\.venv\Scripts\python.exe
uv pip check --python S:\Workfile\MemoryPrintBench\upstream\MemOps\.venv\Scripts\python.exe
```

## 5. API 配置边界

### 5.1 HaluMem + Mem0

已经由官方 `eval/.env-example` 生成：

```text
S:\Workfile\MemoryPrintBench\upstream\HaluMem\eval\.env
```

文件目前只有官方占位符。正式运行前至少需要填写：

- `OPENAI_API_KEY`：用于 HaluMem 的回答生成或 LLM Judge；
- `OPENAI_BASE_URL`：OpenAI-compatible endpoint；
- `OPENAI_MODEL`：固定 Judge/Reader 模型；
- `MEM0_API_KEY`：官方 `eval_memzero.py` 使用 Mem0 云端 `MemoryClient`，初始化时会立即校验该密钥；
- `RETRY_TIMES`、`WAIT_TIME_LOWER`、`WAIT_TIME_UPPER`：失败重试配置。

`.env` 已被官方 `.gitignore` 排除。不得把真实密钥写入 InfoBudget2 文档、Git 提交、实验输出或终端日志。

当前完成的是 Mem0 云端适配器环境，不是本地自托管 Mem0 服务。如果后续必须使用本地 Mem0，应单独建立本地服务配置并适配 HaluMem wrapper，不能假定 `eval_memzero.py` 会自动连接本地实例。

### 5.2 MemOps

运行前在当前 PowerShell 进程中设置：

```powershell
$env:LLM_BASE_URL = "https://your-openai-compatible-endpoint"
$env:OPENAI_API_KEY = "<secret>"
```

如果网关不能提供官方示例中的模型名，需要同步覆盖命令中的 `--model`、`--adjacent-models`、`--long-context-models` 和 `--judge-model`。正式实验中模型名称、endpoint 类型、温度、最大 token 和并发数都必须进入运行清单。

## 6. 官方入口的已验证状态

已完成以下离线验证：

- HaluMem 两个 JSONL 文件均可逐行解析；
- HaluMem 的 `requests`、`jsonlines`、`dotenv`、`tqdm`、`tiktoken`、`pandas`、`pyarrow`、`openai`、`tenacity`、`numpy` 和 `MemoryClient` 均可导入；
- HaluMem `evaluation.py --help` 可正常启动；
- MemOps 的 `tqdm`、`openai` 和 `rank_bm25` 均可导入；
- MemOps `5-test_operation_metrics.py --help` 可正常启动；
- 两个虚拟环境均通过依赖一致性检查。

尚未执行 HaluMem + Mem0 的在线 smoke test，因为官方适配器在导入时就会访问 Mem0 服务并校验真实 `MEM0_API_KEY`。这是凭据尚未配置，不是 Python 或 `uv` 环境故障。

## 7. 获得密钥后的最小运行顺序

### 7.1 HaluMem + Mem0

```powershell
Set-Location S:\Workfile\MemoryPrintBench\upstream\HaluMem\eval
..\.venv\Scripts\python.exe .\eval_memzero.py
..\.venv\Scripts\python.exe .\evaluation.py --frame memzero --version <与脚本中一致的版本>
```

第一条命令调用 Mem0 写入、检索并生成中间结果；第二条命令调用 HaluMem 官方评分器。正式全量运行前，应先构造一个固定的小样本 smoke 子集，确认输出目录、API 模型、并发和费用，再运行冻结探针集。

### 7.2 MemOps

官方仓库已经包含前四阶段数据，可以直接从小规模测试开始：

```powershell
Set-Location S:\Workfile\MemoryPrintBench\upstream\MemOps

.\.venv\Scripts\python.exe .\5-test_operation_metrics.py `
  --output-dir generated_result\5-smoke `
  --model <candidate-model> `
  --adjacent-models <candidate-model> `
  --long-context-models <candidate-model> `
  --no-context-models "" `
  --rag-methods rag_vanilla `
  --rag-retrieval-units turn `
  --max-questions 10

.\.venv\Scripts\python.exe .\5.5-evaluate_operation_metrics.py `
  --input-file generated_result\5-smoke\operation_metrics_all_methods.jsonl `
  --output-dir generated_result\5.5-smoke `
  --judge-model <fixed-judge-model> `
  --eval-workers 4
```

上述官方流程用于生成和评分完整的 MemOps 结果。7 维 MemoryPrint 不直接采用最终 QA Accuracy；需要另行把候选模型输出的 `Fact + source_ids` 与 Gold operation trace 对齐，计算 Current State、Target Binding、Stale Rejection 和 Evidence F1。

## 8. 与 7 维 MemoryPrint 的数据流

```text
外部评测目录
├── HaluMem-Medium + Mem0/候选提取器
│   └── Recall、Target Precision、Memory Accuracy
└── MemOps Gold operation trace + 候选提取器
    └── Current State、Target Binding、Stale Rejection、Evidence F1

                    ↓ 只导入聚合结果和元数据

InfoBudget2
└── 7 维 MemoryPrint + dimension mask + 运行清单
```

导入 InfoBudget2 的最小记录应包含：候选模型 ID、两个官方仓库提交、数据文件 SHA-256、探针样本 ID、Prompt/Schema 版本、Judge 模型、解码参数、每维分数、每维有效样本数、缺失维 mask 和运行时间。原始第三方数据、API 密钥以及大体积中间结果仍保留在 `MemoryPrintBench`。

## 9. 当前状态清单

- [x] 独立外部评测目录已创建；
- [x] HaluMem 和 MemOps 官方仓库已下载并固定提交；
- [x] HaluMem 官方 Medium/Long 数据已下载并校验；
- [x] MemOps 随仓库发布的生成数据已核验；
- [x] 两个独立 `uv` 虚拟环境已创建；
- [x] HaluMem + Mem0 和 MemOps 依赖已安装并通过一致性检查；
- [x] HaluMem 官方数据相对路径已通过硬链接接通；
- [x] HaluMem `.env` 已从官方模板创建；
- [ ] 填写真实 OpenAI-compatible endpoint、Judge 模型和 API key；
- [ ] 填写真实 `MEM0_API_KEY`；
- [ ] 使用冻结的小样本运行两个在线 smoke test；
- [ ] 实现 MemOps 四项 Router-Adapted 指标的对齐与聚合代码；
- [ ] 生成并导入首个候选模型的 7 维 MemoryPrint。
