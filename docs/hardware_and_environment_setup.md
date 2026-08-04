# InfoBudget 硬件与运行环境配置

## 1. 使用方式

项目采用以下开发与实验流程：

```text
Windows 本机
├── 代码开发与单元测试
├── 数据集预处理
└── BERT 主题分段

Linux 服务器
└── 强化学习路由训练

Windows 本机或 Linux
└── QA 检索与最终评估
```

项目要求：

- 使用 `uv` 管理依赖；
- 使用 Python 3.12；
- 同一套代码同时支持 Windows 和 Linux；
- small、medium、large、QA Reader 和 Judge 均通过 API 调用；
- 本地不保存上述 LLM 权重；
- 数据集和本地模型由用户手动放入指定目录；
- 训练和评估过程中不自动下载数据集或模型。

## 2. uv 与 Python

本机已经安装 uv。Windows 和 Linux 均在项目根目录执行：

```text
uv python install 3.12
uv sync --frozen --python 3.12
```

开发环境需要测试依赖时执行：

```text
uv sync --frozen --group dev --python 3.12
```

运行命令统一使用：

```text
uv run python --version
uv run pytest
```

仓库中的 `.python-version` 固定为 `3.12`，`uv.lock` 必须提交 Git；`.venv` 不提交。

Windows 和 Linux 不要复制 `.venv`，应分别使用 `uv sync` 创建各自环境。

## 3. 硬件建议

### 3.1 Windows 本机

用于代码开发、预处理、BERT 分段和可选评估：

| 硬件 | 建议配置 |
|---|---|
| CPU | 8 核或以上 |
| 内存 | 32 GB 或以上 |
| GPU | 可选，建议 NVIDIA GPU |
| 显存 | 8–12 GB 或以上 |
| 磁盘 | 100 GB 以上 SSD 可用空间 |

没有 GPU 时也可以开发、预处理和运行小规模测试，但 BERT 分段和 BGE-M3 向量化速度会较慢。

### 3.2 Linux 训练服务器

用于 Embedding + MLP 路由器训练和完整实验：

| 硬件 | 建议配置 |
|---|---|
| CPU | 16 核或以上 |
| 内存 | 64 GB 或以上 |
| GPU | NVIDIA GPU |
| 显存 | 16–24 GB |
| 磁盘 | 300 GB 以上 NVMe SSD |

当前路由器是冻结 BGE-M3 加 MLP，不需要在 Linux 上部署 small、medium、large、QA Reader 或 Judge 的本地权重。

### 3.3 评估环境

评估可以在 Linux 或 Windows 本机运行。评估环境必须具备：

- 相同的代码版本；
- 相同的 `uv.lock`；
- Python 3.12；
- 相同的配置和提示词；
- 相同的 BGE-M3；
- 训练得到的路由器 checkpoint；
- 对应的 Qdrant L/M/H/S 数据。

## 4. 数据集目录

数据集已经下载，由用户手动放入：

```text
datasets/
├── raw/
│   ├── locomo/
│   │   └── locomo10.json
│   └── longmemeval/
│       └── longmemeval_s_cleaned.json
├── processed/
│   ├── locomo/
│   └── longmemeval/
└── segmented/
    ├── locomo/
    └── longmemeval/
```

- `raw`：用户下载的原始数据；
- `processed`：本机预处理结果；
- `segmented`：本机主题分段结果。

训练前，将 Linux 所需的 `processed`、`segmented` 和 manifest 文件复制到服务器，并保持相同的相对目录。

## 5. 模型目录

### 5.1 已下载的 BERT 模型

```text
seg_models/
├── bert-base-uncased/
└── bert_mlp/
    └── best.pt
```

该模型用于本机主题分段，不需要重新下载。

### 5.2 需要准备的其他本地资源

```text
models/
├── embeddings/
│   └── bge-m3/
├── tokenizers/
│   ├── small/
│   ├── medium/
│   ├── large/
│   ├── qa_reader/
│   └── judge_llm/
└── router/
    └── checkpoints/
```

- BGE-M3 用于 fact、QA query 和主题段向量化；
- tokenizer 用于 buffer 长度预估和 token 分摊；
- small、medium、large、QA Reader 和 Judge 只调用 API，不下载本地权重；
- API 返回的 usage 用于记录真实输入、输出 token 和费用。

Linux 训练和评估时，需要把 BGE-M3、必要 tokenizer 和路由器 checkpoint 放在相同的相对路径。

## 6. Qdrant 与输出目录

LoCoMo 和 LongMemEval 的正式实验共用一个本机 Qdrant server，通过不同 Collection
namespace 隔离。REST 端口为 `127.0.0.1:6333`，gRPC 端口为
`127.0.0.1:6334`。Docker Compose 文件、启动命令和备份要求见
`docs/qdrant_server_deployment.md`，持久化数据默认位于：

```text
deploy/qdrant/storage/
```

实验输出放在：

```text
outputs/rl_router/{dataset}/{split}/{segmentation_method}/
```

Qdrant 数据和实验输出不提交 Git。按 sample 导出的 L/M/H/S JSON 仅供人工查看，
训练和评估仍以 Qdrant server 为正式数据源。

如果评估从 Linux 切换到 Windows，应在 Qdrant 停止时复制完整
`deploy/qdrant/storage` 和实验输出，并保持 Collection namespace、embedding 模型和配置一致。

## 7. Windows/Linux 兼容要求

代码统一遵守：

1. 使用 `pathlib.Path` 处理路径；
2. 配置文件使用项目相对路径；
3. 不硬编码 Windows 盘符或 Linux 用户目录；
4. 文本文件统一使用 UTF-8；
5. 多进程入口使用 `if __name__ == "__main__":`；
6. Python 中不拼接 PowerShell 或 Bash 专属命令；
7. GPU 设备通过配置选择 `auto/cpu/cuda`；
8. 随机种子、模型版本和配置在 Windows/Linux 保持一致。

路径示例：

```python
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "models" / "embeddings" / "bge-m3"
```

## 8. Git 忽略范围

项目根目录的 `.gitignore` 已忽略：

```text
.venv/
.uv-cache/
datasets/raw/
datasets/processed/
datasets/segmented/
models/
seg_models/ 中的模型权重
outputs/
Qdrant 数据
实验结果和临时文件
```

应提交：

```text
pyproject.toml
uv.lock
.python-version
configs/
docs/
infobudget/
scripts/
tests/
datasets/README.md
```

## 9. 简单检查

Windows 本机：

```text
uv run python --version
uv run pytest
检查两个原始数据集
检查 seg_models/bert-base-uncased
检查 seg_models/bert_mlp/best.pt
运行预处理和主题分段
```

Linux 服务器：

```text
uv sync --frozen --group dev --python 3.12
检查 processed 和 segmented 数据
检查 models/embeddings/bge-m3
运行 Embedding + MLP 路由训练
保存 checkpoint 和 Qdrant 输出
```

评估：

```text
加载相同配置和 BGE-M3
加载训练后的路由器 checkpoint
读取 Qdrant S Collection
运行 QA Reader 和 Judge API
保存准确率、token 和费用指标
```
