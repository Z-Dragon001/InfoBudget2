# Datasets

本目录用于放置 InfoBudget 评估使用的数据集原始文件与预处理结果。

目录约定：

```text
datasets/
├── raw/
│   ├── locomo/
│   └── longmemeval/
└── processed/
    ├── locomo/
    │   └── {split}/
    │       ├── manifest.json
    │       ├── samples.jsonl
    │       ├── questions.jsonl
    │       └── sessions.jsonl
    └── longmemeval/
        └── {split}/
            ├── manifest.json
            ├── samples.jsonl
            ├── questions.jsonl
            └── sessions.jsonl
```

说明：

- `raw/`：放原始下载文件，支持 `.json` 和 `.jsonl`。
- `processed/`：由预处理脚本生成 sample / question / session 三层工件。
- 当前支持数据集：`LOCOMO`、`LongMemEval`。
- 统一预处理脚本：`scripts/preprocess_datasets.py`。
