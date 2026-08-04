# Datasets

本目录用于保存 InfoBudget 评估使用的原始数据、冻结的预处理工件和实验划分。

目录约定：

```text
datasets/
├── raw/
│   ├── locomo/
│   └── longmemeval/
├── processed/
│   ├── locomo/
│   │   └── {source_split}/
│   │       ├── manifest.json
│   │       ├── samples.jsonl
│   │       ├── questions.jsonl
│   │       ├── sessions.jsonl
│   │       └── turns.jsonl
│   └── longmemeval/
│       └── {source_split}/
│           ├── manifest.json
│           ├── samples.jsonl
│           ├── questions.jsonl
│           ├── sessions.jsonl
│           └── turns.jsonl
├── segmented/
│   └── {dataset}/{source_split}/{method}/
│       ├── manifest.json
│       └── samples/{sample_id}/
│           ├── segments.jsonl
│           └── segmentation_trace.json
└── splits/
    ├── locomo/
    │   └── cv5_seed42.json
    └── longmemeval/
        ├── fixed_80_10_10_seed42_nsp_text_tiling.json
        └── cv5_360_40_100_seed42_nsp_text_tiling.json
```

说明：

- `raw/`：原始下载文件，只读保存，支持 `.json` 和 `.jsonl`。
- `processed/`：预处理脚本生成的 sample、question、session 和 turn 工件。
- `segmented/`：按 sample 隔离的冻结主题分割结果。
- `splits/`：只保存实验划分中的 sample ID，不复制对话、问题或分割结果。
- 当前支持的数据集：LoCoMo、LongMemEval。
- 统一预处理入口：`scripts/preprocess_datasets.py`。

## LoCoMo 的 `full` 与实验划分

`processed/locomo/full` 和 `segmented/locomo/full` 始终保存全部 10 个 conversation。
这里的 `full` 表示“冻结的基础数据版本”，不是训练集。三档候选记忆也对这 10 个
conversation 各生成一次，并继续使用 segment 中的 `split=full`。

`splits/locomo/cv5_seed42.json` 是独立的实验控制文件：每折列出 8 个 `train` 和
2 个 `test` conversation。路由器训练、特征归一化拟合和策略更新只能读取当前折的
`train` ID；`test` 只能在 checkpoint 固定后用于评估。五折方案没有 validation，
因此超参数必须预先固定；如需调参，应另设开发划分或使用嵌套交叉验证，不能查看
当前折 test 结果后再改参数。

不要生成 `processed/locomo/train`、`processed/locomo/test` 或逐折复制的 JSONL。
复制会造成数据冗余和版本漂移，且容易让同一 conversation 的 `samples`、
`questions`、`sessions` 和 `turns` 不一致。每次训练产生的实验 `manifest.json`
会记录 split manifest 路径、SHA-256、fold 以及 train/validation/test ID，作为
可复现与泄漏审计依据。

LoCoMo 的独立评估入口同样要求 `--split-manifest`、`--fold` 和 `--partition`，
并拒绝不属于指定 partition 的 sample。正式 test 账本写到
`outputs/rl_router/evaluation/locomo/{protocol}/fold_{k}/test/...`，不会与训练期间的
QA reward 账本混合。

## LongMemEval 的固定划分与五折划分

LongMemEval 的 `processed/longmemeval/full`、两套 `segmented/longmemeval/full/{method}`
以及 L/M/H 候选仍保存全部 500 个 question/history sample，不生成物理 train/dev/test
副本。划分生成入口为：

```powershell
uv run python scripts/build_longmemeval_splits.py
```

生成并提交两份版本化 manifest：

- `fixed_80_10_10_seed42_nsp_text_tiling.json`：400 train、50 validation、50 test；
- `cv5_360_40_100_seed42_nsp_text_tiling.json`：每折 360 train、40 validation、100 test。

固定划分的训练问题类型数量严格为 63、106、45、24、56、106；不可回答样本为
24/3/3。五折中每个 sample 恰好进入一次 test，每折 test 含 6 个不可回答样本。

划分器同时执行以下约束：

1. 按 `question_type + is_unanswerable + NSP segment-count quartile` 分层；
2. 把全局 evidence-bearing session 在所有 sample 中的出现位置放入同一组；
3. 把 answerable/`_abs` counterpart 放入同一组；
4. evidence group 不得跨 partition；
5. 纯背景 distractor session 允许跨 partition，其交叉数量写入每折 `audit`；
6. scaler 只使用 train 拟合，validation 只做确定性路由、早停和 checkpoint 选择，
   test 只在最终 checkpoint 冻结后评估。

NSP 分段只作为 segment-count 分层的固定参考，manifest 保存对应 segmentation
manifest 的 SHA-256。NSP 与 BERT-MLP 实验应复用同一划分，避免分段方法对比时同时
改变样本集合。若要求所有背景 distractor 也完全隔离，需要另建去重数据集版本，
不能再把结果标为原始 LongMemEval 基准。
