# LoCoMo 分段 Alpha 与训练 Epoch 实验指南

## 1. 当前实现结论

本轮修改前，分段 manifest 虽然记录了 `adaptive_alpha`，但物理路径只有
`datasets/segmented/<dataset>/<split>/<method>/`。使用不同 alpha 重跑会覆盖同一
`segments.jsonl`。现在 alpha 已进入正式实验身份：

```text
alpha=0.5 -> nsp_text_tiling_alpha_0p5
alpha=0.7 -> nsp_text_tiling_alpha_0p7
```

该身份同时用于：

- 分段目录和 segment ID；
- `segmentation_method` 与 `segmentation_version`；
- extraction campaign scope 与 run ID；
- 训练、部署评估输出路径；
- Qdrant namespace（通过包含 alpha 的 `segmentation_version`）。

不同 epoch 原本由随机 run ID 防止直接覆盖，manifest 也保存了 `epochs`，但目录不可读。
现在默认训练和完整实验路径显式增加 `epochs_<N>`，默认 run ID 也包含 `e<N>`。例如：

```text
outputs/rl_router/training/locomo/cv5/fold_0/
  nsp_text_tiling_alpha_0p5/epochs_3/train_e3_<timestamp>_<id>/
```

自定义 `--output-dir` 仍然受 manifest scope 校验；用不同 epoch 恢复同一目录会被拒绝。

LoCoMo 与 LongMemEval 的预处理、分段、候选记忆提取和完整实验命令均可独立执行，
不会要求同时运行两个数据集。

## 2. LoCoMo 五阶段命令

以下示例使用 PowerShell、NSP TextTiling、`alpha=0.5`、3 epoch、fold 0。
先启动 Qdrant，并按 `configs/models.yaml` 配置相应 API key。

```powershell
$Alpha = 0.5
$Seg = "nsp_text_tiling_alpha_0p5"
$Epochs = 3
$Fold = 0
$Campaign = "qwen_locomo_full_$Seg"
$TrainRun = "locomo_${Seg}_e${Epochs}_fold${Fold}"
$TrainDir = "outputs/rl_router/training/locomo/cv5/fold_$Fold/$Seg/epochs_$Epochs/$TrainRun"
$TrainManifest = "$TrainDir/manifest.json"
$DeployRun = "locomo_${Seg}_e${Epochs}_fold${Fold}_test"
$DeployDir = "outputs/rl_router/deployment_evaluation/locomo/cv5/fold_$Fold/$Seg/epochs_$Epochs/$DeployRun"
```

首次运行前，仅预处理 LoCoMo：

```powershell
uv run python scripts/preprocess_datasets.py --datasets locomo
```

### 阶段 1：信息分段

```powershell
uv run python scripts/segment_datasets.py locomo full `
  --method nsp_text_tiling `
  --alpha $Alpha
```

输出位于：

```text
datasets/segmented/locomo/full/nsp_text_tiling_alpha_0p5/
```

`manifest.json` 保存算法、alpha、完整参数、输入 manifest hash 和分段数量；每个 sample
保存独立的 `segments.jsonl` 与 `segmentation_trace.json`。

### 阶段 2：训练数据记忆提取

```powershell
& .\scripts\build_locomo_rl_candidates.ps1 `
  -Method nsp_text_tiling `
  -Alpha $Alpha `
  -RunPrefix qwen_locomo_full `
  -CampaignId $Campaign
```

该脚本只扫描 LoCoMo 的指定 alpha 目录，初始化不可变 campaign，并为所有 sample
分别提取 small/medium/large 三档候选记忆。正式 Fact 写入按数据集、模型家族、
segmentation version 和 embedding hash 隔离的 Qdrant collections；调用、token、费用和
零 Fact segment 审计写入 SQLite ledger。

### 阶段 3：训练路由

```powershell
uv run python scripts/train_rl_router.py locomo full `
  --method $Seg `
  --campaign-id $Campaign `
  --split-manifest datasets/splits/locomo/cv5_seed42.json `
  --fold $Fold `
  --epochs $Epochs `
  --steps-per-sample 10 `
  --device auto `
  --run-id $TrainRun
```

训练只读取 split manifest 的训练 conversation，复用冻结的 L/M/H 候选及其真实 token
账本。checkpoint、scaler、训练状态、QA ledger 和 manifest 均写入 `epochs_3` 目录。
LoCoMo 当前 `cv5` manifest 没有 validation partition，因此选用预先声明 epoch 的
`final.pt`，不会查看 test fold 选择 checkpoint。

### 阶段 4：评估数据记忆提取

```powershell
$Checkpoint = (Get-Content $TrainManifest -Raw | ConvertFrom-Json).selected_checkpoint

uv run python scripts/evaluate_routed_deployment.py locomo full `
  --method $Seg `
  --split-manifest datasets/splits/locomo/cv5_seed42.json `
  --fold $Fold `
  --checkpoint $Checkpoint `
  --training-manifest $TrainManifest `
  --campaign-id $Campaign `
  --deployment-run-id $DeployRun `
  --output-dir $DeployDir `
  --stage extract
```

该阶段冻结 router 决策，只对 test partition 执行真实分档提取；不调用 QA Reader 或
Judge。完成后 manifest 状态为 `extraction_complete`，保存真实调用、token、费用、
路由决策和 reconciliation 结果。

### 阶段 5：QA 问答与评估

```powershell
uv run python scripts/evaluate_routed_deployment.py locomo full `
  --method $Seg `
  --split-manifest datasets/splits/locomo/cv5_seed42.json `
  --fold $Fold `
  --checkpoint $Checkpoint `
  --training-manifest $TrainManifest `
  --campaign-id $Campaign `
  --resume $DeployRun `
  --output-dir $DeployDir `
  --stage qa
```

该阶段复用阶段 4 已写入的测试记忆，组装 S collection，然后执行 LoCoMo Reader、Judge
和最终评分。结果写入 `samples/<sample_id>/result.json`、`qa/ledger.sqlite3` 和
`aggregate.json`。已完成 sample 可安全恢复，不重复付费。

## 3. 一条命令执行阶段 3–5

完整五折实验可由编排器执行；其内部仍按“训练 → 测试记忆提取 → QA/评估”三个独立
可恢复阶段运行：

```powershell
uv run python scripts/run_routed_cv_experiment.py locomo full `
  --method $Seg `
  --campaign-id $Campaign `
  --split-manifest datasets/splits/locomo/cv5_seed42.json `
  --epochs $Epochs `
  --steps-per-sample 10 `
  --device auto
```

默认输出：

```text
outputs/rl_router/full_experiments/locomo/cv5/
  nsp_text_tiling_alpha_0p5/epochs_3/<experiment_id>/
```

完整聚合同时给出 micro accuracy、各 fold accuracy、fold mean、样本标准差、标准误、
最小值和最大值，可用于报告 `mean ± std`。

## 4. Alpha/Epoch 扫描与汇总

每个 alpha 必须先执行阶段 1 和阶段 2，使用独立 `$Seg` 与 `$Campaign`。同一 alpha 下
可以复用候选 extraction campaign，分别运行不同 epoch：

```powershell
$AlphaValues = @(0.3, 0.5, 0.7)
$EpochValues = @(1, 3, 5, 10)
```

所有完整实验完成后：

```powershell
uv run python scripts/summarize_rl_sweep.py locomo --protocol cv5
```

输出：

```text
outputs/rl_router/full_experiments/locomo/cv5/sweep_summary/summary.json
outputs/rl_router/full_experiments/locomo/cv5/sweep_summary/summary.csv
```

默认排序规则是：fold 平均准确率降序、fold 标准差升序、完整已知成本升序。CSV 只是候选
配置排名，论文表格仍应同时报告成本、调用数、token、micro accuracy 和 fold mean±std。

重要：当前 LoCoMo `cv5_seed42.json` 的 validation 为空。若 alpha/epoch 是在这五个 test
fold 的结果上选择的，这些 test 结果只能作为开发实验，不能再作为无偏最终结果。正式
论文应预先冻结超参数，或另建独立 validation / nested-CV 协议后再只评估一次 test。

## 5. LongMemEval 独立命令

LongMemEval 不会随 LoCoMo 自动运行：

```powershell
uv run python scripts/preprocess_datasets.py --datasets longmemeval
uv run python scripts/segment_datasets.py longmemeval full --method nsp_text_tiling --alpha 0.5
uv run python scripts/build_longmemeval_splits.py --method nsp_text_tiling --alpha 0.5
& .\scripts\build_longmemeval_rl_candidates.ps1 -Method nsp_text_tiling -Alpha 0.5
```

其后使用参数化 method、LongMemEval campaign 和对应 fixed/nested-CV split manifest 调用
同一训练与部署脚本即可。两个数据集拥有不同的 processed/segmented 路径、campaign、
prompt、Qdrant namespace、训练输出和评估输出。
