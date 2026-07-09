# CHANGELOG

## 记录规则

- 从本次开始，每次对项目代码、配置、脚本、测试或文档做出新增、修改、删除时，都必须同步记录到本文件。
- 记录时间统一使用 `YYYY-MM-DD HH:mm` 格式，精确到“年-月-日-时-分”。
- 同一次提问或同一轮开发中，如果包含多个改动，只记录一个时间标题；该时间标题下可以列出多个新增或修改点。
- 每条记录应同时说明“改了什么”与“为什么改”，便于后续追踪设计决策、实验设置和代码演进。

## 2026-07-08 15:40

- 新增 `.gitignore` 忽略策略：忽略 `outputs/` 和 `results/` 实验输出、`datasets/raw/` 原始本机数据、`datasets/processed/` 预处理本机数据、`models/` 本地模型权重与下载产物、Python 缓存、测试缓存、虚拟环境、本地 IDE 配置、外部参考仓库 `.external_refs/`、本地密钥覆盖配置与临时日志文件，避免后续误上传大文件、隐私数据和运行产物。
- 数据目录规则保留 `.gitkeep` 例外，方便仓库继续保留目录结构占位；已被 Git 跟踪的历史数据文件仍需单独执行 `git rm --cached` 后才会真正停止上传。

## 2026-07-08 15:25

- 新增评估阶段 QA answer generator：`infobudget/evaluation/answer_generation.py` 按数据集构造 LoCoMo / LongMemEval 专用 QA prompt，默认用 `evaluation.answer_model_tier: medium` 从 `configs/models.yaml` 的路由模型表取中等模型生成答案，同时保留 `qa_mode: retrieved_top1` 供离线测试和调试切换。
- 新增 `configs/prompts/locomo_answer.txt` 和 `configs/prompts/longmemeval_answer.txt`：LoCoMo 复用 LightMem 的双 speaker、时间推理和短答案提示词，LongMemEval 复用 `question_date + question + retrieved memories` 的 QA 结构，确保两套数据集不再共用同一个回答流程。
- 修改 `infobudget/evaluation/judges.py`：`judge_mode="llm_judge"` 时仍使用 `evaluation.judge_model`，默认 `gpt-4o-mini`；LoCoMo 使用 LightMem `ACCURACY_PROMPT` 并解析 `{"label": "CORRECT|WRONG"}`，LongMemEval 复用 `get_anscheck_prompt()` 的 question_type 分支并按 yes/no 解析，覆盖 temporal、knowledge-update、preference 和 abstention。
- 修改 `infobudget/datasets/longmemeval.py`：LongMemEval 的 abstention 判定从仅识别 `_abs` 结尾调整为与 LightMem 脚本一致的 `abs in question_id`，避免 `abs` 出现在 question id 中间时走错 judge profile。
- 修改 `infobudget/evaluation/dataset_runner.py`：评估阶段只加载已经存好的 memory store 并检索 fact，不改 memory 输出目录；LoCoMo 默认 `top_k=60`，LongMemEval 默认 `top_k=20`，结果仍写入 `outputs/evaluation/{dataset}/{split}/{scoring_mode}/{extraction_mode}/`，memory 仍在 `outputs/memory/{dataset}/{split}/{scoring_mode}/{extraction_mode}/{sample_id}/`。
- 修改 `infobudget/config.py` 与 `configs/config.yaml`：新增 `qa_mode`、`answer_model_tier`、`qa_max_new_tokens`、`locomo_retrieval_top_k`、`longmemeval_retrieval_top_k`，便于后续把 QA 从中等模型切换到大/小模型而不改代码。
- 修改/新增测试文件：`tests/test_answer_generation.py`、`tests/test_llm_judge.py`、`tests/test_datasets.py` 覆盖 medium QA 模型选择、LoCoMo JSON judge、LongMemEval yes/no judge，以及离线 runner 回归。已运行 `python -m pytest tests\test_answer_generation.py tests\test_llm_judge.py tests\test_datasets.py`，结果为 `8 passed`。

## 2026-07-05 18:30

- Replace the old FAISS/Numpy vector index path with local Qdrant collections. `MemoryStore` now writes LightMem-style `MemoryEntry` payloads into Qdrant and still keeps `memory_jsonl/` as a human-readable audit mirror.
- Rename storage config from `faiss_dir/faiss_index_type` to `qdrant_dir/qdrant_memory_collection/qdrant_episode_collection`, add `qdrant-client` as a project dependency, and update memory/evaluation output folders to use `qdrant/`.
- Remove the tracked FAISS index implementation and update tests/docs to verify Qdrant-backed persistence.
- Add a consistency guard for separated evaluation: when JSONL memories exist but the Qdrant point count is missing or stale, the runner rebuilds Qdrant from `memory_entries.jsonl` before retrieval.
- Keep the default hashing fallback for `SentenceTransformerTextEncoder` at the configured embedding dimension, preventing accidental 384/256 dimensionality mismatches between build and evaluation environments.

## 2026-07-05 17:04

- 按 LightMem 的模式选择设计新增 `extractor.extraction_mode`，默认值为 `flat`；`flat` 模式只调用 factual prompt 一次并只生成 `entry_type="factual"` 的事实记忆，`event` 模式调用 factual 与 relational 两个 prompt 各一次并合并为事实记忆和关系记忆，不实现 summary 层。
- 将 factual prompt 与 relational prompt 分离：原 `joint_memory_extraction*.txt` 改为只抽 `fact`，新增 `joint_memory_relation*.txt` 用于 event 模式的第二次关系抽取。prompt 内容对齐 LightMem 的 flat/event 语义：flat 保持事实抽取，event 的 relational prompt 专注支持、鼓励、感谢、共情、兴趣、协作等人际动态。
- `MemoryEntry` 新增 `entry_type` 字段，存储时区分 `factual` 与 `relational`；LLM extractor、mock extractor 和 cost logger 同步区分 `flat_factual`、`event_factual`、`event_relational`，使 event 模式的两次 LLM 调用能在成本日志中体现。
- memory/evaluation 目录新增 extraction mode 层级：`outputs/memory/{dataset}/{split}/{scoring_mode}/{extraction_mode}/{sample_id}/` 与 `outputs/evaluation/{dataset}/{split}/{scoring_mode}/{extraction_mode}/`，避免 flat/event 结果互相覆盖。`build_dataset_memory.py`、`evaluate_dataset_memory.py` 和兼容命令 `run_dataset_evaluation.py` 均新增 `--extraction-mode flat|event`。已运行 `pytest -q`，结果为 `21 passed, 1 skipped`。

## 2026-07-05 15:52

- 将数据集实验流程改为 LightMem-style 的记忆构建与评估分离设计：`DatasetEvaluationRunner` 新增 `build_memories(...)` 和 `evaluate_existing_memories(...)`，前者只抽取并持久化记忆，后者只加载已有记忆做检索和 QA 评估；原 `evaluate(...)` 保留为兼容包装，内部先 build 再 eval。
- 新增两个独立命令：`scripts/build_dataset_memory.py` 只构建 memory store，`scripts/evaluate_dataset_memory.py` 只评估已有 memory store；两者均支持 `--scoring-modes` 和 `--all-scoring-modes`，方便一次性生成/评估 9 种路由评分模式。
- 调整 memory/evaluation 目录布局为 `outputs/memory/{dataset}/{split}/{scoring_mode}/{sample_id}/` 与 `outputs/evaluation/{dataset}/{split}/{scoring_mode}/`，确保不同数据集、split、9 种 scoring mode 和 sample 之间互不覆盖，并新增 `build_manifest.json` 记录记忆构建阶段元数据。
- 新增 `docs/separated_memory_evaluation_design.md` 记录两阶段实验设计、目录结构和命令示例；同步更新数据集测试，验证 build-only 与 eval-existing 可分离运行。已运行 `pytest -q`，结果为 `20 passed, 1 skipped`。

## 2026-07-05 14:57

- 为四份 LightMem-compatible 记忆抽取 prompt 补充 few-shot examples，覆盖 `source_id` 复制、单 turn 拆分多条 atomic facts、跨 topic 输入、fact/relation 区分、以及 InfoBudget 项目配置类事实抽取。这样提示词更接近 LightMem 原始 prompt 的示例驱动风格，也能稳定大模型输出 `{"data": [...]}` 的格式。
- 在 large tier prompt 中补充 “Be exhaustive” 风格 reminder，强调除纯无意义、纯过程性或无依据消息外，应尽量抽取持久事实或有意义关系；small/medium/large 仍保持同一 JSON schema，仅保留不同抽取密度和最大输出条数。已运行 `pytest -q`，结果为 `20 passed, 1 skipped`。

## 2026-07-05 09:34

- 细化 LightMem-compatible 记忆抽取提示词：在保持统一 `{"data": [{"source_id": ..., "fact": "..."}, {"source_id": ..., "relation": "..."}]}` schema 不变的前提下，补充 turn/source_id 处理、factual memory 定义、relational memory 定义、standalone 改写规则、跳过规则、时间锚定、图片描述处理和输出约束。
- 调整 small/medium/large 三档 prompt 的差异为抽取密度与最大输出条数：small 只保留最高价值记忆，medium 平衡召回和精度，large 高召回并保留更多细节；三档均不改变记忆组织格式，以保证和 LightMem 风格存储、检索、评估链路一致。已运行 `pytest -q`，结果为 `20 passed, 1 skipped`。

## 2026-07-05 01:12

- 将主实验长期记忆抽取格式从 InfoBudget 自定义的 `semantic_memory` / `episodic_memory` JSON schema 调整为 LightMem-compatible 的原子记忆格式：大模型只返回 `{"data": [{"source_id": ..., "fact": "..."}, {"source_id": ..., "relation": "..."}]}`，代码侧再按 `source_id` 回填时间戳、weekday、speaker 和 `topic_id`，组织为 LightMem 风格 `MemoryEntry`。
- 修改 `MemoryEntry`、LLM/mock extractor、pipeline、memory store、retriever/judge 相关逻辑，使每个 topic segment 可以产生多条 turn-level atomic memories，并以 `memory` 字段进行向量索引和 QA 检索；成本和路由信息继续通过 `cost_logs.jsonl` 与 pipeline tiers 统计，不混入 memory payload。
- 重写三档 `configs/prompts/joint_memory_extraction_*.txt`：small/medium/large 保持统一 `data[]` schema，只通过最大输出条数和抽取密度控制预算，避免不同 tier 产生不兼容的记忆结构。同步更新单元测试，已运行 `pytest -q`，结果为 `20 passed, 1 skipped`。

## 2026-07-02 10:41

- 为评估阶段新增独立 LLM judge 配置：`configs/config.yaml` 的 `evaluation.judge_model` 默认指向 `gpt-4o-mini`，并在配置加载时转换为 `ModelSpec`。这样评估用模型可以和 small/medium/large 记忆抽取模型分开管理。
- 新增 `LLMJudge` 与 registry 路由：`judge_mode="llm_judge"` 时通过 OpenAI-compatible client 请求 judge 模型并解析 JSON 判定结果；离线测试 bundle 显式切回 `rule_judge`，避免普通单测依赖外部 API。
- 评估 runner 的 manifest 新增 `judge_mode`、`judge_model` 和 `judge_cost_counted: false`，明确 judge 调用成本不计入 InfoBudget 的记忆抽取预算统计；现有 `CostLogger` 仍只记录长期记忆抽取阶段成本。已运行 `pytest -q`，结果为 `20 passed, 1 skipped`。

## 2026-07-02 09:47

- 为数据集评估命令新增单样本/少量样本试跑能力：`scripts/run_dataset_evaluation.py` 新增 `--limit` 和 `--sample-ids` 参数，`DatasetEvaluationRunner.evaluate(...)` 同步支持按 sample id 过滤和按数量截断。这样服务器初次验证 LoCoMo 流程时可以只跑一个样本，避免直接触发全量数据集构建和大模型调用成本。
- 评估 manifest 中新增 `requested_sample_ids` 与 `sample_limit` 元数据，并在使用过滤/截断时将 `num_examples` 写为实际处理样本数，方便后续区分全量评估和临时 smoke test。已运行 `pytest -q`，结果为 `18 passed, 1 skipped`。

## 2026-07-01 23:47

- 按实验室服务器部署需求，将 DashScope API key 从环境变量引用改为在 `configs/models.yaml` 的 large tier 中显式配置。变更记录不写入密钥明文，避免在文档中二次暴露。
- 将 `configs/config.yaml` 中的 `extractor.fallback_to_mock` 改为 `false`，使真实 LLM 抽取失败时直接报错，避免服务器实验结果被启发式 mock fallback 污染。
- 调整离线单元测试：`tests/test_pipeline.py` 与 `tests/test_datasets.py` 在构造临时 bundle 时显式使用 `extractor.mode="mock_joint"`，避免普通 `pytest` 因生产配置的真实 LLM 调用而依赖网络；`tests/test_model_config.py` 改为用临时 `ModelSpec` 验证环境变量解析能力。已运行 `pytest -q`，结果为 `18 passed, 1 skipped`。

## 2026-07-01 22:38

- 新增可选 DashScope 集成测试 `tests/test_dashscope_qwen_integration.py`，用于显式验证阿里云兼容 OpenAI endpoint 上的 `Qwen/Qwen3-Next-80B-A3B-Instruct` 是否可用。该测试默认跳过，避免普通单测依赖网络或 API key；需要真实调用时运行 `RUN_DASHSCOPE_INTEGRATION_TESTS=1 DASHSCOPE_API_KEY=... pytest tests/test_dashscope_qwen_integration.py -q`。
- 测试会校验 large tier 的模型配置、provider-facing 模型名 `qwen3-next-80b-a3b-instruct`、DashScope endpoint，并发起一次短 chat completion 请求，断言返回内容、token usage 和 latency 均有效。已运行 `pytest -q`，结果为 `18 passed, 1 skipped`。

## 2026-07-01 22:14

- 将主题分割 embedding 从 pipeline 中硬编码的 `HashingTextEncoder()` 改为配置驱动创建：新增 `build_text_encoder(...)`，并让 `InfoBudgetPipeline` 根据 `segmentation.embedding_model` 构建编码器。这样分段阶段可以切换真实 embedding 模型，而不是只能使用 256 维哈希向量。
- 新增 `SentenceTransformerTextEncoder`，默认支持 `sentence-transformers/all-MiniLM-L6-v2`，该模型在可用时输出 384 维归一化向量，用于 `LiteTopicSeg` 的相邻 turn cosine similarity 计算；同时保留 `HashingTextEncoder` 作为无外部依赖 fallback，保证本地没有安装依赖或模型权重未缓存时测试和离线调试仍可运行。
- 更新 `configs/config.yaml`：将 `segmentation.embedding_model` 改为 `sentence-transformers/all-MiniLM-L6-v2`。这使主题分割配置与当前希望使用 MiniLM 语义向量的实验设定对齐。
- 更新依赖声明：在 `pyproject.toml` 中新增 `sentence-transformers>=3.0`，用于真实加载 all-MiniLM-L6-v2。
- 补充测试：新增 `tests/test_embeddings.py`，验证 MiniLM 配置会创建 `SentenceTransformerTextEncoder`，并验证外部模型不可用时可以回退到 hashing encoder。已运行 `pytest -q`，结果为 `18 passed`。

## 2026-07-01 18:55

- 接入真实 LLM 联合记忆抽取路径：新增 `infobudget/extractors/llm_joint.py`，实现 OpenAI-compatible chat completions 调用、JSON 解析、schema 规范化、`MemoryEntry` 映射、真实 token usage / latency 成本记录，以及 `LocalJointExtractor`、`APIJointExtractor`、`TieredJointExtractor`。这样 `small / medium / large` 可以按路由结果分别调用本地小/中模型和 API 大模型，不再只能依赖启发式 mock 抽取。
- 修改 pipeline 抽取器接线：`InfoBudgetPipeline` 默认使用 `TieredJointExtractor`，根据 `models.yaml` 中的 `deploy` 字段选择 local 或 API 抽取器；同时保留 `MockJointExtractor` fallback，并新增 `extractor.fallback_to_mock` 配置，保证本地 vLLM 或远程 API 未启动时仍可离线跑通测试和调试。
- 重写 `configs/prompts/joint_memory_extraction_small.txt`、`joint_memory_extraction_medium.txt`、`joint_memory_extraction_large.txt` 和 fallback `joint_memory_extraction.txt`：统一到 InfoBudget 的 semantic memory / episodic memory JSON schema，并吸收 LightMem 的按 turn/source 顺序处理、mention time vs event time 区分、相对时间锚定和图片描述绑定规则；同时保留 BudgetMem model-tier 思路，让三档 prompt 在同一任务结构下体现不同抽取强度。
- 清理旧占位类：移除 `infobudget/extractors/base.py` 中同名 `LocalJointExtractor` / `APIJointExtractor` 的 `NotImplementedError` 占位，避免真实实现和占位实现并存造成导入歧义；统一从 `infobudget.extractors` 导出真实抽取器。
- 更新设计文档：同步修改 `docs/llm_memory_extraction_design.md`，说明当前启发式抽取器已经变为 fallback，主路径已经是 tier-aware OpenAI-compatible LLM 抽取。
- 补充测试：新增 `tests/test_llm_joint_extractor.py`，用 fake client 验证 large tier 会分发到 API extractor、LLM JSON 会映射为 semantic/episodic memory，并记录成本日志。已运行 `pytest -q`，结果为 `16 passed`。

## 2026-07-01 16:02

- 更新三档模型选型配置：`small` 改为本地常驻 `Qwen/Qwen2.5-7B-Instruct` FP16，`medium` 改为本地常驻 `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`，`large` 改为 API 调用 `Qwen/Qwen3-Next-80B-A3B-Instruct`。这样配置层已经与当前实验设定对齐，后续 extractor 可以直接按 tier 读取对应模型。
- 扩展 `ModelSpec`：新增 `api_base_url`、`api_key`、`api_key_env` 和 `request_model_name` 字段，并提供 `effective_model_name` 与 `resolved_api_key()`。这样本地 vLLM 服务和 OpenAI-compatible API 服务都可以用同一套模型注册结构描述，同时避免把真实 API key 明文写入仓库。
- 为本地 small / medium 配置默认 OpenAI-compatible endpoint：`http://localhost:8001/v1` 与 `http://localhost:8002/v1`；为 large 配置 DashScope 兼容模式 endpoint `https://dashscope.aliyuncs.com/compatible-mode/v1`，并使用 `${DASHSCOPE_API_KEY}` / `DASHSCOPE_API_KEY` 从环境变量读取密钥。这样后续接真实 API 时只需要补 extractor 调用逻辑，不需要再改配置结构。
- 同步更新 `prices.yaml` 的模型 key，避免新模型名在成本日志中查不到价格；large 的真实 API 单价暂以 `0.00` 占位，等待确认供应商计费后再更新，避免写入不准确成本假设。
- 新增 `tests/test_model_config.py`，校验三档模型名、部署方式、endpoint、dtype、large API key 环境变量解析和 provider-facing 模型名。已运行 `pytest tests/test_model_config.py tests/test_pipeline.py -q`，结果为 `5 passed`；已运行 `pytest -q`，结果为 `15 passed`。

## 2026-07-01 01:11

- 为 `InfoBudgetPipeline` 新增可选 `run_output_dir` 参数，并在该参数存在时将 memory JSONL、向量索引、segments 和 cost logs 写入该运行目录下的 `memory_jsonl/` 与 `faiss/` 子目录。这样可以按实验运行隔离长期记忆存储位置，同时不修改 `MemoryEntry` 数据结构和现有 JSONL 格式，降低兼容风险。
- 为 pipeline 新增 `save_memory_outputs()` 方法，并让原有 `_save_outputs(...)` 复用该方法。这样评估 runner 可以在不写普通 pipeline debug 输出的情况下，单独持久化 memory 构建结果，避免 evaluation artifacts 和 memory artifacts 混在一起。
- 修改 `DatasetEvaluationRunner`：评估时按 `outputs/memory/{dataset}/{scoring_mode}/{sample_id}/` 为每个样本创建独立 memory 存储根目录；默认 `full` 模式写入 `outputs/memory/{dataset}/full/{sample_id}/`。这样既能按评分模式隔离 memory，又能避免同一数据集多个样本共用目录导致后写样本覆盖先写样本。
- 在评估 `run_manifest.json` 中新增 `memory_output_dir` 元数据，记录该次评估对应的 memory 根目录。这样后续分析某个评分模式的 evaluation 结果时，可以直接定位对应的长期记忆构建产物。
- 补充测试：新增 pipeline 层 `run_output_dir` 覆盖测试，并扩展 dataset runner 测试，断言 `full` 与 `entropy_only` 均会写出独立的 `memory_entries.jsonl`、`segments.jsonl`、`cost_logs.jsonl` 和 `memory.index`。已运行 `pytest tests/test_datasets.py tests/test_pipeline.py -q`，结果为 `6 passed`；已运行 `pytest -q`，结果为 `13 passed`。

## 2026-07-01 01:00

- 新增评分模式选择机制：在 `infobudget/scoring/modes.py` 中定义九种路由评分模式，包括六个单指标模式 `entropy_only`、`lexical_density_only`、`entity_density_only`、`concept_density_only`、`information_gain_only`、`actionability_only`，两个聚合模式 `intrinsic_only`、`utility_only`，以及默认 `full`。这样后续实验可以通过命令行选择不同指标计算阈值得分，而不是只能使用信息含量与效用价值加权和。
- 修改 `InformationScorer` 与 pipeline 接线：`InformationScorer` 新增 `scoring_mode` 参数，仍完整计算 `intrinsic_score`、`utility_score` 和所有 `details`，但根据模式选择写入 `ScoreResult.final_score` 的路由分数；`InfoBudgetPipeline` 和 `from_config_dir(...)` 同步接收并传递该模式。这样 router 仍复用原有 `final_score` 入口，避免改动路由器本身。
- 修改评估命令行：`scripts/run_dataset_evaluation.py` 新增 `--scoring-mode` 参数，合法值限定为九种评分模式；未指定时默认 `full`，保持原有行为。`scripts/run_mock_pipeline.py` 也补充同名参数，便于单独跑端到端 mock pipeline 时选择评分模式。
- 修改评估结果目录：`DatasetEvaluationRunner` 在非 `full` 模式下将评估结果写入对应模式目录，例如 `outputs/evaluation/locomo/entropy_only/`；默认 `full` 模式仍沿用真实 split 目录，例如 `outputs/evaluation/locomo/full/`。同时在 `run_manifest.json` 中记录真实 `split`、`output_label` 和 `scoring_mode`，避免目录标签和数据 split 混淆。
- 补充测试：新增 `entropy_only` 单指标路由分数断言，验证 `final_score` 等于 `details["entropy"]`；扩展数据集评估测试，验证默认 `full` 与 `entropy_only` 输出目录及 manifest 元数据。已运行 `pytest -q`，结果为 `12 passed`；并运行 `python scripts/run_dataset_evaluation.py --help`，确认命令行参数显示九种评分模式。

## 2026-06-30 19:34

- 重构信息效用价值评分结构：将 `scorer.py` 顶层 `details` 从直接暴露 `semantic_novelty`、`entity_novelty`、`episodic_novelty` 改为只暴露 `information_gain` 和 `actionability`，使评分输出与“信息含量四项 + 信息效用两项”的理论结构一致，避免把信息增益的内部子指标和顶层指标混在一起。
- 实现 `InformationGainScorer`：按照当前设计将信息增益定义为三路新增量的最大值，即 `max(G_semantic, G_entity, G_episodic)`；内部复用原有语义、实体、情景新颖性计算，但不再作为顶层 utility 详情输出。这样保留已有 novelty 计算能力，同时把概念命名统一为信息增益。
- 实现 `ActionabilityScorer`：依据文档中的 `Actionability=max(A_frame,A_condition,A_constraint,A_decision)` 思路，新增动作词、条件词、约束词、决策词、阈值正则和弱语气降权规则，用轻量规则近似 Action Frame 完整度、条件明确度、约束强度和决策影响。这样在不新增 Stanza/spaCy 强依赖的前提下，先让可行动性指标可运行、可测试、可解释。
- 更新效用权重配置与 dataclass：将 `configs/weights.yaml` 的 utility 权重从三路 novelty 改为 `information_gain: 0.60` 与 `actionability: 0.40`，并同步修改 `UtilityWeights` 字段，保证配置加载、总分计算和理论指标保持一致。
- 补充评分测试：新增对 `ScoreResult.details` 顶层键集合的断言，确保只包含四个信息含量指标与 `information_gain`、`actionability`；新增可行动性测试，验证明确阈值路由规则得分高于弱意图表达。已运行 `pytest -q`，结果为 `11 passed`。

## 2026-06-30 14:30

- 修改信息含量评分中的 `concept_density` 计算方式：将原先的“去重内容词数 / 总 token 数”改为基于 `extract_idea_units(...)` 的严格 Idea Density 近似实现，即“去重 proposition-like idea units 数 / 总 token 数”。这样做是为了让概念密度不再停留在词汇层面的 type-token ratio，而是更贴近文献中 idea/propositional density 对“语义命题单元密度”的定义。
- 新增 idea unit 抽取工具链：在 `infobudget/utils/text.py` 中补充分词、启发式词性过滤、词形还原/归一化、停用词过滤、同义或别名归并、谓词提示词识别与去重命题单元生成逻辑。这样可以在当前不新增强依赖的情况下，把概念密度升级为可运行的命题密度近似指标。
- 接入可选 `spaCy` 高保真路径：`extract_idea_units(text, spacy_model)` 会在配置了可加载 `spacy_model` 时优先使用 spaCy 的 POS、lemma 与 dependency 信息抽取 idea units；未配置或环境不可用时自动回退到轻量启发式实现。这样保留当前项目的低依赖可运行性，同时为后续更严格的 NLP 实现留出接口。
- 修改 `ConceptDensityScorer` 与 `InformationScorer` 接线：`ConceptDensityScorer` 现在接收 `spacy_model` 配置，并通过 `len(extract_idea_units(text, spacy_model)) / len(tokenize_text(text))` 计算分数；`InformationScorer` 从 `ScoringConfig.spacy_model` 传入该配置。这样保持 `details["concept_density"]` 对外字段不变，但内部语义升级为 Idea Density。
- 补充评分测试：在 `tests/test_scoring_router.py` 中新增 `test_concept_density_uses_deduplicated_idea_units`，验证重复句子中的 idea units 会被去重，且 `models` 与 `LLMs` 能通过别名归并映射为统一概念。已运行 `pytest -q`，结果为 `10 passed`，确认本次评分指标修改没有破坏现有流程。

## 2026-06-29 22:37

- 修改 `LongMemEval` 预处理时间戳策略：将 turn 级时间戳从“直接继承 session 时间”调整为“基于 session 时间按 500ms 递增合成”，与 `LightMem` 当前实现保持一致，便于后续情景记忆、时间排序和检索逻辑对齐。
- 修改通用 session 构建逻辑：`build_sessions_from_flat_turns` 由“按秒偏移”扩展为“按毫秒偏移”，从而同时支持 `LoCoMo` 的 1000ms 递增和 `LongMemEval` 的 500ms 递增，提升数据层设计的可扩展性。
- 补充并更新测试：为 `LongMemEval` 新增 turn 级 `0ms / 500ms` 时间戳断言，同时保留 `LoCoMo` 的既有时间戳行为校验，确保两套数据集预处理规则都能稳定回归。
- 重新运行并验证预处理与测试：确认 `LongMemEval` 处理后的样例已按 `...00.000 / ...00.500` 形式落盘，`pytest tests/test_datasets.py tests/test_segmentation.py -q` 通过，说明本次调整没有破坏现有流程。
- 明确 token 统计口径：保留预处理阶段的 `token_count` 作为分段、信息密度和预算控制的静态估计特征；同时保留运行期 `prompt/completion/total tokens` 作为真实成本统计入口。这样既能服务前置分析，又能兼容未来真实大模型返回的 usage 数据。

## 2026-06-29 22:55

- 完善 LLM 长期记忆提取方案：明确当前 `MockJointExtractor` 只是为了让 InfoBudget 的分割、评分、路由、存储和评估链路先跑通，并不代表最终的真实提取方案；新增文档 [docs/llm_memory_extraction_design.md](/S:/Workfile/InfoBudget/docs/llm_memory_extraction_design.md)，系统说明为什么现在仍是启发式抽取、为什么最终应改为 joint LLM extraction，以及 small / medium / large 三档模型在 InfoBudget 中的职责分工。
- 新增英文版 tier-specific prompts：在 `configs/prompts/` 下补充 `joint_memory_extraction_small.txt`、`joint_memory_extraction_medium.txt`、`joint_memory_extraction_large.txt`，并重写通用 fallback `joint_memory_extraction.txt`。这些提示词统一要求一次输出 summary、semantic memory、episodic memory 和 importance，且输出必须严格遵循 JSON 结构，方便后续直接映射到 `MemoryEntry`。
- 修改 prompt 加载与运行时接线：`infobudget/runtime/prompt_loader.py` 新增批量 prompt 加载能力，`infobudget/runtime/pipeline.py` 改为按 `small / medium / large` 加载不同提取 prompt，让当前 runtime 先具备“按路由层级选择不同提示词”的能力，后续接入真实本地模型和 API 模型时不需要再次重构调用链。
- 修改 `MockJointExtractor` 接口：当前虽然仍是 mock 抽取，但已经支持按 tier 选择不同 prompt。这样一来，哪怕还没有真实 LLM backend，代码结构也已经与未来的真实大中小模型提取方式对齐，后续只需要把 mock 逻辑替换为真实推理调用和 JSON 解析。
- 修复 prompt 渲染细节：由于英文 prompt 中包含大量 JSON 花括号，直接使用 `str.format(...)` 会把 JSON 误判成占位符；因此在 `MockJointExtractor` 中新增了安全渲染逻辑，只替换 `router_level`、`information_score` 和 `segment_text` 三个变量，保证 tier prompt 可以稳定被加载、格式化和统计 token。
- 补充测试：在 `tests/test_pipeline.py` 中新增对 `small / medium / large` prompt 加载的断言，确保运行时确实能够根据路由层级拿到正确的英文提示词模板。

## v0.1.0

- 新增项目骨架：`configs/`、`infobudget/`、`scripts/`、`tests/`、`outputs/`。
- 新增配置系统：支持 `config.yaml`、`weights.yaml`、`models.yaml`、`prices.yaml` 和外置 Prompt。
- 新增 LiteTopicSeg：实现 embedding-only 的主题分段、短段合并与超长切分。
- 新增 Information Scorer：包含 entropy、lexical density、entity density、concept density，以及三类 novelty。
- 新增 P33/P67 Router：固定阈值三档路由。
- 新增 MockJointExtractor：通过单次 joint extraction 接口产出结构化 memory entry。
- 新增 Cost Logger：统一记录 token、latency、tier、backend 和 cost。
- 新增 JSONL Memory Store：保存 memory、episode、segment，并提供向量检索。
- 新增 FAISS 接口：提供 `FaissVectorIndex` 可选实现，同时用 `NumpyFlatIPIndex` 保障本地可运行。
- 新增基础 Evaluation metrics：accuracy、precision、recall、cost、token、latency、router distribution、Pareto front。
- 新增 `datasets/` 目录：补齐 `raw/`、`processed/`、LOCOMO 与 LongMemEval 的目录结构。
- 新增统一数据集层：支持 `DatasetDialogueExample`、`DatasetSession`、`DatasetQAPair`、`DatasetLoader`、`DatasetArtifactStore` 和统一 processed 工件布局。
- 新增 LOCOMO / LongMemEval 预处理器：支持真实原始字段解析、session 时间戳标准化、evidence/session 对齐。
- 新增流式 JSON 数组读取：可处理 `longmemeval_s_cleaned.json` 这类大文件，避免一次性 `json.load()` 爆内存。
- 新增 processed 数据集工件布局：每个 `dataset/split` 输出 `samples.jsonl`、`questions.jsonl`、`sessions.jsonl`、`manifest.json`。
- 新增数据集评估 runner：支持按 dataset / split 运行构建、检索、judge、retrieval trace 与评估结果落盘。
- 新增评估工件布局：每个 `dataset/split` 输出 `metrics.json`、`predictions.jsonl`、`retrieval_traces.jsonl`、`run_manifest.json`。
- 新增设计文档：[docs/dataset_evaluation_design.md](/S:/Workfile/InfoBudget/docs/dataset_evaluation_design.md)

## 修改原因

- 按你的第一阶段需求，先把完整研究骨架跑通，并保留后续论文实验需要的扩展接口。
- 当前环境没有 `faiss`、`spacy`、`sentence-transformers`，因此默认实现选用了可运行的轻量启发式版本，同时没有破坏后续替换真实模型与真实 NLP 组件的接口。
- 原先缺少 `datasets/` 目录与数据集预处理阶段，不满足文档里“预处理代码分开、Loader 统一接口、评估与构建解耦”的要求，因此本次补齐了数据集层。
- 原先的数据集预处理对真实 `LoCoMo` / `LongMemEval` 原始结构支持不足，也没有 session / question 投影与 retrieval trace，因此本次重新设计了数据工件与评估落盘方案。

## 尚未实现的 TODO

- Budget Calibration
- Contextual Bandit / Learned Router
- Attention-based Topic Segmentation
- 基于 spaCy / sentence-transformers / FAISS 的高保真版本
- 面向 LOCOMO / LongMemEval 官方原始字段的更严格字段映射与 judge 逻辑
- LongMemEval 官方 splits 文件接入与多 split 管理

## 下一步建议

- 接入真实 embedding 模型替换 `HashingTextEncoder`
- 启动本地 small / medium vLLM 服务并配置 DashScope API key，跑一次真实 LLM extraction 端到端实验
- 补充数据集 loader 与 QA 评测流程
- 细化中英文实体与事件抽取器，提升 scorer 与 extractor 质量
- 接入真实 QA 生成 / judge 模型，替换当前 mock retrieval-based correctness
