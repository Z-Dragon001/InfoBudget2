# CHANGELOG

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

## 修改原因

- 按你的第一阶段需求，先把完整研究骨架跑通，并保留后续论文实验需要的扩展接口。
- 当前环境没有 `faiss`、`spacy`、`sentence-transformers`，因此默认实现选用了可运行的轻量启发式版本，同时没有破坏后续替换真实模型与真实 NLP 组件的接口。

## 尚未实现的 TODO

- Budget Calibration
- Contextual Bandit / Learned Router
- Actionability / Prediction Gain / Information Gain
- Attention-based Topic Segmentation
- 真正的 LocalJointExtractor / APIJointExtractor
- 基于 spaCy / sentence-transformers / FAISS 的高保真版本

## 下一步建议

- 接入真实 embedding 模型替换 `HashingTextEncoder`
- 接入真实 LLM 后端替换 `MockJointExtractor`
- 补充数据集 loader 与 QA 评测流程
- 细化中英文实体与事件抽取器，提升 scorer 与 extractor 质量
