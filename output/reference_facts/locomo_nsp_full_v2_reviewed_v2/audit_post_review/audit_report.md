# Reference Fact 阶段一自动审计报告

- 自动状态：`AUTOMATED_PASS_MANUAL_PENDING`
- Reference 行数：1054
- 来源 Segment 数：1054
- 冻结 Fact 数：6456
- 零 Fact Segment：47
- 达到 Fact 上限的 Segment：29
- 每段 Fact：min=0，max=15，mean=6.125
- 自动错误：0
- 自动警告：1
- 人工复核项：110

## 自动检查结论

结构、来源 ID、Fact ID、集合哈希、计数、manifest 和 SQLite ledger 自动检查通过。

## 警告

- `incomplete_cost_accounting`：`{"code": "incomplete_cost_accounting", "detail": "不影响 Reference Fact 内容审计，但正式成本报告前需要补齐价格快照。"}`

## 语义近重复扫描

- 是否执行：True
- 阈值：0.9
- 候选对数：30
- 相似度不低于 0.95：4

近重复结果只是人工复核候选。人物方向不同、邀请与接受、计划与完成等高相似文本可能仍然是独立事实，不能自动删除。

## 阶段一通过门槛

阶段一只有在以下条件全部满足后才能标记完成：

1. `error_count == 0`；
2. `manual_review_queue.jsonl` 中全部项目完成复核；
3. 被判定需修改的 Fact 进入新版本 Reference 集合，不能原地静默修改冻结文件；
4. 新版本重新计算 Fact ID、集合哈希和 manifest，并重新运行本审计；
5. 最终审计报告和人工决策文件与 Reference 集合共同归档。

当前状态仍为 `AUTOMATED_PASS_MANUAL_PENDING`，不等同于阶段一最终完成。
