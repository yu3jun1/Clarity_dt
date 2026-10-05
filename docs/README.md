# CLARITY 文档索引

## 当前执行协议与数据

|文档|用途|
|---|---|
|[seed42 reproducibility audit](seed42_reproducibility_audit.md)|A/B/E 同卡双 replicate、确定性、stability gate 与 seed-matched 比较|
|[F 实验方案](teacher_forced_stagewise_ensemble_F_plan.md)|Teacher-forced Stage-wise + 3-member dynamics ensemble，不做 RRT 训练|
|[UCSF 数据集说明](UCSF_POSTOP_GLIOMA_DATASET.md)|本地数据内容、目录、字段和使用限制；不是当前 A–F 消融的数据切换|
|[清理与目录整理计划](repository_cleanup_plan.md)|前置条件、待删除目标、保留结果、备份及提交规则|

## 实验设计背景

原先位于仓库根目录的两份方案已移到 `docs/experiments/`，正文保留，入口如下：

- [原始 v3 研究提案](experiments/Clarity_dt_next_experiment_plan_v3.md)：历史设计背景。早期 A/C endpoint/open-loop 定义不是当前 all-pair 实现。
- [Teacher-forced Stage-wise 消融方案](experiments/teacher_forced_stagewise_ablation_plan.md)：E 的设计依据；当前运行状态和执行协议见上表。

这些文档是设计说明，不是自动执行的指令，也不代表建议中的实验已经完成。
活动代码和组别以 [仓库 README](../README.md)、配置及真实输出 metadata 为准。

`docs/legacy/` 不作为当前文档入口；它与 legacy configs/outputs 的删除仍受清理前置条件约束。
