# CLARITY 文档索引

## 当前执行协议与数据

|文档|用途|
|---|---|
|[seed42 reproducibility audit](seed42_reproducibility_audit.md)|A/B/E 同卡双 replicate、确定性、stability gate 与 seed-matched 比较|
|[F 实验方案](teacher_forced_stagewise_ensemble_F_plan.md)|Teacher-forced Stage-wise + 3-member dynamics ensemble，不做 RRT 训练|
|[UCSF 数据集说明](UCSF_POSTOP_GLIOMA_DATASET.md)|本地数据内容、目录、字段和使用限制；不是当前 A–F 消融的数据切换|
|[目录整理与保留范围](repository_cleanup_plan.md)|保留完整 A–F，仅排除精确指定的首轮 ABCD42；恢复与备份记录|
|[完整实验记录索引](../outputs/ablation_inventory.json)|A–F 全部结果路径、协议区分、指标校验和及重复副本映射|

## 实验设计背景

原先位于仓库根目录的两份方案已移到 `docs/experiments/`，正文保留，入口如下：

- [原始 v3 研究提案](experiments/Clarity_dt_next_experiment_plan_v3.md)：历史设计背景。早期 A/C endpoint/open-loop 定义不是当前 all-pair 实现。
- [Teacher-forced Stage-wise 消融方案](experiments/teacher_forced_stagewise_ablation_plan.md)：E 的设计依据；当前运行状态和执行协议见上表。

这些文档是设计说明，不是自动执行的指令，也不代表建议中的实验已经完成。
活动代码和组别以 [仓库 README](../README.md)、配置及真实输出 metadata 为准。

2026-10-05 已更正此前过宽的清理范围：历史 outputs 和配套 legacy configs/docs 已恢复。
`docs/legacy/` 仅供历史追溯，不作为当前执行协议入口。
当前保留与精确排除范围见 [retention correction](../outputs/reproducibility/seed42_same_gpu/retention_correction.json)；
原 [cleanup manifest](../outputs/reproducibility/seed42_same_gpu/cleanup_manifest.json) 保留为先前操作的事件记录。
