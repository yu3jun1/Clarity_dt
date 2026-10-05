# CLARITY 文档索引

## 当前训练与数据

当前只有普通 seeded training，行为参考 step2400 的 `9871306` 基线；严格训练模式与
同卡严格复跑入口已移除。固定 seed 不承诺逐位一致，GPU/软件版本等来源信息仍记录。
历史配置中的 `deterministic` 键读取时忽略，不修改原保存文件；历史 gate 不是新 run 的前置条件。

|文档|用途|
|---|---|
|[仓库 README](../README.md)|A–F 定义、普通训练命令、共享缓存、metadata 和结果管理|
|[F 实验方案](teacher_forced_stagewise_ensemble_F_plan.md)|Teacher-forced Stage-wise + 3-member dynamics ensemble；普通训练与独立 campaign|
|[UCSF 数据集说明](UCSF_POSTOP_GLIOMA_DATASET.md)|本地数据内容、目录、字段和使用限制；不是当前 A–F 消融的数据切换|

## 历史协议与目录整理记录

|文档|用途|
|---|---|
|[已退役 seed42 reproducibility audit](seed42_reproducibility_audit.md)|旧 A/B/E 同卡双 replicate、严格数值设置、stability gates 与比较的追溯说明；不提供训练入口|
|[目录整理与保留记录](repository_cleanup_plan.md)|历史清理、保留范围、更正及备份决策；不保证当前 outputs 目录存在|

历史 gate/结果分析工具仅读取已有记录，不重新训练或自动恢复已删除的输出。
retention、cleanup manifest 或 inventory 如仍保留，应作为当时的事件/快照读取；
不能据此覆盖用户后续删除、迁移或整理的目录，也不能把旧严格结果冒充普通协议结果。

## 实验设计背景

原先位于仓库根目录的两份方案已移到 `docs/experiments/`，正文保留，入口如下：

- [原始 v3 研究提案](experiments/Clarity_dt_next_experiment_plan_v3.md)：历史设计背景。早期 A/C endpoint/open-loop 定义不是当前 all-pair 实现。
- [Teacher-forced Stage-wise 消融方案](experiments/teacher_forced_stagewise_ablation_plan.md)：E 的设计依据；当前运行方式见仓库 README。

这些文档是设计说明，不是自动执行的指令，也不代表建议中的实验已经完成。
活动代码和组别以 [仓库 README](../README.md)、配置及真实输出 metadata 为准。
`docs/legacy/` 若保留，仅供历史追溯，不作为当前执行协议入口。
