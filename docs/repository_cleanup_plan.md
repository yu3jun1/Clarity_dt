# 仓库实验记录保留与目录整理

## 当前保留范围（2026-10-05 更正）

用户澄清：目标是保留完整 A–F 消融记录，仅删除 A seed42 的 H3 MSE 为
`0.6268193917348981` 所在同一首轮的 A/B/C/D seed42。
此前整族退役范围过宽，已从本地备份恢复其他记录；不是删除所有 seed42，也不是整仓回滚。

精确排除对象全部来自 `git:6584f6d`，同轮启动时间为 `2026-09-30T16:03:39Z`：

|组别|历史 replicate|H3 MSE|
|---|---|---:|
|A seed42|historical_rep01|0.6268193917348981|
|B seed42|historical_rep01|0.14445258025079966|
|C seed42|historical_rep01|0.1262373887002468|
|D seed42|historical_rep01|0.14209353271871805|

这四个目录保持不在 `outputs/reproducibility/seed42_same_gpu/historical/` 中，
过滤后的 `historical_replicates.json` 不再列出它们；含该轮指标的旧 aggregate 日志和旧 A 分类文件不恢复。
Git 历史、备份 tag 和原本地恢复备份不抹除、不重写。
这是用户指定的记录管理范围，不是根据统计显著性作出的实验排除结论。

## 此前已完成的安全整理

- 根目录的原始 v3 提案和 E 消融方案移到 `docs/experiments/`，保留正文并加上执行状态说明。
- 建立 `docs/README.md` 导航；重写仓库 README，只展示活动结果入口和仍需保留的对照。
- `.gitignore` 添加 `outputs/**/*.log`，并只为当前 compact prediction/comparison CSV 放行；history CSV 和权重继续忽略。
- 不改变训练源码、实验配置、缓存、运行中目录或已有结果；先行整理阶段不取消跟踪或提交；正式清理阶段取消跟踪日志并生成本地 commit，不自动推送该 commit。

## 清理与纠正的事件记录

原清理在 A/B/E stability gates、最终 seed-matched comparison 和 F 三个 seeds 完成后执行。
原 [cleanup manifest](../outputs/reproducibility/seed42_same_gpu/cleanup_manifest.json) 与提交 `0606430`
保留为当时的事件记录，不能继续将其 retired_paths 当成当前应删除范围。
当前保留策略、恢复文件数和精确排除对象见
[retention correction](../outputs/reproducibility/seed42_same_gpu/retention_correction.json)。
不可续训的共享内存事故失败记录仍按此前要求删除、不归档；本次没有恢复这些失败运行。

## 此前清理检查的 gate（历史事件，非当前删除规则）

1. A/B/E 的 seed42 双 replicate 全部完成，三个 stability gates 都为 stable。
2. `comparison/seed_matched_summary.json`、`.md` 及 `patient_level_differences.csv` 齐全，覆盖 seeds 42/43/44。
3. F42/43/44 全部训练和评估完成，campaign state=complete 且 `F_seed_summary.json` 存在。
   清理将改变被冻结的 core source hash，F 未结束时同样不能执行。
4. 核对 comparison 的 B43/B44 依赖、配置/源码指纹、工作区和 staging area；不得混入无关改动。
5. 创建 `pre-cleanup-2026-10-03` tag，push 到 origin 并验证 tag 对象及目标 commit；失败则不删除。
6. 对未跟踪的 checkpoints、history、CSV 等保留可恢复本地备份；记录备份位置和 current-result checksums。

## 已恢复的实验与配套目录

```text
outputs/stagewise_recursive/
outputs/pure_rrt_v3/
outputs/pure_rrt_v3_clarity_allpair/
outputs/legacy_open_loop_aux/
outputs/diagnostics/pure_rrt_v3_seed42_survival/
configs/legacy/
docs/legacy/
outputs/reproducibility/seed42_same_gpu/historical/
outputs/reproducibility/seed42_same_gpu/historical_replicates.json
outputs/ablations/teacher_forced_stagewise/       # 旧 protocol E42，保留但非当前 E reproducibility
outputs/pure_rrt_v3_step2400/primary/              # A/B/C/D，seed42/43/44 全部保留后续有效记录
```

`historical/` 保留后续四组 ABCD42 historical_rep02 和旧 E42 historical_rep01；
这五个目录是对应原结果目录的完整快照副本，不能计为额外独立 replicate。
只排除上表四个首轮；任何自动入口都不能从 Git 重新引入它们。

`outputs/pure_rrt_v3_step2400/primary/` 恢复完整 A/B/C/D × seeds 42/43/44，共十二组。
其中 A seed42 的 H3 MSE 为 `0.18340921914204955`，不是被排除的 `0.6268193917348981`。
其余三组后续 seed42 为 B `0.15566894924268126`、C `0.13275389280170202`、D `0.15265309531241655`。
原 summary.json/md 汇总的是这些后续复跑，因此也恢复；最终 E vs B comparison 仍保持原路径与结果不变。
F 的 `historical_controls/D_seed42/43/44` 是独立 compact 快照，仅做历史描述性比较，
不能当成新的确定性 D 复跑，也不能自动替换正式比较中的 B43/B44。

## 必须保留

- 完整 A–F 历史与当前执行记录，仅排除精确指定的首轮；历史协议与确定性协议不混合统计。
- 当前 A/B/E 确定性 replicate、status、完整来源记录、stability gates 和最终 comparison。
- F42/43/44 结果、core/extension provenance、seed summary 与历史对照快照。
- 正常运行中的任务和完整评估结果；不可续训的失败目录按新要求删除，不算正式 replicate 结果。
- `/dev/shm/clarity_mri_cache`、原始 MU/UCSF 数据、split 文件和 pretrained checkpoints。
- 用户已有改动和 staging 内容；恢复 tag/备份不使用 hard reset 或历史重写。

## 完成记录与提交

原等待器 `/tmp/clarity_cleanup_after_pipeline_20261004.py` 已于 2026-10-05 18:56（北京时间）
完成并退出；其 status.json 是先前清理的事件状态，不能重新启动并继续整族删除。

其他旧记录已从 `/tmp/clarity-precleanup-20261004-0c_ou6bk` 复制回原目录，
checkpoint、history 与日志完整恢复在本地；Git 只保留 compact artifacts，输出日志仍不跟踪。
当前 A/B/E/F 的 77 份 compact artifacts 校验和保持不变，源码只修正报告说明，不改变训练或评估算法。
完整 [实验记录索引](../outputs/ablation_inventory.json) 列出协议范围、结果路径、指标校验和及重复快照映射。
`pre-cleanup-2026-10-03` tag 已 push；后续更正使用新的本地提交，不 reset、不改写历史，也不自动 push。
