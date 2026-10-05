# 仓库清理与目录整理

## 此前已完成的安全整理

- 根目录的原始 v3 提案和 E 消融方案移到 `docs/experiments/`，保留正文并加上执行状态说明。
- 建立 `docs/README.md` 导航；重写仓库 README，只展示活动结果入口和仍需保留的对照。
- `.gitignore` 添加 `outputs/**/*.log`，并只为当前 compact prediction/comparison CSV 放行；history CSV 和权重继续忽略。
- 不改变训练源码、实验配置、缓存、运行中目录或已有结果；先行整理阶段不取消跟踪或提交；正式清理阶段取消跟踪日志并生成本地 commit，不自动推送该 commit。

## 正式清理已完成

执行前 A/B/E stability gates、最终 seed-matched comparison 和 F 三个 seeds 均已完成。
不可续训的失败记录已按用户要求删除、不归档；本次清理不是对其性能做统计排除。
实际退役路径、当前结果校验和、远端 backup tag 及可恢复本地备份见
`outputs/reproducibility/seed42_same_gpu/cleanup_manifest.json`。

## 正式清理的 gate

1. A/B/E 的 seed42 双 replicate 全部完成，三个 stability gates 都为 stable。
2. `comparison/seed_matched_summary.json`、`.md` 及 `patient_level_differences.csv` 齐全，覆盖 seeds 42/43/44。
3. F42/43/44 全部训练和评估完成，campaign state=complete 且 `F_seed_summary.json` 存在。
   清理将改变被冻结的 core source hash，F 未结束时同样不能执行。
4. 核对 comparison 的 B43/B44 依赖、配置/源码指纹、工作区和 staging area；不得混入无关改动。
5. 创建 `pre-cleanup-2026-10-03` tag，push 到 origin 并验证 tag 对象及目标 commit；失败则不删除。
6. 对未跟踪的 checkpoints、history、CSV 等保留可恢复本地备份；记录备份位置和 current-result checksums。

## 退役目标（实际执行路径见 manifest）

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
outputs/reproducibility/seed42_same_gpu/old_A_replicate_classification.json
outputs/ablations/teacher_forced_stagewise/       # 旧 protocol E42，非当前 E reproducibility
```

`historical/` 完整删除，包括同一轮旧 A/B/C/D seed42；不自动从 Git 重新创建它们。
删除属于用户指定的目录退役，不是根据测试性能做统计排除的证据。

`outputs/pure_rrt_v3_step2400/` 根据最终 comparison 的真实依赖决定：
若仍引用 B43/B44，仅保留这两组必要对照，其他内容退役；没有依赖则整族退役。
F 的 `historical_controls/D_seed42/43/44` 是独立 compact 快照，仅做历史描述性比较，
不能当成新的确定性 D 复跑，也不能自动替换正式比较中的 B43/B44。

## 必须保留

- 当前 A/B/E 确定性 replicate、status、完整来源记录、stability gates 和最终 comparison。
- F42/43/44 结果、core/extension provenance、seed summary 与历史对照快照。
- 正常运行中的任务和完整评估结果；不可续训的失败目录按新要求删除，不算正式 replicate 结果。
- `/dev/shm/clarity_mri_cache`、原始 MU/UCSF 数据、split 文件和 pretrained checkpoints。
- 用户已有改动和 staging 内容；恢复 tag/备份不使用 hard reset 或历史重写。

## 等待器与提交

当前等待器为 `/tmp/clarity_cleanup_after_pipeline_20261004.py`，
状态在 `/tmp/clarity_cleanup_after_pipeline_20261004/status.json`。
它必须在恢复未完成时保持旧实验族删除关闭，随后重新检查所有 gates；
不得只因为进程消失或显卡空闲就执行清理。

真正清理使用明确验证过的目标并优先移动到本地备份，生成
`outputs/reproducibility/seed42_same_gpu/cleanup_manifest.json`。
先在独立临时副本验证清理后的代码，再执行并复验；compact artifacts 的 checksums 应保持不变。
已有 output logs 最后才执行 `git rm --cached`，本地活跃日志保留。
按之前要求生成本地清理 commit；清理 commit 的 GitHub push 不自动推断为已获批准。
tag push 与 cleanup commit push 是不同操作，最终报告需分别说明状态。
