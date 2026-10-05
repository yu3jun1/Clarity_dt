# F：Teacher-forced Stage-wise Ensemble

## 1. 实验定义与消融问题

F = E 的 Stage-wise + Teacher-forcing + C/D 同形式的 3-member dynamics ensemble。
**不使用 Recursive Rollout Training（RRT）**。沿用当前 MU-Glioma-Post 实验数据；
本轮不引入 UCSF 数据集，不改变 E 的患者划分或训练预算。

|组别|训练输入|Dynamics 成员数|用途|
|---|---|---|---|
|E|每个阶段的真实前状态|1|F 的主要 seed-matched 对照|
|F|每个阶段的真实前状态|3|隔离 teacher-forcing 条件下 ensemble 的影响|
|D|前阶段的预测状态递归反馈|3|辅助比较训练反馈机制；现有 D 为历史协议结果|

E vs F 的唯一模型设计差异是 dynamics 成员数。F 的 ensemble 实现直接复用
`clarity_rrt_v3.model.StagewiseDynamics`，与 C/D 使用同一代码、初始化和融合方式。
MRI encoder、text encoder、survival head 共享；只有三个 dynamics predictor 独立。
因此 F 不是三个完整 CLARITY 模型分别训练后再做后处理融合。

## 2. Teacher-forced 前向与梯度

对每个成员 m，三个阶段分别计算：

```text
pred[m,1] = dynamics[m](s0,                 condition1, delta1)
pred[m,2] = dynamics[m](stopgrad(s1_true),  condition2, delta2)
pred[m,3] = dynamics[m](stopgrad(s2_true),  condition3, delta3)
```

目标为 `stopgrad(s1_true/s2_true/s3_true)`。与 E 一致，真实 s1/s2 作为输入时
也停止梯度；训练及 checkpoint validation 均不把预测状态反馈到下一阶段。
训练通过同一 `Trainer.step` 的 `teacher_forced_stagewise` 分支完成。

- Latent L1 对三个阶段、三个成员等权平均，不能因成员数从 1 增至 3 而直接把 loss 乘三。
- 生存头使用每个成员的 H3 预测、初始 s0 及 full-condition，Cox/BCE 对成员平均。
- Counterfactual loss 使用各阶段真实前状态，对成员和阶段平均，与 E/C/D 保持一致。
- Warmup 前 240 steps 使用 latent + counterfactual；之后使用
  `0.5 × latent + Cox + BCE + counterfactual`。验证不计算 counterfactual。

**不做 RRT 训练，不等于不做递归测试。** 所有组使用统一的 deployment-style 测试：
每个成员独立从 s0 递归预测 H1→H2→H3，之后等权融合状态和风险。
生存概率取各成员 sigmoid 概率的均值。不能给 F 的测试喂入真实 s1/s2，
也不能把平均后的 ensemble 状态反馈给各成员。额外报告成员级误差及 ensemble uncertainty。

## 3. 固定协议

配置：`extensions/teacher_forced_ensemble/configs/F.yaml`。启动时严格检查
F 的 data/model/training/upstream 设置逐项等于当前 E 配置；发现漂移即拒绝运行。

|项目|固定值|
|---|---|
|Seeds|42、43、44，各一次正式 `rep01`；不把 replicate 当成独立 seed|
|MRI cache|`/dev/shm/clarity_mri_cache`；启动前检查完整 manifest，记录 SHA256|
|划分|沿用 `data/splits.json`，按 patient 划分；实际 counts 写入 metadata|
|预算|2400 optimizer steps，warmup 240，每 24 steps 验证|
|Batch size|训练 16，测试 16；num_workers=2|
|学习率|dynamics/survival 2e-4；text 1e-4；text projection 5e-4；vision 5e-5|
|其他|dropout 0.3；survival weight decay 0.01；gradient clipping 2.0；CF cosine margin 0.9|
|主 checkpoint|warmup 后最小 validation total loss 的 `best_val_loss.pt`；相同值保留较早 checkpoint|
|诊断 checkpoint|val C-index checkpoint 仅诊断，不据测试结果选模型|
|确定性|strict deterministic algorithms；TF32 关闭；SDP math-only；固定 Python/NumPy/Torch/DataLoader seeds|

每个 run/evaluation metadata 记录 GPU 型号及 UUID、PyTorch/CUDA/cuDNN、
包版本、数据/配置指纹、训练批次顺序、初始模型指纹和有效预算。
严格确定性只控制当前实现和环境，不承诺不同硬件、库版本之间逐位一致。
F 仍有三倍 dynamics 分支，耗时可以高于 E；相同 step 数并非相同 GPU 时间预算。

## 4. 与运行中 E 隔离的实现

当前 E 的复现性审计冻结了 `src/`、`configs/`、`scripts/` 和 upstream Predictor 源码。
修改这些文件会使运行中的 E source gate 或训练/评估一致性检查失效。
因此 F 在 `extensions/teacher_forced_ensemble/` 中注册自己的 experiment kind，
只在 F 子进程内部扩展 validator/metadata hook；不修改 A–E variant map 或核心文件。

F 的 `source_sha256` 是核心源码 SHA256 与扩展源码 SHA256 的组合；
metadata 同时保存 `core_source_sha256`、`extension_sha256` 和扩展逐文件 SHA256。
campaign 冻结指纹，训练前、训练后及评估时检查，不允许运行期间改源码。
与现有 E 审计共用的延迟清理必须等待 F 完成后才能执行，以免破坏 source gate。

## 5. 调度、路径及启动

预设 F42→GPU5、F43→GPU6、F44→GPU7。三张卡具备资源时并行训练。
已有 E42/E43/E44 或其他用户任务保持运行，不停止任何既有 GPU 进程。
默认只在对应 GPU 空闲（显存占用≤1 GiB、利用率≤5%）时启动；每 30 秒检查。
如用户明确批准共享，才允许在剩余显存≥60000 MiB 的卡上启动。
共享会延长双方任务时长，且显存需求仍须在实际启动后监控，不能视作固定保证。

```bash
export PYTHONPATH="$PWD/extensions/teacher_forced_ensemble:$PWD/src"
/home/tanyuejun/miniconda3/envs/py310/bin/python -m clarity_tf_ensemble campaign --gpus 5 6 7
```

资源共享需明确批准后传 `--allow-shared-gpu`，或由获批准操作写入 campaign
输出目录的 `shared_gpu_approval.json`。没有批准时不自动共享。
campaign/每个 seed 使用独立锁；不覆盖已有 protocol、run 或日志，也不静默续跑。
训练 worker 使用独立 session，避免 coordinator/session 结束连带停止训练。

```text
outputs/ablations/teacher_forced_stagewise_ensemble/
├── protocol.json                     # 冻结来源、cache、GPU 分配及实际启动记录
├── campaign_status.json              # waiting_for_gpu / running / complete / failed
├── F_seed_summary.json               # 三个 seed 完成后汇总
├── historical_controls/              # D42/43/44 compact 快照及校验 manifest
└── primary/
    ├── F_seed42_rep01/
    ├── F_seed43_rep01/
    └── F_seed44_rep01/
```

每个 run 包含 config、status、train/evaluate logs、history、metadata、checkpoints、
metrics 和 recursive predictions。后续 Git 仅保留 compact artifacts，不提交大权重或完整日志。
2026-10-04 恢复修订：如果失败 checkpoint 缺少完整 optimizer/scheduler/RNG 状态，
按用户要求停止对应进程并删除失败 run，不归档；确认目录已删除后，从头重跑预先指定的 rep01。
完整正式结果仍不覆盖；不能将缺少恢复状态的 weights-only checkpoint 当作精确断点续训。
当前 E/F 联合恢复调度最多 3 个并发任务、每次启动前至少 80 GiB 可用 SHM，
E42 两次仍固定原 GPU0，启动需要对应卡空闲。恢复状态见
`outputs/reproducibility/seed42_same_gpu/recovery_shm.json`。

## 6. 结果判断

1. 等 A/B/E stability gates 和 F42/43/44 全部完成，再报告正式 E vs F。
   E42 使用事先指定 rep01，rep02 仅复现性诊断；不能按测试指标择优。
2. 做 `E42/43/44 vs F42/43/44` seed-matched：H1/H2/H3 latent MSE、cosine、
   H3/H1 MSE 比值，以及 C-index、one-year AUC（遵循同一可计算性规则）。
3. 在同一患者/trajectory 上配对，报告 `F − E`（MSE 负值代表改善），
   先在患者内聚合 trajectory，再汇总患者差异。患者 bootstrap 不把重复 trajectory
   当成独立患者；三个 seed 不足以支持强显著性结论。
4. 报告成员误差、ensemble uncertainty、checkpoint step、训练时间和 GPU 差异，
   区分 ensemble 融合收益与单成员 dynamics 质量变化。
5. D 历史结果可辅助描述：campaign 在旧目录清理前保存 D42/43/44 compact 快照。
   它们不是同一确定性执行协议的重跑，不能作为严格控制下的 teacher-forcing 因果比较；
   正式 F vs D 结论需后续获批准的 matched D rerun。
6. MRI encoder 会微调，各 run 的 latent 表征不同；latent MSE 的绝对尺度可能受
   表征改变影响。与 cosine、患者差异、生存指标和成员诊断共同解释，不能仅依据
   单个 seed 的 H3 MSE 宣称 F 更优。F 本轮没有双 replicate stability gate，
   不能把 E 的 gate 通过直接视为 F 自身的重复稳定性证据。
