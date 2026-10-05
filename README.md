# CLARITY：Stage-wise、Teacher-forcing 与 Dynamics Ensemble

当前工作包括 A/B/E 的同卡 seed42 确定性复现，以及独立的 F 消融。
正式比较必须等相关 stability gates 和完整评估产物生成；运行进度与失败状态以各输出目录的 JSON 为准，不能把中间 checkpoint 当作完成结果。

## 实验定义

|组|Dynamics 训练方式|成员数|测试部署|
|---|---|---:|---|
|A|官方 CLARITY true-state all-pairs；每个 i<j pair 独立训练|1|递归 H1/H2/H3|
|B|Pure recursive RRT，预测状态反馈到下一阶段|1|递归 H1/H2/H3|
|C|与 A 相同|3|各成员独立递归，再等权融合|
|D|与 B 相同|3|各成员独立递归，再等权融合|
|E|Teacher-forced stage-wise；输入 s0、stopgrad(s1_true)、stopgrad(s2_true)|1|递归 H1/H2/H3|
|F|与 E 相同，加入 C/D 同形式的 dynamics ensemble；不做 RRT 训练|3|各成员独立递归，再等权融合|

Ensemble 只复制 dynamics predictor；MRI encoder、text encoder 和 survival head 共享。
E vs B 检验训练时 predicted-state feedback 的影响；F vs E 检验 teacher-forcing 条件下 ensemble 的影响。
C/D 定义仍保留在主配置中，但旧执行协议结果不是当前确定性复跑。

各组沿用固定患者划分、可训练 MRI/Text encoders、survival/CF objectives，
2400 optimizer steps、240 warmup steps、每 24 steps 验证、batch size 16。
Latent loss 对阶段/成员等权平均，CF 只参与训练、不参与验证；warmup 期间仍保留 CF。
主 checkpoint 是 warmup 后 validation total loss 最小的 `best_val_loss.pt`，相同值保留较早 checkpoint。
`best_val_cindex.pt` 只作诊断，不据测试性能选择 replicate 或 checkpoint。

## 数据与确定性

当前 A–F 消融使用 MU-Glioma-Post，固定患者级划分见 [data/splits.json](data/splits.json)。
train/validation/test 的 trajectory/patient 数为 74/38、11/5、16/8；
A/C 的 train/validation all-pair/patient 数为 382/38、56/5。
治疗信息严格按 MRI 区间构造，不使用未来治疗；prognosis 使用每位患者最早合格窗口。

MRI 从 `/dev/shm/clarity_mri_cache` 的 float32 `.npy` 缓存读取。
缓存与并发 DataLoader 都占用共享内存，应按并发峰值预留空间；磁盘输出目录清理不会释放 `/dev/shm`。
UCSF 数据集的本地内容、目录结构和使用限制见 [数据集说明](docs/UCSF_POSTOP_GLIOMA_DATASET.md)；本轮 F 不切换数据集。

确定性复跑使用 strict deterministic algorithms、cuDNN deterministic、关闭 TF32/benchmark、
math-only SDPA，以及固定 startup seed、cuBLAS workspace 和 DataLoader worker/generator seeds。
metadata 记录 GPU 型号/UUID、PyTorch/CUDA/cuDNN、配置/数据/源码及初始化/批次顺序指纹。
严格确定性不保证跨硬件或库版本逐位一致。

官方 CLARITY commit 固定为 `dadb82241a24f5ec5e4e4dc994e3116fd4a9da04`。
主配置：[configs/pure_rrt_v3.yaml](configs/pure_rrt_v3.yaml)；
E 配置：[configs/ablations/teacher_forced_stagewise.yaml](configs/ablations/teacher_forced_stagewise.yaml)；
F 配置：[extensions/teacher_forced_ensemble/configs/F.yaml](extensions/teacher_forced_ensemble/configs/F.yaml)。
F 使用独立扩展，不修改运行中 A–E 的冻结源码；同时记录 core 和 extension 指纹。

## 文档与目录

全部说明从 [docs 文档索引](docs/README.md) 进入。原始实验提案集中在 `docs/experiments/`，
当前实现与原始提案不一致时，以活动配置和执行协议为准。

```text
src/clarity_rrt_v3/                     # A–E 核心实现，实验运行期间冻结
configs/                              # 主 factorial 与 E 配置
scripts/                              # 训练及 reproducibility 入口
extensions/teacher_forced_ensemble/    # F 的隔离实现、配置与测试
data/splits.json                       # 固定患者划分，不存放原始 MRI
docs/                                 # 数据集、实验方案、执行协议及清理计划
outputs/
├── reproducibility/seed42_same_gpu/   # A/B/E 当前复现性审计
│   ├── primary/                      # 正式 replicate 与独立 metadata
│   ├── recovery_shm.json             # 重跑队列、资源限制及当前协调状态
│   ├── {A,B,E}_seed42_stability.json  # 生成后才可判断稳定性
│   └── comparison/                   # 全部 gate 通过后才生成正式配对比较
├── ablations/teacher_forced_stagewise_ensemble/
│   ├── primary/                      # F42/43/44
│   ├── historical_controls/          # D compact 快照，仅作历史描述性对照
│   └── campaign_status.json
└── pure_rrt_v3_step2400/primary/B_seed{43,44}/  # 最终 comparison 仍依赖的临时对照
```

最终 E42/43/44 vs B42/43/44 比较 H1/H2/H3、H3/H1 及 patient-level differences。
seed42 正式使用预先指定的 rep01；rep02 仅检查复现性，不能计为另一个 seed。
B43/44 与历史 D 对照存在执行协议差异，应明确标注；3 个 seeds、8 位 test 患者的分析仍属探索性。

## 验证与运行

```bash
conda run -n py310 env PYTHONPATH="$PWD/src:$PWD/extensions/teacher_forced_ensemble" \
  python -m pytest -q tests extensions/teacher_forced_ensemble/tests
```

A/B/E 顺序复现入口为 `scripts/reproducibility/run_same_gpu.sh`；
F 的资源感知调度入口为 `python -m clarity_tf_ensemble campaign --gpus 5 6 7`，
需要设置上述 PYTHONPATH。详细规则见 [复现性方案](docs/seed42_reproducibility_audit.md)
和 [F 实验方案](docs/teacher_forced_stagewise_ensemble_F_plan.md)。
已有 run 不覆盖、不静默续跑；失败后不能直接重复启动同一个 campaign。
恢复协调流程前应核对残留进程、锁、源码指纹和共享内存容量。
用户于 2026-10-04 指定：无法完整续训的失败记录删除、不归档，再从头重跑；完整结果不覆盖。
当前恢复调度合计最多 3 个 E/F 任务，启动前要求至少 80 GiB 可用共享内存；
E42 rep01/rep02 优先在原 GPU0 连续运行，不与其他用户抢占忙卡。

## 结果与清理规则

Git 只提交 compact artifacts：config、metadata、metrics、gates、报告、患者预测和配对 CSV。
`outputs/**/*.log`、权重和训练 history CSV 留在本地；已经 tracked 的日志须在最终清理提交中取消跟踪。

退役目录清理的前置条件、精确目标和安全步骤见 [清理计划](docs/repository_cleanup_plan.md)。
在 A/B/E gates、最终 comparison 和 F 完成之前，不删除旧结果，也不修改冻结源码。
真正删除前须创建并推送 `pre-cleanup-2026-10-03` tag，验证远端成功，并为未跟踪文件保留可恢复本地备份。
只有 `outputs/reproducibility/seed42_same_gpu/cleanup_manifest.json` 存在且清理状态完成，才表示正式清理已执行。
