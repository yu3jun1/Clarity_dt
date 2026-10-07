# CLARITY：Stage-wise、Teacher-forcing 与 Dynamics Ensemble

新增研究：[All-pair Ensemble 可靠性与成员数 v1](docs/ensemble_reliability_size_v1.md)，
使用独立 `clarity_ensemble_study` 入口及 `outputs/ensemble_reliability_size_v1/`；
读取旧 A/C checkpoint 重评估，再训练 M2/M5，不与 A–F 输出混用。

后续研究：[A/C End-to-End Outcome Transfer v1](docs/outcome_transfer_ac_v1.md)，
冻结 A/C checkpoint，对比真实与预测 latent 的 H3 生存结果，输出患者配对误差和 disagreement–survival error 分析。

当前训练采用普通 seeded training，数值执行行为回到历史 step2400 实验对应的
`9871306` 基线；没有默认/显式 strict 两套模式，也不再提供严格确定性训练入口。
保留实验设计、训练预算、共享缓存和来源记录，不承诺同 seed 重跑逐位一致。
历史确定性审计的配置、metadata 和 gate 仅用于追溯，不改写成新协议的记录。

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
F 在训练和 checkpoint validation 使用真实前状态，测试仍是各成员独立递归后等权融合。

各组沿用固定患者划分、可训练 MRI/Text encoders、survival/CF objectives，
2400 optimizer steps、240 warmup steps、每 24 steps 验证、batch size 16。
Latent loss 对阶段/成员等权平均，CF 只参与训练、不参与验证；warmup 期间仍保留 CF。
主 checkpoint 是 warmup 后 validation total loss 最小的 `best_val_loss.pt`，相同值保留较早 checkpoint。
`best_val_cindex.pt` 只作诊断，不据测试性能选择 replicate 或 checkpoint。

## 数据、随机种子与来源记录

当前 A–F 消融使用 MU-Glioma-Post，固定患者级划分见 [data/splits.json](data/splits.json)。
train/validation/test 的 trajectory/patient 数为 74/38、11/5、16/8；
A/C 的 train/validation all-pair/patient 数为 382/38、56/5。
治疗信息严格按 MRI 区间构造，不使用未来治疗；prognosis 使用每位患者最早合格窗口。

MRI 从 `/dev/shm/clarity_mri_cache` 的 float32 `.npy` 缓存读取。
缓存与并发 DataLoader 都占用共享内存，应按并发峰值预留空间；磁盘输出目录清理不会释放 `/dev/shm`。
UCSF 数据集的本地内容、目录结构和使用限制见 [数据集说明](docs/UCSF_POSTOP_GLIOMA_DATASET.md)；
新增数据集说明不表示本轮 A–F 自动切换数据集。

训练固定 Python、NumPy、PyTorch 随机种子及训练 DataLoader 的 shuffle generator，
不额外启用严格 deterministic algorithms、强制 cuDNN/TF32/SDPA 数值模式或严格 worker seeding。
GPU 算子、硬件和库版本仍可能影响重跑结果；相同 seed 不等于逐位一致。
建议从新进程启动；入口不会主动清除父 shell 已设置的环境变量，
也不重置同一进程先前被其他代码改变的 backend 设置。
metadata 继续记录 GPU 型号/UUID、PyTorch/CUDA/cuDNN、配置/数据/源码等来源信息。
读取旧保存配置时，内存中忽略旧 `deterministic` 键，不据此启用另一种执行模式，
也不写回或改写历史配置和结果。

官方 CLARITY commit 固定为 `dadb82241a24f5ec5e4e4dc994e3116fd4a9da04`。
主配置：[configs/pure_rrt_v3.yaml](configs/pure_rrt_v3.yaml)；
E 配置：[configs/ablations/teacher_forced_stagewise.yaml](configs/ablations/teacher_forced_stagewise.yaml)；
F 配置：[extensions/teacher_forced_ensemble/configs/F.yaml](extensions/teacher_forced_ensemble/configs/F.yaml)。
F 使用独立扩展，每次 campaign 冻结启动时的 current core/extension 指纹，
不依赖旧 reproducibility audit 的 protocol 文件。

## 文档与目录

全部说明从 [docs 文档索引](docs/README.md) 进入。原始实验提案集中在 `docs/experiments/`，
当前实现与原始提案不一致时，以活动配置和执行协议为准。

```text
src/clarity_rrt_v3/                     # A–E 核心实现与历史结果分析工具
configs/                              # 主 factorial 与 E 配置
scripts/                              # 常规训练入口；旧严格复跑入口已退役
extensions/teacher_forced_ensemble/    # F 的隔离实现、配置与测试
data/splits.json                       # 固定患者划分，不存放原始 MRI
docs/                                 # 数据集、实验方案、历史执行与整理记录
outputs/                              # 本地实验结果；新实验使用新的 output-root
```

历史审计曾生成 A/B/E seed42 双 replicate、E42/43/44 vs B42/43/44 的配对比较，
以及 F42/43/44 的汇总。这些是历史执行事实，不表示整理或迁移后的每个目录仍存在。
旧 `outputs/pure_rrt_v3_step2400/` 是当前普通 seeded 行为的参考记录；
旧严格协议的记录不能直接与新普通运行混合成 execution-protocol-matched comparison。
历史 stability gate 只解释当时那组结果，不是新普通训练或 F campaign 的启动前提。
历史 retention/cleanup manifest 是操作事件，不用它们自动恢复用户已经删除或迁移的目录。

## 验证与运行

```bash
conda run -n py310 env PYTHONPATH="$PWD/src:$PWD/extensions/teacher_forced_ensemble" \
  python -m pytest -q tests extensions/teacher_forced_ensemble/tests
```

下面只是启动示例，先确认 GPU 空闲且共享内存足够，并使用 replicate 隔离的新输出目录。
训练和评估使用同一个 config、seed、replicate 与 output-root；不要覆盖旧记录。

```bash
conda run -n py310 env CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PWD/src" \
  python -m clarity_rrt_v3.train --config configs/pure_rrt_v3.yaml \
  --variant B --seed 42 --device cuda:0 --replicate rep01 \
  --output-root outputs/runs/step2400_seeded_20261005

conda run -n py310 env CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PWD/src" \
  python -m clarity_rrt_v3.evaluate run --config configs/pure_rrt_v3.yaml \
  --variant B --seed 42 --device cuda:0 --replicate rep01 \
  --output-root outputs/runs/step2400_seeded_20261005

conda run -n py310 env PYTHONPATH="$PWD/extensions/teacher_forced_ensemble:$PWD/src" \
  python -m clarity_tf_ensemble campaign --gpus 5 6 7 \
  --output-root outputs/ablations/teacher_forced_stagewise_ensemble_seeded_20261005
```

A/C/D 使用主配置及对应 `--variant`；E 使用 E 配置及 `--variant E`。
旧 `scripts/reproducibility/run_same_gpu.sh` 和严格复跑 pipeline 不再作为训练入口。
F 的资源感知调度和实验设计见 [F 实验方案](docs/teacher_forced_stagewise_ensemble_F_plan.md)；
历史同卡复现性事实见 [已退役审计说明](docs/seed42_reproducibility_audit.md)。
上述 `--replicate rep01` 训练入口和 F campaign 不覆盖已有 run、不静默续跑；
恢复前应核对进程、锁、源码指纹和共享内存容量。
weights-only checkpoint 缺少 optimizer/scheduler/RNG 状态时，不能当作精确断点续训。

## 结果管理

Git 只提交 compact artifacts：config、metadata、metrics、报告、患者预测和配对 CSV。
`outputs/**/*.log`、权重和训练 history CSV 留在本地，不提交完整训练日志。
实验完成状态以训练预算、评估 metadata 和完整结果产物为准，不能把中间 checkpoint 当作完成结果。
配对比较应匹配 seed、数据划分、训练预算和执行协议；replicate 不算额外独立 seed，
不能按测试指标择优。3 个 seeds、8 位 test 患者的分析仍属探索性。
之前目录整理与保留决策见 [历史整理记录](docs/repository_cleanup_plan.md)；
本次训练行为调整不整理、恢复或改写 outputs。
