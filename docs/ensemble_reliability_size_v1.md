# All-pair Ensemble：可靠性与成员数实验 v1

本轮依据根目录的 `Ensemble_Reliability_and_Size_Analysis_Experiment_Plan.md`，
只研究 all-pair ensemble；不增加 RRT、teacher-forcing、diversity/uncertainty loss 或新生存头。
不启动 UCSF external validation 或 treatment planning。

## 实验组与顺序

|本轮标签|成员数|Seed|来源与操作|
|---|---:|---|---|
|M1|1|42/43/44|读取 step2400 的 A 主 checkpoint，仅重评估|
|M3|3|42/43/44|读取 step2400 的 C 主 checkpoint，仅重评估|
|M2|2|42/43/44|新增三次训练，每次结束后立即评估|
|M5|5|42/43/44|新增三次训练，每次结束后立即评估|

文档中的 E1/E2/E3/E5 在本轮记为 M1/M2/M3/M5，避免与旧 Teacher-forced E 混淆。
先完成六份 A/C checkpoint 的重评估并生成 Step 1 报告，再开始六次 M2/M5 训练。
不因 Step 1 的指标好坏选择 seed、checkpoint 或改变后续训练预算。

固定 MU-Glioma-Post、原患者划分、encoders、生存/CF objectives、优化器及超参数；
2400 optimizer steps、warmup 240、每 24 steps 验证、batch16、num_workers=2。
MRI 使用 `/dev/shm/clarity_mri_cache`。普通 seeded training，不启用严格确定性模式。
主 checkpoint 仍是 warmup 后最小 validation total loss，相同值保留较早者。
Ensemble 只复制 dynamics；encoder 和 survival head 共享，成员独立递归后等权融合。

## 有限的评估与解释

每个 seed、每个 horizon 单独计算以下指标，不混合不同 seed 的 latent 坐标：

- 性能：H1/H2/H3 latent MSE、Cosine；跨三 seed 的均值和样本标准差，H3 MSE std 用于稳定性比较。
- `Uz = mean_member sum_token,feature((z_member - mean_member(z))²)`。
- `Us = Var_member(sigmoid(logit))`，使用 population variance。
- Pearson/Spearman 相关的误差为 `L2(mean_z - true_z)`，不是 MSE。
- Risk–coverage 只取 100/80/60/40%，按 Uz 升序保留 `ceil(coverage × N)` 个窗口，报告 MSE。
- 成员 MSE/Cosine、成员预测向量 flatten 后的两两 Pearson，以及 averaging gain。

M1 的 disagreement 恒零，相关性在数学上未定义，记为 `null`，不画不确定性筛选曲线。
Us 与 latent error 的相关只作辅助信号，不解释为 survival calibration。
原 C-index/Brier365 保留在每份 metrics，仍遵循原 primary survival window 规则。

可靠性分析单位是原 test trajectory-window × horizon（16 个窗口、8 位患者），
不是独立患者试验。报告不增加显著性检验、bootstrap、置信区间、AURC 或新校准模型。
跨 seed 的相关汇总只是 seed 内相关的平均值。不同模型的 MRI encoder 都有微调，
latent 绝对标尺可能不同；不凭单一 MSE 或三个 seed 宣称医疗可靠性得到验证。

## 独立代码与输出

新入口：`src/clarity_ensemble_study/`；新配置：`configs/ensemble_reliability_size_v1.yaml`。
直接复用核心 `train_run` / `evaluate_run` 的训练数学和基础评估，不用 F 的 runtime
monkeypatch，不伪装成 A/C variant，不添加旧路径 fallback 或新兼容层。

```text
outputs/ensemble_reliability_size_v1/
├── protocol.json
├── campaign_status.json
├── M{size}_seed{seed}.log
├── runs/M{size}_seed{seed}/
│   ├── config.yaml / run_metadata.json       # 新训练生成
│   ├── evaluation_metadata.json             # 源 checkpoint 路径、SHA256、GPU/软件版本
│   ├── metrics.json / recursive_predictions.csv
│   ├── best_val_loss.pt                     # 仅 M2/M5；本地权重，不提交 Git
│   └── status.json
└── reports/
    ├── step1_reliability/                    # M1/M3
    └── step2_size/                           # M1/M2/M3/M5
```

报告包括性能表、相关表、四点 risk–coverage 表、M3 各 seed 的 H1–H3 散点图和 coverage 曲线。
成员指标与 pairwise diversity 存在 compact summary；不引入额外统计流水线。
旧 A–F 配置、输出和 checkpoint 不覆写，不复制历史权重到新目录。

## 启动与自动评估

```bash
bash scripts/ensemble_reliability_size/run.sh 2
```

每份重评估在独立子进程运行；每次新训练后自动评估其主 checkpoint。
Step 1 和最终 Step 2 各完成后自动生成对应报告。没有重试或自动续训。
2026-10-05 启动前检查确认 GPU4–7 忙、GPU2 空闲，因此用 GPU2 单卡顺序执行；
不抢占或停止其他任务。SHM 缓存 446 条完整，约 59.3 GiB，启动前剩余约 193 GiB。

2026-10-06 根据用户要求，将剩余的 M5 seed44 放到 GPU1，与 GPU2 的 M5 seed43
并行运行，配置、训练预算和 SHM 缓存不变。仅停止原顺序 campaign 主进程，
不停止 seed43 worker；seed44 使用同一 `worker` 入口训练并自动评估。
`finalize` 入口每 30 秒检查全部 12 组的完成状态，刷新 `campaign_status.json`
中的 `active_runs`，全部完成后生成最终 Step 2 报告，避免顺序队列再次启动 seed44。
运行进程 PID 与 GPU 在 worker 状态及运行 metadata 中记录。

停止原顺序 campaign 后，独立 worker 和收尾入口的命令分别为：

```bash
bash scripts/ensemble_reliability_size/run.sh 1 worker --size 5 --seed 44
bash scripts/ensemble_reliability_size/run.sh 2 finalize
```
