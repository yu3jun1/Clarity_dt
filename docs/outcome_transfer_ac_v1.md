# A vs C: End-to-End Outcome Transfer Analysis v1

## 研究问题与固定设置

检验 A/C 的 latent 预测差异是否传递到下游生存预测。
读取 `outputs/pure_rrt_v3_step2400/primary/{A,C}_seed{42,43,44}/best_val_loss.pt`，
沿用上一轮 ensemble study 的同一批 checkpoint。只做冻结模型推理，不新增训练或选择 checkpoint。
同一 seed 的 A/C 配对，MU-Glioma-Post 和 `data/splits.json` 不变。
MRI 继续使用 `/dev/shm/clarity_mri_cache/`。软件/GPU、源码、数据与 checkpoint SHA256
写入本轮 metadata。使用普通 seeded 推理。

## 三类 survival 输入

1. True latent：用各 checkpoint 的 MRI encoder 编码真实 endpoint MRI，再输入该模型的 survival head。
2. A predicted latent：单成员独立递归 H1→H3，输入 A 的 survival head。
3. C predicted latent：三个成员各自递归，各自输入 C 的共享 survival head，再平均 risk 和 sigmoid 概率。

所有比较固定初始 latent、相应 treatment prefix、clinical context 和本模型 head，只改变 endpoint latent 来源。
A/C 的 encoder/head 分别训练，故 true latent 有 **A-head 和 C-head 两个对照**。
不把一个模型的 latent 直接喂给另一个模型的 head；本轮不是共同 head 下的跨 latent-space 测试。
True latent 是 observed-state 参考，不能假定其指标必然最好，也不用于模型选择。

C 另有一项辅助 fusion control：先平均 predicted latent 再过 head。
它与主部署的“先过 head 再平均结果”不等价；主 A/C 比较始终沿用原部署融合方式。
真实未来 MRI 仅用于 latent target/oracle，不反馈进入递归 rollout。

## 五项分析

|分析|计算单位与输出|
|---|---|
|A/C H1–H3 latent MSE|先给出原 16 trajectory windows 等权指标，跨三个 seeds 汇总 mean±sample SD|
|A/C H3 C-index、Brier365|每患者最早合格四阶段窗口，8 位 test 患者；使用原 C-index 实现和训练参考 IPCW Brier|
|True / A predicted / C predicted|分别记录 H3 生存指标及本模型 predicted−true 差值；另报 C mean-latent 辅助控制|
|Patient-level paired error|先对同一患者同一 horizon 的 windows 平均 latent MSE，再计算 C−A；生存误差使用 primary H3 的 IPCW Brier 贡献及已知结局的平方误差|
|C disagreement vs survival error|每 seed 分别计算 Uz/Us 与已知 365 天结局平方误差的 Pearson/Spearman；保存散点图|

`Uz = sum_token,feature Var_member(z_pred)`；`Us = Var_member(sigmoid(logit))`，均为 population variance。
生存误差为 `(p(alive365) − I(T>365))²`，不是 latent MSE，也不把 risk score 当作概率。
365 天前/当天删失者结局未知，平方误差记为 null，排除于相关性计算。
IPCW Brier 沿用 sksurv 约定：事件≤365 的贡献为 `p²/G_train(T)`；T>365 为
`(1−p)²/G_train(365)`；早期删失贡献为 0。所有 test 患者贡献的均值就是 Brier365，
不是按非零权重重新归一化。该 0 不表示该患者预测正确。
训练删失分布仅来自 train 患者的 primary H3 landmark。

H3 的 survival_time 从 **H3 endpoint** 起算，Brier365 对应 endpoint 后 365 天，
不等同于初始 MRI 后 365 天。治疗条件沿用相应观测区间的 prefix；这是回顾性条件化 outcome transfer。
C-index 是 cohort 层面的排序指标，没有单患者 C-index error。
配对前严格匹配 seed、患者、完整窗口、horizon 和生存标签。
只做上述描述性指标，不增加 p-value、bootstrap、显著性检验或模型重新校准。
不同模型 latent 标尺可能不同；A/C 还包含 head 的训练差异，不能将所有跨模型改进单独归因于 dynamics。

## 代码、启动与输出

独立入口 `src/clarity_outcome_transfer/`，配置 `configs/outcome_transfer_ac_v1.yaml`。
复用核心 `evaluate_run` 加载 checkpoint、数据和 native metrics，新 predictor 同时评估 observed-state 参考。

```bash
# 启动前确认这些卡空闲；三个 seeds 并行，先 A 再 C，每进程只加载一个 checkpoint。
bash scripts/outcome_transfer/run.sh 0 1 2
```

也可只传一张 GPU，六组按顺序完成。所有评估成功后自动生成最终报告。
输出目录必须是新目录；不存在自动覆盖、重试或续训。

```text
outputs/outcome_transfer_ac_v1/
├── protocol.json / campaign_status.json
├── {A,C}_seed{42,43,44}.log
├── runs/{A,C}_seed{42,43,44}/
│   ├── config.yaml / evaluation_metadata.json / status.json
│   ├── metrics.json                 # native metrics + outcome_transfer + patient records
│   └── recursive_predictions.csv    # 三个 horizons 的预测、true reference、disagreement
└── reports/
    ├── summary.json / report.md
    ├── patient_paired_errors.csv
    └── C_disagreement_survival_error.png
```

Git 保留 compact config、metadata、metrics、预测 CSV、配对 CSV 和报告；日志和实时状态忽略。
8 个 test 患者的三个 seed 仍是同一批患者；相关性只描述这批数据，不作泛化或临床可靠性结论。
