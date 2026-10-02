# CLARITY All-Pair vs Pure Stage-wise RRT v3

当前活动实验在 [Clarity_dt_next_experiment_plan_v3.md](Clarity_dt_next_experiment_plan_v3.md) 基础上冻结为 2×2 factorial design：A/C 使用官方 CLARITY 的 true-state all-pairs 训练；B/D 使用 predicted-state recursive RRT 作为主要 dynamics 训练方式。

旧版配置归档在 [configs/legacy/stagewise_recursive_auxiliary_v2.yaml](configs/legacy/stagewise_recursive_auxiliary_v2.yaml)，旧结果保留在 `outputs/stagewise_recursive/`。此前 endpoint/open-loop A/C 的 v3 结果保留在 `outputs/pure_rrt_v3/`；epoch-based factorial 结果冻结在 `outputs/pure_rrt_v3_clarity_allpair/`。当前 step-fair 实验只写入 `outputs/pure_rrt_v3_step2400/`，不会修改或覆盖上述结果。
活动 Python 包位于 `src/clarity_rrt_v3/`；旧 `src/clarity_rrt/` 入口已删除，未保留兼容转发层。


## 实验定义

四组使用相同患者划分、可训练 MRI Encoder、Text Encoder、SurvivalModule、优化器、学习率、optimizer-step 预算与 checkpoint 选择规则。A/C 对每个 all-pair 使用目标端 survival supervision；B/D 使用 H3 survival supervision。Ensemble 只复制 dynamics predictor。两个固定因素分别是 dynamics training（`clarity_all_pair` / `pure_recursive`）与 ensemble size（1 / 3）。训练和评估入口会断言下面的完整矩阵，配置漂移会在模型构建前停止。

| 组 | Dynamics 主训练方式 | 成员数 |
|---|---|---:|
| A: CLARITY-AllPair | 对同一合格患者完整时间线枚举全部 `i<j` pair：`F(s_i^true,c_ij,Δt_ij) -> s_j^true` | 1 |
| B | `s0 -> ŝ1 -> ŝ2 -> ŝ3`，预测状态直接进入下一步 | 1 |
| C: CLARITY-AllPair Ensemble | 与 A 相同，每个 pair 从真实 pre-state 独立开始 | 3 |
| D | 与 B 相同，每个成员独立递归 | 3 |

B/D 的 dynamics 主损失为：

    L_RRT = (L1(ŝ1,s1) + L1(ŝ2,s2) + L1(ŝ3,s3)) / 3

官方 CLARITY drug-swap counterfactual diversity loss 保持启用（`cf_weight=1.0`、`cf_cos_margin=0.9`）：

- A/C 对每个独立 pair 的 interval condition 计算 CF；
- B/D 对 H1/H2/H3 的 stage treatment condition 分别计算并等权平均；
- ensemble 对成员 CF 再等权平均。CF 只参与训练，不参与验证或 checkpoint selection。

四组均训练 2400 次 optimizer update，前 240 step 只优化 dynamics latent loss；CosineAnnealingLR 使用 `T_max=2400`，并在每次 `optimizer.step()` 后推进。每 24 step 验证一次，使四组具有相同验证和 checkpoint 选择机会。batch size 保持 16；`lambda_l1=0.5`、`lambda_cox=1.0`、`lambda_bce=1.0`、`cf_weight=1.0` 均保持不变。

Warmup 结束后同时保存两个 checkpoint：

- `best_val_loss.pt`：按全部 validation trajectory 的 total loss 最小值选择，是预先规定的主 checkpoint；
- `best_val_cindex.pt`：按 patient-level validation C-index 最大值选择，只作诊断。

最终 test 和主汇总只加载 `best_val_loss.pt`，不依赖仅 5 位患者的 validation C-index 选模。

最终推理对 A/B/C/D 统一使用 `s0 -> ŝ1 -> ŝ2 -> ŝ3`。A/C 训练时每个 pair 都从真实 pre-state 独立开始，不传递前一 pair 的 prediction；B/D 训练与部署均递归传递 prediction。B/D 中没有 teacher-forced 主损失、auxiliary rollout loss、`lambda_RRT`、horizon weighting、uncertainty loss或 H4/H5。

## 评估

A/B/C/D 全部从 `s0` 递归部署到 H1/H2/H3。

- Table 1：在全部合格 trajectory 上计算 H1/H2/H3 latent MSE 与 cosine similarity。
- Table 2（主 prognosis 结果）：每位患者只使用时间最早的合格四阶段窗口，计算 H3 C-index 与 IPCW Brier@365；IPCW training reference 使用同一患者级窗口规则。
- Error accumulation：`MSE(H3) - MSE(H1)`，仅在 JSON 中作为辅助描述。
- Representation sanity：observed latent 跨样本方差、相邻真实 latent 的平均 L2、MRI Encoder/LoRA 可训练参数 RMS 更新幅度。
- C/D secondary analysis：除 ensemble mean 外，逐 member 输出 H1/H2/H3 latent MSE 与 cosine；同时输出 latent disagreement、disagreement-error Pearson 相关和 survival probability disagreement。以上均不参与训练或 checkpoint 选择。
- H1/H2 prognosis 仅放在汇总报告附录。

旧 `lambda_RRT` weight ablation 已暂停，没有活动配置或运行脚本。

## Teacher-forced Stage-wise 机制消融

独立的 E 组用于区分 stage-wise supervision 与 recursive predicted-state
training 的贡献，不加入已冻结的 A/B/C/D factorial：

| 组 | Train H1 input | Train H2 input | Train H3 input | Test deployment |
|---|---|---|---|---|
| E: TF-SW | `s0` | `stopgrad(s1_true)` | `stopgrad(s2_true)` | recursive |

E 与 B 复用相同的 `StagewiseTrajectoryDataset`、H1/H2/H3 等权 latent loss、
survival/CF objective、batch size 16、2400 optimizer steps、240 warmup steps、
模型结构、优化器和 checkpoint 选择规则。E 的 CF pre-state 同样使用
`s0/s1_true/s2_true`；测试仍统一执行
`s0 -> ŝ1 -> ŝ2 -> ŝ3`，不使用 teacher forcing。核心比较是 E vs B。

独立配置位于
`configs/ablations/teacher_forced_stagewise.yaml`，输出写入
`outputs/ablations/teacher_forced_stagewise/`，不会改变主实验配置或结果目录。

## 数据与配置

活动配置是 [configs/pure_rrt_v3.yaml](configs/pure_rrt_v3.yaml)。它继续使用固定患者级划分 [data/splits.json](data/splits.json)。
官方 CLARITY 固定为 commit `dadb82241a24f5ec5e4e4dc994e3116fd4a9da04`；训练和评估在构建模型前都会校验 `third_party/CLARITY` 的 HEAD。该 commit 同时写入 resolved config、checkpoint、run metrics 和总汇总。

Cox loss 直接在真实 batch 上一次计算：训练与验证 batch size 均固定为官方设置 16，不使用 gradient accumulation 拼接独立 risk set，也不修改 `SurvivalModule`。按当前样本数，A/C 的训练尾 batch 为 14，B/D 为 10；其余训练 batch 均为 16。

满足四阶段 MRI、严格递增日期及 H1/H2/H3 生存标签的 trajectory / unique patient 数为：

    train / validation / test = 74/38, 11/5, 16/8

A/C 在相同的 38/5 位 train/validation 合格患者上，沿患者完整时间线去重枚举所有有效 pair：

    train / validation all-pairs = 382/38, 56/5

固定 2400 step 后，每个 run 在 resolved `config.yaml`、checkpoint、`metrics.json` 和总汇总中记录 `optimizer_steps`、`warmup_steps`、`samples_seen`、`effective_dataset_passes`。按当前训练集大小与 batch size，A/C 各看到 38200 个样本（100.0 个 dataset pass），B/D 各看到 35520 个样本（480.0 个 dataset pass）；四组 optimizer updates 和 warmup updates 完全相同，数据重复次数透明保留。

长随访患者仍可贡献多个 trajectory 给 dynamics 训练与评估。最终 C-index、Brier 固定为每位患者按起始 MRI 日期排序的第一个合格窗口，因此 test survival metric 使用 8 位患者；patient-level validation C-index 的样本数为 5，但它只选择次级诊断 checkpoint。主 checkpoint 的 validation total loss 对 A/C 使用全部 56 个 validation pair，对 B/D 使用全部 11 条 validation trajectory。A/C 的 survival loss 使用全部 382 个 training pair，B/D 使用全部训练 trajectory。`recursive_predictions.csv` 会用 `primary_survival_window` 标记所选窗口；每个 run 的 `config.yaml`、`metrics.json` 以及总汇总均输出 trajectory/pair count 和 unique patient count。

MRI 直接从 `/dev/shm/clarity_mri_cache` 的 float32 `.npy` 缓存读取。

Stage treatment 严格按 MRI 区间构造。临床时间线把治疗切片记录在区间终点节点，因此 `s_t -> s_{t+1}` 只读取 `s_{t+1}` 上与 `(mri_day_t, mri_day_{t+1}]` 相交的 action。`pair_text`、`step_text`、`prefix_text` 和 `full_text` 均使用显式 `intervals` JSON，不再打包两端的 `pre_actions + post_actions`。纯未来治疗会被排除；只有完整落在当前区间内的疗程才保留 `num_cycles`、总剂量或总分次等全疗程字段。

## 运行

轻量测试：

    conda run -n py310 env PYTHONPATH="$PWD/src" python -m pytest -q

单独运行一组（GPU 0–7）：

    bash scripts/pure_rrt_v3/run_one.sh D 42 4

单独运行 E seed 42：

    bash scripts/teacher_forced_stagewise/run_seed42.sh 3

同时重跑 A/B/C/D seed 42 并在 A 完成后运行 E seed 42：

    tmux new-session -d -s clarity_seed42_rerun_tf \
      'bash scripts/teacher_forced_stagewise/run_seed42_and_rerun_main.sh'

只有确认 seed 42 的方向合理后，才手动补充 seed 43/44：

    bash scripts/pure_rrt_v3/run_remaining_seeds.sh

脚本会在所有已完成 run 上重新生成 `summary.json` 和 `summary.md`。

## 输出隔离

    outputs/
    ├── stagewise_recursive/              # v2 旧结果，只归档
    ├── pure_rrt_v3/                      # endpoint/open-loop A/C 的旧 v3 结果
    ├── pure_rrt_v3_clarity_allpair/      # 已冻结的 epoch-based 2×2 factorial 结果
    ├── pure_rrt_v3_step2400/             # 当前 step-fair 2×2 factorial 输出
        ├── primary/
        │   ├── A_seed42/
        │   ├── B_seed42/
        │   ├── C_seed42/
        │   └── D_seed42/
        ├── summary.json
        └── summary.md
    └── ablations/teacher_forced_stagewise/
        ├── primary/E_seed42/
        ├── summary.json
        └── summary.md
