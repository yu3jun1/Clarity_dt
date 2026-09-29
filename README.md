# CLARITY Pure Stage-wise RRT v3

当前活动实验严格实现 [Clarity_dt_next_experiment_plan_v3.md](Clarity_dt_next_experiment_plan_v3.md)。核心变化是：RRT 不再是 teacher-forced dynamics 旁边的 auxiliary rollout loss，而是 B/D 的主要 dynamics 训练方式。

旧版配置归档在 [configs/legacy/stagewise_recursive_auxiliary_v2.yaml](configs/legacy/stagewise_recursive_auxiliary_v2.yaml)，旧结果保留在 `outputs/stagewise_recursive/`，活动代码不会读取或覆盖它们。v3 结果只写入 `outputs/pure_rrt_v3/`。
活动 Python 包位于 `src/clarity_rrt_v3/`；旧 `src/clarity_rrt/` 入口已删除，未保留兼容转发层。


## 实验定义

四组使用相同患者划分、可训练 MRI Encoder、Text Encoder、SurvivalModule、优化器、学习率、训练轮数、H3 survival supervision 与 checkpoint 选择规则。Ensemble 只复制 dynamics predictor。

| 组 | Dynamics 主训练方式 | 成员数 |
|---|---|---:|
| A | `s0 + full plan + total time -> s3` | 1 |
| B | `s0 -> ŝ1 -> ŝ2 -> ŝ3`，预测状态直接进入下一步 | 1 |
| C | 与 A 相同 | 3 |
| D | 与 B 相同，每个成员独立递归 | 3 |

B/D 的 dynamics 主损失为：

    L_RRT = (L1(ŝ1,s1) + L1(ŝ2,s2) + L1(ŝ3,s3)) / 3

训练和部署均使用同一种递归状态传递。活动实现中没有 teacher-forced 主损失、auxiliary rollout loss、`lambda_RRT`、horizon weighting、uncertainty loss、新模块或 H4/H5。

## 评估

A/B/C/D 全部从 `s0` 递归部署到 H1/H2/H3。

- Table 1：H1/H2/H3 latent MSE 与 cosine similarity。
- Table 2（主 prognosis 结果）：H3 C-index 与 IPCW Brier@365。
- Error accumulation：`MSE(H3) - MSE(H1)`，仅在 JSON 中作为辅助描述。
- Representation sanity：observed latent 跨样本方差、相邻真实 latent 的平均 L2、MRI Encoder/LoRA 可训练参数 RMS 更新幅度。
- C/D secondary analysis：H1/H2/H3 latent disagreement、disagreement-error Pearson 相关、survival probability disagreement；不参与训练或 checkpoint 选择。
- H1/H2 prognosis 仅放在汇总报告附录。

旧 `lambda_RRT` weight ablation 已暂停，没有活动配置或运行脚本。

## 数据与配置

活动配置是 [configs/pure_rrt_v3.yaml](configs/pure_rrt_v3.yaml)。它继续使用固定患者级划分 [data/splits.json](data/splits.json)。

满足四阶段 MRI、严格递增日期及 H1/H2/H3 生存标签的窗口数应为：

    train / validation / test = 74 / 11 / 16

MRI 直接从 `/dev/shm/clarity_mri_cache` 的 float32 `.npy` 缓存读取。

## 运行

轻量测试：

    conda run -n py310 env PYTHONPATH="$PWD/src" python -m pytest -q

单独运行一组（GPU 只能是 4–7）：

    bash scripts/pure_rrt_v3/run_one.sh D 42 4

按计划先在 tmux 中只运行 seed 42：

    tmux new-session -d -s clarity_pure_rrt_v3 \
      'bash scripts/pure_rrt_v3/run_seed42.sh'

只有确认 seed 42 的方向合理后，才手动补充 seed 43/44：

    bash scripts/pure_rrt_v3/run_remaining_seeds.sh

脚本会在所有已完成 run 上重新生成 `summary.json` 和 `summary.md`。

## 输出隔离

    outputs/
    ├── stagewise_recursive/       # v2 旧结果，只归档
    └── pure_rrt_v3/               # v3 活动输出
        ├── primary/
        │   ├── A_seed42/
        │   ├── B_seed42/
        │   ├── C_seed42/
        │   └── D_seed42/
        ├── summary.json
        └── summary.md
