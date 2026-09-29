# CLARITY Stage-wise Recursive Experiment

当前仓库只实现 [CLARITY_RRT_stagewise_recursive_experiment_plan.md](CLARITY_RRT_stagewise_recursive_experiment_plan.md) 定义的实验：比较 CLARITY open-loop dynamics 与 stage-wise recursive transition learning，并分别测试单模型和 dynamics ensemble。

上一版实验的代码路径已经移除。旧方案和旧配置保存在 docs/legacy/、configs/legacy/，原有 12GB 运行产物原样保存在 outputs/legacy_open_loop_aux/。新版产物只写入 outputs/stagewise_recursive/，不会与旧结果混合。

## 实验定义

四组共用官方 MRI encoder、文本 encoder 和 SurvivalModule 架构。Ensemble 只复制 LatentPredictor，组内成员共享 encoder 和 outcome。

| 组 | Dynamics 训练 | λ_RRT | 成员数 |
|---|---|---:|---:|
| A | s0 + 完整治疗轨迹 + 总时间 → s3 | 0 | 1 |
| B | s0→s1, s1→s2, s2→s3 | 0.1 | 1 |
| C | 与 A 相同 | 0 | 3 |
| D | 与 B 相同 | 0.1 | 3 |

每个样本是同一患者连续四次实际观察组成的 s0,s1,s2,s3。A/C 训练时只编码 s0 和 s3，不读取中间 latent；B/D 的 transition 基础损失使用真实阶段输入，RRT 项从第二步起使用各成员自己的预测继续 rollout：

    L_A/C = L1(F(s0, full_plan, dt_total), s3)

    L_B/D = mean_k L1(F(sk, ak, dtk), s{k+1})
            + λ_RRT * mean_{k=2,3} L1(s_hat_k, s_k)

四组均使用同一个官方 outcome 架构和相同的 H3 Cox/BCE 训练目标；没有递归 survival loss、per-horizon head、uncertainty loss 或其他新模块。

测试阶段不评价 direct prediction。所有组统一使用相邻阶段 action 和真实时间间隔，从 s0 递归部署到 H1/H2/H3。输出包括：

- H1/H2/H3 latent MSE 和 cosine similarity；
- MSE(H3) - MSE(H1)；
- H1/H2/H3 C-index 和 IPCW Brier@365；
- C/D 的成员 latent 方差与 latent MSE 的 Pearson 相关；
- C/D 在 25%/50%/75%/100% coverage 下的 selective rollout latent MSE。

λ_RRT 消融在 B、D 上分别运行 0, 0.01, 0.05, 0.1，汇总 H3 latent MSE、C-index 和 Brier@365。

## 数据与配置

唯一当前配置是 [configs/stagewise_recursive.yaml](configs/stagewise_recursive.yaml)。它继续使用固定的患者级 train/validation/test split：

    data/splits.json

在当前数据上，满足四阶段 MRI、严格递增日期及 H1/H2/H3 生存标签要求的窗口数为 train/validation/test = 74/11/16。

MRI 体数据直接从 `/dev/shm/clarity_mri_cache` 的 float32 `.npy` 缓存读取，不再在训练时重复解压 NIfTI。

官方 CLARITY 源码仍作为本地外部依赖放在 third_party/CLARITY，固定代码由当前本地 checkout 提供，不复制进本仓库。

## 运行

先运行轻量测试：

    conda run -n py310 env PYTHONPATH="$PWD/src" python -m pytest -q

运行一组主实验（只允许 GPU 4–7）：

    bash scripts/run_one.sh A 42 4

同时运行 A–D 的 seed 42 并汇总：

    bash scripts/run_single_seed.sh 42

确认单 seed 结果后，在 tmux 中补充 seed 43/44，再运行消融：

    tmux new-session -d -s clarity_stagewise_followup 'bash scripts/run_pipeline.sh'

在 GPU 4–7 上并行运行指定 seed 的 B/D λ_RRT 消融：

    bash scripts/run_ablation.sh 42

也可单独评价已有 checkpoint：

    PYTHONPATH="$PWD/src" conda run --no-capture-output -n py310 \
      python -m clarity_rrt.evaluate run \
      --config configs/stagewise_recursive.yaml \
      --variant A --seed 42 --device cuda:0

新版目录结构：

    outputs/
    ├── legacy_open_loop_aux/       # 上一版结果，只归档不再读取
    └── stagewise_recursive/
        ├── primary/
        │   └── A_seed42/
        ├── ablation/
        │   └── B_lambda0p01_seed42/
        ├── summary.json
        └── summary.md
