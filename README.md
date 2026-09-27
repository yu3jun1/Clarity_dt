# CLARITY + RRT + Dynamics Ensemble

这是基于官方 [DingTianxingjian/CLARITY](https://github.com/DingTianxingjian/CLARITY) 的可复现对比实验仓库。本仓库只管理新增实验代码、配置、固定 split、测试和运行脚本；CLARITY 源码作为本地外部依赖，不使用 Git submodule，也不纳入本仓库当前追踪。实验要求外部源码固定在提交 `dadb82241a24f5ec5e4e4dc994e3116fd4a9da04` 且工作树干净。

实验严格包含四组：

| 组 | Dynamics 成员 | RRT 权重 | 共享组件 |
|---|---:|---:|---|
| A | 1 | 0.0 | 官方 MRI/text encoder 与 SurvivalModule |
| B | 1 | 0.1 | 同上 |
| C | 3 | 0.0 | 成员间共享 encoder 与 SurvivalModule |
| D | 3 | 0.1 | 成员间共享 encoder 与 SurvivalModule |

A/B/C/D 使用同一固定患者划分、同一官方 all-pairs 主任务、相同训练种子 `42/43/44`。B/D 只增加第 2 步起的递归 latent L1；C/D 只复制 `LatentPredictor`。不增加递归 survival loss，不把 ensemble 成员展开成 Cox 风险集中的新患者。

## 当前数据与环境

配置文件 [configs/experiment.yaml](configs/experiment.yaml) 已指向：

- MRI：`/data/tanyuejun/CLARITY/dataset/MU-Glioma-Post`
- 临床时间线：`/data/tanyuejun/CLARITY/clinical/MU_Glioma_Post/clinical_latest.json`
- BrainIAC：`/home/tanyuejun/CLARITY/BrainIAC-main/src/checkpoints/BrainIAC.ckpt`
- MedGemma：`/data/tanyuejun/model/medgemma-4b-it`
- Python：Conda `py310`

固定划分保存在 [data/splits.json](data/splits.json)：132 名符合条件的患者，train/validation/test 为 92/20/20 名；对应 472/83/107 个官方 pair。划分摘要 SHA-256 为 `6f9215facc4975a521e4d6fb31d19c9233978aaf8ba20ddce6306824b6913c4e`。

数据、模型权重、checkpoint 和 `runs/` 均不会提交到 Git。

## 验证与运行

所有命令从仓库根目录执行。首次克隆本实验仓库后，独立准备官方源码（目录已被 `.gitignore` 忽略）：

```bash
git clone https://github.com/DingTianxingjian/CLARITY.git third_party/CLARITY
git -C third_party/CLARITY checkout dadb82241a24f5ec5e4e4dc994e3116fd4a9da04
```

然后执行：

```bash
# 单元测试
conda run -n py310 env PYTHONPATH="$PWD/src" python -m pytest -q

# 路径、依赖、CUDA、上游提交和 split 完整性预检
conda run -n py310 env PYTHONPATH="$PWD/src" \
  python -m clarity_rrt.preflight --config configs/experiment.yaml

# 重新生成固定划分（正式实验开始后不要改变 seed 或重抽 test）
conda run -n py310 env PYTHONPATH="$PWD/src" \
  python -m clarity_rrt.data create-split \
  --timeline-json /data/tanyuejun/CLARITY/clinical/MU_Glioma_Post/clinical_latest.json \
  --mri-data-dir /data/tanyuejun/CLARITY/dataset/MU-Glioma-Post \
  --output data/splits.json --seed 20260927 --ratios 0.70 0.15 0.15
```

先按方案跑 A/seed42：

```bash
bash scripts/run_one.sh A 42 1
```

确认基础复现后运行其他组。批量脚本默认只使用 GPU 0 且只启动一个任务；共享机器上应显式指定空闲 GPU，例如：

```bash
GPU_IDS="1 4" MAX_JOBS=2 bash scripts/run_all.sh
```

单次训练会在 `runs/<variant>_seed<seed>/` 保存：

```text
config.yaml
best.pt
last.pt
history.csv
train.log
predictions.csv
evaluate.log
metrics.json
```

`best.pt` 只按 warmup 后 validation pair C-index 选择。评价使用同一 checkpoint，生成 direct、共同递归 H1/H2/H3、C/D 的 H2 uncertainty 80% 覆盖诊断。主指标为固定方向 risk C-index 和使用训练参考删失分布的一年 IPCW Brier。

若训练已完成但需要单独重评：

```bash
conda run -n py310 env PYTHONPATH="$PWD/src" \
  python -m clarity_rrt.evaluate run --variant A --seed 42 --device cuda:0

conda run -n py310 env PYTHONPATH="$PWD/src" \
  python -m clarity_rrt.evaluate aggregate
```

## 实现边界

- [src/clarity_rrt/data.py](src/clarity_rrt/data.py)：固定患者划分、官方 pair 过滤、连续链附加。
- [src/clarity_rrt/model.py](src/clarity_rrt/model.py)：官方路径等价包装、独立 dynamics 成员、成员内递归 rollout。
- [src/clarity_rrt/train.py](src/clarity_rrt/train.py)：成员损失平均、官方 CF 保留、RRT、完整恢复 checkpoint。
- [src/clarity_rrt/evaluate.py](src/clarity_rrt/evaluate.py)：direct/recursive 预测、C-index、IPCW Brier、uncertainty。

跨组 raw latent MAE/MSE 只作为诊断，因为视觉 LoRA 会让各组表示空间不同。实验不评估治疗推荐，不作因果治疗获益解释。
