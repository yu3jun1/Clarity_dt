# CLARITY + Stage-wise RRT + Ensemble Recursive Deployment 实验方案

## 1. 实验目标

本实验验证：将 CLARITY 的 open-loop disease evolution model 改造成
stage-wise recursive transition model 后，是否能够在 recursive
deployment 场景下获得更准确、更可靠的长期疾病演化预测。

核心比较：

CLARITY:

    s0 + full treatment plan -> sT

Stage-wise RRT:

    s0 + a0 -> s1
    s1 + a1 -> s2
    s2 + a2 -> s3

推理阶段所有方法统一采用 recursive deployment:

    s0 -> s1_hat -> s2_hat -> s3_hat

------------------------------------------------------------------------

## 2. 两种训练方案

### A. CLARITY training

保持 CLARITY 原始训练逻辑。

输入： - 当前 MRI latent - 完整治疗方案 embedding - 时间间隔

输出： - future MRI latent

学习：

    current state + full treatment trajectory
                |
                v
           future state

特点： - open-loop； - 训练时使用真实 pre latent； -
不学习中间阶段状态。

------------------------------------------------------------------------

### B. Stage-wise Recursive Transition Learning (RRT)

提出方法。

将治疗过程拆分为阶段 transition：

    s0 + a0 + dt0 -> s1

    s1 + a1 + dt1 -> s2

    s2 + a2 + dt2 -> s3

学习：

    s(t+1)=F(s(t), action(t), time)

训练阶段允许：

    predicted state -> next transition

使训练过程匹配部署过程。

------------------------------------------------------------------------

## 3. Ensemble Dynamics

多个 transition model：

    F1, F2, ... FM

预测：

    s(t+1)^m

得到：

平均状态：

    mean prediction

不确定性：

    variance across members

------------------------------------------------------------------------

## 4. 实验组

  组   Dynamics                         RRT   Ensemble
  ---- -------------------------------- ----- ----------
  A    CLARITY predictor                No    No
  B    Stage-wise transition            Yes   No
  C    CLARITY predictor                No    Yes
  D    Stage-wise transition ensemble   Yes   Yes

所有组使用相同 outcome model 和 survival evaluation。

------------------------------------------------------------------------

## 5. 统一 Recursive Deployment 推理

不比较 direct prediction。

所有模型测试：

    initial state

    ↓

    predict next state

    ↓

    use prediction as next input

    ↓

    continue rollout

区别：

CLARITY: 训练阶段没有看到 prediction feedback。

RRT: 训练阶段已经学习 prediction -\> next prediction。

------------------------------------------------------------------------

## 6. Latent Prediction Evaluation

作为 RRT 有效性的第一层证据。

比较：

    s1_hat, s2_hat, s3_hat

与真实：

    s1, s2, s3

指标：

### MSE

H1/H2/H3 latent error。

### Cosine similarity

latent direction consistency。

### Error accumulation

观察：

    error(H3)-error(H1)

是否降低。

------------------------------------------------------------------------

## 7. Survival Downstream Evaluation

使用同一个 CLARITY SurvivalModule。

输入：

recursive rollout latent。

指标：

-   C-index
-   Brier@365

分别报告：

H1/H2/H3。

目标：

验证：

latent improvement 是否传递到临床预测任务。

------------------------------------------------------------------------

## 8. Uncertainty Evaluation

仅用于 Ensemble。

不增加 uncertainty loss。

计算：

    member prediction variance

评价：

1.  uncertainty 与 latent prediction error 的相关性；

2.  selective rollout：

低 uncertainty trajectory 是否具有更低预测误差。

------------------------------------------------------------------------

## 9. RRT Weight Ablation

针对 RRT strength：

测试：

    lambda_RRT:
    0
    0.01
    0.05
    0.1

重点观察：

-   H3 latent MSE；
-   H3 survival performance。

目的：

寻找长期 rollout 最优递归监督强度。

------------------------------------------------------------------------

## 10. 不加入的模块

当前阶段不加入：

-   新 outcome architecture；
-   value head；
-   LLM policy；
-   MPC；
-   uncertainty auxiliary loss；
-   trajectory Transformer；
-   per-horizon outcome head。

原因：

避免混淆 dynamics 改进和其他模块改进。

------------------------------------------------------------------------

## 11. 最终论文逻辑

    CLARITY:
    open-loop treatment-conditioned prediction

            ↓

    Stage-wise RRT:
    recursive treatment-conditioned transition learning

            ↓

    Better recursive latent rollout

            ↓

    Better downstream survival prediction

            ↓

    Ensemble uncertainty estimation

核心验证：

训练方式是否应该与真实闭环疾病演化部署方式一致。
