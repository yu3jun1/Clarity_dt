# Clarity_dt 下一步实验方案 v3

> 原始研究提案，保留用于设计溯源；不是当前已冻结执行协议。
> 本文早期 A/C 的 endpoint/open-loop 定义已被 true-state all-pair 训练替代。
> 当前组别、配置和结果入口以 [仓库 README](../../README.md) 为准。

## Pure Stage-wise RRT + Ensemble，在统一 Recursive Deployment 下验证

### 1. 当前研究目标

下一阶段不再把 RRT 定义为“CLARITY 训练上附加一个 recursive auxiliary loss”，而是把它升级为独立、清晰的 **Stage-wise Recursive Transition Learning**。

核心研究问题：

> 在相同 MRI Encoder、文本编码、Outcome 架构和端到端训练框架下，Stage-wise RRT 是否能比 CLARITY-style open-loop dynamics training 更适合 recursive deployment，并进一步改善长期 disease-state prediction 和下游 survival prediction？

本阶段不要求 MRI Encoder 冻结。MRI Encoder 可以继续参与端到端优化，但四组必须保持相同的 Encoder 架构、可训练状态、优化器和学习率设置。

---

## 2. 两种训练方案必须严格区分

### 2.1 Scheme A：CLARITY-style open-loop training

CLARITY-style dynamics 直接从初始状态和完整治疗轨迹预测终点状态：

```text
s0
+
完整治疗方案 A0:A2
+
总时间 Δt0:3
        ↓
       s3
```

形式上：

\[
\hat{s}_3
=
F_{\text{CLARITY}}
(s_0, A_{0:3}, \Delta t_{0:3})
\]

特点：

- open-loop；
- 终点一次预测；
- 不要求预测出的中间状态继续作为后续输入；
- dynamics 与 CLARITY SurvivalModule 联合训练。

这一方案用于 A/C 组。

---

### 2.2 Scheme B：Pure Stage-wise Recursive Transition Learning

RRT 不再作为 auxiliary rollout loss，而直接成为 dynamics 的主要训练方式。

训练过程：

\[
\hat{s}_1
=
F_{\text{RRT}}
(s_0,a_0,\Delta t_0)
\]

\[
\hat{s}_2
=
F_{\text{RRT}}
(\hat{s}_1,a_1,\Delta t_1)
\]

\[
\hat{s}_3
=
F_{\text{RRT}}
(\hat{s}_2,a_2,\Delta t_2)
\]

也就是：

```text
s0 + a0 + dt0
        ↓
       ŝ1
        +
      a1,dt1
        ↓
       ŝ2
        +
      a2,dt2
        ↓
       ŝ3
```

训练阶段和部署阶段使用同一种递归状态传递方式。

推荐的 latent loss：

\[
L_{\text{RRT}}
=
\frac{1}{3}
\left[
L_1(\hat{s}_1,s_1)
+
L_1(\hat{s}_2,s_2)
+
L_1(\hat{s}_3,s_3)
\right]
\]

不再把以下 teacher-forced 路径作为 B/D 的主 dynamics loss：

```text
s1(true) + a1 -> s2
s2(true) + a2 -> s3
```

这一步是下一版实现最重要的修改。

---

## 3. 四组实验

| Group | Dynamics training | Ensemble size | MRI Encoder | Outcome |
|---|---|---:|---|---|
| A | CLARITY-style open-loop | 1 | trainable | CLARITY SurvivalModule |
| B | Pure Stage-wise RRT | 1 | trainable | CLARITY SurvivalModule |
| C | CLARITY-style open-loop | 3 | trainable | CLARITY SurvivalModule |
| D | Pure Stage-wise RRT | 3 | trainable | CLARITY SurvivalModule |

四组必须保持一致：

- patient split；
- MRI Encoder 架构；
- MRI Encoder 是否可训练；
- text encoder / condition representation；
- SurvivalModule；
- optimizer；
- learning-rate policy；
- epochs；
- checkpoint selection；
- survival loss；
- evaluation cohort。

四组之间主要改变的是：

1. dynamics training scheme；
2. ensemble size。

关键比较：

\[
B-A
\]

表示 RRT 在单模型下的增量作用。

\[
C-A
\]

表示 Ensemble 的增量作用。

\[
D-C
\]

表示在 Ensemble 条件下，RRT 是否仍有额外价值。

其中 **D-C 是验证 RRT 增量价值的关键比较**。

---

## 4. MRI Encoder 不冻结

下一版主实验不强制冻结 MRI Encoder。

理由：

研究目标是验证整个 CLARITY-style system 在 recursive deployment 下的最终性能，而不是只验证一个固定 latent space 中的 transition MSE。

如果 Stage-wise RRT 使：

```text
MRI Encoder
+
Dynamics
+
Outcome
```

联合学到更适合 recursive deployment 的表示，这本身可以视为方法收益的一部分。

因此主实验中允许：

```text
MRI Encoder: trainable
Dynamics: trainable
Outcome: trainable
```

但必须保证 A/B/C/D 使用完全相同的 Encoder 设置。

### 4.1 对 latent MSE 的解释边界

由于 Encoder 可训练，不同组最终可能形成不同的 latent representation。

因此：

\[
MSE_A < MSE_B
\]

或

\[
MSE_D < MSE_A
\]

不能被解释成“在完全相同固定 latent coordinate system 中的严格误差差异”。

Latent MSE 和 cosine 应作为 **机制支持证据**。

最终方法有效性的主要证据仍然是：

- recursive deployment 的最终任务表现；
- 尤其是 H3 survival prediction。

### 4.2 必要的 representation sanity checks

不冻结 Encoder 时，只需增加轻量检查，避免 representation collapse：

- observed MRI latent 的跨样本方差；
- \(\|z_{t+1}-z_t\|\) 的平均幅度；
- Encoder / LoRA 参数更新幅度。

只要这些量没有异常塌缩，就不需要额外冻结 Encoder。

---

## 5. 推理阶段统一采用 Recursive Deployment

主实验不再比较 direct inference。

A/B/C/D 全部采用完全相同的递归推理：

\[
s_0
\rightarrow
\hat{s}_1
\rightarrow
\hat{s}_2
\rightarrow
\hat{s}_3
\]

每一步都使用：

- 上一步预测状态；
- 当前 stage 的治疗方案；
- 当前 stage 的真实时间间隔；
- 相同临床上下文。

因此比较的是：

> 不同训练方式，在相同 recursive deployment 下谁更可靠。

这正是 RRT 的核心研究问题。

---

## 6. Primary dynamics evaluation：Recursive Latent Prediction

Latent prediction 仍然必须保留，因为它直接解释 RRT 是否改善了 recursive dynamics。

所有方法统一得到：

```text
ŝ1
ŝ2
ŝ3
```

并与真实：

```text
s1
s2
s3
```

比较。

### 6.1 Latent MSE

报告：

- H1 MSE；
- H2 MSE；
- H3 MSE。

最关注：

\[
MSE@H3
\]

### 6.2 Cosine similarity

报告：

- H1 cosine；
- H2 cosine；
- H3 cosine。

用于检查预测 latent 的方向一致性。

### 6.3 Error accumulation

可以继续报告：

\[
MSE(H3)-MSE(H1)
\]

但只作为辅助描述。

不要假定误差一定随 horizon 单调增加，因为不同 horizon 的真实疾病变化幅度本身不同。

---

## 7. Primary end-to-end evaluation：H3 Survival

下一版将 survival 主结果聚焦到 H3。

原因：

- 研究目标是长期 recursive deployment；
- 当前训练中的 terminal survival supervision 与 H3 最直接对应；
- H1/H2 survival 容易引入额外解释问题。

主报告：

- H3 C-index；
- H3 IPCW Brier@365。

核心链条：

```text
recursive dynamics
        ↓
predicted final state ŝ3
        ↓
CLARITY SurvivalModule
        ↓
H3 prognosis
```

主要希望看到：

\[
MSE^{D}_{H3}<MSE^{C}_{H3}
\]

同时至少一个 downstream 指标改善：

\[
C\text{-index}^{D}_{H3}>C\text{-index}^{C}_{H3}
\]

或：

\[
Brier^{D}_{H3}<Brier^{C}_{H3}
\]

如果成立，就可以较直接地支持：

> 在相同 Ensemble / Encoder / Outcome 条件下，Stage-wise RRT 改善了 recursive deployment。

H1/H2 survival 可以保留到 appendix，不作为主结论。

---

## 8. Ensemble 设计

继续保持：

```text
ensemble_size = 3
```

暂时不扩大到 5。

Ensemble 只复制 dynamics predictor。

共享：

- MRI Encoder；
- Text Encoder；
- SurvivalModule。

D 组每个成员独立递归：

```text
member 1:
s0 -> ŝ1_1 -> ŝ2_1 -> ŝ3_1

member 2:
s0 -> ŝ1_2 -> ŝ2_2 -> ŝ3_2

member 3:
s0 -> ŝ1_3 -> ŝ2_3 -> ŝ3_3
```

最终 prediction 可以使用成员平均。

---

## 9. Uncertainty 暂时作为 Secondary Analysis

当前 ensemble latent variance 与实际 error 的关系不稳定，因此暂时不能把 uncertainty 作为已经成立的贡献。

下一版：

- 不增加 uncertainty loss；
- 不让 uncertainty 影响训练；
- 不影响 checkpoint selection。

只记录：

### 9.1 Latent disagreement

\[
U_H
=
Var_m(\hat{s}^{(m)}_H)
\]

### 9.2 Error association

检查：

\[
U_H
\quad \text{vs} \quad
MSE_H
\]

### 9.3 Survival probability disagreement

建议同时记录：

\[
Std_m(P_m(T>365))
\]

因为 survival probability disagreement 可能比 raw latent variance 更接近临床 uncertainty。

如果 uncertainty 仍然没有稳定关联，则将其放在 supplementary / negative result，不影响 RRT+Ensemble 主实验。

---

## 10. 暂停旧 RRT Weight Ablation

当前旧 ablation：

```text
λ_RRT = 0 / 0.01 / 0.05 / 0.1
```

对应的是：

\[
L_{\text{teacher}}
+
\lambda_{RRT}L_{\text{recursive}}
\]

但下一版已经取消“recursive auxiliary loss”定义。

因此旧 weight ablation 不再是当前主问题。

下一版先采用：

\[
L_{\text{RRT}}
=
\frac{
L_{H1}+L_{H2}+L_{H3}
}{3}
\]

即：

```text
H1 weight = 1
H2 weight = 1
H3 weight = 1
```

只有当纯 RRT 出现明显 H3 不稳定时，再考虑 horizon weighting：

\[
L
=
w_1L_{H1}
+
w_2L_{H2}
+
w_3L_{H3}
\]

例如：

```text
1 / 1 / 2
```

但暂时不做大量超参数搜索。

---

## 11. 实验执行顺序

### Phase 1：修正方法定义

首先修改 B/D：

```text
当前：
teacher-forced stage-wise
+
recursive auxiliary loss
```

改为：

```text
pure recursive stage-wise training
```

A/C 保持 CLARITY-style open-loop training。

MRI Encoder 保持可训练。

### Phase 2：只跑 seed 42

重新训练：

```text
A_seed42
B_seed42
C_seed42
D_seed42
```

检查：

1. B/D 是否正常收敛；
2. representation 没有 collapse；
3. D 的 H3 latent MSE 是否优于 C/A；
4. B 是否不再出现严重的 rollout degradation；
5. H3 survival 是否合理。

### Phase 3：确认后再跑多 seed

如果 seed42 的方向合理，再运行：

```text
seed 42
seed 43
seed 44
```

最终报告：

```text
mean ± std
```

不需要增加复杂 bootstrap 或 nested CV。

---

## 12. H4/H5

主实验仍然只做：

```text
H1 / H2 / H3
```

原因：

H4/H5 样本明显更少。

H4/H5 可以在主方法固定后作为 supplementary stress test。

建议 H4/H5 主要报告：

- latent MSE；
- cosine similarity。

不把稀疏 H4/H5 survival 作为正文主结论。

---

## 13. 下一版主表

### Table 1 — Recursive Latent Dynamics

| Method | H1 MSE ↓ | H2 MSE ↓ | H3 MSE ↓ | H1 Cos ↑ | H2 Cos ↑ | H3 Cos ↑ |
|---|---:|---:|---:|---:|---:|---:|
| A: CLARITY | | | | | | |
| B: RRT | | | | | | |
| C: Ensemble | | | | | | |
| D: RRT + Ensemble | | | | | | |

### Table 2 — H3 End-to-End Prognosis

| Method | H3 C-index ↑ | H3 Brier@365 ↓ |
|---|---:|---:|
| A: CLARITY | | |
| B: RRT | | |
| C: Ensemble | | |
| D: RRT + Ensemble | | |

### Table 3 — Ensemble Reliability（可选）

| Method | H3 latent disagreement-error association | H3 survival-probability disagreement |
|---|---:|---:|
| C: Ensemble | | |
| D: RRT + Ensemble | | |

只有 uncertainty 结果足够稳定时，Table 3 才进入正文。

---

## 14. 最终研究逻辑

最终论文主线应保持简单：

```text
CLARITY-style open-loop training
        ↓
train / deployment mismatch

Pure Stage-wise RRT
        ↓
prediction-conditioned transition learning

Unified recursive deployment
        ↓
better long-horizon latent trajectory

CLARITY SurvivalModule
        ↓
better end-to-end prognosis

Dynamics Ensemble
        ↓
stabilization + optional uncertainty
```

最终需要验证的核心命题：

> **在保持 CLARITY 的 MRI Encoder、文本条件、Outcome architecture 和端到端优化方式一致的前提下，将 open-loop dynamics training 替换为 Pure Stage-wise Recursive Transition Learning，是否能够改善统一 recursive deployment 下的长期 disease-state prediction；Dynamics Ensemble 是否进一步增强这种收益。**

在这个命题稳定成立前，不加入新的 planner、outcome architecture、uncertainty loss 或其他系统模块。
