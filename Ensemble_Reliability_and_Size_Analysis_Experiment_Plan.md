# Ensemble Reliability Analysis & Ensemble Size Analysis 实验设计方案

## 1. 实验目的

当前实验结果显示，Ensemble learning 的潜在价值可能不只是降低 latent prediction error，而是：

1. 降低模型预测方差；
2. 提高 longitudinal prediction 稳定性；
3. 提供 uncertainty signal。

因此下一阶段不继续扩展 RRT 或 Stage-wise，而专注验证：

> Ensemble 是否能够成为可靠 medical world model 的稳定化与不确定性估计组件。

实验分为两个阶段：

- Step 1：Ensemble Reliability Analysis（最高优先级）
- Step 2：Ensemble Size Analysis（敏感性分析）

---

# Step 1：Ensemble Reliability Analysis

## 2. 核心问题

当前已有：

- Single model:
  - A: CLARITY all-pair single

- Ensemble model:
  - C: CLARITY all-pair ensemble

本阶段不重点回答：

> Ensemble 是否显著降低平均误差？

而重点回答：

> Ensemble disagreement 是否能够反映预测可靠性？

即：

如果：

\[
U_i \uparrow
\]

是否意味着：

\[
PredictionError_i \uparrow
\]

---

# 3. 模型设置

固定：

## Primary model

使用：

\[
\text{CLARITY all-pair + Ensemble}
\]

即 C 模型。

原因：

- CLARITY all-pair 是当前强 baseline；
- 避免 RRT / Stage-wise 引入额外变量；
- 专注分析 Ensemble 本身。

---

## Ensemble

默认：

\[
M=3
\]

三个 dynamics predictor：

\[
F_1,F_2,F_3
\]

对于同一个输入：

\[
(s_t,a_t,\Delta t)
\]

得到：

\[
z^{(1)},z^{(2)},z^{(3)}
\]

---

# 4. Uncertainty 定义

## 4.1 Latent uncertainty

ensemble mean：

\[
\bar z=
\frac1M\sum_m z^{(m)}
\]


定义：

\[
U_z=
\frac1M
\sum_m
||z^{(m)}-\bar z||^2
\]


分别计算：

- H1 uncertainty
- H2 uncertainty
- H3 uncertainty

---

## 4.2 Survival uncertainty

如果每个 ensemble member 输出 survival prediction：

\[
p_1,p_2,p_3
\]


定义：

\[
U_s=Var(p_1,p_2,p_3)
\]

分析：

- survival disagreement；
- latent disagreement；
- prediction error。

---

# 5. Reliability Evaluation

## 5.1 Uncertainty-error correlation

目标：

验证：

\[
High\ uncertainty
\rightarrow
High\ error
\]


指标：

### Pearson correlation

\[
corr(U,E)
\]


### Spearman correlation

\[
corr(rank(U),rank(E))
\]


其中：

\[
E_i=
||\bar z_i-z_i^{true}||
\]


分别计算：

- H1
- H2
- H3

---

# 5.2 Risk-Coverage Analysis（重点）

流程：

1. 根据 uncertainty：

\[
U_i
\]

排序。

2. 保留最低 uncertainty 的样本：

- 100%
- 80%
- 60%
- 40%

3. 计算：

\[
MSE@coverage
\]


期望：

如果 uncertainty 有意义：

\[
Coverage\downarrow
\]

同时：

\[
MSE\downarrow
\]


结果：

| Coverage | MSE |
|---|---|
|100%| |
|80%| |
|60%| |
|40%| |

解释：

模型是否能够识别自身不可靠区域。

---

# 5.3 Member disagreement analysis

比较：

单个 member：

\[
MSE_1,MSE_2,MSE_3
\]

与：

ensemble mean：

\[
MSE_{ens}
\]


分析：

ensemble gain 来自：

- averaging；
- member diversity。

---

# 5.4 Member diversity

计算：

\[
corr(F_i,F_j)
\]


分析：

如果：

\[
corr\approx1
\]

说明成员高度相似。

如果：

存在差异：

说明 diversity 有贡献。

---

# 6. Step 1 输出

## Table 1：Prediction performance

| Model | H1 MSE | H2 MSE | H3 MSE |
|-|-|-|-|
|Single| | | |
|Ensemble| | | |


## Table 2：Uncertainty correlation

| Horizon | Pearson | Spearman |
|-|-|-|
|H1|||
|H2|||
|H3|||


## Figure 1

Uncertainty-error scatter plot：

x:

\[
U_z
\]

y:

\[
Error
\]


## Figure 2

Risk-coverage curve：

x:

Coverage

y:

MSE

---

# Step 2：Ensemble Size Analysis

## 7. 实验目的

当前：

\[
M=3
\]

需要验证：

\[
\text{Ensemble size 是否影响性能和 uncertainty quality}
\]

---

# 8. 实验设置

保持：

- 数据；
- 模型；
- optimizer；
- training steps；
- evaluation protocol。

只改变：

\[
M
\]


实验：

| Group | Ensemble size |
|-|-|
|E1|1|
|E2|2|
|E3|3|
|E5|5|

---

# 9. Evaluation Metrics

## 9.1 Prediction accuracy

比较：

\[
MSE(M)
\]

以及：

\[
Cosine(M)
\]


---

## 9.2 Stability

比较：

\[
Std(H3MSE)
\]


验证：

ensemble 是否降低 variance。

---

## 9.3 Uncertainty quality

比较：

\[
corr(U,E)
\]


以及：

risk-coverage。

---

# 10. 结果解释

## Case 1

如果：

\[
M=3\approx M=5
\]

说明：

三个成员已经足够。


---

## Case 2

如果：

\[
M=5>M=3
\]

说明：

更多 ensemble member 有收益。


---

## Case 3

如果：

\[
M=1\approx M=3
\]

说明：

ensemble benefit 不明显。

---

# 11. 推荐执行顺序

## 第一阶段

完成 Step 1：

1. 使用已有 Ensemble 模型；
2. 输出 uncertainty；
3. 完成 reliability analysis。

---

## 第二阶段

完成 Step 2：

训练：

- Ensemble size=2；
- Ensemble size=5。

---

## 第三阶段

根据结果决定：

是否进入：

- UCSF external validation；
- treatment ranking；
- uncertainty-aware decision。

---

# 12. 当前暂不进行

暂不进行：

- RRT tuning；
- Stage-wise modification；
- diversity loss；
- uncertainty loss；
- survival architecture modification。

原因：

当前目标：

\[
验证 Ensemble 是否具有可靠性价值。
\]

---

# 13. 继续研究 Ensemble 的判断标准

至少满足：

## 条件1

降低预测方差：

\[
Std_{ensemble}<Std_{single}
\]


## 条件2

uncertainty 与 error 正相关：

\[
corr(U,error)>0
\]


## 条件3

risk-coverage 有下降趋势：

\[
Coverage\downarrow
\Rightarrow
Error\downarrow
\]


满足后，再进入：

\[
\text{uncertainty-aware treatment planning}
\]

方向。