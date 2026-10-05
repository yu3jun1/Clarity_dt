# Teacher-forced Stage-wise Ablation 实现方案

> E 的机制消融设计文档。当前确定性执行协议见
> [seed42 reproducibility audit](../seed42_reproducibility_audit.md)，
> F 的三成员 ensemble 扩展见 [F 实验方案](../teacher_forced_stagewise_ensemble_F_plan.md)。
> 本文中的建议和命令是设计背景，不表示已完成的实验或清理状态。


## 1. 目的

这个 ablation 的目标不是再引入一个新方法，而是验证当前 RRT 的核心机制：

\[
\boxed{\text{收益究竟来自 stage-wise supervision，还是来自训练时把 predicted state 继续喂回去？}}
\]

当前主实验已经有：

- A：CLARITY all-pair, single model
- B：Pure Recursive RRT, single model
- C：CLARITY all-pair, ensemble
- D：Pure Recursive RRT, ensemble

建议新增一个独立的机制消融：

- **E：Teacher-forced Stage-wise（TF-SW）**

E 不应加入已经冻结的 A/B/C/D 主 factorial，而应作为独立 ablation 实验。

---

## 2. E 与 B 的核心区别

### B：Pure Recursive RRT

当前 B 的训练过程为：

\[
\hat z_1 = F(z_0,c_1,\Delta t_1)
\]

\[
\hat z_2 = F(\hat z_1,c_2,\Delta t_2)
\]

\[
\hat z_3 = F(\hat z_2,c_3,\Delta t_3)
\]

损失：

\[
L_{\mathrm{RRT}}
=
\frac{1}{3}
\left[
L_1(\hat z_1,z_1)
+
L_1(\hat z_2,z_2)
+
L_1(\hat z_3,z_3)
\right]
\]

也就是说，训练和部署阶段都使用 predicted state 递归传播。

---

### E：Teacher-forced Stage-wise

E 的训练改为：

\[
\tilde z_1 = F(z_0,c_1,\Delta t_1)
\]

\[
\tilde z_2 = F(z_1^{true},c_2,\Delta t_2)
\]

\[
\tilde z_3 = F(z_2^{true},c_3,\Delta t_3)
\]

损失仍然完全一样：

\[
L_{\mathrm{TF}}
=
\frac{1}{3}
\left[
L_1(\tilde z_1,z_1)
+
L_1(\tilde z_2,z_2)
+
L_1(\tilde z_3,z_3)
\right]
\]

因此 E 和 B 都保持：

- 相同的四阶段 trajectory；
- 相同的三个 interval treatment；
- 相同的 H1/H2/H3 supervision；
- 相同的 horizon weighting；
- 相同 batch size；
- 相同 2400 optimizer steps；
- 相同 240 warmup steps；
- 相同 dynamics predictor；
- 相同 MRI encoder；
- 相同 text encoder；
- 相同 survival module；
- 相同 optimizer / scheduler；
- 相同 loss 权重。

唯一希望改变的核心变量是：

\[
\boxed{
\text{stage 2/3 的输入是 true state，还是 predicted state}
}
\]

因此真正干净的机制比较是：

\[
\boxed{E\ \text{vs.}\ B}
\]

---

## 3. 数据集不需要修改

当前 `StagewiseTrajectoryDataset` 已经返回：

```python
mri
step_text
prefix_text
full_text
deltas
survival_time
event
```

其中：

```text
mri[:, 0] = s0
mri[:, 1] = s1
mri[:, 2] = s2
mri[:, 3] = s3
```

以及：

```text
step_text[0] = treatment(s0 -> s1)
step_text[1] = treatment(s1 -> s2)
step_text[2] = treatment(s2 -> s3)
```

因此 E 应直接复用和 B 相同的 `StagewiseTrajectoryDataset`。

不要新建 dataset。

这样 E 与 B 使用严格一致的 74 条 training trajectories，能最大限度减少额外混杂因素。

---

## 4. 在 `StagewiseDynamics` 中新增 `teacher_forced()`

当前 `model.py` 已经有：

```python
def rollout(
    self,
    initial,
    step_conditions,
    step_deltas,
):
    ...
```

它会递归使用自己的 prediction。

建议新增一个平行接口：

```python
def teacher_forced(
    self,
    true_inputs: torch.Tensor,
    step_conditions: torch.Tensor,
    step_deltas: torch.Tensor,
) -> torch.Tensor:
    members = []

    for predictor in self.predictors:
        trajectory = []

        for step in range(3):
            prediction = predictor(
                true_inputs[:, step],
                step_conditions[:, step],
                step_deltas[:, step],
            )
            trajectory.append(prediction)

        members.append(torch.stack(trajectory, dim=1))

    return torch.stack(members)
```

输入：

```text
true_inputs shape:
[B, 3, tokens, latent_dim]
```

对应：

```text
stage 0 input = s0
stage 1 input = s1_true
stage 2 input = s2_true
```

输出 shape 与 `rollout()` 保持一致：

```text
[ensemble, B, 3, tokens, latent_dim]
```

这样后续的：

```python
mean_horizon_l1(...)
```

可以完全复用。

---

## 5. Teacher input 建议使用 stop-gradient

当前 MRI encoder 是 trainable。

B 中：

```python
states = encode_mri_stages(...)
targets = states[:, 1:].detach()
```

真正进入 rollout 的初始 state 是：

```python
states[:, 0]
```

因此 MRI encoder 的 input-side gradient 主要来自 \(z_0\)。

如果 E 直接使用：

```python
true_inputs = states[:, :3]
```

那么 \(z_0,z_1,z_2\) 都会作为 predictor input 向 MRI encoder 回传梯度。

这会引入额外 confound：

> E 不仅使用 true state，还获得了更多 teacher-state input gradient。

为了让 E 与 B 更接近，建议：

```python
teacher_inputs = torch.stack(
    [
        states[:, 0],
        states[:, 1].detach(),
        states[:, 2].detach(),
    ],
    dim=1,
)
```

即：

\[
x_1=z_0
\]

\[
x_2=\operatorname{sg}(z_1)
\]

\[
x_3=\operatorname{sg}(z_2)
\]

其中 `sg` 表示 stop-gradient。

target 仍保持：

```python
targets = states[:, 1:].detach()
```

---

## 6. `Trainer.step()` 建议重构为共同 stagewise 分支

目前大致逻辑是：

```python
if self.clarity_all_pair:
    return self._all_pair_step(...)

# 其余默认走 RRT
...
rollout = self.model.rollout(...)
```

不建议复制一套完整 `_teacher_forced_step()`。

更推荐：

```python
scheme = self.config["variants"][self.variant]["training_scheme"]

if scheme == "clarity_all_pair":
    return self._all_pair_step(...)

states = encode_mri_stages(...)
targets = states[:, 1:].detach()

step_conditions = self.model.encode_conditions(
    batch["step_text"],
    batch["clinical_text"],
)

if scheme == "pure_recursive":
    predictions = self.model.rollout(
        states[:, 0],
        step_conditions,
        deltas,
    )

elif scheme == "teacher_forced_stagewise":
    teacher_inputs = torch.stack(
        [
            states[:, 0],
            states[:, 1].detach(),
            states[:, 2].detach(),
        ],
        dim=1,
    )

    predictions = self.model.teacher_forced(
        teacher_inputs,
        step_conditions,
        deltas,
    )
```

然后 B 和 E 共用：

```python
latent = mean_horizon_l1(
    predictions,
    targets,
)
```

这样能防止因为复制代码导致 B/E 在其他训练细节上漂移。

---

## 7. Counterfactual loss 也必须同步 teacher forcing

这是一个容易漏掉的重要点。

当前 `counterfactual_loss()` 的逻辑是：

```python
previous = initial

for stage:
    prediction = member_states[member, :, stage]

    drug_swap_diversity_loss(
        pre_latent=previous,
        ...
    )

    previous = prediction
```

对于当前 RRT，这意味着：

```text
stage 1 CF pre-state = z0
stage 2 CF pre-state = z1_hat
stage 3 CF pre-state = z2_hat
```

如果 E 主 dynamics 改成 teacher forcing，但 CF 仍然递归使用 prediction，那么 E 就不是纯 Teacher-forced 训练。

建议为 `counterfactual_loss()` 增加可选参数：

```python
pre_states=None
```

示意：

```python
def counterfactual_loss(
    model,
    initial,
    member_states,
    conditions,
    deltas,
    treatment_texts,
    margin,
    pre_states=None,
):
    categories = ...

    losses = []

    for member, predictor in enumerate(model.predictors):
        for stage, stage_categories in enumerate(categories):

            if pre_states is None:
                if stage == 0:
                    previous = initial
                else:
                    previous = member_states[member, :, stage - 1]
            else:
                previous = pre_states[:, stage]

            prediction = member_states[member, :, stage]

            losses.append(
                model.clarity.drug_swap_diversity_loss(
                    predictor=predictor,
                    pre_latent=previous,
                    condition_emb=conditions[:, stage],
                    time_delta=deltas[:, stage],
                    pred_latent=prediction,
                    drug_categories=stage_categories,
                    cos_margin=margin,
                )
            )

    return torch.stack(losses).mean()
```

B：

```python
pre_states=None
```

E：

```python
pre_states=teacher_inputs
```

因此：

B 的 CF pre-state：

\[
z_0,\hat z_1,\hat z_2
\]

E 的 CF pre-state：

\[
z_0,z_1^{true},z_2^{true}
\]

这样 CF 与各自的 dynamics training state distribution 一致。

---

## 8. Survival loss 保持与 B 完全一致

建议 E 不修改 survival training。

E 的第三步 prediction：

\[
\tilde z_3
=
F(z_2^{true},a_2,\Delta t_2)
\]

作为：

```python
terminal = predictions[:, :, -1:]
```

继续调用：

```python
risks, logits = self.model.survival(
    initial,
    terminal,
    full_condition,
)
```

总 loss 仍然：

\[
L
=
0.5L_{latent}
+
L_{Cox}
+
L_{BCE}
+
L_{CF}
\]

这样 B 与 E 的训练 objective 保持一致。

不要把 E 改成仅 latent loss，否则 E vs B 不再是 clean ablation。

---

## 9. Warmup 规则保持完全一致

E 应与当前 B 一样：

```text
warmup_steps = 240
total_steps = 2400
batch_size = 16
```

前 240 step 实际训练的是：

\[
L_{latent}+L_{CF}
\]

240 step 后：

\[
0.5L_{latent}
+
L_{Cox}
+
L_{BCE}
+
L_{CF}
\]

不要为 E 单独更改 warmup。

---

## 10. Evaluation 必须仍然使用 recursive rollout

这是整个 ablation 最关键的一点。

E 训练时：

\[
z_0^{true}
\rightarrow \tilde z_1
\]

\[
z_1^{true}
\rightarrow \tilde z_2
\]

\[
z_2^{true}
\rightarrow \tilde z_3
\]

但测试时必须和 A/B/C/D 完全一致：

\[
z_0
\rightarrow
\hat z_1
\rightarrow
\hat z_2
\rightarrow
\hat z_3
\]

即 test 阶段仍然调用：

```python
member_states = model.rollout(
    states[:, 0],
    step_conditions,
    deltas,
)
```

不能在 test 阶段 teacher force。

否则就无法衡量 train-deployment mismatch。

---

## 11. 不建议把 E 加入已经冻结的 A/B/C/D config

当前主实验已经冻结：

```yaml
variants:
  A: {training_scheme: clarity_all_pair, ensemble_size: 1}
  B: {training_scheme: pure_recursive, ensemble_size: 1}
  C: {training_scheme: clarity_all_pair, ensemble_size: 3}
  D: {training_scheme: pure_recursive, ensemble_size: 3}
```

不建议修改成 ABCDE。

建议新增：

```text
configs/ablations/teacher_forced_stagewise.yaml
```

示例：

```yaml
experiment_kind: teacher_forced_stagewise_ablation

output_root: outputs/ablations/teacher_forced_stagewise

variants:
  E:
    training_scheme: teacher_forced_stagewise
    ensemble_size: 1

training:
  total_steps: 2400
  warmup_steps: 240
  validation_interval_steps: 24
  batch_size: 16
  evaluation_batch_size: 16

  seeds: [42, 43, 44]

  lr: 2.0e-4
  text_lr: 1.0e-4
  text_proj_lr: 5.0e-4
  vision_lr: 5.0e-5

  lambda_l1: 0.5
  lambda_cox: 1.0
  lambda_bce: 1.0

  cf_weight: 1.0
  cf_cos_margin: 0.9
```

主实验继续写入：

```text
outputs/pure_rrt_v3_step2400/
```

ablation 独立写入：

```text
outputs/ablations/teacher_forced_stagewise/
```

---

## 12. Experiment assertion 也应独立

不要直接修改当前 `FACTORIAL_VARIANTS` 为 ABCDE。

建议：

```python
MAIN_FACTORIAL_VARIANTS = {
    "A": {
        "training_scheme": "clarity_all_pair",
        "ensemble_size": 1,
    },
    "B": {
        "training_scheme": "pure_recursive",
        "ensemble_size": 1,
    },
    "C": {
        "training_scheme": "clarity_all_pair",
        "ensemble_size": 3,
    },
    "D": {
        "training_scheme": "pure_recursive",
        "ensemble_size": 3,
    },
}

TF_ABLATION_VARIANTS = {
    "E": {
        "training_scheme": "teacher_forced_stagewise",
        "ensemble_size": 1,
    }
}
```

然后根据：

```yaml
experiment_kind:
```

判断不同设计。

示意：

```python
def assert_experiment_design(config):
    kind = config.get(
        "experiment_kind",
        "main_factorial",
    )

    if kind == "main_factorial":
        assert config["variants"] == MAIN_FACTORIAL_VARIANTS

    elif kind == "teacher_forced_stagewise_ablation":
        assert config["variants"] == TF_ABLATION_VARIANTS

    else:
        raise AssertionError(
            f"Unknown experiment kind: {kind}"
        )
```

这样不会破坏已经完成的主实验 reproducibility。

---

## 13. Checkpoint selection 保持不变

E 继续使用：

```text
best_val_loss.pt
```

作为 primary checkpoint。

selection criterion 仍然：

```text
validation_total_loss
```

不要因为 E 是 ablation 就改成：

```text
best H3 MSE
```

更不能使用 test H3 进行选模。

---

## 14. 推荐增加的单元测试

至少增加以下测试：

| Test | 验证内容 |
|---|---|
| `test_teacher_forced_uses_true_pre_states` | stage2 输入确实是 `s1_true`，stage3 是 `s2_true` |
| `test_teacher_forced_does_not_feed_prediction_forward` | H1/H2 prediction 不会作为下一步主输入 |
| `test_teacher_forced_loss_weights_three_horizons_equally` | H1/H2/H3 仍然等权 |
| `test_teacher_forced_cf_uses_true_pre_states` | CF 也使用 true teacher state |
| `test_teacher_forced_eval_is_recursive` | 测试阶段仍走 `rollout()` |
| `test_tf_budget_matches_rrt` | 2400 step / 240 warmup / batch16 |
| `test_main_factorial_config_is_unchanged` | A/B/C/D 主实验冻结不受影响 |

最关键的输入测试可以检查：

```python
torch.testing.assert_close(
    predictor.calls[0],
    states[:, 0],
)

torch.testing.assert_close(
    predictor.calls[1],
    states[:, 1],
)

torch.testing.assert_close(
    predictor.calls[2],
    states[:, 2],
)
```

而当前 RRT 的测试继续验证：

```text
predictor.calls[1] == H1 prediction
predictor.calls[2] == H2 prediction
```

这样可以在代码级把 TF 与 RRT 的关键差异锁死。

---

## 15. 最终主要比较应是 E vs B

论文中的机制 ablation 可以组织为：

| Method | Train H1 input | Train H2 input | Train H3 input | Test deployment |
|---|---|---|---|---|
| A All-pair | true | true | true | recursive |
| E TF-SW | \(z_0\) | \(z_1^{true}\) | \(z_2^{true}\) | recursive |
| B RRT | \(z_0\) | \(\hat z_1\) | \(\hat z_2\) | recursive |

其中最严格的比较是：

\[
\boxed{E\leftrightarrow B}
\]

因为二者：

- 数据完全一致；
- treatment 完全一致；
- horizon supervision 完全一致；
- training budget 完全一致；
- single predictor 完全一致；
- survival loss 完全一致；
- CF loss 完全一致。

核心差别只有：

\[
\text{true previous state}
\]

vs

\[
\text{predicted previous state}
\]

---

## 16. 结果出来后的四种典型情况

### 情况 A：E 的 H1 与 B 接近，但 H2/H3 明显更差

例如：

\[
E_{H1}\approx B_{H1}
\]

但：

\[
E_{H2},E_{H3}\gg B_{H2},B_{H3}
\]

这是最理想的机制证据。

因为 H1 两者都是：

\[
F(z_0,a_0,\Delta t_0)
\]

理论上应该相近。

从 H2 开始：

E 训练看到：

\[
z_1^{true},z_2^{true}
\]

测试却看到：

\[
\hat z_1,\hat z_2
\]

产生 train-deployment mismatch。

而 B 训练和测试都使用 predicted state。

这会直接支持 RRT 的设计动机。

---

### 情况 B：E 与 B 基本一样

说明：

> 当前收益主要来自 stage-wise short-transition supervision，而 predicted-state recursion 本身贡献有限。

此时论文主线需要调整。

---

### 情况 C：E 介于 A 与 B 之间

说明：

> stage-wise decomposition 本身带来一部分收益，而 predicted-state recursive training 又带来额外收益。

这是非常合理，也很容易解释的一种结果。

---

### 情况 D：E 优于 B

说明：

> 在当前小数据条件下，teacher forcing 可能比 recursive self-conditioning 更有效。

这种情况下也不应强行宣称 RRT 胜出，而应重新分析 self-conditioning 的优化代价。

---

## 17. 一个更严格的科学注意点

E vs B 并不只改变一个理论因素。

B：

\[
\hat z_2=F(\hat z_1,\cdots)
\]

意味着 H2/H3 loss 会沿着 recursive graph 向早期 step 反向传播。

也就是说 B 同时包含：

1. predicted-state exposure；
2. recursive credit assignment；
3. backpropagation through time-like behavior。

而 E 的每个 stage 更接近独立 transition。

因此论文中更严谨的措辞应该是：

> **recursive predicted-state training**

而不是：

> **exposure bias alone**

如果以后需要进一步拆分，可以增加一个更细的 ablation：

\[
\hat z_1=F(z_0,\ldots)
\]

\[
\hat z_2=F(\operatorname{sg}(\hat z_1),\ldots)
\]

\[
\hat z_3=F(\operatorname{sg}(\hat z_2),\ldots)
\]

即 predicted state 继续作为下一步输入，但每一步之间 `.detach()`。

这样可以进一步区分：

- self-conditioned state distribution；
- recursive BPTT / credit assignment。

但当前阶段不建议立刻增加这个实验。先完成 TF-SW 即可。

---

# 18. 推荐执行顺序

1. 冻结现有 A/B/C/D 结果；
2. 新增独立 `E = teacher_forced_stagewise`；
3. E 使用与 B 相同的 74 条 trajectory；
4. E 使用相同 2400 optimizer steps；
5. E 使用相同 240 warmup steps；
6. batch size 保持 16；
7. H1/H2/H3 loss 等权；
8. `s1/s2` teacher input 使用 true latent + stop-gradient；
9. CF 同样使用对应 true pre-state；
10. Survival loss 保持与 B 相同；
11. train 阶段 teacher-forced；
12. test 阶段仍然 recursive；
13. 先运行 E seed42；
14. 检查 H1 是否与 B 接近；
15. 检查 H2/H3 是否开始与 B 分叉；
16. 如果实现与方向正常，再跑 E seed43/44；
17. 最终重点比较：
    - H3 MSE；
    - H3 cosine；
    - H3/H1 MSE ratio；
    - error accumulation；
    - cross-seed SD；
    - patient-level H3 improvement。

---

# 19. 最核心的实验判断

最终这个 ablation 主要回答：

\[
\boxed{
\text{stage-wise supervision 是否已经足够？}
}
\]

还是：

\[
\boxed{
\text{必须在训练阶段递归使用 predicted state，才能获得稳定 rollout？}
}
\]

如果 TF-SW 的 H1 与 RRT 相近，但 H2/H3 明显恶化，那么这是目前最直接、最有解释力的 RRT 机制证据。
