# 从空白仓库开始：CLARITY + RRT + Dynamics Ensemble 实验方案

**版本：Fresh Start v1**  
**日期：2026-09-27**  
**官方代码基准：`DingTianxingjian/CLARITY`，提交 `dadb82241a24f5ec5e4e4dc994e3116fd4a9da04`**  
**文档状态：实验与实现设计，尚未实现或运行。**

> 本轮从官方 CLARITY 出发，不迁移 Cloop 的 dynamics、全局 latent adapter、outcome 或 planner。保留官方模型与联合训练主路径，只增加 RRT 辅助分支、Dynamics Ensemble 包装，以及训练后的 uncertainty 评价。
>
> 统计分析保持简单：一份固定患者划分，四组方法，三个训练种子，报告各次结果和均值±标准差。不做多层交叉验证、cross-fitting、复杂显著性检验或新的不确定性校准网络。

---

## 1. 本轮只回答三个问题

1. **RRT 是否有效？** 相比官方 CLARITY，加入递归 latent 重建约束后，长期预测与下游生存预测是否改善？
2. **Ensemble 是否有效？** 集成多个官方 dynamics，是否进一步改善预测？组合是否优于 Ensemble 单独使用？
3. **Uncertainty 是否有用？** 成员对生存概率的分歧，能否帮助识别不可靠预测？

本轮不评估治疗推荐，不接 LLM policy，不实现 MPC，也不把生存预测改善解释为治疗因果获益。

### 1.1 四组主实验

| 组名 | 官方 predictor 数量 | RRT 辅助损失 | Outcome | 训练方式 |
|---|---:|---|---|---|
| A：CLARITY | 1 | 无 | 官方 SurvivalModule | 官方联合训练 |
| B：CLARITY + RRT | 1 | 有 | 相同架构，独立训练 | 官方联合训练 + RRT |
| C：CLARITY + Ensemble | 3 | 无 | 组内共享一个官方 SurvivalModule | 成员损失平均后联合训练 |
| D：CLARITY + RRT + Ensemble | 3 | 有 | 组内共享一个官方 SurvivalModule | 成员联合训练 + RRT |

`3` 个成员是本方案为控制成本提出的起始配置，不是官方参数或已验证最优值。C、D 必须使用相同成员数；第一轮不同时尝试多个成员数。

**四组分别训练参数；同一组的不同时间跨度共享 outcome；ensemble 组的成员也共享 outcome。** 不再为 H1、H2、H3 分别训练 head。

### 1.2 保持不变的模型内容

沿用官方 MRI encoder、MRI token 表示、视觉 LoRA 设置、文本编码器、治疗/临床文本模板、时间编码、LatentPredictor 主干、SurvivalModule，以及原有联合损失。[S1–S4]

尤其不能：

- 换回 Cloop 的 GRU dynamics 或单 global-token 输入。
- 只给改进组增加临床字段、更多患者或新的标签。
- 将 baseline 改成递归 survival 训练后，仍称为未修改的 CLARITY。
- 删除官方启用的 CF 等辅助项，然后只给其中一组恢复。

**允许的共同实验设置**：四组使用一份固定的 train/validation/test 患者名单和同一套评价函数。这是共同 benchmark 协议，不是某一方法独有的修改；结果应与同协议重新运行的 A 比较，而不是直接与论文表格的数字相减。

---

## 2. 新仓库结构：官方代码保留，扩展代码尽量少

建议仓库名称：`clarity-rrt-ensemble`。

```text
clarity-rrt-ensemble/
├── README.md
├── .gitignore
├── third_party/
│   └── CLARITY/                 # 官方代码，固定提交，保留 LICENSE
├── configs/
│   └── experiment.yaml          # 一份公共配置 + 四组开关
├── src/clarity_rrt/
│   ├── __init__.py
│   ├── data.py                  # 固定患者划分、原 pair、附加连续链
│   ├── model.py                 # 官方 predictor 的 ensemble + rollout 包装
│   ├── train.py                 # 复用官方训练逻辑，附加 RRT loss
│   └── evaluate.py              # pair / recursive / uncertainty 三种评价
├── tests/
│   └── test_core.py             # 少量必要的等价性与梯度测试
├── data/
│   └── splits.json              # 固定患者名单，按数据协议管理
└── runs/                        # 配置、checkpoint、日志和预测；不提交大文件
```

不要在新仓库导入 `cloop.*`，也不需要复制历史 `v1/v2/v3/v4` 输出。

以下为可执行的仓库初始化示例；不会自动实现后面的新入口：

```bash
git init clarity-rrt-ensemble
cd clarity-rrt-ensemble

git submodule add https://github.com/DingTianxingjian/CLARITY.git third_party/CLARITY
git -C third_party/CLARITY checkout dadb82241a24f5ec5e4e4dc994e3116fd4a9da04
```

MRI、临床数据和预训练权重按官方说明准备，不应公开提交到代码仓库。记录实际依赖版本、预训练 checkpoint 和官方提交即可，不建立复杂的实验管理平台。

---

## 3. 数据与划分：简单，但不能失去公平性

### 3.1 一份患者级划分

- 已有真正未使用的正式 test 患者时，保留它，不因新建仓库重新抽样。
- 否则在训练前固定一份约 `70% / 15% / 15%` 的患者级 train/validation/test 划分；比例是本方案建议，不是官方原有设置。
- 同一患者的全部 MRI、pair 和递归链只能属于同一个集合。
- 三个训练种子共用这份划分，只改变模型初始化与训练随机性。
- test 只在配置确定后评价；不能因为结果差而重抽 test。

新建仓库不会让已经看过的数据重新成为未见测试数据。如果整个队列此前已反复用于开发，应如实称为开发验证，不声称独立外部验证。

### 3.2 主训练仍使用官方 all-pairs

复用 `GliomaAllPairsTextDataset`，保留其所有合法 `T_i → T_j, i<j` pair，不把训练样本拆成三个独立 horizon 数据集。[S4]

使用官方目标 timepoint 的剩余生存时间与 event 标签，不另外改成起点生存时间。

这代表**目标 landmark 之后的事实条件预后**，不是从最初 MRI 开始的一条治疗方案的全程生存概率。

### 3.3 RRT 只给原 pair 附加连续链，不替换主训练数据

为原始 pair 增加可选的 `chain` 信息。第一版仅给跨度恰好为 2 或 3 个相邻观察间隔的 pair 构造链：

```text
原 pair：T0 → T3
附加链：T0 → T1 → T2 → T3
```

链内必须满足：所有 MRI 可用、日期严格递增、各阶段治疗记录可按官方模板构造。缺少中间信息时，该 pair 仍保留原有官方训练；仅跳过它的 RRT 辅助损失。不能用插值 MRI 或猜测治疗补齐链。

训练批次仍由原 pair loader 产生。每个 batch 最多选取 4 条合法链用于 RRT，选择规则固定；C、D 的主 pair 顺序必须一致。`4` 是控制在线 MRI 显存开销的建议值，不是官方配置。

每个 epoch 记录合法链数和实际 RRT 更新批次数，防止表面开启了 RRT、实际却没有可用长链。

### 3.4 条件信息保持官方语义

原 pair 使用原来的治疗/临床文本；递归每一步用对应相邻 timepoint 的同一文本模板和真实间隔，不能在第一步偷偷附加未来 MRI 信息。

官方文本模板包含 pre/post 端治疗字段。[S4] 因而本轮必须明确是**给定历史事实治疗的条件预测**。这不等于部署时提前知道未来治疗，更不等于自动推荐；后续研究实际推荐时另行审计行动时间语义。

---

## 4. 计算框架：Dynamics 与 Outcome 一起训练

### 4.1 官方主路径保持不变

用 `N` 表示 MRI token 数，`M` 表示 ensemble 成员数，避免混淆。

```text
真实 pre MRI ──→ 官方 MRI encoder ──→ z_pre [B,N,D]
治疗/临床文本 ──→ 官方文本编码器 ──→ condition
                                     │
              ┌──────────────────────┴──────────────────────┐
              │ 官方 LatentPredictor 1 ... LatentPredictor M │
              └──────────────────────┬──────────────────────┘
                                     ↓
                        predicted post [M,B,N,D]
                                     ↓
                      同一个官方 SurvivalModule
                                     ↓
                           risk_m，survival_logit_m
```

真实 post MRI 仅提供 latent 重建目标；官方使用 target latent 的 stop-gradient 约定，第一版保留其处理，不新增 target encoder 机制。[S2]

生存损失从 outcome 经过 predicted post 回到 dynamics。预测 latent 不得 `.detach()`，训练时不得使用上个版本导出的预测缓存。

### 4.2 Ensemble 只复制 dynamics

每组只保留一个官方 MRI encoder、文本 encoder 和 SurvivalModule；复制的只是 `LatentPredictor`。

每个成员独立初始化；不要把完全相同的权重复制三份后在完全相同的路径中训练。A/B 使用相同的单模型初始化，C/D 对应成员使用相同初始化规则，所有组的共同组件初始化规则一致。

不同组之间不共享训练后的权重。同一组内共享编码器与 outcome，意味着本方案是 **shared-encoder/shared-outcome dynamics ensemble**，不应称为全系统独立 deep ensemble。

### 4.3 主路径的成员损失

第 m 个成员：

\[
\hat z^{(m)}_j = F_{\theta_m}(z_i,c_{ij},\Delta t_{ij}),
\qquad
(r_m,\ell_m)=G_\phi(z_i,\hat z^{(m)}_j,c_{ij}).
\]

对每个成员调用相同的官方重建、Cox、BCE 及已启用辅助损失，再求平均：

\[
L_{\mathrm{base}}=\frac1M\sum_{m=1}^M L_{\mathrm{CLARITY}}^{(m)}.
\]

每个成员的 Cox 在同一个患者/pair batch 上计算；**不能把 M 个成员展平成 M 倍患者，放进一个更大的 Cox 风险集。** 这不会增加独立生存样本。

共享 outcome 与共享 encoder 接收成员损失的平均梯度；各 predictor 接收属于自己的梯度。逐成员 loss 是本方案的 ensemble 定义，不保证成员分歧一定有效，需要后续评价。

---

## 5. RRT：只增加递归 latent 重建约束

### 5.1 不新增递归 survival loss

第一版严格采用：

\[
\boxed{L_{\mathrm{total}}=L_{\mathrm{base}}+\lambda_{\mathrm{RRT}}L_{\mathrm{RRT}}.}
\]

原 pair 路径已经通过 Cox/BCE 联合训练 dynamics 与 outcome。新 RRT 分支只重建未来 latent，不另外在递归终点计算 survival loss。

这样，A 仍然是官方原有 pair 训练路径，而不是被改造成共同递归训练的“新 baseline”。

### 5.2 正确递归

对每条长度 H≥2 的链和成员 m：

\[
\hat z^{(m)}_0=z_0,
\quad
\hat z^{(m)}_{k+1}
=F_{\theta_m}(\hat z^{(m)}_k,c_{k,k+1},\Delta t_k).
\]

第二步开始，输入必须是该成员自己的预测，不是真实中间 MRI，也不是 ensemble mean。各成员不得在每步之后重置到共同均值。

第一版只在第 2 至 H 步计算辅助重建，以避免把已有单步监督简单重复加权：

\[
L_{\mathrm{RRT}}
=\frac1M\sum_m\frac1{|\mathcal C|}\sum_{c\in\mathcal C}
\frac1{H_c-1}\sum_{k=2}^{H_c}
\operatorname{MAE}(\hat z^{(m,c)}_k,\operatorname{stopgrad}(z^{(c)}_k)).
\]

`MAE` 在 token 和特征维上求均值，沿用官方 L1 的损失类型。MSE 可以用于评价，但不要顺带把训练主损失从 L1 改成 MSE。

当前 batch 没有合法链时，RRT 项为零；训练仍执行官方主损失。第一步预测仍通过后续递归损失获得梯度，预测链不能截断。

### 5.3 在线编码的处理

真实链内 MRI 用当前模型的官方 encoder 生成 target，并沿用官方 target 端 detach/EMA 配置。初始 pre latent 保持原有梯度路径。

真实 MRI 文件可以缓存读取结果，但 trainable encoder 的输出与当前 predictor 的输出不得作为跨 optimizer step 的固定训练特征缓存。否则训练目标与当前参数会脱节。

---

## 6. 训练配置：先跑一套，不做大规模搜索

### 6.1 以官方运行脚本为公共配置

官方 `run_training.sh` 使用 BrainIAC 在线编码、每模态 8 个 token、32 个总 token、768 维 latent、视觉 LoRA、MedGemma 文本编码器、100 epochs、batch size 16、dropout 0.3、survival weight decay 0.01，以及 L1/Cox/BCE 与 CF 联合训练。[S1]

新仓库先复现这一模式。不要因为已有 Cloop 全局 latent 方便，就只给部分组改成 frozen global latent。

如果设备确实无法支持在线配置，可以让**四组统一**采用官方支持的预提取多模态特征模式；此时应明确命名为 `CLARITY-precomputed` 配置，而不是称为在线官方配置的严格复现。第一轮只选一种模式，不同时铺开两条主线。

### 6.2 建议起始设置

| 设置 | 值/规则 | 来源 |
|---|---|---|
| MRI / text / predictor / outcome 主干 | 官方配置 | 官方 |
| 主训练 pair | 官方 all-pairs | 官方 |
| Epochs / pair batch | 100 / 16 | 官方运行脚本 |
| Outcome dropout / weight decay | 0.3 / 0.01 | 官方运行脚本 |
| 主学习率 | 2e-4；各子模块比例沿用官方 optimizer | 官方运行脚本与训练器 |
| 联合阶段 L1 / Cox / BCE | 0.5 / 1.0 / 1.0 | 官方运行脚本 |
| CF weight | 1.0，保留官方实现 | 官方运行脚本 |
| Ensemble size | 3 | 本方案新增 |
| RRT 链长度 | 2、3 步 | 本方案新增 |
| RRT 每批最多链数 | 4 | 本方案新增 |
| RRT weight | 0.1，作为起始值而非最优结论 | 本方案新增 |
| 训练种子 | 42、43、44 | 本方案实验设置 |
| 数据划分 | 一份固定患者划分 | 本方案共同评价协议 |

如需更改 RRT weight，仅在 validation 上做一次有限比较，例如 0.05、0.1、0.3；Cloop 历史 test 结果不能用于选择。第一轮建议先不搜索。

### 6.3 Warmup 与联合训练

- 前 10 epochs：沿用官方 warmup 的主损失权重，RRT 关闭。
- 第 11 epoch 起：四组启用相同的官方生存联合训练；B、D 额外启用 RRT。
- 所有组按同样的 epoch/原 pair 更新次数训练，记录 ensemble 与 RRT 增加的时间和显存。

**源码细节：**官方 warmup 把主损失中的 Cox/BCE 权重置零，但训练器里另行启用的 CF 等辅助项不会因此自动关闭。[S2–S3] 因而不能把实际配置概括成“前 10 轮绝对只有 L1”，也不要为了符合这句话而修改 A 的代码。

### 6.4 选 checkpoint

主 checkpoint 统一按 warmup 后的 validation pair C-index 选择，沿用官方 `best_c_index` 方向；所有组用同一标准。[S3]

保存初始化指标作诊断，但不将随机初始化结果包装成已经训练好的 survival 模型。若训练后仍不能超过简单参考，照实报告，不因 test 表现另选 epoch。

第一版不需要动态早停策略或多指标复杂评分。保存 `best`、`last` 和训练曲线即可。重新加载时必须恢复共享编码器中训练过的 LoRA/投影、所有 dynamics 成员、outcome 及必要 buffer，确保验证预测可以复现。

---

## 7. 只做三类评价

### 实验一：官方 pair 任务上的下游表现

四组对同一批 test pairs 执行原始 direct prediction，并使用相同 outcome 输入语义。比较：

- **主指标：C-index，使用 risk head。**
- **配套指标：一年 IPCW Brier，使用 survival logit 的 sigmoid。**

主表使用同一个 checkpoint 同时计算两个指标，不为 Brier 再挑一个 test 最优 checkpoint。

```text
| 方法 | C-index ↑ | Brier@365 ↓ | 时间/显存 |
| A    |           |              |           |
| B    |           |              |           |
| C    |           |              |           |
| D    |           |              |           |
```

训练和官方 pair 评价可以保留原来的 pair-level 口径；同时报告患者数与 pair 数，不把多个 pair 当成新增独立患者。这里不另外进行 pair 级显著性检验。

### 实验二：共同递归部署下的长期下游表现

从相同初始 MRI 出发，给四组完全相同的事实治疗序列和间隔，递归预测 H1/H2/H3。

- H 是实际递归调用次数，不是“直接预测时的天数分组”。
- 四组都使用前一步预测递归，baseline 不能每一步偷偷读取真实 MRI。
- 每个 H 固定每名患者的首个合法窗口作为主要递归评价样本；规则在查看性能前确定，四组共用。
- Outcome 始终读取初始 pre、该次预测终点和原 pair 模板生成的起点至终点 condition。不要某组使用最后一步 condition，另一个组使用整段 condition。
- 同一训练好的 outcome 用于所有 H，不重新训练、不按 H 选不同 checkpoint。

报告 C-index、Brier@365；H2 为预先关注的长期任务，H1/H3 支持解释。H3 若事件或样本过少就明确标注，不把随机波动当结论。

```text
| 方法 | C@H1 | C@H2 | C@H3 | Brier@H2 |
| A    |      |      |      |           |
| B    |      |      |      |           |
| C    |      |      |      |           |
| D    |      |      |      |           |
```

这张表检验“原始训练 vs RRT 训练在同一递归部署协议下”的表现，不与 direct prediction 的分数混成一张方法排名表。

### Latent MAE/MSE 的使用边界

可以同步记录各模型自己的预测 latent 与其 target latent 的 MAE/MSE，帮助检查训练与误差累积。

**但主方案沿用可训练视觉 LoRA，不同组的表示空间会随训练变化。** 因而跨组 raw latent MSE 不能单独作为完全统一空间中的精度证明；第一轮将其列为诊断，而以相同真实生存标签上的指标作为主要证据。

若以后必须严格比较 latent MSE，再补一轮“四组使用完全相同、冻结的官方 encoder”实验；它是共同条件下的机制支持实验，不是本轮在线官方配置的替代。不要将某组当前 encoder 的 latent 直接与另一固定 encoder 的 target 强行做 MSE。

### 实验三：不重新训练的 uncertainty 评价

仅对 C、D 进行：

\[
p_m=\sigma(\ell_m),\quad
\bar p=\operatorname{mean}_m p_m,\quad
u=\operatorname{std}_m p_m.
\]

主 ensemble 预测采用成员概率平均；risk score 用共享 head 的成员输出平均。不要把 mean logit 的 sigmoid 与 mean probability 混为同一种聚合。

只保留两个覆盖率：100% 和约 80%。在递归 H2 同一患者集合中，对低 uncertainty 的 80% 预测计算 Brier，再与同样保留人数的随机选择比较。随机选择重复 20 次取均值即可，不做复杂检验。

必须同时报告总患者数、保留人数、实际覆盖率和事件数；排序包含提前删失患者，不能先删掉这些患者再称为全体覆盖率。IPCW 使用训练参考，不能对保留的 test 子集重新拟合删失模型。[S6]

这是**可靠性排序诊断**，不是临床置信区间，也不是已经锁定的拒绝预测阈值。该子集指标不能直接替代全体患者主表。

---

## 8. 统计只做到必要程度

每组 3 个训练种子，报告三次单独结果和均值±标准差。同一份 split 下的种子标准差仅反映训练随机性，不是患者总体置信区间。

第一轮不做 p 值、bootstrap、多折交叉验证、分层亚组、复杂校准损失或综合排名。

**不能省略的只有基本正确性：**患者隔离、固定标签含义、相同 test 样本、正确预测方向，以及对右删失的处理。

一年 Brier 可直接使用 `sksurv.metrics.brier_score`；删除时间/事件支持不足时不能伪造结果或改测试时间点，按预定规则输出 NA 并说明原因。[S6]

### 评价代码的一个必要说明

官方 `compute_auc()` 包含依赖当前标签的自动方向翻转和历史平滑。[S5] 它不适合作为新实验最终 test AUC 的直接实现。本方案主表暂不需要 AUC；若以后补报，统一采用固定方向、无历史平滑的计算，不能根据 test 标签选择反转。

这属于四组共同的评价约定，不是新增模型组件，也不改变官方训练损失。不要从原始日志的平滑 AUC 推导本轮收益。

---

## 9. 实现范围与接口

只需完成四个扩展文件；以下接口名是**待实现规格**，不是官方已经存在的函数。

| 新文件 | 核心职责 | 禁止顺带修改 |
|---|---|---|
| `data.py` | 读取固定 split；包装官方 pair dataset；附加可选 chain | 不重写临床文本、标签或 latent 表示 |
| `model.py` | 保存官方 predictor 成员；共享官方 encoder/head；实现 `forward_pair`、`rollout_chain` | 不新增 GRU、trajectory head 或 value head |
| `train.py` | 沿用官方 optimizer/scheduler/损失；平均成员 loss；附加 RRT；保存整个可恢复系统 | 不把预测缓存当训练输入；不改变 baseline 主目标 |
| `evaluate.py` | direct、recursive、成员分歧；保存预测和两项生存指标 | 不根据结果重选患者、翻转风险方向或调整 test 阈值 |

建议张量契约：

```text
pre_tokens:       [B,N,D]
post_target:      [B,N,D]
member_post:      [M,B,N,D]
member_risk:      [M,B]
member_logit:     [M,B]
chain_targets:   [B_chain,H+1,N,D]
chain_mask:      [B_chain,H]
chain_delta:     [B_chain,H]
```

原始 MRI 输入仍走官方路径；上表描述 encoder 之后的接口。padding 位置不进入重建 loss，不调用非正间隔的 dynamics。

### 9.1 一个简短的训练逻辑示意

```python
# 示意代码；并非可直接运行的官方 API。
pre, target, condition = encode_official_pair(batch)

member_losses = []
for predictor in predictors:
    post_hat = predictor(pre, condition, batch.time_delta)
    risk, logit = official_survival(pre, post_hat, condition)
    member_losses.append(
        official_total_loss(post_hat, target, risk, logit, batch,
                            predictor=predictor)
    )
loss = mean(member_losses)

if epoch > 10 and rrt_weight > 0:
    # 对该原 pair batch 的合法连续链做递归；预测状态不 detach。
    loss = loss + rrt_weight * recursive_latent_loss(batch.chains)

optimizer.zero_grad(set_to_none=True)
loss.backward()
clip_grad_norm_(trainable_parameters, official_clip_value)
optimizer.step()
```

`official_total_loss` 必须包括该配置原来启用的 CF 等项，不能只包 `compute_loss()` 而遗漏训练器追加的损失。所有 optimizer group、成员参数都必须被注册；M 个 predictor 不能只更新第一个。

### 9.2 输出不用复杂化

```text
runs/A_seed42/
    config.yaml
    best.pt
    last.pt
    history.csv
    predictions.csv
    metrics.json
```

记录 upstream commit、模型配置、split 文件摘要和训练种子。`predictions.csv` 至少包含 patient、起点/终点、模式、H、time、event、risk、survival365；ensemble 另存成员概率及其标准差。

### 9.3 配置开关

```yaml
# 新入口的建议配置字段，不是官方 train.py 已有参数。
upstream_commit: dadb82241a24f5ec5e4e4dc994e3116fd4a9da04
variants:
  A: {ensemble_size: 1, rrt_weight: 0.0}
  B: {ensemble_size: 1, rrt_weight: 0.1}
  C: {ensemble_size: 3, rrt_weight: 0.0}
  D: {ensemble_size: 3, rrt_weight: 0.1}
rrt:
  horizons: [2, 3]
  max_chains_per_pair_batch: 4
  start_epoch: 11
  supervise_from_step: 2
  loss: l1
  recursive_survival_loss: false
outcome:
  source: official_survival_module
  share_across_horizons: true
  share_across_members: true
training:
  epochs: 100
  pair_batch_size: 16
  seeds: [42, 43, 44]
evaluation:
  modes: [direct, recursive]
  recursive_horizons: [1, 2, 3]
  metrics: [c_index, ipcw_brier365]
  uncertainty_coverages: [1.0, 0.8]
```

---

## 10. 开始正式训练前，只做六项检查

1. **关闭新增功能能恢复官方路径。** 固定权重、输入及随机性，`M=1、RRT=0` 的前向和损失与官方一致；评价协议不同不能掩盖计算图差异。
2. **联合梯度存在。** 暂时只开 Cox/BCE，至少有事件的 batch 上检查 predictor 与 survival 参数均收到有限梯度。
3. **RRT 真递归。** 第 2 步输入确实等于第 1 步预测；反向能经过中间预测回到早期 dynamics 调用。
4. **成员真实独立。** 参数存储不共享、初始化不相同；C/D 对应成员的初始化规则一致。
5. **数据不混。** chain 不跨患者、集合或错误时间；同一评价模式四组使用相同样本清单。
6. **checkpoint 能恢复。** 保存再加载后，同一批验证输入输出一致，包含已训练的视觉/文本适配参数。

这些是防止实现错误的测试，不是新增统计实验。本文未执行这些仓库集成检查，实施时需要实际运行。

---

## 11. 执行顺序与停止条件

### 第一步：只跑 A，确认官方基础路径

先用一个 seed 跑通官方模型、训练、选模和验证。检查 loss 有限、输出非恒定、checkpoint 可恢复。A 在 validation 上表现不好时，先查数据与基础复现，不要立刻归因于 RRT 或换 outcome。

### 第二步：同一个 seed 跑 B、C、D

检查增加 RRT 后损失尺度是否合理、所有成员是否更新、时间与显存是否可接受。此时只看 validation，test 不用于决定参数。

### 第三步：配置确定后，完成四组 × 三个 seed

共 12 次正式训练；前面相同配置的 seed42 可以计入，不必重复。用相同 test 协议一次性生成 direct 表、recursive 表，以及两个 ensemble 组的 uncertainty 诊断。

### 第四步：按结果作结论，不强行保留贡献

| 结果 | 能支持什么 |
|---|---|
| B 优于 A，D 优于 C | RRT 有额外价值 |
| C 优于 A，D 优于 B | Ensemble 有额外价值 |
| D 优于 A，但不优于 C | 组合优于 baseline，但 RRT 在 ensemble 上的必要性仍未证明 |
| 只在 recursive 评价改善 | 支持递归部署场景收益，不能泛化为所有官方 direct 任务都更好 |
| 只有 latent 误差改善 | 仍未证明下游预后收益 |
| 低 uncertainty 子集优于等人数随机子集 | 支持可靠性排序价值，不是全体患者性能或因果治疗获益 |

主证据应来自相同真实标签上的 C-index/Brier。三个种子只是稳定性检查，不据此宣称强统计显著性或外部临床有效性。

### 只有结果值得进一步解释时，才加一个额外对照

若 RRT 有收益、需要区分“递归训练”与“额外 MRI 监督/计算”的作用，可增加 **CLARITY + 等量 teacher-forced 辅助重建**：使用同样链、相同目标、相近计算，但每步输入真实 latent。

它是一个有针对性的控制，不是第一轮必须扩展到十几组实验。没有这个控制时，可以宣称 RRT 方案有效，但不宜把全部增益严格归因为消除了 exposure bias。

---

## 12. 本轮明确不做

不迁移 Cloop 架构；不做 per-horizon outcome；不冻结 dynamics 后另训 head 作为主实验；不新增 trajectory Transformer、value head、uncertainty loss、校准网络、LLM policy 或 MPC。

不引入 cross-fitting、多层交叉验证或复杂统计推断。也不预设 joint training、共享 head 或 ensemble 必然解决过拟合。

**最终交付就是：一个基于官方 CLARITY 的可复现实验入口，四组联合训练结果，两张性能表，以及一项独立的 uncertainty 可靠性诊断。**

---

## 参考依据与设计边界

以下为本方案核查的来源。标记为“本方案新增”的成员数、RRT 权重、链采样和固定三段划分，是建议实验设置，不是官方已验证配置。

- **[S1] 官方训练配置**：`run_training.sh`，上述固定提交。  
  `https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/run_training.sh`
- **[S2] 官方主/辅助损失与在线编码**：`Predictor/train.py` 中 `_shared_step()`。  
  `https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/train.py`
- **[S3] 官方模型与联合训练**：`Predictor/models/full_model.py` 的 `forward()`、`compute_loss()`，以及 `train.py` 的 warmup/选模逻辑。  
  `https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/models/full_model.py`
- **[S4] 官方 all-pairs、文本字段与目标 landmark 标签**：`Predictor/dataset/dataset_glioma_all_pairs_text.py`。  
  `https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/dataset/dataset_glioma_all_pairs_text.py`
- **[S5] 官方指标实现**：`Predictor/utils/metrics.py` 的 `compute_auc()` 和一年标签函数；AUC 自动翻转/平滑是源码事实，不作为本轮推荐 test 口径。  
  `https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/utils/metrics.py`
- **[S6] 一年 IPCW Brier 的现成实现与支持条件**：scikit-survival `brier_score` API。  
  `https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html`
