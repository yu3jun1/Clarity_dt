# UCSF_POSTOP_GLIOMA_DATASET 数据集说明

核验日期：2026-10-04。本文面向 CLARITY 项目的数据管理与实验接入，依据本地 v1.0 数据、随附 Excel、当前仓库代码及官方资料编写。

本文只记录数据现状和接入注意事项；未修改原始数据、现有训练配置、患者划分或正在运行的实验。

## 1. 数据集身份与用途

本地目录名 `UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0` 对应 **UCSF-ALPTDG**：University of California San Francisco Adult Longitudinal Post-Treatment Diffuse Glioma MRI Dataset，即 UCSF 成人治疗后弥漫性胶质瘤纵向 MRI 数据集。它包含治疗后两次连续随访的多模态 MRI、组织分区及纵向变化标注，以及配套临床资料。名称中的 `POSTOP` 不代表每个病例都接受过切除手术；临床表也存在仅以 MRI 诊断的记录。[官方数据门户](https://imagingdatasets.ucsf.edu/)、[数据集论文](https://doi.org/10.1148/ryai.230182)

它不是 UCSF-PDGM 术前胶质瘤数据集，也不是 MU-Glioma-Post；患者编号、文件组织和随访长度均不能直接混用。

CLARITY 论文第 4.1 节将 UCSF-ALPTDG 用作外部验证队列。这里应区分“论文使用了该数据集”与“本仓库当前实验已接入”：截至核验日期，[主配置](../configs/pure_rrt_v3.yaml)和 [E 配置](../configs/ablations/teacher_forced_stagewise.yaml)仍指向 MU-Glioma-Post，不能把当前 A/B/E 结果描述为 UCSF 结果。[CLARITY 论文](https://arxiv.org/html/2512.08029v1#S4.SS1)

## 2. 本地存放位置与规模

实际路径是以 `/` 开头的绝对路径，而不是仓库内的 `data/tanyuejun/...` 相对路径。

```text
解压目录：
/data/tanyuejun/CLARITY/dataset/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0/

原始压缩包：
/data/tanyuejun/CLARITY/dataset/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0.zip

临床及影像汇总表：
/data/tanyuejun/CLARITY/dataset/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0/
  UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx
```

以下数量来自本地文件清点，不含 ZIP 本身：

|项目|本地核验结果|
|---|---:|
|患者目录|298|
|每位患者的 MRI 时点|2：`time1`、`time2`|
|MRI 检查时点总数|596|
|每位患者的 NIfTI 文件|16|
|NIfTI 文件总数|4,768|
|Excel 文件|1|
|全部文件总数|4,769|
|文件内容总大小|30,179,666,034 bytes，约 28.11 GiB|
|患者编号范围|`100001` 至 `100302`，不连续|

编号范围中不存在 `100071`、`100091`、`100159`、`100242` 四个目录。现有 298 个病例都具有完整的 16 种预期文件；编号不连续本身不能作为下载缺失的证据。

上述容量是文件大小之和，不是磁盘分配块数，也不是解压后的未压缩体素容量。

## 3. 目录结构与文件命名

所有患者均采用同一平铺结构，例如：

```text
UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0/
├── UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx
├── 100001/
│   ├── 100001_time1_t1.nii.gz
│   ├── 100001_time1_t1ce.nii.gz
│   ├── 100001_time1_t2.nii.gz
│   ├── 100001_time1_flair.nii.gz
│   ├── 100001_time1_t1ce-t1.nii.gz
│   ├── 100001_time1_seg.nii.gz
│   ├── 100001_time2_t1.nii.gz
│   ├── 100001_time2_t1ce.nii.gz
│   ├── 100001_time2_t2.nii.gz
│   ├── 100001_time2_flair.nii.gz
│   ├── 100001_time2_t1ce-t1.nii.gz
│   ├── 100001_time2_seg.nii.gz
│   ├── 100001_flair_subtraction.nii.gz
│   ├── 100001_flair_subtraction_seg.nii.gz
│   ├── 100001_t1ce_subtraction.nii.gz
│   └── 100001_t1ce_subtraction_seg.nii.gz
├── 100002/
└── ...
```

`time1`、`time2` 是两次治疗后随访，不是“术前、术后”的固定配对。

|文件后缀|含义|每位患者数量|
|---|---|---:|
|`time{1,2}_t1`|非增强 T1 MRI|2|
|`time{1,2}_t1ce`|增强 T1 MRI|2|
|`time{1,2}_t2`|T2 MRI|2|
|`time{1,2}_flair`|FLAIR MRI|2|
|`time{1,2}_t1ce-t1`|同一时点增强 T1 减非增强 T1 的派生图像|2|
|`time{1,2}_seg`|单时点组织分区掩码|2|
|`flair_subtraction`|两时点 FLAIR 的纵向差分图像|1|
|`t1ce_subtraction`|两时点增强差分图像的纵向差分|1|
|`{flair,t1ce}_subtraction_seg`|纵向变化分区掩码|2|

作者定义的变化图像关系为：

```text
time{k}_t1ce-t1 = T1CE(time{k}) - T1(time{k})
flair_subtraction = FLAIR(time2) - FLAIR(time1)
t1ce_subtraction = [T1CE(time2) - T1(time2)]
                  - [T1CE(time1) - T1(time1)]
```

因此 `t1ce_subtraction` 不是简单的 `time2_t1ce - time1_t1ce`。这些关系依据作者说明，本文没有逐体素重算所有差分文件。[数据集论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294954/)、[作者基准仓库](https://github.com/rachitsaluja/UCSF-ALPTDG-benchmarks)

## 4. 影像几何、数据类型与标签

### 4.1 全量头信息核验

对全部 4,768 个 NIfTI 读取头信息，结果如下：

|属性|结果|
|---|---|
|数组尺寸|全部为 `(240, 240, 155)`|
|体素间距|全部为 `(1.0, 1.0, 1.0)` mm|
|由有效 affine 识别的轴向|全部为 `LPS`|
|同一患者内尺寸和 affine 一致性|全部通过；affine 比较绝对容差 `1e-5`|
|NIfTI 头读取失败|0|
|非分割图像存储类型|3,576 个 `float64`|
|分割文件存储类型|595 个 `float64`、596 个 `float32`、1 个 `uint16`|

几何一致不等于已经人工确认每个病例的解剖配准质量。`qform_code`、`sform_code` 并不完全一致，读取时应使用 NIfTI 有效 affine，不能忽略方向或仅凭数组形状判断对齐。

图像可按现有模型接口转为 `float32`。掩码虽然可能以浮点类型存储，但语义是离散类别；转整数之前应验证标签值，重采样掩码必须使用最近邻插值。

### 4.2 单时点组织分区

`time1_seg`、`time2_seg` 的类别对应本地 Excel 的 `Volume_Label*` 字段：

|标签值|本地名称|含义|
|---:|---|---|
|0|Background|背景，即未标注的上述组织区域|
|1|NCR|本地表称 necrotic core；论文描述使用 NETC，即非增强肿瘤核心|
|2|SNFH|周围非增强 FLAIR 高信号区域|
|3|ET|增强组织|
|4|RC|手术切除腔|

NCR 与 NETC 的术语差异应保留在数据字典中，不应据此将全部非增强核心解释为组织学坏死。也不应把 SNFH 全部解释为单纯水肿。[官方门户的组织名称](https://imagingdatasets.ucsf.edu/)、[论文的组织定义](https://doi.org/10.1148/ryai.230182)

本地抽查 `100001`、`100006`、`100100`、`100200` 的部分单时点掩码，观察到与上述对应关系一致的标签和体积；不是对全部掩码体素的标签审计。个别病例不出现某类别是允许的。

不要直接套用 BraTS 常见标签映射：本数据的 `4` 是切除腔，而不是增强肿瘤。

### 4.3 纵向变化分区

`flair_subtraction_seg`、`t1ce_subtraction_seg` 使用另一套标签：

|标签值|含义|
|---:|---|
|0|未标注变化区域|
|1|增加区域|
|2|减少区域|

该映射经过本地抽样核对：`100006_flair_subtraction_seg` 中标签 1、2 分别有 12,212、1,880 个体素，与表内 `SNFH_Inc_Volume`、`SNFH_Dec_Volume` 一致；其 ET 变化掩码标签 1 有 507 个体素，与 `ET_Inc_Volume` 一致。另检查了 `100003` 的 ET 减少区域。

同一患者可以同时具有增加与减少区域。全零变化掩码也确实存在，不能仅因全零就认定文件损坏。

在 1 mm 等体素网格上，体素计数数值等于体积的 mm³ 数值；上述抽样与 Excel 一致。若需 mL，应将 mm³ 除以 1,000；若重采样，须用新体素体积重新计算，不能沿用原计数。

## 5. Excel 工作表与数据字典

本地文件名中的版本为 `V5.0`，影像目录版本为 `v1.0`；二者属于不同命名，不能互相替代。

### 5.1 工作表概览

|工作表|有效数据记录|主键/用途|
|---|---:|---|
|`UCSF-LPTDG`|596 行，298 位患者|`SubjectID + Timepoint`；影像、人口统计和分区体积|
|`Clinical Info`|298 行|`SubjectID`；诊断、分子、治疗与时间信息|
|`Clinical_Dictionary`|20 条|临床字段解释，无单独标题行|
|`TrainTestSplit`|298 行|`SubjectID`；248 TRAIN、50 TEST|
|`UCSF-PDGM Mapping`|62 行|术后队列与术前 UCSF-PDGM 的编号映射，存在下述质量问题|

前三个涉及病例的主表，即影像表、临床表、划分表，患者 ID 集合均与本地 298 个目录一致。读取时应剔除空白尾行，不应把 Excel 的 `max_row` 当作样本数；各工作表命名中的 `LPTDG` 也应按原名读取。

### 5.2 影像表字段

|字段组|原始字段|说明|
|---|---|---|
|标识|`SubjectID`、`Timepoint`、`Full ID`|每位患者对应两个时点|
|诊断|`WHO 2021 Diagnosis`|患者诊断分类|
|扫描设备|`Scanner_ID`、`Scanner`|596 行均为 `GE 3.0 T Discovery MR750`；有 4 个 Scanner_ID|
|人口统计|`Patient Sex`、`Patient Age`|每个时点重复列出；统计患者人数时须去重|
|单区域体积|`NCR_Volume_Label1`、`SNFH_Volume_Label2`、`ET_Volume_Label3`、`RC_Volume_Label4`|分别对应标签 1、2、3、4|
|汇总体积|`WT_Volume_Label1+2+3`、`TC_Volume_Label_1+2`|TC 列名存在疑点，见第 6 节|
|SNFH 变化|`SNFH_CHANGE_OVERALL`、`SNFH_Inc_Volume`、`SNFH_Dec_Volume`、`SNFH_OverallChange_Volume`|纵向类别及变化体积|
|ET 变化|`ET_CHANGE_OVERALL`、`ET_Inc_Volume`、`ET_Dec_Volume`、第 24 列|第 24 列与 SNFH 的净变化列同名，见第 6 节|

### 5.3 临床表字段

以下列号为 Excel 中从 1 起计数的物理列号。说明保留源数据含义；涉及时间方向者应同时查阅第 6 节，不能直接当作已清洗的标签。

|列|原始字段|说明|
|---:|---|---|
|1|`SubjectID`|患者 ID；跨表关联时统一为字符串|
|2|`Days from 1st surgery/DX  to 1st scan`|首次手术/诊断与首扫之间的天数；表头与字典方向不一致|
|3|`Days from 1st scan to 2nd scan`|第二次扫描减第一次扫描的天数|
|4|`EOR at 1st SX`|首次手术切除范围；出现 GTR、STR、BX、MRI 和空白|
|5|`WHO 2021 Diagnosis`|WHO 2021 诊断分类|
|6|`Presumed Diagnosis`|推定诊断；不能与上一列无条件互换|
|7|`Grade`|分级；有缺失|
|8|`MGMT`|`positive`、`unmethylated` 或缺失|
|9|`MGMT methylation index`|甲基化指数；有大量缺失|
|10|`IDH`|`mut`、`WT` 或缺失|
|11|`1p19q`|`codel`、`intact` 或缺失|
|12|`ATRX`|`loss`、`intact` 或缺失|
|13|`Days from death to 1st scan`|字典定义为首扫日期减死亡日期；不能直接视为正的剩余生存天数|
|14|`Days from 1st scan to next progression`|字典定义为后续首次进展日期减首扫日期|
|15|`Days from 1st scan to 1st RT start (neg = RT first)`|首扫与首次放疗开始的相对时间；表头提示与字典方向相反|
|16|`1st Chemo type`|首次化疗类型自由文本，存在 58 种含空值的取值|
|17|`Days from 1st chemo start to 1st scan`|字典定义为首扫日期减首次化疗开始日期|
|18|`On tx at 1st scan (c = chemo, rt, n = none)`|首扫时治疗状态：c=化疗，rt=放疗，n=未处于治疗中|
|19|`On tx at 2nd scan (c = chemo, rt, n = none)`|第二次扫描时治疗状态，编码同上|
|20|`Number of surgeries`|字典称首扫前手术次数；时序使用前须核实|
|21|`Days from 1st scan to 2nd SX`|字典定义为首扫日期减第二次手术日期|
|22|`Days from 1st scan to 3rd SX`|字典定义为首扫日期减第三次手术日期；空值说明存在拷贝笔误|

GTR、STR、BX 通常分别指全切、次全切、活检；MRI 记录对应未手术、由影像诊断的情形，应保留独立编码而非强行赋予切除程度。

作为本地患者级统计，男性 177 人、女性 121 人，年龄范围 21–83 岁；两次扫描间隔中位数约 64.94 天，范围约 17.99–413.08 天。

部分重要临床字段的空白数如下。这些是未知/未提供，不应自动编码为阴性、野生型或无治疗：

|字段|空白人数 / 298|
|---|---:|
|MGMT|169|
|MGMT methylation index|229|
|IDH|79|
|1p19q|200|
|ATRX|222|
|死亡相对时间|172|
|后续进展相对时间|121|

## 6. 本地质量问题与处理边界

下列问题已经通过本地只读核验观察到，本文没有自动修正或删除相应记录。

### 6.1 时间字段存在方向冲突与极端值

- 首次手术/诊断至首扫：表头表达“首扫减首次手术/诊断”，字典却写“首次手术/诊断减首扫”。本地 298 个值中有 2 个负值，最大约 41,745.45 天；不能仅按表头统一翻转。
- 放疗相对时间：表头说“负数表示先放疗”，但字典定义“首扫减放疗开始”，按该定义先放疗应为正数；本地最大约 328,372.36 天。
- 死亡相对时间：126 个非空值中 123 个为负，另 3 个大于 36,525 天。负数与字典的“首扫减死亡”方向一致，但极大正值需单独核对。
- 后续进展相对时间：177 个非空值中有 14 个负值，与“首扫之后的首次进展”的字段描述冲突，最小约 -3,612.42 天。

日期录入、公式或缺失日期参与计算可能解释部分异常，但这只是待核实的可能原因，不能据此直接恢复真实日期。应保留原值、质量标记和转换规则；不要统一取绝对值、截断为零或将异常值解释为超长生存。

仅依据“死亡日期未列出”不能推导充分的删失生存标签。生存任务还需可靠的末次随访/删失时间、事件定义和起算时点；缺失死亡时间不能直接设为生存 365 天。第二次 MRI 的日期也不应默认为末次生存随访日期。

### 6.2 重复表头与组织体积列名

- 影像表第 19 列和第 24 列均名为 `SNFH_OverallChange_Volume`。第 24 列位于 ET 字段组，可能是 ET 净变化列的表头笔误；在确认前应按物理列位置保留两列，不能依赖同名键覆盖。Pandas 也可能给第二列自动加后缀，应显式核对。
- `TC_Volume_Label_1+2` 的表头可疑：在 198 行 NCR、ET、TC 都为数值的记录中，TC 等于 NCR+ET，即标签 1+3；未观察到它等于 NCR+SNFH。不要直接按“1+2”重建肿瘤核心。
- 影像体积单元格还出现字面字符串 `None`，与真正空白并存。缺失值处理需区分原始编码及任务语义，不能把所有 `None` 无条件替换成 0。

### 6.3 UCSF-PDGM 映射不是已验证的一一映射

本地映射表有 62 行，但只有 61 个不同的 UCSF-LPTDG ID 和 61 个不同的 UCSF-PDGM ID：

- 源 ID `100015` 出现两次。
- 目标 ID `UCSF-PDGM-088` 出现两次。
- 表中存在源 ID `100159`，但当前发布目录和其他三个病例主表均不包含它。

不能把“62 行”直接当成当前本地队列中 62 位已经确认的重叠患者。跨术前/术后数据集联合训练时应先核实映射，并以真实患者为分组单位去重、划分，避免同一患者进入训练和测试两侧。

## 7. 官方划分与实验任务边界

`TrainTestSplit` 工作表给出患者级划分：248 TRAIN、50 TEST；所有本地患者都恰有一条划分记录。该划分用于作者的分割基准，并不自动等价于 CLARITY 外部验证协议，也不包含独立 validation 集。[作者数据卡](https://atlas.rsna.org/cards/48595a48-e1e6-4ab5-8844-e895a3482c01)

复现作者分割基准时应沿用该划分；另设验证集应只从 TRAIN 内划出并记录患者清单。若复现 MU 训练模型的 UCSF 外部验证，应单独声明是否冻结模型、使用哪些 UCSF 患者、是否用 UCSF 调参；不能使用外部测试结果选择 checkpoint。

本数据适合研究两时点组织变化、单步影像/潜在状态预测和经过标签核验的患者级结局任务。每位患者只有两个真实 MRI 时点，所以：

- 可以形成 `time1 → time2` 的一次真实观测转移。
- 不能直接构造当前四时点 `s0,s1,s2,s3` 的 teacher-forced H1/H2/H3 训练窗口。
- 不能在缺少 time3/time4 真值时，把递归生成的后续状态当成真实 H2/H3 验证目标。

这些是由本地时点数量与当前 [StagewiseTrajectoryDataset](../src/clarity_rrt_v3/data.py) 的四观测窗口要求推得的接入限制，不是额外的官方数据集标签。

另一个重要边界是未来信息：`time2`、跨时点 subtraction 和 longitudinal change mask 均含后续影像信息。研究“给定 time1 预测未来”时，它们可以作训练目标或事后评价依据，不能作为 time1 的已知输入。基线临床上下文也不能包含后续死亡、进展或尚未发生的治疗信息。

## 8. 与本仓库 CLARITY 接口的关系

当前配置明确使用：

```yaml
data:
  timeline_json: /data/tanyuejun/CLARITY/clinical/MU_Glioma_Post/clinical_latest.json
  mri_data_dir: /data/tanyuejun/CLARITY/dataset/MU-Glioma-Post
  mri_cache_dir: /dev/shm/clarity_mri_cache
  split_file: data/splits.json
```

现有 [CachedMRIVolumeLoader](../src/clarity_rrt_v3/data.py) 读取缓存的 manifest 和 `.npy`，并不直接解析 UCSF 的 Excel 或原始命名。因此仅把 `mri_data_dir` 改成 UCSF 路径，不会完成数据集切换。

如果后续单独实现 UCSF 接入，需要至少完成：患者 ID/时点映射、临床表转换、明确的时间和生存标签规则、独立患者划分、适用于两时点任务的数据加载器，以及单独的缓存目录和 manifest。不得复用或覆盖正在服务 MU 实验的 `/dev/shm/clarity_mri_cache`；本次说明没有建立 UCSF 缓存。

现有上游 `dataset_glioma_all_pairs_text.py` 的模态顺序为 `t1c, t2w, t1n, t2f`。与 UCSF 文件的对应关系应显式固定：

|现有模态名|UCSF 文件模态|顺序|
|---|---|---:|
|`t1c`|`t1ce`|1|
|`t2w`|`t2`|2|
|`t1n`|`t1`|3|
|`t2f`|`flair`|4|

据此堆叠的单时点数组为 `[4, 240, 240, 155]`，转换为 `float32` 后对应现有接口的 `[4,H,W,D]`。这属于当前代码接口映射，不代表 UCSF 或其他基准模型规定了统一模态顺序。

## 9. 最小只读读取示例

以下示例只读取一个病例的两个真实时点，不生成缓存或实验结果，也不构造未经核验的生存标签。当前 `py310` 环境已包含 `nibabel`、`numpy`、`openpyxl`。

```python
from pathlib import Path
import nibabel as nib
import numpy as np
import openpyxl

root = Path(
    "/data/tanyuejun/CLARITY/dataset/"
    "UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0"
)
subject = "100001"
modalities = ("t1ce", "t2", "t1", "flair")

def load_timepoint(timepoint):
    images = [
        nib.load(root / subject / f"{subject}_time{timepoint}_{mod}.nii.gz")
        for mod in modalities
    ]
    assert all(image.shape == images[0].shape for image in images)
    assert all(
        np.allclose(image.affine, images[0].affine, rtol=0, atol=1e-5)
        for image in images
    )
    volumes = [image.get_fdata(dtype=np.float32) for image in images]
    assert all(np.isfinite(volume).all() for volume in volumes)
    return np.stack(volumes, axis=0)

x1 = load_timepoint(1)
x2 = load_timepoint(2)
print(x1.shape, x2.shape)  # (4, 240, 240, 155), float32

book = openpyxl.load_workbook(
    root / "UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx",
    read_only=True,
    data_only=True,
)
print(book.sheetnames)
book.close()
```

该示例的有限性断言会显式暴露非有限体素，不会静默填补；本文的全量检查仅覆盖文件清点和头信息，未证明所有影像的所有体素都有限，也未重新验证 ZIP 全量 CRC。

## 10. 使用条件、引用与版本留存

官方门户说明数据供非商业研究使用，发表成果应引用数据集并注明来自 UCSF Ci2 Datasets for Medical Imaging repository；数据卡注明需要遵守数据使用协议。应保留实际同意的协议，以该协议确定具体权限；CLARITY 代码的 MIT 许可证不能替代数据集权限。本说明不保存此前下载链接中的临时签名参数。[UCSF 官方门户](https://imagingdatasets.ucsf.edu/)、[作者数据卡的使用条件](https://atlas.rsna.org/cards/48595a48-e1e6-4ab5-8844-e895a3482c01)

数据集论文：Brandon K. K. Fields 等，2024，《UCSF 成人治疗后弥漫性胶质瘤纵向 MRI 数据集》（题名中文译述），*Radiology: Artificial Intelligence*，6(4):e230182。[原题名与 DOI: 10.1148/ryai.230182](https://doi.org/10.1148/ryai.230182)

临床表 SHA-256，便于确认本文统计对应的版本：

```text
dfd35d77effb034ae8182633685e1cbe56179f6d413d2db9703772a81a4d1b0d
```

后续实验应另行记录实际纳入患者数、排除原因、训练/验证/测试清单、原始表与转换文件指纹、预处理和标签构造规则。298 位患者是发布规模，不代表经临床标签质控后任一任务的最终有效样本数。
