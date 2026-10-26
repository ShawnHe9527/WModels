# L5PC_exp.py 使用手册

> 基于 Hay et al. 2011 / Beniaguev et al. 2021 的 L5b 锥体细胞多隔室模型。
> 本手册对应 `L5PC2021/L5PC_exp.py`。

---

## 目录

1. [脚本是什么](#1-脚本是什么)
2. [环境准备](#2-环境准备)
3. [快速开始](#3-快速开始)
4. [编号系统详解](#4-编号系统详解)
5. [如何选取想要的 Section 和 Segment](#5-如何选取想要的-section-和-segment)
6. [如何控制脉冲输入频率与输入对象](#6-如何控制脉冲输入频率与输入对象)
7. [三类数据产出说明](#7-三类数据产出说明)
8. [绘图功能](#8-绘图功能)
9. [形态标注功能](#9-形态标注功能)
10. [在 Python 中编程使用（进阶）](#10-在-python-中编程使用进阶)
11. [完整命令行参数表](#11-完整命令行参数表)
12. [输出文件清单](#12-输出文件清单)
13. [可调常量说明](#13-可调常量说明)
14. [常见问题](#14-常见问题)

---

## 1. 脚本是什么

`L5PC_exp.py` 是一个封装好的实验脚本，实现三大目标：

1. **编号系统** —— 对胞体(soma)、轴突(axon)、基树突(basal)、顶树突(apical)分开编号，并提供 section→segment 的完整映射。
2. **Section 级刺激控制** —— 延续模型自带的分段策略（`nseg = 1+2*int(L/40)`），但以 **section 为控制节点**：激活某个 section = 激活它内部的所有 segment。
3. **一次性产出三类数据**：
   - 阈下特征（连续值）
   - 二值激活（0/1）
   - 胞体放电标签 + 胞体阈下电压轨迹

---

## 2. 环境准备

### 2.1 依赖

| 依赖 | 说明 |
|------|------|
| NEURON (Python 版) | 模型运行引擎，`.mod` 文件需已编译 |
| numpy / pandas | 数据处理 |
| matplotlib | 绘图（已设置为 Agg 后端，无需 GUI） |

### 2.2 重要：必须使用带 NEURON 的 Python 环境

系统的 `miniconda3/bin/python` **没有** NEURON 模块，无法运行。本机已确认可用的环境是 **`worm_sim`**：

```bash
# 检查哪个 Python 能 import neuron
/home/neuron/miniconda3/envs/worm_sim/bin/python -c "import neuron; print('OK')"
```

因此所有运行命令都推荐写成：

```bash
/home/neuron/miniconda3/envs/worm_sim/bin/python L5PC_exp.py [参数]
```

> 如果你在 VS Code 里运行，请在右下角选择 `worm_sim` 解释器。

### 2.3 工作目录

脚本会自动 `os.chdir` 到自身所在目录（`L5PC2021/`），所以必须保证以下文件都在该目录下：

```
L5PC2021/
├── L5PC_exp.py              ← 本脚本
├── L5PCbiophys5b.hoc        ← 生物物理模型（离子通道）
├── L5PCtemplate_2.hoc       ← 细胞模板（形态加载 + 分段）
├── morphologies/cell1.asc   ← 形态学文件
└── mods/                    ← 离子通道/突触机制（需已编译）
```

---

## 3. 快速开始

### 3.1 单次仿真（默认 1000ms）

```bash
cd /home/neuron/L5PC/L5PC2021
/home/neuron/miniconda3/envs/worm_sim/bin/python L5PC_exp.py
```

运行后会：
1. 加载模型并打印编号统计；
2. 保存编号目录（section/segment）；
3. 打印几个编号示例；
4. 随机选取一部分 section 激活，跑一次仿真；
5. 提取特征、绘图、保存到 `L5PC_exp_outputs/`。

### 3.2 生成小数据集（多次随机仿真）

```bash
/home/neuron/miniconda3/envs/worm_sim/bin/python L5PC_exp.py --trials 20
```

每次 trial 会随机选取不同数量、不同组合的 section 激活，模拟不同的输入模式。

### 3.3 自定义仿真时长

```bash
/home/neuron/miniconda3/envs/worm_sim/bin/python L5PC_exp.py --duration 2000
```

---

## 4. 编号系统详解

### 4.1 编号规则

| 编号类型 | 格式 | 示例 | 含义 |
|---------|------|------|------|
| Section ID | `{区域}_{区域内序号}` | `basal_3` | 基树突的第 3 个 section |
| Segment ID | `{section_id}_{段内序号}` | `basal_3_0` | 基树突第 3 个 section 的第 1 个 segment |

四大区域：

| 区域 | region 前缀 | Section 数 |
|------|------------|-----------|
| 胞体 | `soma` | 1 |
| 轴突 | `axon` | 2 |
| 基树突 | `basal` | 84 |
| 顶树突 | `apical` | 109 |

### 4.2 关键约定

- **区域内序号 `i`**：按 NEURON 内部 SectionList 的迭代顺序（从胞体向外），每次加载稳定一致、可复现。
- **段内序号 `j`**：从 0 开始，**`j=0` 表示最靠近胞体（x=0 端）的 segment**，`j` 越大越靠近树突末梢。

例如 `apical_0`（L=46.1µm，nseg=3）：

```
apical_0_0   x=0.167   距胞体 7.7µm    ← 近胞体
apical_0_1   x=0.500   距胞体 23.1µm
apical_0_2   x=0.833   距胞体 38.4µm   ← 远胞体
```

### 4.3 编号目录文件

运行一次后，`L5PC_exp_outputs/` 下会生成两个 CSV，方便你离线查阅：

| 文件 | 内容 |
|------|------|
| `section_catalog.csv` | 每个 section 的编号、区域、长度、nseg、距胞体距离等 |
| `segment_catalog.csv` | 每个 segment 的编号、所属 section、x 位置、距胞体距离等 |

---

## 5. 如何选取想要的 Section 和 Segment

### 5.1 方法一：直接在命令行传给 `--mark-*` / `--plot-sections`

你只需要知道 section/segment 的编号字符串即可。编号来自上面的规则和 CSV 目录。

### 5.2 方法二：在 Python 里交互式查询（推荐）

进入 Python（用 worm_sim 解释器），逐行执行：

```python
from L5PC_exp import L5PCExperiment

exp = L5PCExperiment()   # 加载模型 + 编号 + 放突触

# ① 列出某区域的所有 section 编号
exp.section_ids('basal')    # → ['basal_0', 'basal_1', ..., 'basal_83']
exp.section_ids('apical')   # → ['apical_0', ..., 'apical_108']

# ② 查看某个 section 的详细信息（长度、距胞体距离、内部 segment 列表）
exp.show_section('apical_50')

# ③ 列出某个 section 的所有 segment 编号
exp.segments_of('apical_50')   # → ['apical_50_0', 'apical_50_1', 'apical_50_2']
```

### 5.3 按"距胞体距离"批量选取（常见需求）

```python
import numpy as np

cat = exp.section_catalog

# 选取顶树突中，近端距离在 600~700 µm 之间的所有 section
mask = (cat['region'] == 'apical') & \
       (cat['proximal_dist_um'] >= 600) & \
       (cat['proximal_dist_um'] <= 700)
selected = cat.loc[mask, 'section_id'].tolist()
print(selected)
```

---

## 6. 如何控制脉冲输入频率与输入对象

### 6.1 输入对象（激活哪些 section）

由 `run()` 的 `section_ids` 参数决定：

```python
# 只激活 3 个 section
record = exp.run(section_ids=['basal_3', 'apical_50', 'apical_51'])
```

- 传什么，就激活什么；
- 每个选中的 section 会被**自动展开为它内部的所有 segment**，对每个 segment 施加 Poisson 脉冲。

**命令行方式**（用 `--activate-sections` 指定要激活的 section）：

```bash
# 只激活顶树突第 20 号 section
python L5PC_exp.py --activate-sections apical_20

# 激活多个指定 section
python L5PC_exp.py --activate-sections apical_20 basal_3 apical_50
```

> 不指定 `--activate-sections` 时，脚本会**随机**选取 section 激活（每次 trial 不同组合）。
> 指定后，所有 trial 都固定激活这些 section。

### 6.1b 直接激活特定 segment（segment 级控制）

用 `--activate-segments` 直接指定要激活的 segment（**不需要展开 section**）：

```bash
# 只激活顶树突第 20 号 section 的第 0、2 号 segment
python L5PC_exp.py --activate-segments apical_20_0 apical_20_2
```

> 可以同时混合使用 section 和 segment：
> ```bash
> python L5PC_exp.py --activate-sections basal_3 --activate-segments apical_20_0
> ```
> 会同时激活 `basal_3`（展开为其所有 segment）+ `apical_20_0`（单个 segment）。

### 6.2 脉冲频率

频率有**两个通道**：

| 通道 | 参数 | 默认值 |
|------|------|--------|
| 兴奋性输入 | `ex_rate_hz`（或命令行 `--ex-rate`） | 500 Hz |
| 抑制性输入 | `inh_rate_hz`（或命令行 `--inh-rate`） | 400 Hz |

> ⚠️ 频率单位是**每个 segment** 的 Poisson 平均发放率，不是整个 section 的总和。
> 一个 section 有 nseg 个 segment，其总输入 ≈ `nseg × rate`。

**命令行方式：**

```bash
python L5PC_exp.py --ex-rate 800 --inh-rate 100
```

**Python 方式：**

```python
record = exp.run(
    section_ids=['basal_3'],
    ex_rate_hz=800.0,    # 兴奋性 800Hz/segment
    inh_rate_hz=100.0,   # 抑制性 100Hz/segment
)
```

---

## 7. 三类数据产出说明

每次仿真产出以下数据：

### 7.1 阈下特征 + 二值激活（`trial_{i}_section_features.csv`）

每行是一个 section，关键列：

| 列名 | 含义 | 类型 |
|------|------|------|
| `section_id` | 编号 | str |
| `region` | 区域 | str |
| `mean_subthreshold_v` | 平均阈下电压（超过阈值的部分被截掉后求平均） | 连续值 |
| `peak_v` | 峰值电压 | 连续值 |
| `depolarization_area` | 去极化面积 ∫max(0, V-V_rest)dt | 连续值 |
| `activated` | 是否超阈值发放（峰值 > 阈值记为 1） | 0/1 |

### 7.2 胞体标签 + 轨迹（`trial_{i}_soma.npz`）

用 `numpy.load` 读取，包含：

| 键 | 含义 |
|----|------|
| `time` | 时间轴 (ms) |
| `soma_v` | 胞体原始电压轨迹 |
| `soma_sub_v` | 胞体阈下电压轨迹（截掉超阈值部分） |
| `soma_spike` | 胞体是否放电（0/1） |
| `active_sections` | 本次激活的 section 列表 |

读取示例：

```python
import numpy as np
d = np.load('L5PC_exp_outputs/trial_0_soma.npz', allow_pickle=True)
print(d['soma_spike'])
print(d['soma_v'])
```

### 7.3 编号目录（只生成一次）

`section_catalog.csv`、`segment_catalog.csv`（见第 4 节）。

---

## 8. 绘图功能

### 8.1 电压轨迹图

用 `--plot-sections` 指定要画的 section，会生成一张多子图：

- 左图：**胞体电压轨迹**（含发放阈值红线、静息电位蓝虚线，标题标注是否放电）；
- 右侧：每个选定 section 内部**各 segment 的电压轨迹**叠加。

```bash
# 指定画哪些 section
python L5PC_exp.py --plot-sections basal_0 apical_0 apical_50

# 不指定则自动选"峰值最高的 4 个 section"
python L5PC_exp.py --plot

# 关闭绘图
python L5PC_exp.py --plot false
```

输出：`L5PC_exp_outputs/trial_{i}.png`

### 8.1b 绘制特定 segment 的电压轨迹

用 `--plot-segments` 指定要画的 segment（紫色粗线，每个 segment 一个子图）：

```bash
# 画胞体 + 两个指定 segment
python L5PC_exp.py --plot-segments apical_20_0 apical_20_2

# 混合：画胞体 + 指定 section + 指定 segment
python L5PC_exp.py --plot-sections basal_3 --plot-segments apical_20_0
```

### 8.2 形态标注图

用 `--mark-sections` / `--mark-segments` 在神经元形态（X-Y 俯视投影）上标注你选中的部位。

```bash
# 只高亮 section
python L5PC_exp.py --mark-sections basal_0 basal_2 apical_50

# 同时标注具体 segment
python L5PC_exp.py \
    --mark-sections basal_0 apical_50 \
    --mark-segments apical_50_0 apical_50_2
```

标注效果：

| 对象 | 显示 |
|------|------|
| 整个形态 | 各区域基础色细线（绿=胞体、橙=轴突、红=基树突、蓝=顶树突） |
| 选中 Section | 红色粗线 + 白色标签 |
| 选中 Segment | 蓝色圆点 + 蓝色数字 |

输出：`L5PC_exp_outputs/trial_{i}_markup.png`

---

## 9. 形态标注功能

（内容同 8.2，这里补充编程调用方式）

```python
from L5PC_exp import plot_morphology_markup

fig = plot_morphology_markup(
    exp,
    section_ids=['basal_0', 'apical_50'],   # 高亮的 section
    segment_ids=['apical_50_0'],            # 标注的 segment
    save_path='my_markup.png',              # 保存路径（None 则只返回 fig）
    projection='xy',                         # 'xy' 俯视 / 'xz' 侧视
)
```

---

## 10. 在 Python 中编程使用（进阶）

除了命令行，你可以把 `L5PC_exp.py` 当库来用，实现完全自定义的实验流程。

### 10.1 完整自定义流程示例

```python
from L5PC_exp import L5PCExperiment

exp = L5PCExperiment()   # 加载模型、编号、放突触

# 1. 选取要激活的 section（例如所有顶树突远端）
cat = exp.section_catalog
distal = cat[(cat['region'] == 'apical') &
             (cat['proximal_dist_um'] > 500)]['section_id'].tolist()

# 2. 运行仿真
record = exp.run(
    section_ids=distal,
    duration_ms=1500,
    ex_rate_hz=600.0,
    inh_rate_hz=200.0,
    seed=123,
)

# 3. 提取特征
features = exp.extract_features(record)

# 4. 查看结果
print("胞体是否放电:", features['soma_spike'])
print(features['section_features'].head())

# 5. 保存
from L5PC_exp import save_outputs
save_outputs('my_outputs', 0, features,
             exp.section_catalog, exp.segment_catalog,
             record['active_sections'])
```

### 10.2 可复用的主要方法与属性

| 成员 | 类型 | 说明 |
|------|------|------|
| `exp.cell` | NEURON 对象 | 底层细胞模型 |
| `exp.section_catalog` | DataFrame | section 编号目录 |
| `exp.segment_catalog` | DataFrame | segment 编号目录 |
| `exp.section_segments` | dict | section_id → segment_id 列表 |
| `exp.segment_obj` | dict | segment_id → (Section, seg) |
| `exp.section_ids(region)` | 方法 | 取 section_id 列表 |
| `exp.segments_of(section_id)` | 方法 | 取某 section 的 segment_id |
| `exp.show_section(section_id)` | 方法 | 打印 section 详情 |
| `exp.run(...)` | 方法 | 刺激 + 运行仿真 |
| `exp.extract_features(record)` | 方法 | 提取三类特征 |

---

## 11. 完整命令行参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--trials` | int | 1 | 仿真次数（随机选取激活 section） |
| `--duration` | int | 1000 | 每次仿真时长 (ms) |
| `--ex-rate` | float | 500.0 | 每个 segment 兴奋性输入率 (Hz) |
| `--inh-rate` | float | 400.0 | 每个 segment 抑制性输入率 (Hz) |
| `--outdir` | str | L5PC_exp_outputs | 输出目录 |
| `--plot` | bool | True | 是否绘制电压轨迹图 |
| `--plot-sections` | 多个 str | None | 要绘制电压轨迹的 section_id |
| `--mark-sections` | 多个 str | None | 要在形态图上高亮的 section_id |
| `--mark-segments` | 多个 str | None | 要在形态图上标注的 segment_id |
| `--activate-sections` | 多个 str | None | 指定要激活的 section_id（覆盖随机选取） |
| `--activate-segments` | 多个 str | None | 指定要直接激活的 segment_id |
| `--plot-segments` | 多个 str | None | 指定要绘制电压轨迹的 segment_id |

---

## 12. 输出文件清单

一次运行后，`L5PC_exp_outputs/` 目录结构：

```
L5PC_exp_outputs/
├── section_catalog.csv              ← section 编号目录（只生成一次）
├── segment_catalog.csv              ← segment 编号目录（只生成一次）
├── trial_0_section_features.csv     ← 阈下特征 + 二值激活
├── trial_0_soma.npz                 ← 胞体标签 + 轨迹
├── trial_0.png                      ← 电压轨迹图（--plot 开启时）
├── trial_0_markup.png               ← 形态标注图（--mark-* 指定时）
├── trial_1_...
└── ...
```

---

## 13. 可调常量说明

这些常量定义在脚本顶部，按需修改：

| 常量 | 默认值 | 含义 |
|------|--------|------|
| `SPIKE_THRESHOLD` | -20.0 | 超阈值发放判据 (mV)，与 `run_L5PC_gui.count_spikes` 一致 |
| `REST_VOLTAGE` | -76.0 | 静息电位基线 (mV)，用于算去极化面积 |
| `V_INIT` | -76.0 | 仿真初始化膜电位 |
| `DT` | 0.025 | 积分步长 (ms) |
| `CELS` | 34.0 | 温度 (deg C) |

---

## 14. 常见问题

### Q1：`ModuleNotFoundError: No module named 'neuron'`

你用的是不带 NEURON 的 Python。请改用：

```bash
/home/neuron/miniconda3/envs/worm_sim/bin/python L5PC_exp.py
```

### Q2：如何知道某个编号对应神经元的哪个位置？

用 `--mark-sections` 把该 section 标到形态图上，直接看图定位；或 `exp.show_section('basal_3')` 看它的距胞体距离。

### Q3：为什么胞体/soma 有时也显示 `activated=1`？

`soma` 的 `peak_v` 取的是胞体电压峰值，所以胞体一旦放电，`soma` 这一行的 `activated` 就是 1。这是设计使然（把胞体也当作一个"观测点"）。

### Q4：如何做 segment 精细度研究？

脚本已支持 segment 级控制：

```bash
# 激活特定 segment
python L5PC_exp.py --activate-segments apical_50_0 apical_50_2
# 绘制特定 segment
python L5PC_exp.py --plot-segments apical_50_0
# 在形态图上标注特定 segment
python L5PC_exp.py --mark-segments apical_50_0
```

编程方式：`exp.segments_of(section_id)` 可拿到任意 section 的 segment 列表；`run(segment_ids=[...])` 可直接按 segment 刺激。

### Q5：图太多/太慢怎么办？

- 关闭绘图：`--plot false`；
- 减少 `--trials`；
- 缩短 `--duration`。

---

*手册结束。脚本路径：`L5PC2021/L5PC_exp.py`*
