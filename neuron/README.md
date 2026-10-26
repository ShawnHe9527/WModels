# neuron/ — NEURON 模型资产（官方原样）

## 来源与真实性

本目录下所有文件**逐字节**来自 Beniaguev et al. 2021（*Neuron*）
官方仓库 <https://github.com/SelfishGene/neuron_as_deep_net> 的
`L5PC_NEURON_simulation/` 子目录。

已用文件大小 + `diff` 双向核对（2026-10-19）：

| 文件 | 字节数 | 与官方一致 |
|---|---|---|
| `hoc/L5PCtemplate_2.hoc` | 5829 | ✅ |
| `hoc/L5PCbiophys5b.hoc` | 2513 | ✅ |
| `hoc/init.hoc` | 1288 | ✅ |
| `mods/*.mod`（18 个） | 见各文件 | ✅ |
| `morphologies/cell1.asc` | 396544 | ✅ |

> ⚠️ **不要修改这些文件。** 任何改动都会让我们与论文的仿真设置脱钩，
> 使得结果无法与原文对照。

## 目录内容

```
hoc/
├── init.hoc              NEURON 启动入口（GUI demo 菜单）
├── L5PCbiophys5b.hoc     通道分布 + 几何（biophys()）
└── L5PCtemplate_2.hoc    L5PCtemplate 类：形态导入、nseg 计算、locateSites
mods/                     18 个机制文件
morphologies/cell1.asc    形态（196 个 section）
reference/                官方参考脚本（只读，用于对照）
```

## 首次使用：编译机制

```bash
cd WModel2/neuron/mods
PATH=/home/neuron/miniconda3/envs/worm_sim/bin:$PATH nrnivmodl .
# 产出 x86_64/libnrnmech.so（约 198 KB，已编译好，一般无需重做）
```

## 加载顺序（硬约束）

```python
from neuron import h
h.nrn_load_dll("WModel2/neuron/mods/x86_64/libnrnmech.so")  # 必须最先
h.load_file("stdrun.hoc")          # 其次
h.load_file("import3d.hoc")        # 形态导入
h.load_file("L5PCbiophys5b.hoc")   # 通道
h.load_file("L5PCtemplate_2.hoc")  # 模板类
h.celsius = 34.0                   # 体温
cell = h.L5PCtemplate("neuron/morphologies/cell1.asc")
```

⚠️ 不先 `nrn_load_dll` 会报 `Im is not a MECHANISM`。
⚠️ 顺序错了会段错误，不是抛异常。

## 已实测可用的属性名（避免踩坑）

| 想读的量 | 正确写法 | 错误写法 |
|---|---|---|
| 钙浓度 | `seg.cai` | ~~`seg.cai_CaDynamics_E2`~~（hasattr=False）|
| Ca_HVA 的 m | `seg.m_Ca_HVA` | — |
| Ih 的 m | `seg.m_Ih` | — |
| Ca_LVAst 的 h | `seg.h_Ca_LVAst` | — |
| 机制列表 | `h.psection(sec=s)` **返回字符串** | ~~`['density_mechs']`~~（不是 dict）|

WModel2 实测（2026-10-19）：196 sections，642 compartments，
`apic[51]` nseg=11，父段 `apic[50]@1.0`，子段 `apic[52]`+`apic[53]`。

## 关键模型事实（实测，非文档抄录）

| 项 | 值 |
|---|---|
| section 总数 | 196（soma 1 + axon 2 + basal 84 + apical 109）|
| basal compartment | 262 |
| apical compartment | 377 |
| `apical_51` | L = 215.48 µm，nseg = **11**，diam 1.54→1.29 µm |
| nseg 公式（官方） | `1 + 2*int(L/40)` |
| `apical_51` 的 x=0 | 距胞体 **700.30 µm**（近端），连父段 `apic[50]`(1.0) |
| `apical_51` 的 x=1.0 | 距胞体 **915.78 µm**（远端），**分叉点**（`apic[52]` + `apic[53]`）|
| 段内电紧张衰减 | **仅 4%**（x=0 → x=1 的稳态去极化保留 104%）|

## 机制清单（18 个）

**离子通道（11）**：
`CaDynamics_E2`、`Ca_HVA`、`Ca_LVAst`、`Ih`、`Im`、`K_Pst`、`K_Tst`、
`NaTa_t`、`NaTs2_t`、`Nap_Et2`、`SK_E2`、`SKv3_1`

**突触（6）**：
`ProbAMPANMDA2`、`ProbAMPANMDA_3`、`ProbAMPANMDA_EMS`、
`ProbGABAAB_EMS`、`ProbUDFsyn2`

**其他（1）**：`epsp`（注入电流的点过程，不是突触）

## 突触在哪？—— 重要澄清

**`.hoc` / `.mod` 里没有任何突触实例。** 突触由官方 Python 脚本
`reference/simulate_L5PC_and_create_dataset.py` 在运行时动态挂到
**每一个 segment** 上：

```python
for segInd, segment in enumerate(allSegments):
    exSynapse  = DefineSynapse_NMDA(segment)    # 每段 1 个兴奋
    inhSynapse = DefineSynapse_GABA_A(segment)  # 每段 1 个抑制
```

`allSegments` = 遍历 `basal + apical` 的**每个 section 的每个 segment**。

⇒ `apical_51` 有 `nseg = 11` ⇒ **11 对（22 个）突触**。

> 教训：模型 = 形态 + 通道（`.hoc`/`.mod`）+ **实验装置（Python）**。
> 只 grep `.hoc` 会得出"没有突触"的错误结论。

## 官方突触参数（必须照抄）

| | 兴奋（默认 NMDA） | 抑制（GABA_A） |
|---|---|---|
| 机制 | `ProbAMPANMDA2` | `ProbUDFsyn2` |
| `gmax` | **0.0004 µS** | **0.001 µS** |
| τ_rise | AMPA 0.3 / NMDA 2.0 ms | 0.2 ms |
| τ_decay | AMPA 3.0 / NMDA 70.0 ms | 8 ms |
| `e` | 0 mV | **−80 mV** |
| 可塑性 | `Use=1, u0=0, Dep=0, Fac=0`（关）| 同 |
| 驱动方式 | `NetCon(None, syn).event(t)` | 同 |

NMDA 的 Mg²⁺ 阻塞由 `ProbAMPANMDA2.mod` 内的 `mggate` 实现（电压依赖），
这正是论文"复杂度需要 5–8 层"的来源。

## 官方仿真协议（供对照）

| 项 | 值 |
|---|---|
| 时长 | 6 s × 128 trials/seed |
| 输入 | 每个 segment 一条独立棘波串（空格 + 时间都随机）|
| 兴奋率范围 | 0–800 spikes / 100 ms / 整棵树 |
| 抑制率 | 兴奋 + [−600, +200] |
| 平滑 | 高斯核 σ ∈ {25…600} ms，随时间分段重采样 |
| 记录 | soma V、nexus V（`apic[50]` 的 0.9 处）、全部 segment V |
| 输出 | soma 阈上（>−25 mV）局部极大时刻 ⇒ **spike 输出** |
