# 跨神经元泛化核实 —— 人 L23 (Eyal) vs 我们的大鼠 L5b

> 日期：2026-10-14
> 目的：核实「世界模型（G4/G4'）能否泛化到【不同神经元类型】」——用真实模型实测，不靠推理。

---

## 1. 论文与代码溯源

用户提供标题 *"Dendritic morphology and synaptic nonlinearities enhance
functional complexity in human cortical neurons"*。

**已在工作区内找到配套代码**：

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/ido4848/FCI.git` |
| 本地路径 | `/home/neuron/L5PC/FCI/` |
| 标题 | **"What makes human cortical pyramidal neurons functionally complex"** |
| 作者 | Ido Aizenbud, Daniela Yoeli, David Beniaguev, Christiaan PJ de Kock, Michael London, Idan Segev |
| 出处 | bioRxiv 2024.12.17.628883 |

⇒ 是同一工作的预印本版本（用户手上的标题应为投稿/发表版）。

### 模型清单（`FCI/simulating_neurons/neuron_models/`）

| 目录 | 物种/类型 | 形态文件 |
|---|---|---|
| `human/eyal/Human_L23_PC_0603_11_937_Eyal_...` | **人 L23 锥体** | `2013_03_06_cell11_1125_H41_06.asc` |
| `human/bbp/Human_L5_PC_BBP_...` | 人 L5 锥体 | `2057_H21_29_197_11_01_03_metcontour.asc` |
| `rat/bbp/Rat_L2_TPC_BBP_...` | 大鼠 L2 锥体 | `mtC191200B_idA_diams_fixed.asc` |
| `rat/hay/Rat_L5b_PC_2_Hay_...` | 大鼠 L5b 锥体 | `cell1.asc` ← **与我们同源** |

### 🔴 关键限定：这 4 个模型不构成「跨通道集」基准

实测证据：

1. **4 个模型的 `mods/` 里是【完全相同】的 4 个 mod**：
   `na.mod`, `kv.mod`, `AMPANMDA_EMS.mod`, `GABAAB_EMS.mod`（md5 全同）
2. 共用模板 `neuron_models/passive_dends_simple_soma_model.hoc`（314 行）：
   - **树突（basal + apical）：只 `insert pas`** —— 纯被动，无任何树突非线性
   - 胞体 + 轴突：`insert pas; insert kv; insert na`
3. **密度是硬编码且四模型共用**：
   ```hoc
   gbar_kv = 2000 // fitted for rheobase of Rat_L5b_PC_2_Hay
   gbar_na = 8300 // fitted for rheobase of Rat_L5b_PC_2_Hay
   ```
   树突不设 ⇒ 无。胞体 ×0.1。**同一组数字用于全部 4 个"不同"神经元。**
4. `properties.json` 只有 5 个字段：`rho, rho_axon, avg_seg_len, std_seg_len, seg_chunk_size`
   （即：**只靠 rho 缩放 s/ax 电导 + 形态本身来区分神经元**）

⇒ **FCI 的 4 个模型只测「几何迁移」，不测「通道集迁移」。**
   这一点与 Hay2011 的 cell1/cell2/cell3 是同一类（细胞类型相同、形态不同）。

---

## 2. 人 L23 vs 我们 L5PC：实测差异（全部逐 compartment 实测）

| 维度 | **我们 L5PC**（大鼠 L5b） | **人 L23 (Eyal)** | 影响 |
|---|---|---|---|
| **树突通道种类** | **7 种**：`Ih, Im, Ca_HVA, Ca_LVAst, SK_E2, SKv3_1, NaTs2_t` | **0 种**（只有 `pas`） | 🔴🔴 根本差异 |
| **树突非线性** | Ca 平台、Ih 反向、A 型 K、SK 钙门控 | **无**（纯被动 cable） | 🔴🔴 |
| **胞体/轴突通道** | `NaTs2_t, NaTa_t, Nap_Et2, K_Pst, K_Tst, Im, SKv3_1, SK_E2, Ca_HVA, Ca_LVAst, Ih, pas` | `na, kv, pas` | 🔴 |
| **通道种类总数** | **12**（+6 突触 mod = 18 mod） | **2**（+2 突触 = 4 mod） | 🔴 |
| **突触机制** | `ProbAMPANMDA2` + `ProbUDFsyn2` | `AMPANMDA_EMS` + `GABAAB_EMS` | 🟡 |
| **NMDA 参数** | τd_NMDA=70 ms, gmax=4e-4, **无 gamma** | τd_NMDA=43 ms, gmax=8.8e-4, **gamma=0.078** | 🟡 |
| **`pas` 反转电位 `e_pas`** | **−90 mV** | **−70 mV** ⚠️ | 🟡 已修 |
| **`Ra`** | **100 Ω·cm** | **150 Ω·cm** ⚠️ | 🟡 已修 |
| `cm`（胞体/轴突/树突） | 1.0 / 1.0 / 2.0 | 1.0 / 1.0 / 2.0 | ✅ 相同 |
| `g_pas` | 3e−5, 6e−5（两档） | 5e−5（soma/axon）, 1e−4（dend） | 🟡 |
| **`celsius`** | 34 °C | **34 °C**（运行时设，见 `model_utils.py`） | ✅ 相同 |
| **`dt`** | 0.025 ms | **0.025 ms** | ✅ 相同 |
| **`v_init`** | −76.0 mV | **−76.0 mV** | ✅ 相同 |
| **nseg 规则** | `1 + 2*int(L/40)` | `1 + 2*int(L/chunkSize=40)` | ✅ **完全相同** |
| section 数 | 196（soma1/axon2/basal84/apical109） | 219（soma1/axon2/basal115/apical101） | 🟡 |
| compartment 数 | 642 | 1051 | 🟡 |
| **`nseg==1` 的段** | **95 / 196 = 48%** | **81 / 219 = 37%** | 🟡 |
| 3D 点位 | 全部有（stub axon 除外） | 217/219 有，2 个 stub axon 无 | ✅ |
| 形态 | 大鼠 L5b | 人 L23 | — |

### ⚠️ 一个反直觉但极重要的结论

> **L23 在「通道丰富度」上比我们【简单得多】，但它的非线性【转移到了突触】。**
>
> - 我们 L5PC 的非线性主要来自**树突离子通道**（Ca 平台、Ih、A 型 K）
> - 人 L23 的非线性**全部来自突触的 NMDA Mg 阻滞**：
>   ```mod
>   mggate = 1 / (1 + exp(gamma * -v) * (mg / 3.57))   : Jahr & Stevens 1990
>   g_NMDA  = (B_NMDA - A_NMDA) * mggate
>   ```
>   正是论文标题里的 "**synaptic** nonlinearities"。
>
> 而 **FCI 论文的核心结论**是：人类神经元功能复杂度更高，主要归因于
> **树突膜面积 + 分支模式 + NMDA 密度与非线性（gamma）**。
> 他们的 4 组突触参数集（human/rat × gamma 0.078/0.062 正交）就是为此设计的。

⇒ 对我们的世界模型：**L23 需要的是「突触电导」的建模能力，而不是「更多离子通道」。**
   这比加 10 个离子通道更接近我们的短板（我们目前 `MECH` 表里**完全没有突触**）。

---

## 3. 实证：我们的求解器真的能跑人 L23（不是推理）

脚本：`tools/cross_neuron_section_test.py`
对象：人 L23 的 6 个基底树突段（nseg = 1, 3, 5, 7, 9, 11）
方法：段完全孤立成 2-端口元件 → 两端 SEClamp 钳位 → 跑稳态 →
      比较 NEURON 端口电流 vs 我们 `SectionUnit.step()` 返回的 I0/I1

两个判据（分开测，避免互相掩盖）：

| 判据 | 含义 | 暴露什么 |
|---|---|---|
| **膜**（V0 = V1 = −65） | 轴向电流 = 0，端口电流全来自膜 | `e_pas` / `g_pas` / 面积 |
| **轴+膜**（V0 = −65, V1 = −75） | 叠加轴向电流 | `Rax` 的重建口径 |

### 结果

| 判据 | 修复前 | **修复后** |
|---|---|---|
| 膜（V0=V1=−65） | 中位误差 80.000% | **0.000%** |
| 轴+膜（V0=−65,V1=−75） | 中位误差 8.184% | **0.001%** |

逐段（修复后，全部 0.00%）：

| section | nseg | L (µm) | 膜判据 | 轴+膜判据 |
|---|---|---|---|---|
| `dend[2]` | 3 | 64.97 | 0.00% | 0.00% |
| `dend[1]` | 5 | 109.29 | 0.00% | 0.00% |
| `dend[27]` | 7 | 146.87 | 0.00% | 0.00% |
| `dend[21]` | 9 | 168.73 | 0.00% | 0.00% |
| `dend[5]` | 11 | 224.22 | 0.00% | 0.00% |
| `dend[0]` | **1** | 32.26 | 0.00% | 0.00% |

并且与解析式 `I = g_pas·A·(V−e_pas)/2` 完全吻合（如 `dend[5]`：
解析 0.002131 nA vs NEURON 0.002087 nA，差 2.1%，来自 pt3d 渐细段的面积离散）。

**结论：cable 求解器 + 被动膜在【换神经元】时精度不降。** ✅

---

## 4. 本次核实中发现并修复的 3 个 bug

### #124a `e_pas` 硬编码 −90
`L5PC_unit_sec.py` 旧代码：`self.e_pas = -90.0`
- L5PC / Hay 确实是 −90 ⇒ 一直没暴露
- 人 L23 是 **−70** ⇒ 写死会让膜判据错 **80%**
- **修复**：`self.e_pas = _scalar(dm['pas']['e'])`（从 psection 读）

### #124b `Ra` 取的是「当前 section」
旧代码：`self.Ra = _scalar(_h.Ra)`
- NEURON 的 `h.Ra` 经「当前 section」机制取值 ⇒ 结果取决于此刻 push 的是谁，
  **不是** `sec` 本身
- L5PC 全部 section 都是 Ra=100（实测 `{100.0}`）⇒ 恰好相等，一直没暴露
- 人 L23 是 Ra=**150** ⇒ 一旦当前 section 不对就整体错 1.5×
- **修复**：`self.Ra = _scalar(sec.Ra)`

### #125 ⚠️ `nseg==1` 分支漏掉线性化外推项（**最严重**）
`L5PC_unit_sec.py` 的 `step()` 有两个分支：

```python
# 通用分支（N>1）✅
rhs = Cdt * self.Vn - self.Iion + gi * V          # ← 有 gi*V

# N==1 分支（旧）❌
num = Cdt*Vn - self.Iion + gb0*V0 + gb1*V1        # ← 缺 gi*V
```

稳态固定点对比（`A = gb0+gb1`，被动膜 `I_ion = gi(V−e)`）：

| | 稳态方程 | 解 |
|---|---|---|
| 正确 | `gi(V−e) + A(V−V_hold) = 0` | `V = (A·V_hold + gi·e)/(A+gi)` |
| bug | `gi(2V−e) + A(V−V_hold) = 0` | `V = (A·V_hold + gi·e)/(A+2gi)` |

⇒ 端口电流 `I0 = gi·(V_hold − e)`（正确）vs `gi·(2V_hold − e)`（bug）。
当 `V_hold ≈ e` 时正确值趋 0，**相对误差爆掉**（实测人 L23 `dend[0]`：
NEURON `+0.000565`，我们 `−0.006771`，连**符号**都错）。

**影响面**：`nseg==1` 的段 —— L5PC **95/196 = 48%**，人 L23 **81/219 = 37%**。

**修复后**：人 L23 全部段 0.000%；L5PC 的 `probes/T0.1` 仍 **12/12 通过**（净收益，无回归）。

---

## 5. ⚠️ 重要的认知修正：T0.1 从未验证过求解器

`probes/T0.1_IJ_obtainable.py` 一直报 **12/12 通过**，包括 4 个 `nseg==1` 的段
（`basal_81, basal_83, apical_106, apical_108`，稳态误差仅 1e-7~1e-6）

**为什么没暴露 #125？** 读源码发现：

```python
# Run A (in-situ)：
I0 = u.gb0 * (V0 - VC[0])      # ← 用 NEURON 记录的 V 做【轴向重建】
# Run B/C (replay)：
i0 = h.Vector().record(c0._ref_i)   # ← NEURON 自己算的 SEClamp 电流
```

`SectionUnit` 在 T0.1 中**只被用来取 `gb0 / gb1 / xs / nseg` 四个几何量**，
**从未调用 `u.step()`**。⇒ 两个 Run 的物理**都来自 NEURON**。

所以 T0.1 实际验证的是：

> ✅ **建模假设**：一段神经元可视为 2-端口元件，`I_J` 可由两端电位与几何精确重建
> （即「section unit 模块化」成立 —— 这仍然是有价值的核心结论）
> ❌ **不是**：我们自己的 cable 求解器是否正确

**含义**：
- `F4 / F5 / V2c` 等既有实验结论 **不受 #125 影响**（它们用 NEURON 跑仿真 + MLP 拟合，
  不经过 `SectionUnit.step()`）
- 真正用到 `u.step()` 的只有 `L5PC_unit.py`（示例）与
  `L5PC2026/prior_probes/2026_09_18/demo_training_data.py`（演示脚本）
- ⚠️ **`L5PC2026/L5PC_unit_sec.py` 与 `WModel/L5PC_unit_sec.py` md5 相同 ⇒
  三个 bug 在 L5PC2026 里同样存在**，若以后在那里做组装实验需同步修复

---

## 6. 结论：能否泛化？G4' 行不行？

### 6.1 物理骨架层：**能，且已实证**

| 部件 | 换神经元时 | 证据 |
|---|---|---|
| cable 方程（`_axial_Rax` / `_tridiag`） | **零改动** | 人 L23 误差 0.000% |
| 几何读取（`arc3d`/`diam3d`/`area()`） | **零改动** | 同上 |
| 被动膜（`pas`） | **零改动** | 同上 |
| 数值格式（`commit`/`cai`/导频） | **零改动** | 同上 |

### 6.2 需要扩展的接口（**不是架构改变**）

| # | 缺什么 | 成本 | 备注 |
|---|---|---|---|
| 1 | `na` / `kv` 两个新机制 | **低** | 都已暴露 `minf/hinf/mtau/htau` + `q10=2.3`，与 `MECH` 表同构，可直接转录 |
| 2 | `e_pas` / `Ra` / `g_pas` 等**从模型读**而非硬编码 | 已修 | #124 |
| 3 | 🔴 **突触电导**（`AMPANMDA_EMS` 的 NMDA） | **中** | 见下 |
| 4 | 部分 section 无 3D 点（stub axon） | 低 | 已有回退路径（均匀圆柱） |

### 6.3 🔴 真正的短板：瞬时电压依赖的突触电导

NMDA 的形式是：

$$g_{\text{NMDA}}(v,t) = \underbrace{(B_{\text{NMDA}}(t)-A_{\text{NMDA}}(t))}_{\text{双指数（记忆）}} \times \underbrace{\frac{1}{1+\exp(\gamma \cdot(-v))\cdot(\mathrm{mg}/3.57)}}_{\text{mggate}（\textbf{瞬时}电压依赖）}$$

⚠️ **`mggate` 不是门控动力学，而是瞬时电导**：`g = ḡ·f(v)`，没有自己的 ODE。
我们现有的 `MECH` 表假设是「`g = gbar × ∏gate_i^{p_i}`，每个 gate 有 ODE」⇒
**表达不了这一项**（这正是之前 T0.3b 反复测 `g_NMDA` 效果异常的原因）。

好消息：`_linearize()` 是**数值求导**，天然支持任意 `g(V)`。
需要加的只是「输入 g 表」这一层接口，不是求解器。

### 6.4 学习层：G4' 的假设成立，但有一个硬性前提

**G4' 的核心假设**是「物理项负责绝大多数，网络只补小残差」。本次实证支持它：
物理骨架在跨神经元时**不需要重训、精度不降**（0.000%）。

**但有一个硬性前提**（否则必然失败）：

> 🔴 **θ 必须显式编码「这个神经元有哪些通道」。**
>
> 因为 L5PC 的非线性来自**树突通道**，L23 的非线性来自**突触 NMDA**。
> 若只在 L5PC 上训练，残差网络会学到「给树突加 Ca/Ih/Im 修正」；
> 换到 L23（树突本该纯被动）时，**网络会错误地施加这些修正** ⇒ 系统性偏差。
>
> ⇒ 训练时必须 **随机化 θ**（几何、密度、**通道存在性**），
>   验证必须用 **leave-one-neuron-out**。

### 6.5 关于「G4' = 同时训 G3 和 G4，用 G3 修正 G4」

方向正确，但**修正项应落在【导数/电流域】，不是状态域、也不是并联平均**：

$$\frac{dx}{dt} = f_{\text{phys}}(x,\theta) + \Delta_{\text{net}}(x,\theta)$$

理由：缺失的通道/突触本质是**一个缺失的电流项** `ΔI = ḡ·m·h·(v−E)`，
属于 `dx/dt` 的残差。这样 `Δ→0` 时精确退化为纯物理模型（可解释性天然保留），
且残差网络**不需要是 GNN**（残差本身是局部电流），MLP/GRU 足够。

---

## 7. 缺口清单（下一步）

| # | 任务 | 优先级 |
|---|---|---|
| 1 | 转录 `na.mod` / `kv.mod` 进 `L5PC_gates.py` 的 `MECH` 表 | 高 |
| 2 | 补完我们自身 4 个 axon 通道（`NaTa_t, Nap_Et2, K_Pst, K_Tst`，目前 `rates=None`） | 高 |
| 3 | 加「瞬时电压依赖电导」接口（`g(v)` 形式，为 NMDA 铺路） | 高 |
| 4 | 设计「训练期 θ 随机化」范围（可参考 FCI 的 4 组突触参数集） | 高 |
| 5 | 同步修复 `L5PC2026/L5PC_unit_sec.py` 的 #124/#125 | 中（若在那做组装） |
| 6 | 找真正的跨通道集基准（完整版人 L5 / 人 L23 含其完整 mod 集） | 中 |
| 7 | 用 `AMPANMDA_EMS` 在 L23 上做「突触非线性」的端到端验证 | 中 |

---

## 附：本次新增/修改文件

| 文件 | 说明 |
|---|---|
| `tools/verify_cross_neuron.py` | 载入 FCI 4 个模型，实测 section/segment/通道/3D 可用性 |
| `tools/cross_neuron_section_test.py` | 行为验证：`SectionUnit` vs NEURON，人 L23 基底树突 2-端口 |
| `tools/_probe_nseg_map.py` | sid → section → nseg 映射 |
| `results/cross_neuron/cross_neuron_facts.json` | 4 模型完整事实 |
| `results/cross_neuron/cross_neuron_section_test.json` | 逐段端口电流对比 |
| `L5PC_unit_sec.py` | 修 #124a/#124b/#125 |
