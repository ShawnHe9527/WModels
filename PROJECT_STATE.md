# PROJECT_STATE.md — WModel2 当前状态

> 最后更新：2026-10-26（清理与文档收敛后）
> 本文件是**当前进度与决策的唯一入口**；稳定事实与规则见 `AGENTS.md`。
> 读法：先看“目标/已完成/进行中”，再看“已知问题/待确认问题”。

## 1. 目标

### 1.1 科学目标

在官方 L5PC NEURON 模型上，对目标段 `apical_51`（11 个 compartment）学习一个 latent 世界模型：

```
Enc(s_t, θ) -> z_t        Pred(z_t, u_t, θ) -> z_{t+1}        Dec(z_{t+1}) -> ŝ_{t+1}
```

- `s`：7 个慢门控 + `cai` + `V`，逐 compartment ⇒ 9×11 = **99 维**
- `u`：目标段 11 + 邻段 17 个 compartment 的 `[AMPA, NMDA_ungated, GABA]`，加 1 维 bAP ⇒ **85 维**
- 目标：预测下一步状态，重点看 **Ca 平台/Ca 事件**与长程闭环稳定性，而不是只拟合短程噪声

### 1.2 当前验收目标

以配对评测 `te_s3` 的 `model/L0`（持续性基线）为主：

- **硬指标**：`k=200`（5 ms）`model/L0 < 1.0`；长视界不出现 `DIVERGED`
- **当前扩展目标**：`k=4000/8000`（100/200 ms）也不输 L0；plateau/inter 档稳定优于 L0
- **物理合理性**：plateau 的 `V_max` 落在 **13–17 mV**（真值约 13.35）；Ca 峰 F1 ≥ 0.30
- **工程目标**：每个新实验可复现，按 `AGENTS.md` 的 `runs/<YYYY-MM-DD>-<commit7>/` 归档

### 1.3 非目标

- 不改 `neuron/` 官方资产；不重写 L5PC 生物物理模型
- 不追求“所有 regime 都赢 L0”：linear 档接近噪声主导，允许输
- 不靠加大 `d_z` 刷指标：`d_z=16` 是降维目标的一部分
- 不做跨神经元泛化（暂缓）

## 2. 已完成

### 2.1 数据

| 数据集 | 规模 | `dt` | `u` 维 | plateau | trial 门禁 | 用途 |
|---|---|---|---|---|---|---|
| `data/tr_s1` | 200 条 × 40000 帧 | 0.025 ms | 85 | 44.5% | 82% | 训练 |
| `data/va_s2` | 200 条 × 40000 帧 | 0.025 ms | 85 | 48.5% | 87% | 验证 |
| `data/te_s3` | 200 条 × 40000 帧 | 0.025 ms | 85 | 41.0% | 88% | 测试 |

- 三组数据由 `world.gen.datagen2` 调 NEURON 生成，总计约 **17 GB**
- `va_s2` 的 48.5% 略高于 45% 门限，但数据完整、三组同协议分布，已决定**不重跑**（重跑会引入样本差异）
- `data/*/ub.dat`（v3 边界电位）已删除；v4 不再使用

### 2.2 代码与文档

- v4 活跃代码：`world/core/{safety,progress,config}`、`world/gen/{datagen2,sim2,protocol,protocol2,secinfo}`、`world/learn/{data2,model,train2}`
- 旧 v3 代码/工具已归档到 `archive/legacy_v3/`；旧 v3 结果在 `results/archive/legacy_v3/`
- 根 `README.md`、`world/README.md`、`docs/README.md`、`results/README.md` 已改为 v4 现状
- `docs/DESIGN_v4.md` 已标为定档；§9.7 明确是 **R0 首轮基线**，不再把 0.786 当当前最优
- `world/core/config.py` 已收敛到 v4 常量（`DT=0.025`、`SECTION=apical_51` 等），删除旧 `data/wm2` 路径与旧阶段常量
- `run_v4.sh` 的 `v6`/`diag` 已改指向当前最优 R2（`t2v5_events`）；`v5` 改名 `t2v4_baseline`，避免与归档重名

### 2.3 训练/评测轮次

| 轮次 | run | σ | model/L0 @200 | @8000 | plateau Vmax | Ca 峰 F1 | 结论 |
|---|---|---|---|---|---|---|---|
| R0 | `t2v4_anchor` | 0.000362 | 0.786 | 22.025 | 56.0 | 0.000 | 首轮基线 |
| R1 | `t2v5_ancC` | 0.000301 | 0.654 | 2.368 | 12.7 | 0.162 | anchor-cl 有效 |
| R1b | `t2v5_noancC` | 0.000351 | 0.777 | 15.824 | 2.2 | 0.000 | 消融：关掉 anchor-cl 长程崩 |
| **R2** | **`t2v5_events`** | **0.000310** | **0.562** | **0.911** | 19.7 | 0.294 | **当前最优** |
| R3 | `t2v5_events_w5a2` | 0.000316 | 0.605 | 1.473 | 10.8 | 0.250 | 不采纳（V 不足） |
| R4 | `t2v5_events_w8` | 0.000346 | 0.648 | **DIV** | 14.3 | 0.324 | 不采纳（长程发散） |

R2 的完整 `model/L0`（总体）：

| k | 10 | 40 | 200 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|---|---|
| ratio | 0.600 | 0.585 | **0.562** | 0.515 | 0.547 | 0.625 | **0.911** |

- R2 **所有总体视界都 <1**，且 k=8000 未发散；这是目前唯一做到这一点的配置
- R2 分档：plateau 0.538→0.636（k=10→8000），inter 0.502→0.703，均 <1；linear 在 k=4000/8000 为 1.089/1.720，属于“无结构档不硬造”的预期
- R4 的 plateau 档 43 个 trial 中 2 个在 k=8000 数值爆炸到 3.8e18，整体判 `DIV`

### 2.4 清理与验证（2026-10-26）

- 删除所有 `__pycache__`/`.pyc`、空 `world/eval/`、旧 `ub.dat`
- 归档：v2/v3 文档 → `docs/archive/`；13 个旧 ckpt、12 个旧日志、18 个旧 report、旧 figures/scaler → `results/archive/legacy_v3/`；旧代码/工具 → `archive/legacy_v3/`
- 验证：
  - `bash -n run_v4.sh` ✅
  - 所有活动 Python `py_compile`/import ✅
  - `datagen2`、`train2`、`round_report`、`diag_latent` 的 `--help` 均 `rc=0` ✅
  - 三组数据可加载：`200 × 40000 × 85`，`th=(200,27)` ✅
  - `tools.rounds_index`、`tools.fig_rounds_summary` 可重生成 ✅

### 2.5 评估闸门与 R5 零训练标定（2026-10-26）

**环境**：WSL2 Ubuntu 22.04，20 逻辑核，MemAvailable ≈ 6.1 GB，磁盘 ≈ 981 GB；
`tools/v0_check --n 1` 通过（5.2 s/trial，5 门禁 1/1）。  
⚠️ `tools/verify_env.py` 仍按旧目录布局写，v4 下 10/13 失败（找不到 `mods/`、`morphologies/`、`probes/`）；
这是工具过时，不是环境不可用；以 `v0_check` 为准。

**评估闸门**（`runs/2026-10-26-be60d2d-eval-gate/`）：

- `va_s2`：128 trial × 2 个 t0（4000/16000），256 个配对样本；R2_last 在全部视界都优于 R0–R4。
- R2_last：k=200 **0.511** [0.457,0.580]，k=2000 **0.497** [0.422,0.580]，k=8000 **0.849** [0.733,0.983]。
- R2_best：k=8000 **1.970**（2/256 DIV）；R1 2.606；R4 2.713（33/256 DIV）。⇒ **用 last.pt，不用 best.pt**。
- `te_s4`（新盲测，seed=4）：R2_last k=200 **0.562**，k=2000 **0.594**，k=8000 **0.899** [0.783,1.031]；R2_best k=8000 2.022（4/256 DIV）。
- **闸门结论：条件通过**。R2 是当前最好的模型，短/中程（≤50 ms）在 va 和 blind te_s4 上都赢 L0/L1/L2/L3；
  但 **200 ms 不稳健**：k=8000 的 95% CI 在 te_s4 包含 1，且 L2 ridge 更好（va 0.820，te_s4 0.869 vs R2 0.849/0.899）；
  L1 是非因果上界，也更好（0.632/0.693）。不能再说“所有视界都赢全部基线”。

**te_s4**：200 条 × 40000 帧，16.8 min，plateau 45.5%（略高于 45%），trial 门禁 88%；
数据已接受、未用测试集调参；`data/te_s4` 未进 Git。

**R5 零训练标定**（`runs/2026-10-26-be60d2d-r5-calib/`）：**失败**。

- 在 va_s2 上按 plateau trial 拟合 rest-relative gain（V 以 −76 mV、cai 以 20 分位为基线）；
- 拟合得到 V gain **1.067**、cai gain **1.000**（没有压缩）；
- 在 te_s4 的 128 trial 上：Vmax 17.49 → **23.72**（真值 19.81）；Ca 峰 F1 0.224 → **0.224**；
- model/L0 @2000 0.536 → 0.537，@8000 0.721 → 0.726；
- 结论：**全局/逐因子线性标定不能解决幅度问题**，且 `Vmax 13–17 mV` 这个判据不跨数据集成立（te_s4 真值 Vmax 已 19.81）。
  需要训练侧改动，或重新定义物理判据。

## 3. 进行中

- **没有训练/评测进程在跑**；最后一次训练仍是 2026-10-23 20:43（R4）
- **R5 零训练标定已做并失败**；训练回退（幅度软约束）尚未启动，等待确认损失/λ/判据
- **完整结果叙述文档尚未写**：当前入口是 `results/rounds/README.md` + R2/R3/R4 的 `CONCLUSIONS.md`；`DESIGN_v4.md` 原先引用的 `EXPERIMENTS_RESULTS.md` 不存在，引用已改到 `results/rounds/README.md`
- **`runs/` 已开始使用**：`eval-gate` 与 `r5-calib` 两个目录已有 commit/config/metrics/stdout；但训练代码仍写 `results/checkpoints/`，需人工复制
- **GitHub 已上传**：`main` 已 push 到 `https://github.com/ShawnHe9527/WModels.git`；
  首个 commit `f201cf3`（164 个文件，9.9 MB）；`data/`、`results/checkpoints/`、`results/archive/`、`*.pt` 均未上传

## 4. 关键决策与理由

| # | 决策 | 理由 |
|---|---|---|
| D1 | `u` 用 85 维（目标段 33 + 邻段 51 + bAP 1），不再用边界电位 | 边界电位是目标段状态的“回声”而非原因（v3 实测 corr 0.95–0.98）；邻段必须由自己的突触驱动 |
| D2 | 只存 T2，T1 由 `sim2.to_t1()` 派生 | 同一仿真的同一份记录，T1/T2 完全配对；零额外仿真/存储 |
| D3 | `s` 含 `V`（99 维），`cai` 存 log10 | 自由膜电位是状态；log10 才能覆盖多个数量级 |
| D4 | 残差输出，起点用真值 `s0` | 解码器有约 1.6e-3 系统性偏差，相减抵消；用 `Dec(z0)` 会一开始就吃偏差 |
| D5 | DeepSets + `u_glob`（bAP），`d_z=16` | 参数量不随 compartment 数增长；bAP 是全局信号，不能被 mean/max 池化抹平 |
| D6 | 不拆分快慢因子，单一 16 维 latent | D-a 实测 `cai/快门控 = 0.98×`，未达 2× 分组阈值 |
| D7 | T2 用闭环课程 + anchor-cl + 事件加权 + 档加权/hot 采样 | R0→R1 证明 anchor-cl 是长程不崩的必要条件；R2 证明事件/采样组合能到所有总体视界 <1 |
| D8 | 当前以 R2（`t2v5_events`）为最优，R4 不采纳 | R2 `a(200)=0.562`、k=8000=0.911、无 DIV；R4 虽然 Vmax/F1 更好，但 k=8000 发散 |
| D9 | 评测以 L0（复制上帧）为主、配对、按 regime 分档 | linear 档会稀释总体；单 trial/不分档会误判 |
| D10 | 旧 v3 代码/结果归档而非永久删除 | 保留可追溯性；活动树只留 v4，避免 import/文档污染 |

## 5. 已知问题

### 5.1 科学/模型

1. **R2 的 V 上限过冲**：在 te_s3 上 19.7 mV vs 真值 13.35（+47%）；R3 降到 5 又变成 10.8（方向对但过头）。但在 blind te_s4 的 128 trial 上真值 Vmax 已是 19.81，说明“13–17 mV”不是跨数据集的固定判据。
2. **R2 的 Ca 峰 F1 偏低**：te_s3 0.294，blind te_s4 0.224（R4 在 te_s3 0.324 更高）；固定阈值 P/R 为 0.696/0.867（k=2000），与无阈值峰指标不能混用。
3. **R2 长程不稳健**：va_s2 k=8000=0.849 [0.733,0.983]，te_s4 k=8000=0.899 [0.783,1.031]（CI 含 1）；L2 ridge 更好（0.820/0.869），L1 非因果上界也更好（0.632/0.693）。linear 档在 te_s3 k=4000/8000 为 1.089/1.720，va k=8000 为 2.077。
4. **R4 长程发散**：2/43 plateau trial 到 3.8e18；R4 的 `anchor-cl=0.1` + `event-weight=8` 组合不稳定。
5. **`train2 eval` 仍只有 L0**：新 `tools/eval_gate.py` 已实现 L1（逐 trial 均值，非因果）、L2（岭回归）、L3（逐维 AR(1)），但训练/归档流程尚未接入；不能只用 `train2 eval` 声称“赢过 L1–L3”。
6. **T1 的 target 可疑**：`train2.py` 的 T1 用 `s0` 在整段窗口重复作为重构目标（`s_flat = s0.expand(...)`），注释却写“逐帧重构”；这会让 encoder 倾向忽略 `u`，尚未做对照实验确认影响。
7. **数据集门禁略超上界**：`va_s2` plateau 48.5%、`te_s4` 45.5%（目标 15–45%）；两者数据均已接受并记录，未按测试集调参。`datagen2` 仍会返回 1，`run_v4.sh v3` 在 va_s2 后会停。

### 5.2 工程/复现

8. **`datagen2 --resume` 不可复现**：续跑只把 `Policy2` 的 RNG 重放 `done` 次，但原循环还消耗了 `make_spike_trains` / `make_tuft_bundles` / `make_bap_times` 的 RNG；断点后的协议与一次跑完不一致。此外 `th_arr` 续跑时没有加载旧值。
9. **内存护栏未真正接入训练循环**：`--mem-mb` 只传给 `safety.preflight()` 打印；实测训练 RSS 3.4–4.0 GB，`MemGuard` 没有在 step 循环里 `check()`。
10. **权重没有进 GitHub**：`.gitignore` 排除了 `results/checkpoints/`、`*.pt`、`results/rounds/**/ckpt/`；
   当前 R2 权重只在本地，需用 GitHub Release 附件或 Git LFS 分享。
11. **`runs/` 规范只部分落地**：`eval-gate` / `r5-calib` 已有 commit/config/metrics/stdout；但 `train2` 仍写 `results/checkpoints/<run>`，训练实验需要人工建目录、复制 ckpt。
12. **完整结果文档缺失**：只有 `results/rounds/README.md` 索引与 R2/R3/R4 的人工结论；R0/R1/R1b 没有 `CONCLUSIONS.md`，也没有统一叙述版 `EXPERIMENTS_RESULTS.md`。
13. **`world/gen/secinfo.py` 仍有旧辅助函数**：`new_cell`/`read_state` 等不是 v4 主路径，且内部还有相对路径 `morphologies/cell1.asc`、`L5PC_exp`；当前只通过 `sim2.theta()` 调 `build_theta`，暂未清理，避免影响 θ 提取。
14. **`run_v4.sh v5` 仍是初始基线配置**，不是 R2；当前最优的完整命令只写在 `results/rounds/R2_event_weighting/CONCLUSIONS.md`，脚本与最优配置尚未统一。
15. **`tools/verify_env.py` 过时**：仍按旧根目录布局找 `mods/`、`morphologies/`、`probes/`，v4 下 10/13 失败；环境是否可用以 `tools/v0_check` 为准。
16. **R5 零训练标定失败**：rest-relative gain 拟合出 V=1.067、cai=1.000；te_s4 上 Vmax 17.49→23.72（真值 19.81），Ca 峰 F1 0.224→0.224，model/L0 几乎不变。全局线性标定不能解决幅度问题。

## 6. 待确认问题

1. **R2 是否作为当前基线冻结？** 评估闸门确认 R2_last 是当前最好的模型，但 200 ms 不稳健（te_s4 CI 含 1，L2 更好）。是否接受“R2 只赢短/中程，200 ms 交给 L2/后续 R6”？
2. **R5 训练回退怎么做？** 零训练标定已失败；需要确认训练侧改动：
   - A. 幅度软约束：`L_amp = λ·relu(|Δŝ_cai| - k·|Δs_cai|)^2`（R4 结论推荐，最小改动）
   - B. 长程正则（直接压 k>4000 发散）
   - C. 先修 T1 `s0` target / `datagen2 --resume`，再重跑对照
   需要你定 λ、k 和判据；我未改 `train2.py`。
3. **`runs/` 规范怎么落地？** 已有两个 run 目录；是否现在建一个不改训练代码的 shell 包装器？
4. **R0–R4 是否回填 `runs/`？** 还是保留在 `results/rounds/` 作为历史布局，新实验才用 `runs/`？
5. **L1/L2/L3 定义是否接受？** `tools/eval_gate.py` 当前：L1=逐 trial 全轨迹均值（非因果上界），L2=岭回归 `[s_t,u_t,θ]→s_{t+k}`，L3=逐维 AR(1)。若定义要改，历史结论需重算。
6. **是否新建 `EXPERIMENTS_RESULTS.md`？** 当前引用已改到 `results/rounds/README.md`；若需要论文式叙述版，放在根目录还是 `results/`？
7. **归档是否永久删除？** `archive/legacy_v3` + `results/archive/legacy_v3` 约 43 MB；当前保留可追溯，确认后可删。
8. **`run_v4.sh v5` 是否改成 R2 配置？** 改成 R2 会覆盖/重训当前最优，需明确“复现”还是“新实验”。
9. **物理判据要不要改？** `Vmax 13–17 mV` 在 te_s4 不成立（真值 19.81）；是否改成 `pred_Vmax/true_Vmax ∈ [0.8,1.2]` + Ca 峰 F1 + 长期 vs L2？改判据需要你确认。

## 7. 下一步（待确认后执行）

### 7.1 已完成的标定（失败）

R5 零训练标定已经在 `runs/2026-10-26-be60d2d-r5-calib/` 执行：

- va_s2 拟合 rest-relative gain：V=1.067、cai=1.000；
- te_s4 128 trial：Vmax 17.49→23.72（真值 19.81），Ca F1 0.224→0.224；
- model/L0 @2000 0.536→0.537，@8000 0.721→0.726。

⇒ **零训练标定失败，不能靠输出缩放解决幅度问题。**

### 7.2 R5 训练回退（建议，未执行）

建议只改一个变量：在 R2 配置上加入 **幅度软约束**，不碰事件权重、anchor-cl、数据：

```
L = L_R2 + λ · mean( relu(|Δcai_pred| - k·|Δcai_true|)^2 )
```

- 目的：只惩罚“比真值更陡”的 cai 变化，压事件幅度过冲，不改事件/静止帧配比；
- 建议起点：`λ=0.1`、`k=1.0`，只在闭环段（cl>0）生效；其余参数与 R2 完全一致；
- 预算：9000 步，约 52 min，RSS 3.4–4.0 GB，单进程；
- 成功判据（**待你确认**）：
  - te_s4 k=2000/8000 的 model/L0 不比 R2 差 >0.02；
  - k=8000 不 DIV，且不劣于 L2（te_s4 当前 L2=0.869）；
  - `pred_Vmax/true_Vmax ∈ [0.8,1.2]`（不再用绝对的 13–17 mV）；
  - Ca 峰 F1 不低于 R2（te_s3 0.294 / te_s4 0.224）。
- 失败预案：回退 R2，不再连续调 λ；转为检查 L2 基线与长程正则。

### 7.3 工程修复（与 R5 分开）

1. 修 `datagen2 --resume` 的 RNG/`th_arr`；
2. 用 4 个窗口对照 T1 target=`s0` 与 `s_true`，确认后再决定是否重训 T1；
3. 把 `MemGuard.check()` 接进训练循环；
4. 把 `train2` 的输出接入 `runs/<date>-<commit7>/`（或写一个 shell 包装器）。
