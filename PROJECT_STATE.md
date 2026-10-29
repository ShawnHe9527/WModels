# PROJECT_STATE.md — WModel2 当前状态

> 最后更新：2026-10-27（R5-B 之后；C+A 计划与迁移准备）
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

### 2.6 R5-H1 视界 pilot（2026-10-27）

`runs/2026-10-27-f81f09c-r5-h1/`：保持 R2 配方，只把 `length` 128→512、`cl_steps` 64→128；
2000 步（stage A 压缩为 500 步，stage B 1500 步）。训练 3579 s，最终 σ=0.000437，
峰值 RSS≈4.6 GB。

`te_s4` 盲测（128 trial × 2 t0）：

| model | k=10 | k=200 | k=2000 | k=8000 |
|---|---:|---:|---:|---:|
| R2_last | **0.605** | **0.562** | **0.594** | **0.899** |
| H1_last | 0.783 | 0.833 | 2.025 | 2.960 |
| H1_best | 0.824 | 0.828 | 2.037 | 10.617 |

**结论：H1 没有改善，反而全面变差；未启动 H2。**

- H1 单步 σ=0.000437 vs R2 0.000310，首先就是欠训练；
- 2000 步 + stage A 从 2000 压到 500，不是干净的“只改视界”对照；
- 长闭环训练后期 loss/grad 多次尖峰（grad 最高 >30），H1_last 在 200 ms 严重漂移；
- H1_best 长程更差（10.6），不能因为单步 σ 选它。

⇒ 这次 pilot **不能证明“长视界有害”**，只能说明：在 2000 步、压缩课程下，拉长训练视界没有带来改善。要做干净判断，需要 9000 步 + 原课程，成本约 3–4 h（L=512）。

### 2.7 R5-B 事件中心采样 + V 加权（2026-10-27）

`runs/2026-10-27-d2a3c49-r5b/`：保持 R2 视界（length=128=3.2 ms，cl=64=1.6 ms）和 9000 步；
改动为 **50% 窗口以 Ca episode 中心开窗** + **V 事件加权 weight=5, tau=0.02**。
训练 3494 s（58.2 min），最终 σ=0.000354，最佳 σ=0.000315，峰值 RSS≈5.9 GB。

`te_s4` 盲测（128 trial × 2 t0）：

| model | k=10 | k=200 | k=2000 | k=8000 | DIV |
|---|---:|---:|---:|---:|---:|
| **R2** | **0.605** | **0.562** | **0.594** | **0.899** | 0/256 |
| B_last | 0.652 | 0.705 | 1.414 | 3.199 | **28/256** |
| B_best | 0.635 | 0.622 | 0.973 | 1.858 | 3/256 |

物理指标（plateau n=43，k=8000，t0=4000）：

| model | Vmax pred | Vmax true | Ca peak F1 | amp ratio |
|---|---:|---:|---:|---:|
| R2 | 18.85 | 19.81 | **0.329** | 0.923 |
| B_best | 28.82 | 19.81 | 0.163 | 1.045 |

**结论：B 没有改善；短/中程也变差，长程更差且 B_last 大量发散。R2 仍是当前最优。**

- 事件中心窗把 128 步窗口压在 Ca episode 上，但窗口仍只有 3.2 ms，容易只记住事件片段；
- V 事件加权放大了 V 的陡变帧，B_best 的 Vmax 反而升到 28.82；
- B 的训练 loss/grad 随 cl 增大持续上升（见 `fig_training.png`），闭环稳定性没有改善。

### 2.8 工作站环境验证（2026-10-28，a6000middle / 192.168.1.94）

- 仓库 `/home/Hesj/WModel`：`main` 与 `origin/main` 同步，HEAD `9bd4844`，工作区干净
- 存储：`data` → `/home/Hesj/QH1_dataHe/WM`，`results/checkpoints` → 同目录 `checkpoints/`；
  CIFS `//192.168.1.65/QH1_Labdata/dataHe`，**45 TB 可用**，NAS 侧可写（沙箱说明见 5.2-21）
- 环境 `/home/Hesj/envs/worm_sim`（venv，Python 3.10.20）：torch 2.4.1+cpu（`cuda=False`）、
  numpy 2.0.2、scipy 1.15.3、matplotlib 3.10.8、NEURON 8.2.0
- `neuron/mods` 已 `nrnivmodl`，`x86_64/libnrnmech.so` 已生成（平台相关，未进 Git）
- `tools/v0_check --n 1 --seed 99 --mem-mb 2500`：**1/1 通过**，5 门禁全过，9.8 s/trial
- 机器 32 逻辑核 / 502 GB 内存；`/home` 只剩 9.4 GB（99%），env 占 1.2 GB
- 权重已就位（2026-10-28 19:17 由 WSL `scp` 传入）：`t1v4_dz16/best.pt`（step 10000，val_rel 0.0314）、
  `t2v5_events/{last,best,stageA}.pt`（last step 9000 σ=3.098e-4；best step 6500 σ=2.688e-4）；
  三者均 `load_state_dict(strict=True)` 通过；`scaler_v4.json` 与 R2 ckpt 内嵌 scaler 逐字段一致
- NAS 布局改为分类目录：`datasets/{1s,2s}/<name>`、`checkpoints/`、`scalers/`、`logs/`、`README.md`；
  仓库 `data/` 仍指向 `WM`，命令用 `--data datasets/2s/<name>`
- 2 s 数据已生成（2026-10-28 19:29–20:33，三组并行，18.6 s/trial，62 min/组，共 34.5 GB）：
  `datasets/2s/{tr_s1_2s,va_s2_2s,te_s4_2s}`；`s`(200,80000,9,11)、`u`(200,80000,85)、`th`(200,27)，无 NaN/Inf
- 数据集门禁：plateau **45.5 / 47.5 / 49.0%**（目标 15–45%，略超，与 1 s 的 `va_s2` 48.5% 同类）；
  单 trial 门禁 **64 / 66 / 67%**（1 s 为 82/87/88%）。下降是**窗口长度效应**：G1 的 `up_frac ≥ 0.005`
  按 1 s 校准，2 s 分母翻倍；抽查 7 个 trial，2 s 的 up_frac 与 1 s 前 40000 帧相当
  （如 `tr` trial40：0.712 vs 0.643），plateau 档仍是 39/39 有平台 ⇒ 按 PLAN §2 约定记录后接受，不重跑
- `scaler_v4_2s.json` 已生成（`make_scaler`，38 s）：u 通道与 1 s 差 1–9%，`SKv3_1.m`/`SK_E2.z`
  的 loc 差 -46%/-20%，其余因子在 ±13% 内
- 未开始：C+A 训练、`te_s4_2s` 评测

### 2.9 R6 C+A（2 s 数据 + L=512）已执行：**不采纳**（2026-10-28/29）

`runs/2026-10-28-9bd4844-r6-ca-rerun/`：2 s 数据 + `length=512` + 9000 步（stage A 2000），
训练 **4.91 h**（17690 s），最终 σ=**3.24e-4**（门禁 0.008），RSS 峰值 **12.4 GB**
（新护栏 32 GB 通过；旧 6 GB 阈值会在 stage B 误杀）；无发散。

`te_s4_2s` 盲测（128 trial × 2 t0 = 256 样本）：

| k | 10 | 200 | 2000 | 8000 |
|---|---:|---:|---:|---:|
| R2（1 s，zero-shot） | **0.625** | **0.568** | **0.562** | **0.890** |
| C+A | 0.673 | 0.605 | 0.652 | 0.931 |

分档（plateau）：R2 0.509/0.460/0.454/0.562 vs C+A 0.563/0.517/0.549/0.596，**全档更差**。

物理指标（plateau n=37，k=8000，t0=4000）：peak F1 **0.359 vs 0.303**；
幅度比 0.934 vs **1.339**（C+A 过冲）；nrmse 0.351 vs 0.499；峰时误差 1.50 vs 2.60 ms。

⇒ 按 PLAN §8，**停止 C 路线**。R2 在 2 s 数据上仍是最好模型（zero-shot 与 1 s `te_s4` 基本一致：
0.562/0.890 vs 0.594/0.899），说明数据本身没有变难，C+A 是实打实变差。

### 2.10 工程修复（2026-10-29）

- `train2 t2` 新增 `--save-every`（默认 500 步覆盖存 `last.pt`）、`--rss-cap-mb` / `--mem-floor-mb`
  （进程内 `MemGuard`，触发时**先落盘再以退出码 2 结束**）。冒烟测试：正常 2 步存 best/last；
  故意把 cap 设 50 MB → step 1 存 `last.pt`（含 `aborted` 字段）并 `rc=2`
- `tools/eval_gate.py` 新增 `--scaler`（默认 `results/data/scaler_v4.json`），修掉 1 s/2 s scaler 混用；
  本地 `results/data/scaler_v4.json` 已恢复为 1 s scaler
- `datagen2 --resume` 修复：meta **每条 trial 原子落盘**、记录 `rng_state`、续跑恢复 RNG 状态并载入
  `th.npy`；2-trial 对照（跑一半 SIGKILL → 续跑）验证 **s/u/th 与一次跑完逐字节一致**
- 新增 `tools/guard_run.sh`：启动前检查（磁盘 ≥10 GB、可用内存 ≥16 GB）+ 60 s 采样 +
  自动停机（RSS >32 GB / 可用 <16 GB / 磁盘 <3 GB），日志写 `RUN_DIR/monitor.log`
- 新增 `tools/new_run.sh`：按 AGENTS §5 建 `runs/<date>-<commit7>-<name>/` 并写 `commit.txt`（拒绝覆盖）
- `.gitignore`：`results/logs/`、run 目录的 `stdout.log`/`monitor.log` 不进 Git（本地保留）

### 2.11 诊断 A：误差随视界/因子/事件帧的分解（2026-10-29）

工具 `tools/diag_a.py`；run `runs/2026-10-29-3f06c05-diag-a-rerun2/`（**用 va_s2_2s，不碰盲测**，
128 trial，t0=4000，9 个视界一次 rollout）。

- R2 的 `model/L0`：0.636（k=10）→ **0.440（k=2000）** → 0.562（4000）→ 0.732（8000）；
  linear 档 0.73→**1.67**（k=2000→8000）是长程恶化的主要来源
- 逐因子（k=8000，按归一化绝对误差的份额）：**Ca_HVA.h 19.4%**（model/L0=1.91）、V 13.9%（0.69）、
  SK_E2.z 11.4%、SKv3_1.m 10.4%、Im.m 10.2%、Ca_LVAst.h 9.8%、Ca_LVAst.m 8.7%、
  cai 8.5%（0.60）、Ih.m 7.8%
- bAP 帧只占 V 误差的 **2–4%**；Ca 事件帧检测阈值过严（0 帧），但 cai 因子本身 0.60
- latent：`|z|` 5.5→6.7，`cos(z, z_true)` 0.93→0.77（k=2000→8000）⇒ 长程 latent 漂移
- C+A 在 k=8000 的慢门控更差：Ih.m 1.43、Ca_LVAst.h 1.61
- ⚠️ 指标口径：对近常数因子（帧间 diff 极小），`model/L0` 比值会被“持久化几乎完美”放大；
  判断重要性要看**绝对归一化误差份额**，不能只看比值

### 2.12 方案 1：latent 线性探针（2026-10-29）

工具 `tools/probe_latent.py`；run `runs/2026-10-29-45df95c-probe-latent/`。
`va_s2_2s`，128 trial × 60 个时间点，按 trial 96/32 划分 train/test，ridge 探针：

| 模型 | Ih.m | Ca_LVAst.m | Ca_LVAst.h | Ca_HVA.h | SKv3_1.m | SK_E2.z | Im.m | cai | V | 99 维整体 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2 | 0.994 | 0.986 | 0.992 | **0.980** | 0.915 | 0.970 | 0.966 | 0.988 | 0.989 | 0.966 |
| C+A | 0.994 | 0.986 | 0.990 | **0.969** | 0.901 | 0.971 | 0.965 | 0.986 | 0.988 | 0.963 |

shuffled-z 控制 ≈ 0（R² = -0.004）。

**结论：latent 装得下慢状态**（连最难的 `Ca_HVA.h` 都能线性恢复 0.97–0.98），
所以不是编码容量/`d_z` 问题；问题在 **predictor/rollout 把它带偏**。
⇒ 下一步应改预测器训练（慢因子损失加权 / delta 收缩），而不是加宽 latent。

### 2.13 E0：逐因子 delta 收缩（推理侧，2026-10-29）

工具 `tools/probe_shrink.py`；run `runs/2026-10-29-6d892af-shrink/`。R2，`va_s2_2s`，128 trial，t0=4000：

| 配置 | k=200 | k=2000 | k=4000 | k=8000 | linear@8k | plateau@8k |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.597 | 0.440 | 0.562 | 0.732 | 1.673 | 0.551 |
| slow3=0.85 | 0.591 | 0.434 | 0.550 | 0.695 | 1.534 | 0.544 |
| slow3=0.70 | 0.591 | 0.433 | 0.546 | 0.670 | 1.414 | 0.547 |
| slow3=0.50 | 0.597 | 0.440 | 0.548 | **0.653** | 1.282 | 0.564 |
| Ca_HVA.h=0.70 | 0.586 | 0.425 | 0.539 | 0.680 | 1.435 | 0.562 |
| slow3=0（冻结） | 0.646 | 0.492 | 0.601 | 0.696 | **1.139** | 0.639 |

- 中等收缩（α≈0.5）把 k=8000 从 0.732 降到 0.653，主要来自 linear（1.673→1.282）；
  短/中程几乎不变 ⇒ **漂移假设成立**

### 2.14 E1：慢因子损失权重（C-only）——无效，且 2 s + L=128 基座不稳定（2026-10-29）

- 基线 `runs/2026-10-29-e8e54b2-r6-c-only/`：σ=3.56e-4，4021 s（1.12 h），RSS 10.5 GB
- 加权版 `runs/2026-10-29-5ad6b9c-r6-c-only-fw/`：`Ih.m=2, Ca_LVAst.h=2, Ca_HVA.h=3`，
  σ=3.48e-4，3771 s
- `va_s2_2s`（128 trial，t0=4000）：

| model | k=2000 | k=8000 | DIV |
|---|---:|---:|---:|
| R2（1 s 数据） | **0.440** | **0.732** | 0/128 |
| C-only 基线（2 s） | 0.834 | 2.789 | 1/128 |
| C-only + 权重 | 0.807 | 2.687 | 3/128 |

- 在**训练集**上基线的 k=8000 也是 **3.47**（0 发散）⇒ 不是泛化问题，是闭环本身不稳定
- 权重只带来 2–3% 的变化，且在噪声内 ⇒ **该干预无效**（不是“加权有害”的结论，而是基座不成立）
- 两个混杂因素：
  1. **事件密度**：2 s 数据下 128 帧窗口包含事件的概率约为 1 s 的一半（同样事件摊到 2 倍时长）；
     C+A 用 L=512（4× 窗口）就稳定，与这个解释一致
  2. **scaler 估计噪声**：2 s scaler 只用 400 个单帧估计，`SKv3_1.m` 的 scale 比 2000 样本小 36%、
     V 小 8%（重算 `tools/make_scaler --n-sample 2000` 可复现）
- ⇒ 2 s + L=128 不能当对照基座；下一步应回到 **1 s 数据**复现 R2 配方，或先解决 scaler/事件密度
- 冻结（α=0）linear 更好但 plateau 从 0.551 恶化到 0.639 ⇒ **不能一律按住**，
  训练侧应做“只在真值静止的帧惩罚增量”，而不是固定 α

## 3. 进行中

- **没有训练/评测进程在跑**；最后一次训练 2026-10-28 21:37–02:32（R6 C+A，9000 步，4.91 h），评测已完成
- **R6 C+A 不采纳**：`te_s4_2s` 上全视界、全档差于 R2，物理指标也更差；按 PLAN §8 停止 C 路线
- **R5 零训练标定失败；R5-H1 未改善；R5-B 也未改善**；H2 未启动；下一步需选更长数据/完整 9000 步干净对照/工程修复
- **完整结果叙述文档尚未写**：当前入口是 `results/rounds/README.md` + R2/R3/R4 的 `CONCLUSIONS.md`；`DESIGN_v4.md` 原先引用的 `EXPERIMENTS_RESULTS.md` 不存在，引用已改到 `results/rounds/README.md`
- **`runs/` 已开始使用**：`eval-gate`、`r5-calib`、`r5-h1`、`r5-b` 四个目录已有 commit/config/metrics/stdout/REPORT；但训练代码仍写 `results/checkpoints/`，需人工复制
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
3. **R2 长程不稳健、且对 t0 敏感**：va_s2 k=8000=0.849 [0.733,0.983]，te_s4 k=8000=0.899 [0.783,1.031]（CI 含 1）；L2 ridge 更好（0.820/0.869），L1 非因果上界也更好（0.632/0.693）。te_s4 若只用 t0=4000 单窗口，k=8000=0.721；加上 t0=16000 后升到 0.899 ⇒ 单窗口结果偏乐观。linear 档在 te_s3 k=4000/8000 为 1.089/1.720，va k=8000 为 2.077。
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
17. **R5-H1 视界 pilot 失败**：L=512/cl=128 只跑 2000 步就评估，te_s4 k=2000 2.025、k=8000 2.960（R2 0.594/0.899）；单步 σ=0.000437（R2 0.000310）。既有欠训练，也有压缩课程/长闭环不稳定，不能当成干净的“视界有害”证据。
18. **R5-B 事件中心采样 + V 加权失败**：9000 步后 te_s4 k=2000 1.414（B_last）/0.973（B_best），k=8000 3.199/1.858（R2 0.594/0.899）；B_last 28/256 发散；Vmax 28.82（真值 19.81），Ca F1 0.163（R2 0.329）。事件片段过拟合 + V 加权放大陡变，未解决长程。
19. **迁移准备**：`sim2.py` 已从硬编码 `/home/neuron/...` 改为项目相对路径；`train2 t2` 新增 `--scaler`；`tools/make_scaler.py` 可为 2 s 数据算新 scaler。但 `neuron/mods/x86_64/libnrnmech.so` 是平台相关产物，工作站必须重新 `nrnivmodl`；`data/`、`results/checkpoints/` 不进 Git。
20. **工作站 `DISPLAY` 不可达会把 NEURON 卡死**：工作站 `DISPLAY=192.168.31.175:0.0`，但该 X server 不通；`import neuron` 会阻塞在 X 连接（strace 可见 connect 6000 端口），`nrnivmodl` 同样受影响。所有 NEURON/训练命令须 `env -u DISPLAY`（或 `NEURON_MODULE_OPTIONS=-nogui`）。
21. **DSH 会话沙箱把 `/home` 与 CIFS 挂载显示为只读**：真实挂载是 rw（`danger-full-access` 下 `touch` 通过）；默认 `workspace-write` 下 `touch data/...` 报 `Read-only file system`。写 NAS/环境需放宽沙箱，不要把“只读”误判成挂载故障。
22. **RSS 6 GB 阈值会误杀工作站训练**：HANDOFF §18 / PLAN §8 原先沿用 WSL 7.8 GB 机器的 `RSS > 6 GB` 停止条件；工作站 502 GB 下 L=512 训练实测 RSS 4.3–6.2 GB（峰值 6.19 GB、MemAvailable 始终 >400 GB），2026-10-28 21:06 在 step 771/9000 被误杀。已改为 **RSS > 32 GB 或 MemAvailable < 16 GB** 才停。
23. **本地 `results/data/scaler_v4.json` 目前放的是 2 s scaler**：`tools/eval_gate.py` 没有 `--scaler` 参数，评 `te_s4_2s` 时把 `WM/scalers/scaler_v4_2s.json` 复制到了该路径。以后评 1 s 数据前必须换回 1 s scaler（NAS `WM/scalers/scaler_v4.json`），否则 L2/L3 基线归一化会用错。

## 6. 待确认问题

1. **R2 是否作为当前基线冻结？** 评估闸门确认 R2_last 是当前最好的模型，但 200 ms 不稳健（te_s4 CI 含 1，L2 更好）。是否接受“R2 只赢短/中程，200 ms 交给 L2/后续 R6”？
2. **R5-B 之后已选 C+A**：2 s 数据 + `L=512` + 完整 9000 步；C-only 归因，A-only/3 s 暂缓。
   完整命令、数据生成、scaler、迁移清单见
   [`docs/PLAN_R6_C_A_AND_MIGRATION.md`](docs/PLAN_R6_C_A_AND_MIGRATION.md)。
   当前未启动新实验；等待工作站/确认后执行。
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

### 7.2 R5-H1 已执行（失败），H2 未启动

`runs/2026-10-27-f81f09c-r5-h1/`：L=512、cl=128、batch=24、2000 步（stage A 压缩为 500）。
`te_s4` 结果：

| model | k=200 | k=2000 | k=8000 |
|---|---:|---:|---:|
| R2_last | **0.562** | **0.594** | **0.899** |
| H1_last | 0.833 | 2.025 | 2.960 |
| H1_best | 0.828 | 2.037 | 10.617 |

⇒ **H1 没有改善；按约定不启动 H2（length=1024）。**

解释边界：H1 只跑 2000 步（R2 是 9000），且 stage A 从 2000 压到 500，
所以这是一次“低成本方向测试”，不能证明长视界本身有害；但它也没有给出继续加长到 1024 的理由。
长闭环后期 loss/grad 多次尖峰，H1_last 在 200 ms 严重漂移。

### 7.3 已定计划：C+A → C-only（见计划文档）

2 s 数据 + **C+A**（`L=512`，完整 9000 步，stage A 2000）为首选；
若 C+A 有改善，再跑 **C-only**（2 s，`L=128`，9000 步）区分“数据长度”与“视界”。
A-only 和 3 s 数据暂缓。完整命令、资源预算、判据、工作站迁移清单：

[`docs/PLAN_R6_C_A_AND_MIGRATION.md`](docs/PLAN_R6_C_A_AND_MIGRATION.md)
和 [`docs/HANDOFF_WORKSTATION_AGENT.md`](docs/HANDOFF_WORKSTATION_AGENT.md)

关键前置：

1. 2 s 三组数据（约 35 GB）生成后，先用 `tools/make_scaler.py` 写
   `results/data/scaler_v4_2s.json`，训练时传 `--scaler`；
2. 工作站上先 `cd neuron/mods && nrnivmodl`，再跑 `tools/v0_check`；
3. 不并行跑数据生成和训练；当前 WSL 只适合单次 C+A/C-only，完整 2×2 建议在工作站。

### 7.4 工程修复（与 R5 分开）

1. ✅ 修 `datagen2 --resume` 的 RNG/`th_arr`（2026-10-29，见 2.10；逐字节一致已验证）；
2. ⬜ 用 4 个窗口对照 T1 target=`s0` 与 `s_true`，确认后再决定是否重训 T1；
3. ✅ 把 `MemGuard.check()` 接进训练循环（`--rss-cap-mb`，触发先落盘）；
4. ✅ `runs/` 接入：`tools/guard_run.sh`（监控/停机）+ `tools/new_run.sh`（建目录/commit.txt）；
   周期 `last.pt` 也已加上，崩溃最多丢 `--save-every` 步。

