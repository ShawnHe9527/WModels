# R6 实验计划：C+A / C-only 与工作站迁移（2026-10-27）

> 本文件是 **R5-B 之后**的执行计划。当前结论：R2 仍是唯一在所有总体视界 <1 的配置；
> R5 零训练标定、H1（视界 pilot）、B（事件中心 + V 加权）均失败。
> 本计划只做 **C（2 s 数据）** 和 **A（L=512 完整 9000 步）**，不再做 B/H2。

## 1. 要回答的问题

|  | `L=128`（R2 视界） | `L=512`（A） |
|---|---|---|
| 1 s 数据 | R2（已有） | A-only（可选） |
| **2 s 数据** | **C-only（归因）** | **C+A（首选）** |

- **C+A**：2 s 数据 + `length=512`，追求效果；
- **C-only**：2 s 数据 + R2 视界，用来区分“数据长度”与“视界”的贡献；
- **A-only**：1 s 数据 + `length=512`，只有预算充足时才做，用来单独验证视界。

**执行顺序**：2 s 数据 → C+A →（若有改善）C-only →（可选）A-only。

## 2. 数据（2 s）

三组数据，**不改仿真协议**，只改 `t_test`：

```bash
PY=/home/neuron/miniconda3/envs/worm_sim/bin/python
export PYTHONPATH=$PWD

$PY -u -m world.gen.datagen2 --name tr_s1_2s --n-trial 200 --seed 1 \
    --t-settle 100 --t-test 2000 --every-secs 20
$PY -u -m world.gen.datagen2 --name va_s2_2s --n-trial 200 --seed 2 \
    --t-settle 100 --t-test 2000 --every-secs 20
$PY -u -m world.gen.datagen2 --name te_s4_2s --n-trial 200 --seed 4 \
    --t-settle 100 --t-test 2000 --every-secs 20
```

- `T = 80000` 帧 / trial（2 s），三组约 **35 GB**；
- 生成约 **1.5–2 h**（5 s/trial 的 2 倍左右）；
- `datagen2` 若因 `plateau_frac` 返回 1，**数据仍完整**；记录后接受，不用测试集调参；
- `te_s4_2s` 生成后**不要用于调参**，只做最终盲测；
- 不要用 `--resume`（当前 RNG 续跑不可复现）。

## 3. 新 scaler（必须）

新数据不能直接复用 `results/data/scaler_v4.json`。先算新 scaler，再训练：

```bash
$PY -m tools.make_scaler --data tr_s1_2s \
    --out results/data/scaler_v4_2s.json --seed 0
```

`train2` 新增了 `--scaler`，C+A/C-only 必须显式传入：

```text
--scaler results/data/scaler_v4_2s.json
```

T1 权重首轮复用 `t1v4_dz16/best.pt`；如果新 scaler 的 `loc/scale` 与旧值差很多，
再决定是否重训 T1（重训约 1 h）。不要覆盖旧 `scaler_v4.json`。

## 4. C+A：2 s + L=512 + 完整 9000 步

```bash
$PY -u -m world.learn.train2 t2 \
  --data tr_s1_2s --val va_s2_2s --run r6_ca_2s_l512 --tier t2 \
  --init t1v4_dz16/best.pt --scaler results/data/scaler_v4_2s.json \
  --steps 9000 --batch 24 --length 512 \
  --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 128 \
  --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 \
  --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --mem-mb 5000
```

- **不使用** `--event-center-frac` / `--event-v-weight`（B 已失败）；
- stage A 保持 **2000 步**，不要像 H1 那样压到 500，否则不是干净对照；
- 预计 **4–5 h**，峰值 RSS **4.5–5.5 GB**；同一时刻只跑一个训练。

## 5. C-only（归因，C+A 有改善后再跑）

```bash
$PY -u -m world.learn.train2 t2 \
  --data tr_s1_2s --val va_s2_2s --run r6_c_only_2s --tier t2 \
  --init t1v4_dz16/best.pt --scaler results/data/scaler_v4_2s.json \
  --steps 9000 --batch 24 --length 128 \
  --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 \
  --cl-weight 1.0 --cl-weight-start 0.1 \
  --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 \
  --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --mem-mb 5000
```

- 预计 **约 1 h**，RSS 约 4 GB；
- 与 C+A 的差异 = 视界贡献；与 R2 的差异 = 数据 + 视界组合。

## 6. 评测（同一套协议）

先在同一批 `te_s4_2s` 上评 R2（zero-shot）和 C+A/C-only：

```bash
$PY -m tools.eval_gate \
  --data te_s4_2s --t0 4000,16000 --ks 10,200,2000,8000 \
  --n-trials 128 --batch 16 \
  --model R2=t2v5_events/last.pt \
  --model CA=r6_ca_2s_l512/last.pt \
  --out runs/<date>-<commit>-r6-ca/eval_te_s4_2s.json
```

- 同时报 `te_s4`（旧 1 s）上的结果，确认没有退化；
- 分 regime + L2 ridge + Vmax + Ca peak F1；
- 最终盲测只用 `te_s4_2s` 一次；不要根据它继续调参。

**判据**：

- `te_s4_2s` k=2000/8000 ≤ R2（R2 在旧 `te_s4` 为 0.594/0.899；在新测试集要重测）；
- 无 `DIVERGED`；
- Vmax 比值 `pred/true ∈ [0.8,1.2]`（不要再用绝对 13–17 mV）；
- Ca 峰 F1 不低于 R2；如果 C+A 只改善长程但物理指标变差，需要记录，不直接接受。

## 7. 工作站迁移清单

### 7.1 代码

- 克隆 `https://github.com/ShawnHe9527/WModels.git`；
- `sim2.py` 已改为**项目相对路径**，不再硬编码 `/home/neuron/...`；
- 需要在工作站重编译 NEURON 机制：

```bash
cd neuron/mods
nrnivmodl
```

  `x86_64/` 未进 Git，必须重建；`libnrnmech.so` 不能跨平台拷贝。
- Python 环境按 `AGENTS.md` 重装（`worm_sim`，Python 3.10，NEURON 8.2.0，
  numpy 2.0.2，torch 2.4.1+cpu），或者按工作站实际版本更新 `AGENTS.md`。

### 7.2 数据与权重

Git 不含：

- `data/`（17 GB，1 s；2 s 约 35 GB）；
- `results/checkpoints/`（R2、T1、C+A 等权重）；
- `results/archive/`。

迁移方式二选一：

1. 拷贝 `data/` 的 `tr_s1` / `va_s2` / `te_s4`，以及 `results/checkpoints/t1v4_dz16`、
   `t2v5_events`、`t2v4_anchor` 等；
2. 在新机器重新生成 1 s/2 s 数据（约 1.5–2 h/组）+ 重新训练 R2（约 1 h）。

推荐 **拷贝 1 s 数据和 R2 checkpoint**，2 s 数据在工作站生成（避免 35 GB 传输）。

### 7.3 运行前检查

```bash
$PY tools/v0_check --n 1 --seed 99 --mem-mb 2500
$PY -m tools.make_scaler --data tr_s1 --out /tmp/scaler_check.json --n-sample 20
```

- `v0_check` 必须 1/1 通过；
- 若 `nrnivmodl` 后 neuron 加载失败，先查 `neuron/mods/x86_64/libnrnmech.so` 是否存在；
- 工作站上更新 `AGENTS.md` 的环境事实（Python 路径、conda 名、GPU/CPU）。

### 7.4 资源预算

| 项 | 当前 WSL | 工作站建议 |
|---|---|---|
| 内存 | 7.8 GB，训练 4.5–5.5 GB | ≥32 GB（64 GB 更稳） |
| CPU | 20 逻辑核，CPU only | 8 核以上；GPU 可选 |
| 磁盘 | 908 GB 可用 | ≥500 GB 用于 1 s+2 s 数据 |
| 2 s 数据 | 35 GB | 同 |
| C+A | 4–5 h | 视 CPU/GPU 而定；CPU 代码不自动变快 |

**GPU 不是即插即用**：`train2` 目前是 CPU 代码；要用 GPU 需要改数据/模型 `.to(device)`
和 rollout 循环，并重新验证数值一致性。

## 8. 停止条件

- 2 s 数据生成失败/NaN：停止，不训练；
- C+A 在 `te_s4_2s` k=2000/8000 比 R2 差，且物理指标没有改善：停止 C 路线；
- C+A 有改善但 C-only 几乎一样：说明主要是数据长度贡献，不是视界；
- C+A 有改善且 C+A > C-only：再评估 A-only 或 3 s 数据；
- 任何一次训练 RSS > 6 GB 或系统可用 <600 MB：立即停止，降低 batch/迁移工作站。

## 9. 产出

- `runs/<date>-<commit7>-r6-ca/`：`commit.txt` / `config.yaml` / `stdout.log` / `metrics.json`；
- 结果图与说明：沿用 `tools/plot_r5b.py` 的图型，扩成 R6 报告（训练曲线、视界、Vmax/F1、trace）；
- 文档：`PROJECT_STATE.md` 更新；本文件保留为计划与迁移依据。

---

**当前决定**：先做 **2 s 数据 + C+A**；只有 C+A 有改善才做 C-only。A-only/3 s 暂不启动。
