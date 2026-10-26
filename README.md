# WModel2 — L5PC 树突世界模型（v4）

在官方 L5PC NEURON 模型上，把 `apical_51` 这一段（11 个 compartment）当作受突触驱动的动力系统，学习：

```
Enc(s_t, θ) -> z_t        Pred(z_t, u_t, θ) -> z_{t+1}        Dec(z_{t+1}) -> ŝ_{t+1}
```

- `s`：7 个慢门控 + `cai` + `V`，逐 compartment ⇒ 9×11 = **99 维**
- `u`：目标段 11 + 邻段 17 个 compartment 的 AMPA / NMDA(去门控) / GABA 电导，加 1 维 bAP 指示 ⇒ **85 维**
- 数据：`tr_s1` / `va_s2` / `te_s3` 各 200 条，`dt=0.025 ms`，每条 40000 帧；由 `world.gen.datagen2` 现场调 NEURON 生成，共约 17 GB，**不是下载的数据**

## 当前状态

- 实验方案：`docs/DESIGN_v4.md`（已定档，2026-10-22 执行）
- 数据：三组 v4 数据已生成，位于 `data/`
- 训练：R0–R4 六轮已完成；**当前最优为 R2**（`t2v5_events`，`model/L0@200=0.562`，`model/L0@8000=0.911`，无发散）
- R4（`t2v5_events_w8`）在 k=8000 出现数值发散，不采纳；详见 `results/rounds/R4_event_w8/CONCLUSIONS.md`
- 结果入口：`results/rounds/README.md`（跨轮索引）；`results/rounds/R*/CONCLUSIONS.md`（人工结论）

## 目录地图

| 路径 | 内容 |
|---|---|
| `neuron/` | 官方 NEURON 资产（hoc / mod / morphology / reference），**只读，禁止修改** |
| `world/core/` | 资源护栏、进度条、v4 常量 |
| `world/gen/` | `datagen2.py`（生成）、`sim2.py`（NEURON 仿真）、`protocol.py` + `protocol2.py`（驱动协议）、`secinfo.py`（θ_s） |
| `world/learn/` | `data2.py`（memmap / token）、`model.py`（Enc/Pred/Dec）、`train2.py`（T1/T2/eval） |
| `tools/` | 当前入口工具：`round_report.py`、`rounds_index.py`、`fig_rounds_summary.py`、`ca_event_metrics.py`、`latent_ci.py`、`diag_latent.py`、`plot_v4.py`、`verify_env.py`、`v0_check.py`、两个探针 |
| `docs/` | 当前设计/事实/文献；历史文档在 `docs/archive/` |
| `results/` | 当前 ckpt / rounds / logs / reports / figs；旧 v3 产物在 `results/archive/legacy_v3/` |
| `archive/legacy_v3/` | 已归档的 v3 代码与旧工具，**不再 import，不要作为当前入口** |
| `data/` | 大数据集，可重生成；当前 17 GB |

## 怎么跑

所有命令在 `WModel2/` 下执行：

```bash
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

# 已有数据，直接训练/评测
bash run_v4.sh v4     # T1 自编码
bash run_v4.sh v5     # T2 两阶段课程
bash run_v4.sh v6     # 配对评测（te_s3）
bash run_v4.sh all    # v3 -> v4 -> v5 -> v6 -> diag -> v6b
```

`run_v4.sh` 已固定使用：

```text
/home/neuron/miniconda3/envs/worm_sim/bin/python
```

重新生成数据用 `bash run_v4.sh v3`。注意：`va_s2` 的 `plateau_frac=48.5%` 略高于 45% 门限，`datagen2` 会返回 1，`set -euo pipefail` 会让脚本在 va_s2 后停下；现有数据是完整的，训练/评测不必重生成。

## 环境事实

- 解释器：`/home/neuron/miniconda3/envs/worm_sim/bin/python`（3.10.20）；新终端默认 `base`，没有 torch
- NEURON 8.2.0；加载顺序 `stdrun.hoc` → `import3d.hoc` → `L5PCbiophys5b.hoc` → `L5PCtemplate_2.hoc`
- numpy 2.0.2（用 `np.ptp`，不要 `arr.ptp()`）；torch 2.4.1+cpu
- 内存：总 7.8 GB；训练进程实测 RSS 约 3.4–4.0 GB，同一时刻只跑一个训练
- 线程数必须在 import numpy/torch 之前设好（`world.core.safety.apply_thread_caps`）

## 文档与清理约定

- 当前设计以 `docs/DESIGN_v4.md` 为准；事实以 `docs/MODEL_FACTS.md` 为准；代码/实测冲突时以代码和实跑输出为准
- v2/v3 设计、迁移记录、旧事故调查在 `docs/archive/`；旧 v3 代码在 `archive/legacy_v3/`；旧 v3 结果在 `results/archive/legacy_v3/`
- 归档区只作历史追溯，不参与当前训练/评测
