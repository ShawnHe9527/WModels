每次会话开始，先读 PROJECT_STATE.md

# AGENTS.md — WModel2 项目事实与工作约束

> 本文件只写**长期稳定的事实与规则**；实验进度、当前结论、待确认问题一律写在
> `PROJECT_STATE.md`。两者冲突时：代码/实跑输出 > `PROJECT_STATE.md` > 本文件。

## 1. 一句话说明

在官方 L5PC NEURON 模型上，把 `apical_51`（11 个 compartment）当作受突触驱动的动力系统，
学习 `Enc(s_t,θ)→z_t`、`Pred(z_t,u_t,θ)→z_{t+1}`、`Dec(z_{t+1})→ŝ_{t+1}` 的 latent 世界模型，
并预测 9 个状态因子（7 个慢门控 + `cai` + `V`）的下一步演化。

## 2. 环境

- conda 环境名：`worm_sim`
- Python 绝对路径：`/home/neuron/miniconda3/envs/worm_sim/bin/python`（3.10.20）
- 当前版本：`numpy==2.0.2`、`torch==2.4.1+cpu`、`NEURON==8.2.0`、`scipy==1.15.3`、`matplotlib==3.10.8`
- 当前仓库**没有** `requirements.txt` / `pyproject.toml`；不要凭记忆升级版本，先对照本节的实测版本。

复现当前环境的命令（不要在其他已有环境上直接升级；装包前先确认）：

```bash
/home/neuron/miniconda3/bin/conda create -n worm_sim python=3.10 -y
/home/neuron/miniconda3/envs/worm_sim/bin/python -m pip install --upgrade pip
/home/neuron/miniconda3/envs/worm_sim/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu
/home/neuron/miniconda3/envs/worm_sim/bin/python -m pip install \
  numpy==2.0.2 NEURON==8.2.0 scipy==1.15.3 matplotlib==3.10.8
```

- 线程数必须在 import numpy/torch **之前**设置；脚本里已通过
  `world.core.safety.apply_thread_caps()` 处理。
- 训练进程实测 RSS 约 **3.4–4.0 GB**（R2；L512/事件加权可到 5–6 GB）；同一时刻只允许一个训练/评测进程。
- 迁移到工作站：`sim2.py` 已用项目相对路径；`neuron/mods/x86_64/libnrnmech.so` 必须在目标机
  重新 `nrnivmodl`；2 s 数据用 `tools/make_scaler.py` 生成新 scaler，训练传 `--scaler`。
- 新终端默认是 `base`，没有 torch；不要依赖 `conda activate`，一律用上面的绝对路径。

## 3. 常用命令

先设置：

```bash
cd /home/neuron/L5PC/WModel2
PY=/home/neuron/miniconda3/envs/worm_sim/bin/python
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

### 3.1 跑单次实验

现有流水线（会写 `results/checkpoints/<run>/`，属于历史布局）：

```bash
bash run_v4.sh v4     # T1 自编码
bash run_v4.sh v5     # T2 初始基线
bash run_v4.sh v6     # 评测当前最优 R2（t2v5_events）
```

直接训练一个 T2：

```bash
$PY -u -m world.learn.train2 t2 \
  --data tr_s1 --val va_s2 --run <run-name> --tier t2 \
  --init t1v4_dz16/best.pt --steps 9000 --batch 24 --length 128 \
  --d-z 16 --hidden 192 --n-layers 2 --lr 1e-3 \
  --cl-stage-a-steps 2000 --cl-start 4 --cl-steps 64 --cl-weight 1.0 \
  --cl-weight-start 0.1 --anchor-weight 0.1 --anchor-every 4 --anchor-cl 0.1 \
  --event-weight 10 --event-tau 0.2 \
  --regime-weights "0.25,0.35,0.40" --hot-frac 0.5 \
  --stage-b-lr 2e-4 --stage-b-grad-clip 0.3 \
  --sigma-gate 0.008 --mem-mb 3300
```

> 当前代码还不能直接把 ckpt 写到 `runs/`；按第 5 节规范，训练后用 `cp` 把 ckpt
> 复制进本次 `runs/<date>-<commit7>/ckpt/`，并在 `metrics.json` 里记录 ckpt 相对路径。
> 不要为了省事只把结果留在 `results/checkpoints/`。

### 3.2 跑测试/自检

项目没有 pytest 套件；最低验证按下面顺序：

```bash
# 1) NEURON 资产、hoc 加载顺序、机制、单位等 13 项自检
$PY tools/verify_env.py

# 2) 所有活动 Python 文件语法
$PY -m compileall -q world tools

# 3) 真实 NEURON 端到端（2 条 trial，约几十秒；会加载 mod）
$PY -m tools.v0_check --n 2 --seed 99 --mem-mb 2000
```

如果只改文档/README，也必须至少跑第 2 项，确认没有误改 import。

> ⚠️ `tools/verify_env.py` 目前仍按旧根目录布局找 `mods/`、`morphologies/`、`probes/`，
> 在 v4 下会 10/13 失败；**不要把它当成当前门禁**。环境可用性以 `tools/v0_check` 为准。

### 3.3 出图/汇总

```bash
# 单轮：评估一个 ckpt，写 results/rounds/<round>/
$PY -u -m tools.round_report --run t2v5_events --round R2_event_weighting \
  --title "当前最优 R2" --k 2000 --mem-mb 3200

# 跨轮索引 + 对比图
$PY -m tools.rounds_index

# 六轮总结图
$PY -m tools.fig_rounds_summary

# 单 ckpt 的诊断图（误差-视界、latent、空间剖面等）
$PY -m tools.plot_v4 --ckpt t2v5_events/last.pt --data te_s3
```

注意：`fig_rounds_summary` 和 `rounds_index` 没有 `--help`，运行即写文件；
不要在“只读检查”时误跑。

## 4. 目录结构

```text
WModel2/
├── run_v4.sh                 # 当前流水线入口：v3 / v4 / v5 / v6 / v6b / diag / all
├── neuron/                   # 官方 NEURON 资产，只读
│   ├── hoc/                  # stdrun/import3d/L5PCbiophys5b/L5PCtemplate_2
│   ├── mods/                 # 18 个 .mod + 已编译的 x86_64/libnrnmech.so
│   ├── morphologies/cell1.asc
│   └── reference/            # 官方仿真脚本与 L5PC_exp.py
├── world/
│   ├── core/                 # safety（线程/内存）、progress、config（v4 常量）
│   ├── gen/                  # datagen2 / sim2 / protocol / protocol2 / secinfo
│   └── learn/                # data2 / model / train2
├── tools/                    # 当前工具与两个保留探针
├── docs/                     # DESIGN_v4、MODEL_FACTS、文献/资产参考
│   └── archive/              # v1/v2/v3 设计与旧调查，仅历史追溯
├── data/                     # v4 数据集，约 22.5 GB，可重生成但很贵
│   ├── tr_s1/ va_s2/ te_s3/ te_s4/  # s.dat u.dat th.npy meta.json
├── results/
│   ├── checkpoints/          # 当前 v4 权重（t1v4 + R0–R4）
│   ├── rounds/               # R0–R4 归档 + 总索引 + 对比图
│   ├── logs/ reports/ figs/  # 当前日志、JSON、诊断图
│   ├── data/                 # scaler_v4.json
│   └── archive/legacy_v3/    # 旧 v3 结果，仅历史追溯
├── archive/legacy_v3/        # 旧 v3 代码/工具，仅历史追溯
└── runs/                     # 新规范：每个实验一个 <YYYY-MM-DD>-<commit7>/（已建多个）
```

## 5. 实验规范（从本文件生效起执行）

每个实验必须独立目录：

```text
runs/<YYYY-MM-DD>-<commit7>/
├── commit.txt        # git rev-parse HEAD；有未提交改动时追加 git status --porcelain
├── config.yaml       # 本次全部参数：data/seed/模型/优化器/CL/评测视界/ckpt 路径
├── metrics.json      # 机器可读结果：sigma、各 k 的 model/L0、diverged、分 regime、Ca 指标
├── stdout.log        # 完整 stdout+stderr（不要用 grep 丢掉 Traceback）
└── ckpt/             # 可选：best.pt / last.pt 副本
```

规则：

1. **一个实验一个目录，不覆盖**。重跑用新日期/新后缀，例如 `2026-10-27-abc1234-r5`。
2. `commit7` 取 `git rev-parse --short=7 HEAD`；当前仓库已初始化 git（`main`），
   `origin` 指向 `https://github.com/ShawnHe9527/WModels.git`，`main` 已 push。
   本机可直接 `git push`；新机器需要 PAT/SSH。不要再用 `nogit`。
3. `config.yaml` 必须足以复现：数据目录、seed、`tier`、`d_z`、`hidden`、`lr`、`batch`、
   `length`、CL 课程参数、`event_weight`、`anchor_*`、评测视界、ckpt 来源。
4. `metrics.json` 必须同时保留总体和 `linear/inter/plateau` 分档；发散点标 `diverged: true`，
   不能只写一个漂亮的总数。
5. 失败实验也保留目录和 `stdout.log`；不要删失败记录，否则无法归因。
6. 一次只改一个变量；同时改两处必须先在 `PROJECT_STATE.md` 的“待确认问题”里写清楚并等确认。
7. `results/rounds/` 是 R0–R4 的历史归档，不往里面塞新实验；新实验只进 `runs/`。
8. 注意保护CPU，内存和磁盘空间，避免实验失败。

## 6. 禁区

### 6.1 不能改/不能删的文件

- `neuron/**`：官方模型资产，**只读**；不要改 `.hoc`、`.mod`、`cell1.asc`、
  参考脚本，也不要删 `neuron/mods/x86_64/libnrnmech.so`。
- `data/{tr_s1,va_s2,te_s3,te_s4}/`：当前 v4 数据集，约 22.5 GB；不要覆盖、截断、删除。
  重生成一组约 17 分钟且会改变样本，必须先确认。
- `results/checkpoints/{t1v4_dz16,t2v4_anchor,t2v5_ancC,t2v5_noancC,t2v5_events,t2v5_events_w5a2,t2v5_events_w8}/`：
  当前 T1 与 R0–R4 权重；不要原地覆盖。
- `results/rounds/R0_baseline_t2v4` … `R4_event_w8/`：历史结果；`metrics.json` 和自动 `README.md`
  不要手改；`CONCLUSIONS.md` 是人工结论，修改前必须确认。
- `docs/DESIGN_v4.md`、`docs/MODEL_FACTS.md`、`docs/README.md`、根 `README.md`、`world/README.md`：
  设计与事实来源；修改前必须给方案并确认。
- `archive/**`、`results/archive/**`：历史归档，只读；不要为了“省空间”擅自删除。

### 6.2 不能擅自装的包/不能擅自做的变更

- 不要 `pip install` / `conda install` / 升级 `numpy`、`torch`、`NEURON`、`scipy`、`matplotlib`；
  需要新包先说明用途、版本、对现有环境的影响，等确认。
- 不要改评估协议、数据划分、`d_z=16`、9 个状态因子的定义；这些是实验结论可比性的前提。
- 不要启用 scheduled sampling（历史上爆炸）；需要时先论证并确认。
- 不要改 `world/gen/protocol.py`、`sim2.py`、`datagen2.py`、`world/learn/data2.py`、`model.py`、
  `train2.py` 的核心行为而不先出方案；这些会同时影响数据、训练和评测。
- 不要在 `/mnt/c` 下运行任何东西；不要依赖 `conda activate`；不要删除 `data/` 或 `results/`。
- 不要用 `rm -rf` 处理拼接出来的路径；删除前必须确认绝对路径、列出目标、得到明确同意。

## 7. 汇报约定

- 实验结束三行：**结论 / 证据（具体数字）/ 下一步建议**。
- 有图用 `present` 或 markdown 图片，不要只用文字描述。
- 收工前更新 `PROJECT_STATE.md`。
- 发现别的问题先记录到 `PROJECT_STATE.md` 的“已知问题/待确认问题”，不要顺手扩大范围。
