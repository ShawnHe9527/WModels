# world/ — 当前 v4 代码

只保留 v4 活跃模块。v3 的 `datagen.py` / `sim.py` / `data.py` / `train.py` /
`metrics.py` 与 `world/viz/` 已归档到 `archive/legacy_v3/code/`，不参与当前入口。

## 模块地图

| 路径 | 作用 | 状态 |
|---|---|---|
| `core/safety.py` | 线程数、内存预检、RSS 读取 | ✅ 当前 |
| `core/progress.py` | 终端进度条（非 tty 自动逐行） | ✅ 当前 |
| `core/config.py` | 只保留 NEURON 常量、目标 section、状态/通道/区域清单 | ✅ 当前 |
| `gen/protocol.py` | 目标段棘波串、burst、refractory（v3 标定，v4 复用） | ✅ 当前依赖 |
| `gen/protocol2.py` | `Policy2` 分层采样、tuft 簇、bAP 时刻表 | ✅ 当前 |
| `gen/sim2.py` | `Sim2`：目标段 22 突触 + 邻段 34 突触 + 胞体脉冲池；输出 `s(9,11)`、`u(85)` | ✅ 当前 |
| `gen/secinfo.py` | NEURON 初始化、θ_s 提取（`sim2.theta()` 调用 `build_theta`） | ✅ 当前（含部分旧辅助函数，仅 θ 路径在用） |
| `gen/datagen2.py` | memmap 写盘 + 续跑 + 门禁；CLI 入口 | ✅ 当前 |
| `learn/data2.py` | `Dataset2` / `Sampler2`：85 维 → T2(28 token) / T1(14 token) | ✅ 当前 |
| `learn/model.py` | `Encoder` / `Predictor` / `Decoder` / `WorldModel.rollout`（残差输出） | ✅ 当前 |
| `learn/train2.py` | `t1` / `t2` / `eval`，两阶段课程 + 闭环尾 + anchor | ✅ 当前 |

## 数据流

```
protocol2.Policy2
   ├─ protocol.make_spike_trains     目标段 11 E + 11 I
   ├─ protocol2.make_tuft_bundles    邻段 17 E + 17 I（簇状）
   └─ protocol2.make_bap_times       胞体脉冲池
                ↓
sim2.Sim2.run -> s (T,9,11), u (T,85)
                ↓
datagen2 -> data/<name>/{s.dat,u.dat,th.npy,meta.json}
                ↓
data2.Dataset2 / Sampler2 -> tokens
                ↓
train2 t1 -> t2 -> eval
```

## 命令

```bash
cd /home/neuron/L5PC/WModel2
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

python -m world.gen.datagen2 --help
python -m world.learn.train2 --help
```

`run_v4.sh` 是完整流水线；当前数据已生成，通常只跑 `bash run_v4.sh v4/v5/v6`。

## 关键约定

- `u` 是 comp-major `[AMPA, NMDA_ungated, GABA]` × 28 个 compartment + bAP，共 85 维；T1 用 `sim2.to_t1()` 派生，不再单独存
- `s` 的 `cai` 存 **log10**；`V` 是自由膜电位（不是钳位）
- 闭环输出用残差：`ŝ_{t+1} = ŝ_t + [Dec(ẑ_{t+1}) - Dec(ẑ_t)]`，起点用真值 `s0`
- `config.py` 不再放实验窗口/优化超参；窗口和超参由 CLI 决定
- `secinfo.py` 中旧 `new_cell` / `read_state` 等函数不是 v4 主路径；当前只通过 `sim2.theta()` 调 `build_theta`
