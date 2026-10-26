"""v4 数据生成：三类驱动（目标段 / tuft 簇 / 胞体 bAP）+ 门禁 + memmap 写盘。

## 数据集布局（`data/<name>/`）

| 文件 | 形状 | dtype | 大小（200 条）|
|---|---|---|---|
| `s.dat`  | `(N, T, 9, 11)` | float32 | 15.8 MB/条 ⇒ **3.2 GB** |
| `u.dat`  | `(N, T, 85)`    | float32 | 13.6 MB/条 ⇒ **2.7 GB** |
| `th.npy` | `(N, 27)`       | float32 | 21 KB |
| `meta.json` | 配置 + per-trial 协议、门禁、统计 | — | — |

T = t_test / dt（默认 1000 / 0.025 = 40000）。

### ⚠️ 只存 85 维（T2），不存 43 维（T1）

T1 是 T2 的**确定性函数**（按 section 对 compartment 求和），
用 `sim2.to_t1(U)` 在训练/评测时派生即可。好处：

1. 零额外存储、零额外仿真
2. **完全配对** —— 两次实验看的是同一个仿真的同一份记录，
   T1 vs T2 的差异纯粹来自观测分辨率（若存两份数据，两次生成用
   不同 RNG，差异会混杂样本差异，无法归因）
3. 还能顺便派生 T3（只用 E 总量 = 37 维）做第三档消融

### 为什么用 memmap

单条轨迹 s+u ≈ 29 MB，200 条约 5.9 GB，远超可用 RAM（1.5–2.7 GB）。
`np.memmap` 逐条写盘，内存峰值只有一条轨迹。

## 门禁（DESIGN_v4 §5.3 与 §9.1）

单 trial 门禁（不通过则只记录、不重采，保证可复现）：

| 门 | 判据 | 防什么 |
|---|---|---|
| G1 | 上行时间占比 ≤ 60% | 锁死上行态 |
| G2 | 有中间态（`−50<V<−25` 占比 ≥ 0.5%）| 退化成硬二值 |
| G3 | tuft 事件数 ≤ 20 | 事件泛滥 |
| G4 | bAP 数 ≤ 25 | 胞体过密 |
| G5 | 无 NaN、\|V\| < 150 mV | 数值发散 |

数据集级门禁（不合格则退出码 1，提示调采样范围）：

| 门 | 目标 |
|---|---|
| `plateau_frac` | **15%–45%** |

⚠️ `plateau` 是**逐 trial 布尔**（$V_{\max} > -30$ mV **且** cai 跨度 > 0.5），
`plateau_frac` 是**数据集比例**。两者不要混。
**cai 存的是 $\log_{10}$ 值**，所以 0.5 是 log10 口径。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m world.gen.datagen2 \
        --name tr_s1 --n-trial 200 --seed 1

    # 续跑（已写好的前 k 条跳过）
    ... --resume
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..core import safety
from ..core import progress

THREADS = safety.apply_thread_caps(4)

from . import protocol as P1            # noqa: E402
from . import protocol2 as P2           # noqa: E402
from . import sim2 as SM                # noqa: E402

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data")

UP_THR = -45.0
MID_LO, MID_HI = -50.0, -25.0
V_LIMIT = 150.0
PLATEAU_V = -30.0
PLATEAU_CAI = 0.5            # log10 口径


# ==============================================================================
def _open(path: str, shape: tuple, mode: str):
    return np.memmap(path, dtype=np.float32, mode=mode, shape=shape)


def _episodes(v, thr):
    m = (v > thr).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return int((d == 1).sum())


def _peaks(v, thr, min_gap):
    idx = np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])
                   & (v[1:-1] > thr))[0] + 1
    out: list[int] = []
    for i in idx:
        if not out or i - out[-1] >= min_gap:
            out.append(int(i))
    return out


def _gates(pol, st):
    """单 trial 门禁（按档位，linear 档按定义就是亚阈的）。"""
    g = {}
    if pol.regime == "linear":
        g["G1"] = st["up_frac"] <= 0.10
        g["G2"] = True
    else:
        g["G1"] = 0.005 <= st["up_frac"] <= 0.60
        g["G2"] = st["mid_frac"] >= 0.005
    g["G3"] = st["tuft_n_ev"] <= 20
    g["G4"] = st["n_bap"] <= 25
    g["G5"] = (not st["nan"]) and st["absmax"] < V_LIMIT
    return g


# ==============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="world.gen.datagen2")
    ap.add_argument("--name", type=str, required=True,
                    help="数据集名，如 tr_s1 / va_s2 / te_s3")
    ap.add_argument("--n-trial", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--t-settle", type=float, default=100.0)
    ap.add_argument("--t-test", type=float, default=1000.0)
    ap.add_argument("--nb-n-e", type=int, default=1,
                    help="邻段每 compartment 兴奋突触数（D4：1=O1，2=O2）")
    ap.add_argument("--nb-n-i", type=int, default=1)
    ap.add_argument("--mem-mb", type=float, default=1800.0)
    ap.add_argument("--resume", action="store_true",
                    help="已有前 k 条则跳过（续跑）")
    ap.add_argument("--every-secs", type=float, default=3.0,
                    help="进度刷新间隔（非 tty 时每 N 秒一行）")
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=a.mem_mb)

    outdir = os.path.join(DATA_ROOT, a.name)
    os.makedirs(outdir, exist_ok=True)

    T = int(round(a.t_test / a.dt))
    N = a.n_trial
    nseg_expected, nf = 11, SM.N_FACTORS

    # ---- 续跑：读已有 meta，决定从哪条开始 ----
    meta_path = os.path.join(outdir, "meta.json")
    done = 0
    meta = None
    if a.resume and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        done = len(meta.get("trials", []))
        print(f"续跑：已有 {done} 条，从第 {done} 条继续")

    sim = SM.Sim2(dt=a.dt, t_settle=a.t_settle, t_test=a.t_test,
                  nb_n_e=a.nb_n_e, nb_n_i=a.nb_n_i)

    shapes = {
        "s": (N, T, nf, nseg_expected),
        "u": (N, T, SM.U_DIM),
    }
    print(f"\n数据集 {a.name}: N={N}  T={T}  nseg={sim.nseg}  "
          f"u_dim={SM.U_DIM}  nb_e={sim.n_nb_e}  nb_i={sim.n_nb_i}")
    tot_mb = 0.0
    for k, sh in shapes.items():
        mb = np.prod(sh) * 4 / 1e6
        tot_mb += mb
        print(f"  {k}.dat  {sh}  ≈ {mb:.1f} MB")
    print(f"  合计 ≈ {tot_mb:.0f} MB（memmap 逐条写，内存峰值 = 1 条轨迹）")

    mode = "r+" if done > 0 else "w+"
    starts = {k: _open(os.path.join(outdir, f"{k}.dat"), sh, mode)
              for k, sh in shapes.items()}
    th_arr = np.zeros((N, 27), dtype=np.float32)

    th = sim.theta()
    print(f"θ_s (27 维) 前几个: {np.round(th[:6], 4)}")

    if meta is None:
        meta = dict(
            version="v4", name=a.name, n_trial=N, T=T, dt=a.dt,
            t_settle=a.t_settle, t_test=a.t_test,
            nseg=sim.nseg, factor_names=SM.FACTOR_NAMES,
            u_dim=SM.U_DIM, u_layout="comp-major [AMPA,NMDA_ungated,GABA]",
            u_slices=dict(own=[0, 33], nb_apic50=[33, 42],
                          nb_apic52=[42, 57], nb_apic53=[57, 84],
                          bap=[84, 85]),
            nb_order=[int(i) for i in SM.NB_IDX],
            nb_n_e=a.nb_n_e, nb_n_i=a.nb_n_i,
            section=SM.SECTION, seed=a.seed,
            gmax_e=SM.G_E, gmax_i=SM.G_I,
            driver=dict(tuft="bundle (recruit fraction f)",
                        bap="soma IClamp pulse pool, real AP",
                        own="ProbAMPANMDA2 + ProbUDFsyn2"),
            t1_derivation="sim2.to_t1(U85) -> (T,43)",
            trials=[],
        )

    rng = np.random.default_rng(a.seed)
    # 续跑时把 RNG 推进到相同状态（前 done 条已消耗）
    for _ in range(done):
        _ = P2.sample_policy2(rng)

    prog = progress.Progress(max(N - done, 1),
                             desc=f"生成 {a.name}",
                             every_secs=a.every_secs)
    t_all = time.time()

    for i in range(done, N):
        pol = P2.sample_policy2(rng)

        own_e, own_i, _ = P1.make_spike_trains(
            rng, sim.nseg, a.t_settle, a.t_settle + a.t_test,
            pol.to_policy1())
        tuft_e, tuft_i, bundles = P2.make_tuft_bundles(
            rng, sim.n_nb_e, sim.n_nb_i, a.t_settle,
            a.t_settle + a.t_test, pol)
        baps = P2.make_bap_times(rng, bundles, a.t_settle,
                                 a.t_settle + a.t_test, pol,
                                 max_n=SM.MAXP)

        t0 = time.time()
        S, U = sim.run(own_e, own_i, tuft_e, tuft_i, baps,
                       bap_amp=pol.bap_amp, bap_width=pol.bap_width)
        wall = time.time() - t0

        starts["s"][i] = S
        starts["u"][i] = U
        th_arr[i] = th

        # ---- 统计 ----
        V = S[:, 8, :]
        cai = S[:, 7, :]                    # 已经是 log10
        vmean = V.mean(axis=1)
        tuft_v = np.concatenate([d["V"] for d in sim.last_nb
                                 if d["idx"] in SM.CHILD_IDX])
        v_soma = sim.last_v_soma
        cai_span = float(cai.max() - cai.min())

        st = dict(
            up_frac=float((vmean > UP_THR).mean()),
            mid_frac=float(((vmean > MID_LO) & (vmean < MID_HI)).mean()),
            tuft_n_ev=_episodes(tuft_v.mean(axis=0), UP_THR),
            n_bap=int(len(baps)),
            nan=bool(np.isnan(S).any() or np.isnan(U).any()),
            absmax=float(max(np.abs(S).max(), np.abs(v_soma).max())),
        )
        gates = _gates(pol, st)

        rec = dict(
            i=i, wall_s=round(wall, 2),
            regime=pol.regime, policy=pol.to_dict(),
            V_max=float(V.max()), V_min=float(V.min()),
            V_span=float(V.max() - V.min()),
            cai_span=cai_span,
            cai_max=float(cai.max()),
            up_frac=st["up_frac"], mid_frac=st["mid_frac"],
            tuft_vmax=float(tuft_v.max()),
            tuft_n_ev=st["tuft_n_ev"],
            soma_spk=int(len(_peaks(v_soma, 0.0, int(1.0 / a.dt)))),
            n_bap=st["n_bap"],
            n_tuft_event=int(sum(len(x) for x in tuft_e)),
            plateau=bool(V.max() > PLATEAU_V and cai_span > PLATEAU_CAI),
            gates=gates, pass_=bool(all(gates.values())),
        )
        meta["trials"].append(rec)

        gs = "".join("1" if gates[k] else "0"
                     for k in ("G1", "G2", "G3", "G4", "G5"))
        prog.update(i - done + 1,
                    trial=f"{i}/{N}",
                    reg=pol.regime[:4],
                    Vmax=f"{rec['V_max']:.0f}",
                    up=f"{100*st['up_frac']:.1f}%",
                    wall=f"{wall:.1f}s",
                    gate=gs)

        if (i + 1) % 20 == 0:
            for v in starts.values():
                v.flush()

    for v in starts.values():
        v.flush()
    prog.close()
    np.save(os.path.join(outdir, "th.npy"), th_arr)

    # ---- 数据集级门禁 ----
    tr = meta["trials"]
    pl = np.array([r["plateau"] for r in tr], dtype=bool)
    ps = np.array([r["pass_"] for r in tr], dtype=bool)
    vmax = np.array([r["V_max"] for r in tr])
    vspan = np.array([r["V_span"] for r in tr])
    up = np.array([r["up_frac"] for r in tr])
    regs = [r["regime"] for r in tr]

    meta["gate"] = dict(
        plateau_frac=float(pl.mean()),
        pass_frac=float(ps.mean()),
        V_max_p50=float(np.median(vmax)),
        V_span_p50=float(np.median(vspan)),
        up_frac_p50=float(np.median(up)),
        wall_total_s=float(sum(r["wall_s"] for r in tr)),
        n_trial=len(tr),
        by_regime={r: dict(
            n=int(sum(1 for x in regs if x == r)),
            # ⚠️ 必须同时过滤 y，否则每个档都会报「全部 plateau」
            plateau=int(sum(1 for x, y in zip(regs, pl) if x == r and y)),
            pass_=int(sum(1 for x, y in zip(regs, ps) if x == r and y)),
        ) for r in ("linear", "inter", "plateau")},
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    g = meta["gate"]
    wall = g["wall_total_s"]
    print("\n" + "=" * 70)
    print(f"完成：{g['n_trial']} 条，总耗时 {wall/60:.1f} min "
          f"（{wall/max(g['n_trial'],1):.2f} s/条）")
    print(f"  plateau trial 比例  {g['plateau_frac']*100:.1f}%   (目标 15–45%)")
    print(f"  单 trial 门禁通过   {g['pass_frac']*100:.0f}%")
    print(f"  V_max 中位          {g['V_max_p50']:.1f} mV")
    print(f"  V_span 中位         {g['V_span_p50']:.1f} mV")
    print(f"  上行时间占比 中位   {g['up_frac_p50']*100:.2f}%")
    print("  分档：")
    for r, d in g["by_regime"].items():
        if d["n"]:
            print(f"    {r:<8} n={d['n']:>3}  plateau={d['plateau']:>3}  "
                  f"pass={d['pass_']:>3}")
    ok = 0.15 <= g["plateau_frac"] <= 0.45 and g["pass_frac"] >= 0.60
    print(f"  门禁: {'✅ 通过' if ok else '❌ 需调采样范围'}")
    print(f"⇒ {outdir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
