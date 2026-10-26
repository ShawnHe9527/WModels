"""数据生成：批量跑轨迹并写 memmap。

## 数据集布局（`data/<name>/`）

| 文件 | 形状 | dtype | 大小(200条) |
|---|---|---|---|
| `s.dat` | `(N, T, 9, 11)` | float32 | 15.8 MB/条 ⇒ 3.2 GB |
| `u.dat` | `(N, T, 3, 11)` | float32 | 5.3 MB/条 ⇒ 1.1 GB |
| `ub.dat` | `(N, T, 3)` | float32 | 4.8 MB |
| `th.npy` | `(N, 27)` | float32 | 21 KB |
| `meta.json` | 全局配置 + per-trial 协议与统计 | — | — |
| `scaler.json` | 逐因子标准化尺度（**只用训练集算**）| — | — |

T = 1000 / 0.025 = 40000。

## 为什么用 memmap

单条轨迹 21 MB，200 条约 4.3 GB。全部放内存会超出可用 RAM（1.3–1.8 GB）。
用 `np.memmap` 直接写盘，逐条填充。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m world.gen.datagen \
        --name tr_s1 --n-trial 200 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..core import config as C
from ..core import safety
from ..core import progress

THREADS = safety.apply_thread_caps(4)

from . import protocol as P            # noqa: E402
from . import sim as SM                # noqa: E402

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data")


# ==============================================================================
def _open_memmap(path: str, shape: tuple, mode: str):
    return np.memmap(path, dtype=np.float32, mode=mode, shape=shape)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="world.gen.datagen")
    ap.add_argument("--name", type=str, required=True,
                    help="数据集名，如 tr_s1 / va_s2 / te_s3")
    ap.add_argument("--n-trial", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--t-settle", type=float, default=100.0)
    ap.add_argument("--t-test", type=float, default=1000.0)
    ap.add_argument("--mem-mb", type=float, default=1800.0)
    ap.add_argument("--resume", action="store_true",
                    help="已有前 k 条则跳过（续跑）")
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=a.mem_mb)

    outdir = os.path.join(DATA_ROOT, a.name)
    os.makedirs(outdir, exist_ok=True)

    T = int(round(a.t_test / a.dt))
    N = a.n_trial
    sim = SM.Sim(dt=a.dt, t_settle=a.t_settle, t_test=a.t_test)
    nseg = sim.nseg
    nf = SM.N_FACTORS

    shapes = {
        "s": (N, T, nf, nseg),
        "u": (N, T, 3, nseg),
        "ub": (N, T, 3),
    }
    print(f"数据集 {a.name}: N={N}  T={T}  nseg={nseg}  nfactor={nf}")
    for k, sh in shapes.items():
        mb = np.prod(sh) * 4 / 1e6
        print(f"  {k}.dat  {sh}  ≈ {mb:.1f} MB")

    starts = {k: _open_memmap(os.path.join(outdir, f"{k}.dat"), sh, "w+")
              for k, sh in shapes.items()}
    th_arr = np.zeros((N, 27), dtype=np.float32)

    th = sim.theta()
    print(f"θ_s (27 维) 前几个: {np.round(th[:6], 4)}")

    meta = dict(
        name=a.name, n_trial=N, T=T, dt=a.dt, t_settle=a.t_settle,
        t_test=a.t_test, nseg=nseg, factor_names=SM.FACTOR_NAMES,
        u_channels=["AMPA", "NMDA_ungated", "GABA"],
        boundary=["apic[50](1.0)", "apic[52](0.0)", "apic[53](0.0)"],
        section=SM.SECTION, seed=a.seed, gmax_e=0.0004, gmax_i=0.001,
        trials=[],
    )

    rng = np.random.default_rng(a.seed)
    prog = progress.Progress(N, desc=f"生成 {a.name}")

    for i in range(N):
        pol = P.sample_policy(rng)
        ex, inh, info = P.make_spike_trains(
            rng, nseg, a.t_settle, a.t_settle + a.t_test, pol)

        t0 = time.time()
        S, U, UB = sim.run(ex, inh)
        dt_wall = time.time() - t0

        starts["s"][i] = S
        starts["u"][i] = U
        starts["ub"][i] = UB
        th_arr[i] = th

        Vs = S[:, 8, :]          # (T, nseg) 膜电位
        cai = S[:, 7, :]
        rec = dict(
            i=i, wall_s=round(dt_wall, 2),
            regime=getattr(pol, "regime", ""),
            policy=pol.to_dict(),
            **{k: (float(v) if isinstance(v, (int, float, np.floating))
                   else v) for k, v in info.items()},
            V_max=float(Vs.max()), V_min=float(Vs.min()),
            V_span=float(Vs.max() - Vs.min()),
            cai_span=float(cai.max() - cai.min()),
            # 平台/尖峰：必须同时满足「去了很高」且「钙真的累积」
            # ⚠️ 只卡 V_max>-50 会把「强去极化但未过阈」也算进来（实测
            #    rate=60,ρ=0.6 时 Vmax=−46.8 但 cai 跨度仅 0.092）
            plateau=bool(Vs.max() > -30.0 and (cai.max() - cai.min()) > 0.5),
        )
        meta["trials"].append(rec)
        prog.update(i + 1, wall=f"{dt_wall:.1f}s",
                    Vmax=f"{rec['V_max']:.1f}")

        # 定期 flush，避免断电丢全部
        if (i + 1) % 20 == 0:
            for v in starts.values():
                v.flush()

    for v in starts.values():
        v.flush()
    np.save(os.path.join(outdir, "th.npy"), th_arr)

    # ---- 数据门禁 ----
    pl = [r["plateau"] for r in meta["trials"]]
    vmax = np.array([r["V_max"] for r in meta["trials"]])
    vspan = np.array([r["V_span"] for r in meta["trials"]])
    meta["gate"] = dict(
        plateau_frac=float(np.mean(pl)),
        V_max_p50=float(np.median(vmax)),
        V_span_p50=float(np.median(vspan)),
        wall_total_s=float(sum(r["wall_s"] for r in meta["trials"])),
    )
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    g = meta["gate"]
    print("\n" + "=" * 70)
    print(f"完成：{N} 条，总耗时 {g['wall_total_s']/60:.1f} min")
    print(f"  平台区占比   {g['plateau_frac']*100:.1f}%  (判据 15–45%)")
    print(f"  V_max 中位   {g['V_max_p50']:.1f} mV")
    print(f"  V_span 中位  {g['V_span_p50']:.1f} mV")
    ok = 0.15 <= g["plateau_frac"] <= 0.45
    print(f"  门禁: {'✅ 通过' if ok else '❌ 平台区占比越界，需调参数范围'}")
    print(f"⇒ {outdir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
