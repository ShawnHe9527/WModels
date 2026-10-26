"""探针：整细胞仿真 vs 隔离段仿真 —— 计时 + 物理差异性。

回答一个具体问题：**做单段世界模型时，应该跑整细胞还是把段隔离出来？**

## 判据

1. **耗时**：整细胞 196 段 vs 单段 1 段
2. **边界电位的真实性**：隔离段的邻居波形必须人为给定；整细胞里是算出来的
3. **边界电位的信息量**：它跟 22 个突触电导有多冗余

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m tools.probe_full_vs_isolated \
        --mode full --tstop 1000
    cd WModel2 && PYTHONPATH=$PWD python -m tools.probe_full_vs_isolated \
        --mode iso  --tstop 1000
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

MODDIR = "/home/neuron/L5PC/WModel2/neuron/mods/x86_64/libnrnmech.so"
HOCODIR = "/home/neuron/L5PC/WModel2/neuron/hoc"
MORPH = "/home/neuron/L5PC/WModel2/neuron/morphologies/cell1.asc"

G_E = 0.0004      # µS，官方 gmax（兴奋）
G_I = 0.0010      # µS，官方 gmax（抑制）
RATE_E = 30.0     # spikes/s per segment
RATE_I = 20.0     # spikes/s per segment

# ⚠️ NEURON 的 FInitializeHandler 通过 nrnpython() 调这个函数，
#    必须是**模块级**函数 + 模块级列表（不能用闭包）
EVENTS: list = []


def add_all_events():
    """在 finitialize 之后投递所有突触事件（官方脚本同一做法）。"""
    for nc, ts in EVENTS:
        for t_ in ts:
            nc.event(float(t_))


def build(tstop: float, dt: float, mode: str, seed: int = 1):
    from neuron import h
    h.nrn_load_dll(MODDIR)
    cwd = os.getcwd()
    os.chdir(HOCODIR)
    for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc",
              "L5PCtemplate_2.hoc"):
        h.load_file(f)
    os.chdir(cwd)
    h.celsius = 34.0
    h.dt = dt

    cell = h.L5PCtemplate(MORPH)
    sec = cell.apic[51]
    par = cell.apic[50]
    ch1 = cell.apic[52]
    ch2 = cell.apic[53]

    if mode == "iso":
        # 隔离：切断父段连接（子段本来就挂在本段，也一并切断）
        # ⚠️ 必须在建 NetCon / 记录之前做
        for c in list(sec.children()):
            c.disconnect()
        sec.disconnect()

    nseg = int(sec.nseg)
    rng = np.random.default_rng(seed)
    keep = []
    syn_g = {"e": [], "i": []}
    events = []
    for i in range(nseg):
        x = (i + 0.5) / nseg
        seg = sec(x)

        e = h.ProbAMPANMDA2(seg)
        e.tau_r_AMPA, e.tau_d_AMPA = 0.3, 3.0
        e.tau_r_NMDA, e.tau_d_NMDA = 2.0, 70.0
        e.gmax, e.e = G_E, 0
        nc_e = h.NetCon(None, e)
        nc_e.delay, nc_e.weight[0] = 0.0, 1
        times_e = np.sort(rng.uniform(0, tstop,
                                      rng.poisson(RATE_E * tstop / 1000)))
        events.append((nc_e, times_e))

        ii = h.ProbUDFsyn2(seg)
        ii.tau_r, ii.tau_d = 0.2, 8.0
        ii.gmax, ii.e = G_I, -80
        nc_i = h.NetCon(None, ii)
        nc_i.delay, nc_i.weight[0] = 0.0, 1
        times_i = np.sort(rng.uniform(0, tstop,
                                      rng.poisson(RATE_I * tstop / 1000)))
        events.append((nc_i, times_i))

        syn_g["e"].append(h.Vector().record(e._ref_g_AMPA))
        syn_g["i"].append(h.Vector().record(ii._ref_g))
        keep += [e, nc_e, ii, nc_i]

    rec = dict(
        t=h.Vector().record(h._ref_t),
        V=h.Vector().record(sec(0.5)._ref_v),
        V_par=h.Vector().record(par(1.0)._ref_v),
        V_ch1=h.Vector().record(ch1(0.0)._ref_v),
        V_ch2=h.Vector().record(ch2(0.0)._ref_v),
        V_soma=h.Vector().record(cell.soma[0](0.5)._ref_v),
        g_ex=syn_g["e"], g_in=syn_g["i"],
    )
    return h, cell, rec, keep, events


def main(argv=None) -> int:
    global RATE_E, RATE_I
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("full", "iso"), default="full")
    ap.add_argument("--tstop", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--v-init", type=float, default=-76.0)
    ap.add_argument("--rate-e", type=float, default=RATE_E)
    ap.add_argument("--rate-i", type=float, default=RATE_I)
    a = ap.parse_args(argv)

    RATE_E, RATE_I = a.rate_e, a.rate_i
    h, cell, rec, keep, events = build(a.tstop, a.dt, a.mode, a.seed)
    nsec = sum(1 for _ in cell.all)
    ncomp = sum(int(s.nseg) for s in cell.all)

    global EVENTS
    EVENTS.clear()
    EVENTS.extend(events)
    h('objref fih')
    h.fih = h.FInitializeHandler('nrnpython("add_all_events()")')

    t0 = time.time()
    h.v_init = a.v_init          # ⚠️ 默认 -65，不设会被拉到 -65
    h.tstop = a.tstop
    h.run()
    el = time.time() - t0

    V = np.array(rec["V"])
    t = np.array(rec["t"])
    Vp, Vc1, Vc2 = (np.array(rec["V_par"]), np.array(rec["V_ch1"]),
                    np.array(rec["V_ch2"]))
    Vs = np.array(rec["V_soma"])
    nspk = int(((Vs[1:-1] > 0) & (Vs[1:-1] >= Vs[:-2]) &
                (Vs[1:-1] >= Vs[2:])).sum())

    print(f"=== mode={a.mode}  tstop={a.tstop}  dt={a.dt} ===")
    print(f"  section 数        : {nsec}")
    print(f"  compartment 数    : {ncomp}")
    print(f"  仿真耗时          : {el:.2f} s")
    print(f"  apic51(0.5) V     : [{V.min():.1f}, {V.max():.1f}]  "
          f"跨度 {V.max()-V.min():.1f} mV")
    print(f"  父段 V            : [{Vp.min():.1f}, {Vp.max():.1f}]")
    print(f"  子段52 V          : [{Vc1.min():.1f}, {Vc1.max():.1f}]")
    print(f"  子段53 V          : [{Vc2.min():.1f}, {Vc2.max():.1f}]")
    print(f"  soma V            : [{Vs.min():.1f}, {Vs.max():.1f}]  spike={nspk}")

    # 边界电位与突触电导的相关性（判断信息是否冗余）
    gE = float(np.mean([np.array(v) for v in rec["g_ex"]], axis=0).mean())
    print(f"  平均 g_AMPA       : {gE:.5f} µS")

    np.savez(f"/tmp/probe_{a.mode}.npz", t=t, V=V, V_par=Vp,
             V_ch1=Vc1, V_ch2=Vc2, V_soma=Vs,
             g_ex=np.array([np.array(v) for v in rec["g_ex"]]),
             g_in=np.array([np.array(v) for v in rec["g_in"]]))
    print(f"  ⇒ 轨迹已存 /tmp/probe_{a.mode}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
