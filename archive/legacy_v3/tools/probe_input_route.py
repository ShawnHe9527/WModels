"""探针：向邻居段注入电流 vs 邻居段自身有突触 —— 对目标段的效果有何不同。

## 要回答的问题（用户 2026-10-21 提出）

> "我从上下游邻居段的临近目标段的 segment 注入电流让它传入目标段，不行吗？"

## 三种条件，同一个目标段 apical_51

| 条件 | 做法 | 输入性质 |
|---|---|---|
| **A 目标段自身有突触**（现状）| 22 个突触只挂 apical_51 | 边界电位是"自造"的 |
| **B 邻居段加突触** | apic50/52/53 上也挂突触 | 物理最真实，但输入维数暴涨 |
| **C 邻居段注电流**（用户提议）| IClamp 打在邻居段贴近 junction 的位置 | 我们**完全控制**的电流 |

## 关键判据

1. **`corr(V_边界, V_目标)` 是否下降**
   现状 0.95–0.98（说明边界≈自己的 V，输入是假的）。
   若 C 能把它降下来 ⇒ **输入变成真外生**。

2. **目标段 V 是否被真正驱动**
   看跨度、是否出现 Ca 尖峰。

3. **C 与 B 的目标段响应是否可比**
   若可比 ⇒ 可以用 C 替代 B（成本低得多）。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m tools.probe_input_route --cond C
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.core import safety

THREADS = safety.apply_thread_caps(2)

import numpy as np                    # noqa: E402

MODDIR = "/home/neuron/L5PC/WModel2/neuron/mods/x86_64/libnrnmech.so"
HOCODIR = "/home/neuron/L5PC/WModel2/neuron/hoc"
MORPH = "/home/neuron/L5PC/WModel2/neuron/morphologies/cell1.asc"

G_E = 0.0004
G_I = 0.0010
EVENTS: list = []
KEEP: list = []


def add_all_events():
    for nc, ts in EVENTS:
        for t_ in ts:
            nc.event(float(t_))


def spikes(rng, rate_hz, t0, t1):
    n = rng.poisson(rate_hz * (t1 - t0) / 1000)
    return np.sort(rng.uniform(t0, t1, n))


def build(cond, t0=100.0, t1=1100.0, rate=150.0, i_amp=0.15, seed=3):
    """返回 h, cell, 记录容器, 目标段截面。"""
    from neuron import h
    h.nrn_load_dll(MODDIR)
    cwd = os.getcwd()
    os.chdir(HOCODIR)
    for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc",
              "L5PCtemplate_2.hoc"):
        h.load_file(f)
    os.chdir(cwd)
    h.celsius = 34.0
    h.dt = 0.025

    cell = h.L5PCtemplate(MORPH)
    KEEP.append(cell)
    tgt = cell.apic[51]
    par = cell.apic[50]
    ch1 = cell.apic[52]
    ch2 = cell.apic[53]
    nseg = int(tgt.nseg)
    rng = np.random.default_rng(seed)

    def add_syn(sec, i):
        n = int(sec.nseg)
        seg = sec((i + 0.5) / n)
        KEEP.append(seg)
        e = h.ProbAMPANMDA2(seg)
        e.tau_r_AMPA, e.tau_d_AMPA = 0.3, 3.0
        e.tau_r_NMDA, e.tau_d_NMDA = 2.0, 70.0
        e.gmax, e.e = G_E, 0
        ne = h.NetCon(None, e)
        ne.delay, ne.weight[0] = 0.0, 1
        EVENTS.append((ne, spikes(rng, rate, t0, t1)))
        ii = h.ProbUDFsyn2(seg)
        ii.tau_r, ii.tau_d = 0.2, 8.0
        ii.gmax, ii.e = G_I, -80
        ni = h.NetCon(None, ii)
        ni.delay, ni.weight[0] = 0.0, 1
        EVENTS.append((ni, spikes(rng, rate * 0.65, t0, t1)))
        KEEP.extend([e, ne, ii, ni])

    # 目标段挂 22 个突触（条件 D 除外 —— 那是"目标段完全静默，只被邻居驱动"）
    if cond != "D":
        for i in range(nseg):
            add_syn(tgt, i)

    if cond == "B":
        for sec in (par, ch1, ch2):
            for i in range(int(sec.nseg)):
                add_syn(sec, i)

    if cond == "D":
        # 只注电流，目标段自己的 22 个突触全部关闭 ⇒ 测"输入是否真外生"
        for sec, x in ((par, 0.95), (ch1, 0.05), (ch2, 0.05)):
            seg = sec(x)
            KEEP.append(seg)
            ic = h.IClamp(seg)
            ic.delay, ic.dur, ic.amp = t0, t1 - t0, i_amp
            KEEP.append(ic)

    if cond == "C":
        # 在邻居段**贴近 junction 的一端**打电流钳
        # apic_51 的 x=0 接 par 的 x=1.0 ⇒ 打在 par(0.95)
        # apic_51 的 x=1 接 ch 的 x=0.0 ⇒ 打在 ch(0.05)
        for sec, x in ((par, 0.95), (ch1, 0.05), (ch2, 0.05)):
            seg = sec(x)
            KEEP.append(seg)
            ic = h.IClamp(seg)
            ic.delay, ic.dur, ic.amp = t0, t1 - t0, i_amp
            KEEP.append(ic)

    # 记录：目标段 11 个 compartment 的 V + 3 个边界电位
    tsegs = [tgt((i + 0.5) / nseg) for i in range(nseg)]
    KEEP.extend(tsegs)
    rec_v = [h.Vector().record(s._ref_v) for s in tsegs]
    b_par = par(1.0); b_c1 = ch1(0.0); b_c2 = ch2(0.0)
    KEEP.extend([b_par, b_c1, b_c2])
    rec_b = [h.Vector().record(b_par._ref_v),
             h.Vector().record(b_c1._ref_v),
             h.Vector().record(b_c2._ref_v)]
    rec_t = h.Vector().record(h._ref_t)
    return h, cell, rec_v, rec_b, rec_t, (t0, t1)


def run_case(cond, **kw):
    EVENTS.clear(); KEEP.clear()
    h, cell, rec_v, rec_b, rec_t, (t0, t1) = build(cond, **kw)
    h('objref fih_p')
    h.fih_p = h.FInitializeHandler('nrnpython("add_all_events()")')
    KEEP.append(h.fih_p)
    import __main__
    __main__.add_all_events = add_all_events

    h.v_init = -76.0
    h.tstop = t1
    h.finitialize(-76.0)
    h.dt = 0.025
    h.continuerun(t1)

    t = np.asarray(rec_t)
    m = t >= t0
    V = np.asarray([np.asarray(v)[m] for v in rec_v])       # (P,T)
    B = np.asarray([np.asarray(v)[m] for v in rec_b])       # (3,T)
    return V, B


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("A", "B", "C", "D"), default="C")
    ap.add_argument("--i-amp", type=float, default=0.15,
                    help="条件 C 的注入电流 (nA)")
    ap.add_argument("--rate", type=float, default=150.0)
    a = ap.parse_args(argv)

    V, B = run_case(a.cond, i_amp=a.i_amp, rate=a.rate)
    Vmid = V[V.shape[0] // 2]
    print(f"=== 条件 {a.cond}  (rate={a.rate:.0f} Hz, "
          f"I={a.i_amp:.3f} nA) ===")
    print(f"  目标段 V   : [{V.min():.1f}, {V.max():.1f}]  "
          f"跨度 {np.ptp(Vmid):.2f} mV")
    print(f"  是否过阈   : {'是' if V.max() > -50 else '否'}")
    for i, nm in enumerate(("apic50(1.0)", "apic52(0)", "apic53(0)")):
        c = float(np.corrcoef(B[i], Vmid)[0, 1])
        print(f"  {nm:<12}: 跨度 {np.ptp(B[i]):6.2f} mV  "
              f"corr(边界, 目标) = {c:+.3f}")
    # 关键指标：边界电位中有多少是"目标自己解释不了的"
    bc = np.mean([np.corrcoef(B[i], Vmid)[0, 1] for i in range(3)])
    print(f"\n  ★ 平均 corr(边界, 目标) = {bc:+.3f}")
    print(f"     （现状 A 实测 0.95–0.98 ⇒ 边界≈自己的 V ⇒ 输入是假的）")
    print(f"     1 - corr² = {1-bc**2:.3f} ⇒ 边界中"
          f" {(1-bc**2)*100:.1f}% 的方差是目标解释不了的")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
