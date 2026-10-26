"""探针：邻居段的驱动方式如何影响目标段收到的「信息」。

## 用户的核心关切（2026-10-21）

> "我希望注入的电流不是钳制的，钳制会使电流持续恒定地影响，
>  这会使得输入跟以前钳制一样被强行拉平。
>  我希望模仿上下两端由 Ca 尖峰、胞体放电而达到的电信号传入。"

## 三种驱动方式，**平均电流相同**

| 条件 | 形态 | 模仿什么 |
|---|---|---|
| `dc` | 恒定电流 | （对照）钳制式，无时间结构 |
| `pulse` | 1.5 ms 窄脉冲，20–80 Hz 不规则 | 胞体 AP 回传（bAP）|
| `burst` | 每次 5 个脉冲 @ 100 Hz，间隔 1.5 ms | Ca 尖峰 / 簇状放电 |
| `syn` | Exp2Syn + Poisson 事件（τd=20 ms）| 真实突触输入 |

## 判据：目标段收到多少「新信息」

只比较 V 的跨度是不够的（DC 也能推高）。
关键指标是**目标段的响应能否由它自己的过去预测**：

1. **线性自回归残差**：用目标段自己的 V(t-1..t-5) 预测 V(t)，
   剩下的残差占比 = 外部输入带来的「新信息」。
   DC 驱动下这个残差应该很小（因为只是把整条曲线平移）。
2. **响应的事件数**：V 上有多少个「峰」（说明输入有时间结构）。
3. **一阶差分的重尾性**：脉冲式驱动会让 ΔV 出现尖峰（突触/尖峰特征），
   DC 驱动的 ΔV 非常平滑。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m tools.probe_drive_modes --mode burst
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


def _poisson(rng, rate, t0, t1):
    n = rng.poisson(rate * (t1 - t0) / 1000.0)
    return np.sort(rng.uniform(t0, t1, n))


def build(mode, t0=100.0, t1=1100.0, i_mean=0.05, rate=30.0, seed=5):
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
    dur = t1 - t0

    # ---- 目标段：22 个突触（与其他实验一致，保持不变）----
    for i in range(nseg):
        seg = tgt((i + 0.5) / nseg)
        KEEP.append(seg)
        e = h.ProbAMPANMDA2(seg)
        e.tau_r_AMPA, e.tau_d_AMPA = 0.3, 3.0
        e.tau_r_NMDA, e.tau_d_NMDA = 2.0, 70.0
        e.gmax, e.e = G_E, 0
        ne = h.NetCon(None, e)
        ne.delay, ne.weight[0] = 0.0, 1
        EVENTS.append((ne, _poisson(rng, 120.0, t0, t1)))
        ii = h.ProbUDFsyn2(seg)
        ii.tau_r, ii.tau_d = 0.2, 8.0
        ii.gmax, ii.e = G_I, -80
        ni = h.NetCon(None, ii)
        ni.delay, ni.weight[0] = 0.0, 1
        EVENTS.append((ni, _poisson(rng, 80.0, t0, t1)))
        KEEP.extend([e, ne, ii, ni])

    # ---- 邻居段：三种驱动方式，平均电流相同 ----
    # 注入点在贴近 junction 的一端
    sites = ((par, 0.95), (ch1, 0.05), (ch2, 0.05))
    n_pulse = 3 * int(round(rate * dur / 1000.0))

    if mode == "dc":
        for sec, x in sites:
            s = sec(x); KEEP.append(s)
            ic = h.IClamp(s)
            ic.delay, ic.dur, ic.amp = t0, dur, i_mean
            KEEP.append(ic)

    elif mode in ("pulse", "burst"):
        for sec, x in sites:
            s = sec(x); KEEP.append(s)
            ic = h.IClamp(s)
            KEEP.append(ic)
            if mode == "pulse":
                # 单脉冲，宽度 1.5 ms，高度 = 平均电流 / 占空比
                w = 1.5
                amp = i_mean * dur / (n_pulse / 3.0 * w) / 3.0
                ts = _poisson(rng, rate, t0, t1 - w)
                for tt in ts:
                    ic.delay, ic.dur, ic.amp = float(tt), w, amp
                    # ⚠️ IClamp 只能有一个时段；用多个 IClamp 实现脉冲串
                    ic2 = h.IClamp(s)
                    ic2.delay, ic2.dur, ic2.amp = float(tt), w, amp
                    KEEP.append(ic2)
                ic.amp = 0.0
            else:  # burst: 5 个 1.5 ms 脉冲 @ 100 Hz
                w, nb = 1.5, 5
                n_burst = max(1, int(round(rate * dur / 1000.0 / nb)))
                amp = i_mean * dur / (n_burst * nb * w)
                ts = _poisson(rng, rate / nb, t0, t1 - nb * w)
                for tt in ts:
                    for k in range(nb):
                        ic2 = h.IClamp(s)
                        ic2.delay = float(tt + k * 10.0)
                        ic2.dur, ic2.amp = w, amp
                        KEEP.append(ic2)
                ic.amp = 0.0

    elif mode == "syn":
        for sec, x in sites:
            s = sec(x); KEEP.append(s)
            syn = h.Exp2Syn(s)
            syn.tau1, syn.tau2, syn.e = 0.5, 20.0, 0.0
            # 平均电导 ≈ rate * gpeak * tau_d(ms) * 1e-3
            gpk = i_mean / 70.0 / (rate * 20.0 * 1e-3) / 3.0
            nc = h.NetCon(None, syn)
            nc.delay, nc.weight[0] = 0.0, gpk
            EVENTS.append((nc, _poisson(rng, rate, t0, t1)))
            KEEP.extend([syn, nc])

    # ---- 记录 ----
    tsegs = [tgt((i + 0.5) / nseg) for i in range(nseg)]
    KEEP.extend(tsegs)
    rec_v = [h.Vector().record(s._ref_v) for s in tsegs]
    b = []
    for sec, x in ((par, 1.0), (ch1, 0.0), (ch2, 0.0)):
        s = sec(x); KEEP.append(s)
        b.append(h.Vector().record(s._ref_v))
    rec_t = h.Vector().record(h._ref_t)
    return h, rec_v, b, rec_t, (t0, t1)


def analyze(V, B, t, t0):
    m = t >= t0
    V = V[:, m]
    B = B[:, m]
    Vm = V[V.shape[0] // 2]

    d = np.diff(Vm)
    # 「新信息」：用自身过去 20 步（0.5 ms）预测 **40 步后**（1 ms）的值。
    # ⚠️ 不能预测 t+1：膜时间常数 6–18 ms，相邻步几乎相同 ⇒ R² 恒为 1，无区分度。
    L, H = 20, 40
    n = len(Vm) - L - H
    X = np.stack([Vm[i:i + n] for i in range(L)], axis=1)
    y = Vm[L + H:L + H + n]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1.0 - resid.var() / max(y.var(), 1e-12)

    # 峰计数（说明时间结构）
    pk = ((Vm[1:-1] > Vm[:-2]) & (Vm[1:-1] > Vm[2:]) &
          (Vm[1:-1] > Vm.mean() + Vm.std()))
    return dict(
        vmin=float(V.min()), vmax=float(V.max()), vspan=float(np.ptp(Vm)),
        dv_max=float(np.abs(d).max()), dv_med=float(np.median(np.abs(d))),
        tail=float(np.percentile(np.abs(d), 99.9) /
                   max(np.median(np.abs(d)), 1e-12)),
        n_peak=int(pk.sum()), self_r2=float(r2),
        new_info=float(1.0 - r2),
        bc=float(np.mean([np.corrcoef(B[i], Vm)[0, 1] for i in range(3)])),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dc", "pulse", "burst", "syn"),
                    default="burst")
    ap.add_argument("--i-mean", type=float, default=0.05,
                    help="邻居段平均注入电流 (nA)，所有模式相同")
    ap.add_argument("--rate", type=float, default=30.0)
    a = ap.parse_args(argv)

    h, rv, rb, rt, (t0, _) = build(a.mode, i_mean=a.i_mean, rate=a.rate)
    h('objref fih_d')
    h.fih_d = h.FInitializeHandler('nrnpython("add_all_events()")')
    KEEP.append(h.fih_d)
    import __main__
    __main__.add_all_events = add_all_events
    h.v_init = -76.0
    h.tstop = 1100.0
    h.finitialize(-76.0)
    h.dt = 0.025
    h.continuerun(1100.0)

    V = np.asarray([np.asarray(v) for v in rv])
    B = np.asarray([np.asarray(v) for v in rb])
    t = np.asarray(rt)
    s = analyze(V, B, t, t0)
    print(f"=== mode={a.mode}  平均电流={a.i_mean:.3f} nA  "
          f"rate={a.rate:.0f} Hz ===")
    print(f"  目标段 V        : [{s['vmin']:.1f}, {s['vmax']:.1f}]  "
          f"跨度 {s['vspan']:.2f} mV")
    print(f"  ΔV 最大         : {s['dv_max']:.5f} mV/步")
    print(f"  ΔV 中位         : {s['dv_med']:.6f} mV/步")
    print(f"  ΔV 重尾比(p99.9/中位) : {s['tail']:.1f}")
    print(f"  峰数            : {s['n_peak']}")
    print(f"  ★ 自回归 R²(1ms)  : {s['self_r2']:.4f}  ⇒ "
          f"外部输入贡献 {s['new_info']*100:.1f}% 的不可预测方差")
    print(f"  corr(边界, 目标) : {s['bc']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
