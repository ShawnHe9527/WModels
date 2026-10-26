"""探针：邻居段能否「自发」产生 Ca 尖峰？bAP 如何衰减到目标段？

## 两个问题（用户 2026-10-21）

1. `apic[52]` / `apic[53]`（目标段的两个下游子段）在**突触驱动**下能否
   自发产生 Ca 尖峰（即：不是我们画的波形，而是通道自己再生的）？
2. 胞体放电（bAP）沿胞体 → 顶树突 → `apic[50]` → `apic[51]` 衰减多少？
   在目标段还剩多少 mV？

## 三个实验

| 实验 | 驱动 | 判据 |
|---|---|---|
| `syn`  | 远端 tuft 同步 Poisson 突触，rate 30–400 Hz | V 峰 > −40 mV 且 V>−45 的持续时间 > 5 ms ⇒ Ca 平台 |
| `regen`| `apic[52]` 远端 2 ms 短脉冲，幅度 0.02–1.0 nA 扫描 | 全或无跃变（幅度翻倍但响应饱和）⇒ 再生性 |
| `bap`  | 胞体 IClamp 1.0 nA × 300 ms | 各 apic 段的 V 峰 → 衰减曲线 |

⚠️ 关键：「自发」必须与「被动滤波」区分开。判据是**再生性**：
短脉冲引起的响应远长于脉冲本身，且对幅度呈全或无。

用法：
    cd WModel2 && PYTHONPATH=$PWD python -m tools.probe_neighbor_caspike \
        --exp syn regen bap
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

EVENTS: list = []
KEEP: list = []
_DLL = False
_FIH = False


def add_all_events():
    for nc, ts in EVENTS:
        for t_ in ts:
            nc.event(float(t_))


def _poisson(rng, rate, t0, t1):
    n = rng.poisson(rate * (t1 - t0) / 1000.0)
    return np.sort(rng.uniform(t0, t1, n))


def _peaks(v, thr, min_gap):
    idx = np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > thr))[0] + 1
    out: list[int] = []
    for i in idx:
        if not out or i - out[-1] >= min_gap:
            out.append(int(i))
    return out


def _episodes(v, thr):
    """返回 V > thr 的连续段 [(起点, 终点)]，用于识别「去极化事件」。"""
    m = (v > thr).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


# ==============================================================================
class Probe:
    def __init__(self, dt=0.025, t_settle=100.0, t_test=400.0):
        global _DLL, _FIH
        from neuron import h
        self.h = h
        if not _DLL:
            h.nrn_load_dll(MODDIR)
            _DLL = True
        cwd = os.getcwd()
        os.chdir(HOCODIR)
        for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc",
                  "L5PCtemplate_2.hoc"):
            h.load_file(f)
        os.chdir(cwd)
        h.celsius = 34.0

        self.dt = dt
        self.t_settle = t_settle
        self.t_test = t_test
        self.cell = h.L5PCtemplate(MORPH)
        KEEP.append(self.cell)

        self.sec = {}
        for idx in (50, 51, 52, 53):
            self.sec[idx] = self.cell.apic[idx]

        # ---- 几何 ----
        self.geo = {}
        for idx, s in self.sec.items():
            self.geo[idx] = dict(L=float(s.L), diam=float(s.diam),
                                 nseg=int(s.nseg), x0=float(s.x3d(0)),
                                 y0=float(s.y3d(0)), z0=float(s.z3d(0)))

        # ---- 记录：所有 apic 段中点 V ----
        apic = self.cell.apic
        try:
            self.napic = len(apic)
        except TypeError:
            self.napic = 0
            while True:
                try:
                    apic[self.napic]
                    self.napic += 1
                except Exception:
                    break
        self.apic_segs = [apic[i](0.5) for i in range(self.napic)]
        KEEP.extend(self.apic_segs)
        self.rec_v_apic = [h.Vector().record(sg._ref_v) for sg in self.apic_segs]

        # ---- 记录：tuft + 目标段的 V、cai，逐段 ----
        self.tuft_idx = (52, 53)
        self.detail = {}
        for idx in (50, 51, 52, 53):
            n = int(self.sec[idx].nseg)
            segs = [self.sec[idx]((i + 0.5) / n) for i in range(n)]
            KEEP.extend(segs)
            self.detail[idx] = dict(
                n=n, segs=segs,
                V=[h.Vector().record(s._ref_v) for s in segs],
                cai=[h.Vector().record(s._ref_cai) for s in segs],
            )

        # ---- soma ----
        # ⚠️ cell.soma 是 SectionList（len=1），不是 Section ⇒ 必须 [0]
        self.soma = self.cell.soma[0]
        self.soma_seg = self.soma(0.5)
        KEEP.append(self.soma_seg)
        self.rec_v_soma = h.Vector().record(self.soma_seg._ref_v)
        self.rec_t = h.Vector().record(h._ref_t)

        # ---- tuft 突触：每个 segment 4 个 ProbAMPANMDA2（同步）----
        # 记录 (section_idx, seg_idx, NetCon) 以便按簇选取
        self.tuft_syn = []          # [(idx, i, NetCon), ...]
        for idx in self.tuft_idx:
            n = int(self.sec[idx].nseg)
            for i in range(n):
                seg = self.sec[idx]((i + 0.5) / n)
                KEEP.append(seg)
                for _ in range(4):
                    e = h.ProbAMPANMDA2(seg)
                    e.tau_r_AMPA, e.tau_d_AMPA = 0.3, 3.0
                    e.tau_r_NMDA, e.tau_d_NMDA = 2.0, 70.0
                    e.gmax, e.e = 0.0004, 0.0
                    nc = h.NetCon(None, e)
                    nc.delay, nc.weight[0] = 0.0, 1.0
                    KEEP.extend([e, nc])
                    self.tuft_syn.append((idx, i, nc))

        # ---- 远端短脉冲 IClamp（apic[52] 远端 x=1.0）----
        tip = self.sec[52](1.0)
        KEEP.append(tip)
        self.ic_tip = h.IClamp(tip)
        KEEP.append(self.ic_tip)
        # ---- 胞体 IClamp ----
        self.ic_soma = h.IClamp(self.soma_seg)
        KEEP.append(self.ic_soma)
        # ---- 胞体脉冲 IClamp 池（⚠️ Vector.play 在 IClamp.amp 上不生效，改用脉冲池）----
        self.MAXP = 32
        self.soma_pulses = [h.IClamp(self.soma_seg) for _ in range(self.MAXP)]
        KEEP.extend(self.soma_pulses)
        for icp in self.soma_pulses:
            icp.delay, icp.dur, icp.amp = 1e9, 0.1, 0.0

        # ---- FInitializeHandler（事件必须在 finitialize 内投递）----
        if not _FIH:
            h('objref fih_gen')
            h.fih_gen = h.FInitializeHandler('nrnpython("add_all_events()")')
            KEEP.append(h.fih_gen)
            import __main__
            __main__.add_all_events = add_all_events
            _FIH = True

    # ------------------------------------------------------------------
    def set_soma_pulses(self, times, amp=2.0, width=2.0):
        """在指定时刻给胞体 2 ms 短脉冲（诱发 bAP）。超出脉冲池则截断。"""
        times = list(times)[:self.MAXP]
        for j, icp in enumerate(self.soma_pulses):
            if j < len(times):
                icp.delay, icp.dur, icp.amp = float(times[j]), width, amp
            else:
                icp.delay, icp.dur, icp.amp = 1e9, 0.1, 0.0
        self._n_pulse = len(times)

    # ------------------------------------------------------------------
    def run(self, ex_events=None, tip_amp=0.0, soma_amp=0.0, v_init=-76.0):
        h = self.h
        EVENTS.clear()
        if ex_events is not None:
            for nc, ts in ex_events:
                EVENTS.append((nc, ts))
        else:
            for _, _, nc in self.tuft_syn:
                EVENTS.append((nc, np.empty(0)))

        # 远端短脉冲
        if tip_amp > 0:
            self.ic_tip.delay = self.t_settle + 20.0
            self.ic_tip.dur = 2.0
            self.ic_tip.amp = tip_amp
        else:
            self.ic_tip.amp = 0.0
        # 胞体持续注入
        if soma_amp > 0:
            self.ic_soma.delay = self.t_settle
            self.ic_soma.dur = self.t_test
            self.ic_soma.amp = soma_amp
        else:
            self.ic_soma.amp = 0.0

        h.v_init = v_init
        h.tstop = self.t_settle + self.t_test
        h.finitialize(v_init)
        h.dt = self.dt
        h.continuerun(h.tstop)

        t = np.asarray(self.rec_t)
        i0 = int(round(self.t_settle / self.dt))
        i1 = int(round((self.t_settle + self.t_test) / self.dt))
        out = dict(t=t[i0:i1], v_soma=np.asarray(self.rec_v_soma)[i0:i1],
                   v_apic=np.stack([np.asarray(v)[i0:i1]
                                    for v in self.rec_v_apic]))
        det = {}
        for idx, d in self.detail.items():
            det[idx] = dict(V=np.stack([np.asarray(v)[i0:i1] for v in d["V"]]),
                            cai=np.stack([np.asarray(v)[i0:i1] for v in d["cai"]]))
        out["detail"] = det
        return out

    # ------------------------------------------------------------------
    def tuft_summary(self, out):
        det = out["detail"]
        V = np.concatenate([det[i]["V"] for i in self.tuft_idx])
        cai = np.concatenate([det[i]["cai"] for i in self.tuft_idx])
        thr = -45.0
        T = V.shape[1] * self.dt
        # 全段整体上行占比
        n_above = int((V > thr).sum())
        frac_above = n_above / V.size
        # 事件：整段平均 V 的连续上行段
        vmean = V.mean(axis=0)
        epi = _episodes(vmean, thr)
        durs = [(e - s) * self.dt for s, e in epi]
        npk = sum(len(_peaks(v, -40.0, int(5.0 / self.dt))) for v in V)
        return dict(vmin=float(V.min()), vmax=float(V.max()),
                    span=float(V.max() - V.min()),
                    cai_max=float(cai.max()),
                    cai_base=float(np.median(cai[:, :200])),
                    frac_above=float(frac_above),
                    n_epi=len(epi),
                    epi_dur=float(np.median(durs)) if durs else 0.0,
                    T=T, n_peaks=npk)


# ==============================================================================
def exp_syn(p: Probe, rates, seed=7):
    rng = np.random.default_rng(seed)
    t0, t1 = p.t_settle, p.t_settle + p.t_test
    print("\n=== 实验 1：远端 tuft 均匀 Poisson 突触驱动 ===")
    print(f"{'rate Hz':>8} {'Vmax':>8} {'span':>7} {'cai_max':>8} "
          f"{'>−45占比':>10} {'事件数':>7} {'事件时长ms':>11}")
    rows = []
    for r in rates:
        ev = [(nc, _poisson(rng, r, t0, t1)) for _, _, nc in p.tuft_syn]
        out = p.run(ex_events=ev)
        s = p.tuft_summary(out)
        rows.append((r, s))
        print(f"{r:>8.0f} {s['vmax']:>8.1f} {s['span']:>7.1f} "
              f"{s['cai_max']:>8.4f} {s['frac_above']*100:>9.0f}% "
              f"{s['n_epi']:>7d} {s['epi_dur']:>11.1f}")
    print("\n  基线 cai ≈ %.5f（静息）" % rows[0][1]["cai_base"])
    return rows


def exp_bundle(p: Probe, sizes, rate=200.0, width=5.0, bundle_hz=2.0, seed=11):
    """突触**簇**：每次 k 个突触在 width ms 内齐射。

    这是生理的 tuft 输入形态（皮层传入是同步集合放电，不是均匀 Poisson）。
    k 扫描可测「全或无」：若 Vmax 对 k 饱和，则为再生性 Ca 尖峰。
    """
    rng = np.random.default_rng(seed)
    t0, t1 = p.t_settle, p.t_settle + p.t_test
    n_b = int(round(bundle_hz * (t1 - t0) / 1000.0))
    btimes = np.sort(rng.uniform(t0 + 20, t1 - 20, n_b))
    print(f"\n=== 实验 1b：突触簇驱动（每簇 {width:.0f} ms，{bundle_hz:.0f} Hz ⇒ {n_b} 簇）")
    print(f"{'k 突触':>7} {'总电导µS':>9} {'Vmax':>8} {'cai_max':>8} "
          f"{'上行时长ms':>10} {'上行占比':>9} {'事件数':>7} "
          f"{'目标ΔV':>9} {'目标cai':>8}")
    rows = []
    nsyn = len(p.tuft_syn)
    for k in sizes:
        ev = []
        for (_, _, nc) in p.tuft_syn:
            ev.append((nc, np.empty(0)))
        # 每簇随机选 k 个突触齐射
        sel_times = [[] for _ in range(nsyn)]
        for tb in btimes:
            sel = rng.choice(nsyn, size=min(k, nsyn), replace=False)
            for j in sel:
                sel_times[j].append(float(rng.uniform(tb, tb + width)))
        ev = [(p.tuft_syn[j][2], np.sort(np.array(sel_times[j])))
              for j in range(nsyn)]
        out = p.run(ex_events=ev)
        s = p.tuft_summary(out)
        # 目标段（apic[51]）看到的响应
        v51 = out["detail"][51]["V"]
        c51 = out["detail"][51]["cai"]
        base = float(np.median(v51[:, :200]))
        s["tgt_vmax"] = float(v51.max())
        s["tgt_dv"] = float(v51.max() - base)
        s["tgt_cai"] = float(c51.max())
        rows.append((k, s))
        print(f"{k:>7d} {k*0.0004:>9.4f} {s['vmax']:>8.1f} "
              f"{s['cai_max']:>8.4f} {s['epi_dur']:>10.1f} "
              f"{s['frac_above']*100:>8.0f}% {s['n_epi']:>7d} "
              f"{s['tgt_dv']:>9.1f} {s['tgt_cai']:>8.4f}")
    print("\n  判据：k 增大但 Vmax 饱和 ⇒ 再生性（全或无）；Vmax 随 k 线性增长 ⇒ 被动求和。")
    return rows


def exp_regen(p: Probe, amps):
    print("\n=== 实验 2：远端 2 ms 短脉冲幅度扫描 → 是否再生（全或无）===")
    print(f"{'amp nA':>8} {'Vmax':>8} {'峰后持续>−50ms':>15} {'cai_max':>8}")
    rows = []
    for a in amps:
        out = p.run(tip_amp=a)
        det = out["detail"]
        v52 = det[52]["V"].max(axis=0)
        cai = det[52]["cai"].max()
        # 脉冲后仍维持 > −50 mV 的时长
        post = v52[(int(20.0 / p.dt)):]
        dur = float((post > -50).sum() * p.dt)
        rows.append((a, float(v52.max()), dur, float(cai)))
        print(f"{a:>8.3f} {v52.max():>8.1f} {dur:>15.2f} {cai:>8.4f}")
    print("\n  判据：若 dur 对 amp 呈全或无跃变（而非随 amp 线性增长），则为再生性 Ca 尖峰。")
    return rows


def exp_bap(p: Probe, soma_amp=1.0):
    print("\n=== 实验 3：胞体放电（bAP）沿顶树突的衰减 ===")
    out = p.run(soma_amp=soma_amp)
    v_soma = out["v_soma"]
    v_apic = out["v_apic"]
    t = out["t"]
    # 胞体峰数
    spk = _peaks(v_soma, 0.0, int(1.0 / p.dt))
    print(f"  胞体脉冲数 = {len(spk)}；胞体 Vpeak = {v_soma.max():.1f} mV")
    base = np.median(v_soma[:200])
    print(f"  胞体静息 = {base:.2f} mV")
    print(f"\n{'apic idx':>9} {'L µm':>8} {'静息':>7} {'Vmax':>7} "
          f"{'ΔV峰':>7} {'衰减%':>7}")
    rows = []
    soma_dv = float(v_soma.max() - base)
    for i in range(p.napic):
        v = v_apic[i]
        b = float(np.median(v[:200]))
        dvm = float(v.max() - b)
        frac = 100.0 * dvm / max(soma_dv, 1e-9)
        rows.append((i, p.geo.get(i, {}).get("L", np.nan), b,
                     float(v.max()), dvm, frac))
    # 只打印关键段 + 每 5 段
    for (i, L, b, vm, dvm, frac) in rows:
        if i % 5 == 0 or i in (50, 51, 52, 53):
            Ls = f"{L:.0f}" if L == L else "-"
            print(f"{i:>9d} {Ls:>8} {b:>7.2f} {vm:>7.2f} "
                  f"{dvm:>7.2f} {frac:>6.1f}%")
    # 目标段细节
    det = out["detail"]
    v51 = det[51]["V"]
    print(f"\n  ★ apical_51（目标段）: ΔV 峰 = {v51.max(axis=0).max() - np.median(v51[:, :200]):.2f} mV"
          f"  (段内 {v51.shape[0]} 个 compartment，最大值−最小值 = "
          f"{float(v51.max() - v51.min()):.2f} mV)")
    v50 = det[50]["V"]
    print(f"    apic[50]（近端邻居）: ΔV 峰 = "
          f"{v50.max(axis=0).max() - np.median(v50[:, :200]):.2f} mV")
    return rows


def exp_bap_rates(p: Probe, amps):
    """胞体持续注入不同电流 ⇒ 不同放电率 ⇒ 目标段每击的响应。"""
    print("\n=== 实验 3b：胞体放电率 vs 目标段响应 ===")
    print(f"{'I nA':>7} {'胞体Hz':>8} {'胞体峰值':>9} {'目标ΔV峰':>10} "
          f"{'目标cai_max':>11} {'每次衰减ms':>11} {'目标Vmax':>9}")
    rows = []
    for a in amps:
        out = p.run(soma_amp=a)
        v_soma = out["v_soma"]
        v51 = out["detail"][51]["V"]
        c51 = out["detail"][51]["cai"]
        spk = _peaks(v_soma, 0.0, int(1.5 / p.dt))
        rate = len(spk) / (p.t_test / 1000.0)
        base = float(np.median(v51[:, :200]))
        # 每击在目标段引起的 ΔV（取击后 30 ms 内的最大），以及衰减到 37% 的时间
        dv = []
        decay = []
        w = int(30.0 / p.dt)
        pre_w = int(10.0 / p.dt)
        for i in spk:
            lo = max(0, i - pre_w)
            seg = v51[:, i:min(i + w, v51.shape[1])].mean(axis=0)
            if len(seg) < 5 or i < pre_w:
                continue
            # ⚠️ 不能拿「刺激前 5 ms」当基线：持续注入会把整条树突抬高
            pre = float(np.median(v51[:, lo:i].mean(axis=0)))
            pk = float(seg.max())
            dv.append(pk - pre)
            fin = np.where(seg < pre + 0.37 * (pk - pre))[0]
            decay.append(float(fin[0] * p.dt) if len(fin) else float("nan"))
        rows.append((a, rate, dv, decay))
        mdv = float(np.median(dv)) if dv else float("nan")
        mdec = float(np.median(decay)) if decay else float("nan")
        print(f"{a:>7.2f} {rate:>8.1f} {v_soma.max():>9.1f} {mdv:>10.2f} "
              f"{c51.max():>11.4f} {mdec:>11.1f} {v51.max():>9.1f}")
    print("\n  注：「目标ΔV峰」为单次 bAP 在 apical_51 上引起的中位峰值去极化；"
          "「每次衰减ms」为回落到 base+2 mV 的时间。")
    return rows


def exp_interact(p: Probe, sizes, seeds=2, n_bundle=6, width=5.0, bap_amp=4.0,
                 bap_lead=3.0, seed=23):
    """bAP 与 tuft 输入能否**协同**跨过 Ca 阈值？

    对每个 k，比较：无 bAP / tuft 簇前 3 ms 给一个 2 ms 胞体脉冲。
    若带 bAP 时临界 k 下降 ⇒ 目标段是**符合检测器**，不是线性求和器。
    """
    rng0 = np.random.default_rng(seed)
    nsyn = len(p.tuft_syn)
    print(f"\n=== 实验 1c：bAP × tuft 簇协同（{n_bundle} 簇 × {seeds} seed）===")
    print(f"{'k':>4} {'bAP':>6} {'平台概率':>9} {'tuft_Vmax':>10} "
          f"{'上行时长ms':>10} {'目标ΔV':>9} {'目标cai':>8}")
    rows = []
    for k in sizes:
        for use_bap in (False, True):
            prob, vmax, dur, tdv, tcai = [], [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(seed + 100 * s + k)
                t0 = p.t_settle + 50
                t1 = p.t_settle + p.t_test - 50
                bt = np.sort(rng.uniform(t0, t1, n_bundle))
                sel_times = [[] for _ in range(nsyn)]
                for tb in bt:
                    sel = rng.choice(nsyn, size=min(k, nsyn), replace=False)
                    for j in sel:
                        sel_times[j].append(float(rng.uniform(tb, tb + width)))
                ev = [(p.tuft_syn[j][2], np.sort(np.array(sel_times[j])))
                      for j in range(nsyn)]
                if use_bap:
                    p.set_soma_pulses(bt - bap_lead, amp=bap_amp, width=2.0)
                else:
                    p.set_soma_pulses([], amp=bap_amp, width=2.0)
                out = p.run(ex_events=ev)
                sm = p.tuft_summary(out)
                v51 = out["detail"][51]["V"]
                c51 = out["detail"][51]["cai"]
                base = float(np.median(v51[:, :200]))
                prob.append(1.0 if sm["vmax"] > 0.0 else 0.0)
                vmax.append(sm["vmax"])
                dur.append(sm["epi_dur"])
                tdv.append(float(v51.max() - base))
                tcai.append(float(c51.max()))
            rows.append((k, use_bap, np.mean(prob), np.mean(vmax),
                         np.mean(dur), np.mean(tdv), np.mean(tcai)))
            print(f"{k:>4d} {('有' if use_bap else '无'):>6} "
                  f"{np.mean(prob)*100:>8.0f}% {np.mean(vmax):>10.1f} "
                  f"{np.mean(dur):>10.1f} {np.mean(tdv):>9.1f} "
                  f"{np.mean(tcai):>8.4f}")
    print("\n  判据：带 bAP 时同一 k 的平台概率上升 ⇒ 协同（符合检测）。")
    return rows


# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["syn", "regen", "bap"],
                    choices=["syn", "bundle", "regen", "bap", "baprate",
                             "interact"])
    ap.add_argument("--t-test", type=float, default=400.0)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[2, 5, 10, 20, 30, 60, 100])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[2, 4, 8, 12, 16, 24, 32, 40, 56])
    ap.add_argument("--bundle-width", type=float, default=5.0)
    ap.add_argument("--bundle-hz", type=float, default=2.0)
    ap.add_argument("--amps", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00])
    ap.add_argument("--soma-amp", type=float, default=1.0)
    ap.add_argument("--soma-amps", type=float, nargs="+",
                    default=[0.3, 0.45, 0.6, 0.8, 1.0, 1.3])
    ap.add_argument("--inter-sizes", type=int, nargs="+",
                    default=[8, 12, 16, 20, 24])
    ap.add_argument("--inter-seeds", type=int, default=2)
    ap.add_argument("--n-bundle", type=int, default=6)
    ap.add_argument("--bap-amp", type=float, default=4.0,
                    help="胞体脉冲幅度 nA（3–5 nA × 2 ms ⇒ 单发 bAP）")
    ap.add_argument("--bap-lead", type=float, default=3.0)
    a = ap.parse_args()

    p = Probe(t_test=a.t_test)
    print("=== 几何 ===")
    for idx, g in p.geo.items():
        print(f"  apic[{idx}]: L={g['L']:.1f} µm  diam={g['diam']:.2f} µm  "
              f"nseg={g['nseg']}  位置=({g['x0']:.0f},{g['y0']:.0f},{g['z0']:.0f})")
    print(f"  总 apic 段数 = {p.napic}；tuft 突触数 = {len(p.tuft_syn)} "
          f"({len(p.tuft_syn)*0.0004:.4f} µS 总量)")

    if "syn" in a.exp:
        exp_syn(p, a.rates)
    if "bundle" in a.exp:
        exp_bundle(p, a.sizes, width=a.bundle_width, bundle_hz=a.bundle_hz)
    if "regen" in a.exp:
        exp_regen(p, a.amps)
    if "bap" in a.exp:
        exp_bap(p, a.soma_amp)
    if "baprate" in a.exp:
        exp_bap_rates(p, a.soma_amps)
    if "interact" in a.exp:
        exp_interact(p, a.inter_sizes, seeds=a.inter_seeds,
                     n_bundle=a.n_bundle, bap_amp=a.bap_amp,
                     bap_lead=a.bap_lead)


if __name__ == "__main__":
    main()
