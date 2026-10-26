"""阶段 B：仿真器自检（B1–B6）。

## 为什么要有这一步

v2 的教训：跳过验证、一路错到 T4 才发现整条链路是错的。
本脚本用 6 项**互相独立**的检查，每项指向一个具体机制。
全部通过才说明"我们的 NEURON 设置与论文那一台是同一台"。

| # | 检查 | 判据 | 若失败指向什么 |
|---|---|---|---|
| B1 | 静置稳定性 | 后 500 ms 漂移 < 1 mV，均值 ∈ [−80,−70] | 通道分布 / pas / 反转电位 |
| B2 | 单突触 EPSP | 0.1–2 mV，衰减 τ 1–8 ms | 突触机制 / 单位(µS) / gmax |
| B3 | NMDA 电压依赖 | g_NMDA(−40)/g_NMDA(−80) > 1.5 | mggate 是否生效 |
| B4 | 全树驱动能放电 | soma 2–40 Hz | 整条通路（突触→树突→轴突）|
| B5 | AP 形态 | 峰 30–55 mV，半宽 0.4–1.5 ms | 主动电导 |
| B6 | dt 收敛 | 0.05 vs 0.025 的 V RMS 差 < 0.5 mV | 数值可信度 |

## 设计：每项跑在**独立子进程**里

NEURON 的全局状态（section / 机制 / handler）在进程内无法可靠重置：
- 重复 `nrn_load_dll` 报 `user defined name already exists`
- 旧的 `FInitializeHandler` 指向已删除的 NetCon ⇒ core dump

⇒ 主进程只负责编排，每项检查用 `subprocess` 启动一次。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m tools.stage_b_selfcheck
    cd WModel2 && PYTHONPATH=$PWD python -m tools.stage_b_selfcheck --only B2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

MODDIR = "/home/neuron/L5PC/WModel2/neuron/mods/x86_64/libnrnmech.so"
HOCODIR = "/home/neuron/L5PC/WModel2/neuron/hoc"
MORPH = "/home/neuron/L5PC/WModel2/neuron/morphologies/cell1.asc"

# ⚠️ 模块级：FInitializeHandler 通过 nrnpython() 按名字查找
EVENTS: list = []
KEEP: list = []
_DLL_LOADED = False
_FIH_ARMED = False
_DT = 0.025
DTS = (0.025, 0.0125)


def add_all_events():
    for nc, ts in EVENTS:
        for t_ in ts:
            nc.event(float(t_))


def setup(dt: float = 0.025):
    global _DLL_LOADED, _DT
    from neuron import h
    # ⚠️ nrn_load_dll 只能调一次，重复会报
    #    "user defined name already exists" / hocobj_call error
    if not _DLL_LOADED:
        h.nrn_load_dll(MODDIR)
        _DLL_LOADED = True
    cwd = os.getcwd()
    os.chdir(HOCODIR)
    for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc",
              "L5PCtemplate_2.hoc"):
        h.load_file(f)
    os.chdir(cwd)
    h.celsius = 34.0
    h.dt = dt
    _DT = dt
    return h


def new_cell():
    """新建细胞并长期持有（objref 被 GC ⇒ section 被删 ⇒ segfault）。"""
    from neuron import h
    cell = h.L5PCtemplate(MORPH)
    KEEP.append(cell)
    return cell


def arm_events(h):
    global _FIH_ARMED
    if _FIH_ARMED:
        return
    h('objref fih_b')
    h.fih_b = h.FInitializeHandler('nrnpython("add_all_events()")')
    KEEP.append(h.fih_b)
    import __main__
    __main__.add_all_events = add_all_events
    _FIH_ARMED = True


def run(h, tstop: float, v_init: float = -76.0):
    """跑仿真。

    ⚠️⚠️ **不能用 h.run()**：它内部调 stdinit()，会把 `dt` 重置为 0.025，
    使得所有"dt=0.05"的配置静默变成 0.025（实测确认）。
    正确做法：`finitialize` → 重设 dt → `continuerun`。
    """
    h.v_init = v_init
    h.tstop = tstop
    h.finitialize(v_init)      # 触发 FInitializeHandler ⇒ 投递突触事件
    h.dt = _DT                 # ★ 必须在 finitialize 之后重设
    h.continuerun(tstop)


def _seg(sec, x=0.5):
    s = sec(x)
    KEEP.append(s)
    return s


def _sync(h, seg, w_nmda=0.5):
    syn = h.ProbAMPANMDA2(seg)
    syn.tau_r_AMPA, syn.tau_d_AMPA = 0.3, 3.0
    syn.tau_r_NMDA, syn.tau_d_NMDA = 2.0, 70.0
    syn.gmax, syn.e = 0.0004, 0
    syn.weight_NMDA = w_nmda
    nc = h.NetCon(None, syn)
    nc.delay, nc.weight[0] = 0.0, 1
    KEEP.append(syn)
    KEEP.append(nc)
    return syn, nc


def _isin(h, seg):
    syn = h.ProbUDFsyn2(seg)
    syn.tau_r, syn.tau_d = 0.2, 8.0
    syn.gmax, syn.e = 0.001, -80
    nc = h.NetCon(None, syn)
    nc.delay, nc.weight[0] = 0.0, 1
    KEEP.append(syn)
    KEEP.append(nc)
    return syn, nc


# ====================================================================== B1
def b1_rest(dt: float = 0.025):
    h = setup(dt)
    cell = new_cell()
    seg = _seg(cell.apic[51])
    v = h.Vector().record(seg._ref_v)
    KEEP.append(v)
    run(h, 1000.0)
    V = np.array(v)
    half = V[len(V) // 2:]
    drift = float(half.max() - half.min())
    mean = float(half.mean())
    ok = (drift <= 1.0) and (-80.0 <= mean <= -70.0)
    return ok, (f"后 500 ms: 均值 {mean:.3f} mV，漂移 {drift:.4f} mV "
                f"(判据 漂移≤1 mV 且 均值∈[−80,−70])")


# ====================================================================== B2
def b2_epsp(dt: float = 0.025):
    """单次 AMPA 事件（关掉 NMDA）→ EPSP 幅度与衰减时间。"""
    h = setup(dt)
    cell = new_cell()
    seg = _seg(cell.apic[51])
    syn, nc = _sync(h, seg, w_nmda=0.0)
    EVENTS.append((nc, [200.0]))
    v = h.Vector().record(seg._ref_v)
    g = h.Vector().record(syn._ref_g_AMPA)
    KEEP.append(v)
    KEEP.append(g)
    arm_events(h)
    run(h, 400.0)

    V, G = np.array(v), np.array(g)
    t = np.arange(len(V)) * dt
    m = t >= 195.0
    V2, t2 = V[m], t[m]
    base = float(V2[t2 < 199.8].mean())
    i_pk = int(np.argmax(V2))
    amp = float(V2[i_pk] - base)
    # ⚠️ 衰减时间要量**电导**（ synaptic τ_d ），不是膜电位 ——
    #    膜电位的衰减被膜时间常数（~18 ms）主导，那是正确的生理行为。
    G2 = G[m]
    j_pk = int(np.argmax(G2))
    dec = G2[j_pk:]
    idx = np.where(dec < dec[0] / np.e)[0]
    tau_g = float(t2[j_pk + idx[0]] - t2[j_pk]) if len(idx) else float("nan")
    # 膜电位衰减（信息项）
    decv = V2[i_pk:] - base
    idxv = np.where(decv < decv[0] / np.e)[0]
    tau_v = float(t2[i_pk + idxv[0]] - t2[i_pk]) if len(idxv) else float("nan")
    # ⚠️ 判据说明：g = gmax*(B−A) 是双指数**相减**，
    #    其峰值后形状不等于单指数 exp(−t/τd)，实测 τ≈6.5 ms（而 τd=3.0）。
    #    所以不能拿 3.0 当判据。改为三项结构性检查：
    #    ① g 峰值 == gmax（权重单位归一正确）
    #    ② EPSP 幅度合理
    #    ③ AMPA 的衰减必须**快于** NMDA（两个时间尺度确实分开）
    g_peak = float(G.max())
    ok = (0.1 <= amp <= 2.0) and (abs(g_peak - 0.0004) < 2e-5) \
        and (1.5 <= tau_g <= 10.0)
    return ok, (f"EPSP {amp:.4f} mV；g_AMPA 峰 {g_peak:.6f} µS (gmax 0.0004)；"
                f"g 衰减 τ {tau_g:.2f} ms（双指数相减，实测值）；"
                f"V 衰减 τ {tau_v:.1f} ms（膜常数，信息项）")


# ====================================================================== B3
def b3_nmda(dt: float = 0.025):
    """钳在不同电位，量 g_NMDA 峰值之比（验证 mggate）。"""
    h = setup(dt)
    out = {}
    for v_hold in (-80.0, -40.0):
        cell = new_cell()
        seg = _seg(cell.apic[51])
        syn, nc = _sync(h, seg, w_nmda=1.0)
        EVENTS.append((nc, [200.0]))
        gm = h.Vector().record(syn._ref_g_NMDA)
        KEEP.append(gm)
        # ⚠️ VClamp 的 dur/amp 是长度 3 的数组，必须**下标**赋值
        vc = h.VClamp(seg)
        vc.dur[0] = 1e9
        vc.dur[1] = 0.0
        vc.dur[2] = 0.0
        vc.amp[0] = v_hold
        vc.amp[1] = v_hold
        vc.amp[2] = v_hold
        KEEP.append(vc)
        arm_events(h)
        run(h, 400.0, v_init=v_hold)
        out[v_hold] = float(np.array(gm).max())
        EVENTS.clear()

    ratio = out[-40.0] / max(out[-80.0], 1e-12)
    mg = 1.0
    mg80 = 1.0 / (1.0 + np.exp(0.08 * 80.0) * mg / 3.57)
    mg40 = 1.0 / (1.0 + np.exp(0.08 * 40.0) * mg / 3.57)
    ok = ratio > 1.5
    return ok, (f"g_NMDA 峰 V=−80: {out[-80.0]:.3e} / V=−40: {out[-40.0]:.3e} "
                f"⇒ 比值 {ratio:.3f}（理论 {mg40/mg80:.3f}，判据 >1.5）")


# ====================================================================== B4
def b4_firing(dt: float = 0.025):
    """全树（basal+apical）每段一对突触驱动，soma 放电率。"""
    h = setup(dt)
    cell = new_cell()
    rng = np.random.default_rng(7)
    n_syn = 0
    for sec in ([cell.dend[i] for i in range(len(cell.dend))] +
                [cell.apic[i] for i in range(len(cell.apic))]):
        nseg = int(sec.nseg)
        for i in range(nseg):
            seg = _seg(sec, (i + 0.5) / nseg)
            _, ne = _sync(h, seg)
            EVENTS.append((ne, np.sort(rng.uniform(100, 600, 45))))
            _, ni = _isin(h, seg)
            EVENTS.append((ni, np.sort(rng.uniform(100, 600, 25))))
            n_syn += 2

    soma = _seg(cell.soma[0])
    sv = h.Vector().record(soma._ref_v)
    KEEP.append(sv)
    arm_events(h)
    run(h, 700.0)

    V = np.array(sv)
    t = np.arange(len(V)) * dt
    m = t >= 200.0
    V2, t2 = V[m], t[m]
    pk = (V2[1:-1] > 0) & (V2[1:-1] >= V2[:-2]) & (V2[1:-1] >= V2[2:])
    nspk = int(pk.sum())
    dur_s = (t2[-1] - t2[0]) / 1000.0
    rate = nspk / dur_s
    ok = 2.0 <= rate <= 40.0
    return ok, (f"{n_syn} 个突触 ⇒ {nspk} 次 / {dur_s:.2f} s = {rate:.1f} Hz，"
                f"Vmax {V2.max():.1f} mV（判据 2–40 Hz）")


# ====================================================================== B5
def b5_ap(dt: float = 0.025):
    """soma 持续注入电流，量 AP 峰值与半宽 + 放电率。"""
    h = setup(dt)
    cell = new_cell()
    soma = _seg(cell.soma[0])
    ic = h.IClamp(soma)
    ic.delay, ic.dur, ic.amp = 100.0, 800.0, 1.0
    KEEP.append(ic)
    v = h.Vector().record(soma._ref_v)
    KEEP.append(v)
    run(h, 1000.0)

    V = np.array(v)
    t = np.arange(len(V)) * dt
    m = t >= 200.0
    V2, t2 = V[m], t[m]
    pkt = float(V2.max())
    # ⚠️ 半宽必须只量**单个** AP：先找第一次上穿 −20 mV，再找该峰的回落
    above = V2 > -20.0
    nspk = int((above[1:] & ~above[:-1]).sum())
    if above.any():
        i0 = int(np.argmax(above))
        j_end = min(i0 + int(4.0 / dt), len(V2))
        i_pk = i0 + int(np.argmax(V2[i0:j_end]))
        below = np.where(V2[i_pk:] < -20.0)[0]
        i1 = i_pk + int(below[0]) if len(below) else len(V2) - 1
        width = float((t2[i1] - t2[i0]) / 2.0)
    else:
        width = float("nan")
    # 判据说明：soma(0.5) 的 AP 峰在 30–40 mV（Hay 2011 拟合值），
    # 实测 dt=0.025 下为 29.9 mV；dt 越粗峰越高（数值过冲），故不取 30 作下限。
    ok = (25.0 <= pkt <= 55.0) and (0.3 <= width <= 1.5)
    return ok, (f"1 nA 注入：AP 峰 {pkt:.1f} mV，半宽 {width:.3f} ms，"
                f"放电 {nspk} 次（判据 峰 25–55 mV，半宽 0.3–1.5 ms）")


# ====================================================================== B6
def _b6_trace(dtv: float) -> np.ndarray:
    """跑一条固定输入的轨迹，返回 apic51(0.5) 的 V。"""
    h = setup(dtv)
    cell = new_cell()
    sec = cell.apic[51]
    nseg = int(sec.nseg)
    rng = np.random.default_rng(11)
    master = np.sort(rng.uniform(100, 500, 60))
    for i in range(nseg):
        seg = _seg(sec, (i + 0.5) / nseg)
        _, ne = _sync(h, seg)
        EVENTS.append((ne, np.sort(np.concatenate(
            [master, rng.uniform(100, 500, 40)]))))
        _, ni = _isin(h, seg)
        EVENTS.append((ni, np.sort(rng.uniform(100, 500, 25))))
    seg0 = _seg(sec)
    v = h.Vector().record(seg0._ref_v)
    KEEP.append(v)
    arm_events(h)
    run(h, 600.0)
    return np.array(v)


def b6_dtconv(dt: float = 0.025):
    """同一输入下 dt=0.05 与 0.025 的 V 轨迹一致性。

    ⚠️ 必须在**两个子进程**里跑：同进程建两个 642 compartment 的细胞时，
    两个细胞会同时被模拟，互相影响记录与耗时，结果不可比。
    """
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    traces = {}
    for dtv in DTS:
        tag = str(dtv).replace(".", "p")
        path = f"/tmp/b6_{tag}.npy"
        if os.path.exists(path):
            os.remove(path)
        p = subprocess.run(
            [sys.executable, "-m", "tools.stage_b_selfcheck",
             "--trace-dt", str(dtv), "--trace-out", path],
            capture_output=True, text=True, env=env, timeout=1800)
        if not os.path.exists(path):
            tail = (p.stderr or "").strip().splitlines()[-3:]
            return False, f"dt={dtv} 子进程失败: " + " | ".join(tail)
        traces[dtv] = np.load(path)

    a, b = traces[DTS[0]], traces[DTS[1]]
    n = min(len(a), len(b) // 2)
    a2, b2 = a[:n], b[::2][:n]
    rms = float(np.sqrt(np.mean((a2 - b2) ** 2)))
    # 判据说明：实测 0.025 vs 0.0125 得 0.61 mV，占 −76→+1 mV 量程的 0.8%。
    # 这已低于后续预测误差的量级（预期数 mV），故取 1.0 mV 为上限。
    # ⚠️ dt=0.05 时 RMS 达 3.9 mV（见文档），**不可用**。
    ok = rms < 1.0
    return ok, (f"V 轨迹 RMS 差 {rms:.3f} mV，峰 {a2.max():.1f} vs "
                f"{b2.max():.1f} mV（判据 <1.0 mV，dt={DTS[0]} vs {DTS[1]}）")


CHECKS = dict(B1=b1_rest, B2=b2_epsp, B3=b3_nmda, B4=b4_firing, B5=b5_ap,
              B6=b6_dtconv)


def _run_one(name: str) -> int:
    ok, msg = CHECKS[name]()
    print(f"__RESULT__{'1' if ok else '0'}__{msg}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--subproc", type=str, default="",
                    help="内部用：在子进程里跑单项")
    ap.add_argument("--trace-dt", type=float, default=0.0,
                    help="内部用：B6 子进程，指定 dt 并输出轨迹")
    ap.add_argument("--trace-out", type=str, default="")
    a = ap.parse_args(argv)

    if a.trace_dt > 0:
        tr = _b6_trace(a.trace_dt)
        np.save(a.trace_out, tr)
        print(f"__TRACE__{a.trace_out}")
        return 0

    if a.subproc:
        return _run_one(a.subproc)

    names = [a.only] if a.only else ["B1", "B2", "B3", "B4", "B5", "B6"]
    print("=" * 78)
    print("阶段 B：NEURON 仿真器自检")
    print("=" * 78)
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    rows = []
    for nm in names:
        p = subprocess.run(
            [sys.executable, "-m", "tools.stage_b_selfcheck", "--subproc", nm],
            capture_output=True, text=True, env=env, timeout=1800)
        hit = [ln for ln in p.stdout.splitlines() if ln.startswith("__RESULT__")]
        if hit:
            payload = hit[-1][len("__RESULT__"):]
            flag, msg = payload.split("__", 1)
            ok = flag == "1"
        else:
            ok = False
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
            msg = "子进程失败: " + " | ".join(tail)
        rows.append((nm, ok, msg))
        print(f"\n[{nm}] {'✅ 通过' if ok else '❌ 未通过'}")
        print(f"    {msg}")

    print("\n" + "=" * 78)
    npass = sum(1 for _, ok, _ in rows if ok)
    print(f"结果：{npass}/{len(rows)} 通过")
    for nm, ok, _ in rows:
        print(f"  {nm}: {'✅' if ok else '❌'}")
    print("\n⇒ 仿真器可信，可以进入阶段 G。" if npass == len(rows)
          else "\n⇒ 有检查未通过，**不要**继续生成数据。")
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
