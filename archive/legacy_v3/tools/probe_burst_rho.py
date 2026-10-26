"""探针：burst（时间簇状）与 ρ（段间相关）是否必要。

## 为什么单独测

这两个是**正交**旋钮：
- `burst`：时间维度，同一通道内一次来 3–6 个棘波（间隙 3–8 ms）
- `rho`  ：空间维度，段与段的电导同步涨落

需要回答：加上它们，`apical_51` 的状态是否会进入**现有协议覆盖不到的区域**？
如果状态分布几乎不变，那它们就是可有可无的调参项；
如果状态进入了新区域（更大动态范围、新时间尺度、Ca 平台），那就是必要的。

## 四个条件

    base    : 独立 Poisson，无 burst，ρ=0
    burst   : 独立 renewal + burst，ρ=0
    rho     : 独立 Poisson，ρ=0.6
    both    : burst + ρ=0.6

## 判据

1. **V 动态范围**：新条件是否显著拓宽（Ca 平台需要 V 上到 −50 mV 以上）
2. **状态 PCA 维数**：新条件是否引入新方向（维数上升 = 新自由度）
3. **`Ca_HVA.h` 的变化量**：它是 Ca 平台的门控，跨度大说明平台被激活
4. **`cai` 的累积**：钙累积是平台记忆的指标

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m tools.probe_burst_rho \
        --cond base --tstop 1000 --seed 1
"""
from __future__ import annotations

import argparse
import os

import numpy as np

MODDIR = "/home/neuron/L5PC/WModel2/neuron/mods/x86_64/libnrnmech.so"
HOCODIR = "/home/neuron/L5PC/WModel2/neuron/hoc"
MORPH = "/home/neuron/L5PC/WModel2/neuron/morphologies/cell1.asc"

G_E = 0.0004
G_I = 0.0010
RHO = 0.6
RATE_E = 40.0      # Hz per synapse
RATE_I = 28.0

GATES = (
    ("Ih", "m"), ("Ca_LVAst", "m"), ("Ca_LVAst", "h"),
    ("Ca_HVA", "h"), ("SKv3_1", "m"), ("SK_E2", "z"), ("Im", "m"),
)

EVENTS: list = []
KEEP: list = []


def add_all_events():
    for nc, ts in EVENTS:
        for t_ in ts:
            nc.event(float(t_))


# ---------------------------------------------------------------- 棘波生成
def renewal_train(rng, rate_hz, tstop, regularity=0.0, refrac=2.0,
                  burst_prob=0.0, burst_n=(3, 6), burst_isi=(3.0, 8.0)):
    """gamma-renewal 发放 + 绝对不应期 + 可选 burst。

    regularity=0 ⇒ CV=1（Poisson）；regularity→1 ⇒ CV 减小（更规则）。
    """
    mean_isi = 1000.0 / max(rate_hz, 1e-6)
    shape = max(1.0, 1.0 / max(1.0 - 0.9 * regularity, 1e-6))
    out = []
    t = float(rng.uniform(0, mean_isi))
    n_max = int(tstop / mean_isi * 4) + 200
    for _ in range(n_max):
        if t >= tstop:
            break
        out.append(t)
        # burst：主棘波后追加一簇
        if burst_prob > 0 and rng.random() < burst_prob:
            n = int(rng.integers(burst_n[0], burst_n[1] + 1))
            tb = t
            for _ in range(n):
                tb += float(rng.uniform(*burst_isi))
                if tb < tstop:
                    out.append(tb)
        isi = float(rng.gamma(shape, mean_isi / shape))
        t += max(isi, refrac)
    return np.sort(np.array(out, dtype=float))


def shared_env(rng, tstop, block_ms=100.0, sigma_ms=80.0):
    n_blk = int(np.ceil(tstop / block_ms))
    blk = rng.uniform(0.3, 1.7, n_blk)
    nb = max(3, int(4 * sigma_ms))
    k = np.exp(-0.5 * (np.arange(-nb, nb + 1) / sigma_ms) ** 2)
    k /= k.sum()
    grid = np.repeat(blk, int(block_ms))[:int(tstop)]
    return np.maximum(np.convolve(grid, k, mode="same"), 0.0)


def build(cond: str, tstop: float, dt: float, seed: int):
    global EVENTS, KEEP
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
    KEEP.append(cell)                 # ⚠️ 必须持有，否则 section 被删 ⇒ segfault
    sec = cell.apic[51]
    nseg = int(sec.nseg)
    segs = [sec((i + 0.5) / nseg) for i in range(nseg)]
    KEEP += segs

    rng = np.random.default_rng(seed)
    use_burst = cond in ("burst", "both")
    use_rho = cond in ("rho", "both")
    rho = RHO if use_rho else 0.0

    # 空间相关：先算共享包络，再按 rho 调制每个通道的率
    if use_rho:
        # ⚠️ master 必须在这里算**一次**，不能在每通道里重采样
        n_master = int(rng.poisson(rho * RATE_E * tstop / 1000))
        master_e = np.sort(rng.uniform(0, tstop, n_master))
        env = {"rho": rho, "master": master_e}
    else:
        env = None
    # 抑制用另一个 master（兴奋/抑制不应完全同步）
    if use_rho:
        n_mi = int(rng.poisson(rho * RATE_I * tstop / 1000))
        env_i = {"rho": rho, "master": np.sort(rng.uniform(0, tstop, n_mi))}
    else:
        env_i = None

    EVENTS.clear()
    KEEP.clear()
    KEEP.append(cell)
    KEEP += segs

    grec = {"a": [], "n": [], "g": []}
    for i, seg in enumerate(segs):
        # ---- 兴奋率：空间抖动 + ρ 共享调制 ----
        r_e = RATE_E * float(rng.uniform(0.9, 1.1))

        e = h.ProbAMPANMDA2(seg)
        e.tau_r_AMPA, e.tau_d_AMPA = 0.3, 3.0
        e.tau_r_NMDA, e.tau_d_NMDA = 2.0, 70.0
        e.gmax, e.e = G_E, 0
        nc_e = h.NetCon(None, e)
        nc_e.delay, nc_e.weight[0] = 0.0, 1

        if use_rho:
            ts_e = _train_modulated(rng, r_e, tstop, env, use_burst)
        else:
            ts_e = renewal_train(rng, r_e, tstop,
                                 burst_prob=0.15 if use_burst else 0.0)
        EVENTS.append((nc_e, ts_e))

        ii = h.ProbUDFsyn2(seg)
        ii.tau_r, ii.tau_d = 0.2, 8.0
        ii.gmax, ii.e = G_I, -80
        nc_i = h.NetCon(None, ii)
        nc_i.delay, nc_i.weight[0] = 0.0, 1
        r_i = RATE_I * float(rng.uniform(0.9, 1.1))
        if use_rho:
            ts_i = _train_modulated(rng, r_i, tstop, env_i, use_burst)
        else:
            ts_i = renewal_train(rng, r_i, tstop,
                                 burst_prob=0.15 if use_burst else 0.0)
        EVENTS.append((nc_i, ts_i))

        grec["a"].append(h.Vector().record(e._ref_g_AMPA))
        grec["n"].append(h.Vector().record(e._ref_g_NMDA))
        grec["g"].append(h.Vector().record(ii._ref_g))
        KEEP += [e, nc_e, ii, nc_i]

    st = {}
    for (m_, g_) in GATES:
        attr = f"{g_}_{m_}"
        st[f"{m_}.{g_}"] = [h.Vector().record(getattr(sg, "_ref_" + attr))
                            for sg in segs]
    st["cai"] = [h.Vector().record(sg._ref_cai) for sg in segs]
    st["V"] = [h.Vector().record(sg._ref_v) for sg in segs]
    tvec = h.Vector().record(h._ref_t)
    return h, cell, grec, st, tvec, nseg


def _train_modulated(rng, rate_hz, tstop, env, use_burst):
    """⚠️ 真·空间相关：一半来自共享主棘波串（master），一半独立。

    只调制**率**是不够的 —— Poisson 噪声会淹没相关性（实测 r 仅 0.047）。
    正确做法是让各通道**复制**同一个 master 棘波串的一部分。
    """
    rho = float(env["rho"])
    master = env["master"]            # ★ 所有通道共享同一条 master 串
    n_ind = rng.poisson((1.0 - rho) * rate_hz * tstop / 1000)
    indep = rng.uniform(0, tstop, n_ind)
    ts = list(master) + list(indep)
    if use_burst:
        extra = []
        for x in ts:
            if rng.random() < 0.15:
                n = int(rng.integers(3, 7))
                tb = x
                for _ in range(n):
                    tb += float(rng.uniform(3.0, 8.0))
                    if tb < tstop:
                        extra.append(tb)
        ts = sorted(ts + extra)
    return np.sort(np.array(ts, dtype=float))


# ---------------------------------------------------------------- 统计
def pca_dims(X, ths=(0.90, 0.95, 0.99)):
    Xc = X - X.mean(0, keepdims=True)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    v = s ** 2
    cum = np.cumsum(v) / v.sum()
    return {t: int(np.searchsorted(cum, t) + 1) for t in ths}


def main(argv=None) -> int:
    global RATE_E, RATE_I, RHO
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=("base", "burst", "rho", "both"),
                    default="base")
    ap.add_argument("--tstop", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--rate-e", type=float, default=RATE_E)
    ap.add_argument("--rate-i", type=float, default=RATE_I)
    ap.add_argument("--v-init", type=float, default=-76.0)
    a = ap.parse_args(argv)
    RATE_E, RATE_I, RHO = a.rate_e, a.rate_i, a.rho

    h, cell, grec, st, tvec, nseg = build(a.cond, a.tstop, a.dt, a.seed)
    h('objref fih')
    h.fih = h.FInitializeHandler('nrnpython("add_all_events()")')
    KEEP.append(h.fih)
    import __main__
    __main__.add_all_events = add_all_events
    h.v_init = a.v_init
    h.tstop = a.tstop
    h.run()

    t = np.array(tvec)
    m = t >= 100.0
    V = np.array(st["V"])[:, m]
    Vmid = V[nseg // 2]
    cai = np.log10(np.maximum(np.array(st["cai"])[:, m], 1e-12))
    hHVA = np.array(st["Ca_HVA.h"])[:, m]

    n_ev = sum(len(ts) for _, ts in EVENTS)
    print(f"=== cond={a.cond}  seed={a.seed}  tstop={a.tstop} ===")
    print(f"  突触事件总数     : {n_ev}")
    print(f"  每通道平均       : {n_ev/(2*nseg):.1f} 次 / {a.tstop:.0f} ms "
          f"= {n_ev/(2*nseg)/(a.tstop/1000):.0f} Hz")
    print(f"  apic51(0.5) V    : [{Vmid.min():.1f}, {Vmid.max():.1f}]  "
          f"跨度 {np.ptp(Vmid):.1f} mV")
    print(f"  全段 V 最大跨度   : {np.median(np.ptp(V, axis=1)):.1f} mV")
    print(f"  Ca_HVA.h 跨度    : "
          f"{np.median(np.ptp(hHVA, axis=1)):.4f}  "
          f"(中位 0.63 → 越小越接近平台态)")
    print(f"  h 的最小值        : {hHVA.min():.4f}")
    print(f"  cai log10 跨度   : {np.median(np.ptp(cai, axis=1)):.4f}")

    # 状态 PCA（用 V + 7 门控 + cai）
    names, blocks = [], []
    for k, vs in st.items():
        A = np.array(vs)[:, m]
        if k == "cai":
            A = np.log10(np.maximum(A, 1e-12))
        blocks.append(A)
        names.append(k)
    S = np.vstack(blocks).T
    Ss = (S - S.mean(0)) / np.maximum(S.std(0), 1e-12)
    d = pca_dims(Ss)
    print(f"  状态 99 维 PCA   : " +
          "  ".join(f"{k*100:.0f}%⇒{v}" for k, v in d.items()))

    # 输入相关性
    ga = np.array([np.array(v)[m] for v in grec["a"]])
    C = np.corrcoef(ga)
    off = C[~np.eye(nseg, dtype=bool)]
    print(f"  AMPA 通道间 r    : 中位 {np.median(off):+.3f}  最大 {off.max():+.3f}")

    np.savez(f"/tmp/br_{a.cond}_{a.seed}.npz", V=Vmid, cai=cai[:, nseg // 2],
             hHVA=hHVA[nseg // 2], t=t[m], ga=ga)
    print(f"  ⇒ /tmp/br_{a.cond}_{a.seed}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
