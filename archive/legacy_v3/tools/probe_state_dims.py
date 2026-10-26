"""探针：状态维度与水维数实测 —— 决定 d_z 能压到多小。

## 回答两个问题

1. **$s_t$（99 维）的有效维数是多少？** ⇒ 决定 $d_z$ 下限
2. **$u_t$ 的有效维数是多少？** ⇒ 判断是否需要 36 维

## 方法

跑一条真实轨迹（整细胞，突触只挂 `apical_51` 的 11 个 compartment，
共 11 对 = 22 个），记录：
- 状态：9 因子 × 11 compartment = 99 维
- 输入：11 AMPA + 11 NMDA(去门控) + 11 GABA = 33 维

然后：
- 每因子的**空间变异系数 CV**（11 个 compartment 之间）
- 每个因子对**时间**的方差占比
- 整体 PCA：解释 90/95/99% 方差需要几维
- 输入侧同样做 PCA

⚠️ 注意：上一次测"11 点冗余 92–99%"是在**无突触**电压钳下做的。
   现在每段挂了自己的突触（Poisson 独立采样），空间差异会变大，
   必须重测。

## 用法

    cd WModel2 && PYTHONPATH=$PWD python -m tools.probe_state_dims \
        --tstop 1000 --rate-e 120 --rate-i 80
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

# ⚠️ 必须模块级：FInitializeHandler 通过 nrnpython() 调用
EVENTS: list = []
KEEP: list = []


def add_all_events():
    for nc, ts in EVENTS:
        for t_ in ts:
            nc.event(float(t_))


GATES = (
    ("Ih", "m"), ("Ca_LVAst", "m"), ("Ca_LVAst", "h"),
    ("Ca_HVA", "h"), ("SKv3_1", "m"), ("SK_E2", "z"), ("Im", "m"),
)


def _spikes_indep(rng, rate_hz, tstop):
    """完全独立：每段自己的恒定率 + Poisson。"""
    return np.sort(rng.uniform(0, tstop, rng.poisson(rate_hz * tstop / 1000)))


def _envelope(rng, tstop, block_ms=100.0, sigma_ms=80.0):
    """论文式的**共享慢包络**：分段常数 + 高斯平滑。"""
    n_blk = int(np.ceil(tstop / block_ms))
    blk = rng.uniform(0.3, 1.7, n_blk)
    nb = max(3, int(4 * sigma_ms))
    k = np.exp(-0.5 * (np.arange(-nb, nb + 1) / sigma_ms) ** 2)
    k /= k.sum()
    n_ms = int(tstop)
    grid = np.repeat(blk, int(block_ms))[:n_ms]
    env = np.convolve(grid, k, mode="same")
    return np.maximum(env, 0.0)


def _spikes_shared(rng, rate_hz, tstop, env, scale):
    """给定共享包络下的 Poisson 采样（通道间只通过慢包络相关）。"""
    n_ms = len(env)
    p = np.maximum(rate_hz * 1e-3 * env * scale, 0.0)   # 每 ms 的发射概率
    hit = rng.random(n_ms) < p
    t = np.nonzero(hit)[0].astype(float)
    t += rng.random(len(t))                              # 亚 ms 抖动
    return np.sort(t[t < tstop])


def build(tstop, dt, rate_e, rate_i, seed=1, mode="indep"):
    global EVENTS, KEEP
    from neuron import h
    h.nrn_load_dll(MODDIR)
    # ⚠️ 必须 chdir 到 hoc 目录再用**相对**文件名 load_file。
    #    用绝对路径加载 L5PCtemplate_2.hoc 会导致 h.finitialize() segfault
    #    （该 hoc 里有 xopen/execute 类相对上下文）。
    cwd = os.getcwd()
    os.chdir(HOCODIR)
    for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc",
              "L5PCtemplate_2.hoc"):
        h.load_file(f)
    os.chdir(cwd)
    h.celsius = 34.0
    h.dt = dt

    cell = h.L5PCtemplate(MORPH)
    KEEP.append(cell)          # ⚠️ 模块级持有，防 hoc objref 被析构
    sec = cell.apic[51]
    nseg = int(sec.nseg)
    rng = np.random.default_rng(seed)

    EVENTS.clear()
    KEEP.clear()
    # ⚠️⚠️ 必须先创建并长期持有所有 Segment 对象。
    #   若在循环里用 `sec((i+0.5)/n)` 的临时对象挂突触/record，
    #   临时对象下一轮就被 GC ⇒ pnt 指针悬空 ⇒ h.finitialize() segfault
    #   （无 Python traceback，只在 C 层 core dump）。
    segs = [sec((i + 0.5) / nseg) for i in range(nseg)]
    KEEP += segs

    g_rec = {"ampa": [[] for _ in range(nseg)],
             "nmda": [[] for _ in range(nseg)],
             "gaba": [[] for _ in range(nseg)]}

    # ---- 输入生成策略 ----
    if mode == "shared":
        env = _envelope(rng, tstop)
        seg_scale_e = rng.uniform(0.5, 1.5, nseg)
        seg_scale_i = rng.uniform(0.5, 1.5, nseg)
    else:
        env = None
        seg_scale_e = seg_scale_i = np.ones(nseg)

    n_ev = 0
    for i, seg in enumerate(segs):

        e = h.ProbAMPANMDA2(seg)
        e.tau_r_AMPA, e.tau_d_AMPA = 0.3, 3.0
        e.tau_r_NMDA, e.tau_d_NMDA = 2.0, 70.0
        e.gmax, e.e = G_E, 0
        nc_e = h.NetCon(None, e)
        nc_e.delay, nc_e.weight[0] = 0.0, 1
        if mode == "shared":
            ts_e = _spikes_shared(rng, rate_e, tstop, env, seg_scale_e[i])
        else:
            ts_e = _spikes_indep(rng, rate_e, tstop)
        EVENTS.append((nc_e, ts_e))
        n_ev += len(ts_e)

        ii = h.ProbUDFsyn2(seg)
        ii.tau_r, ii.tau_d = 0.2, 8.0
        ii.gmax, ii.e = G_I, -80
        nc_i = h.NetCon(None, ii)
        nc_i.delay, nc_i.weight[0] = 0.0, 1
        if mode == "shared":
            ts_i = _spikes_shared(rng, rate_i, tstop, env, seg_scale_i[i])
        else:
            ts_i = _spikes_indep(rng, rate_i, tstop)
        EVENTS.append((nc_i, ts_i))
        n_ev += len(ts_i)

        g_rec["ampa"][i].append(h.Vector().record(e._ref_g_AMPA))
        g_rec["nmda"][i].append(h.Vector().record(e._ref_g_NMDA))
        g_rec["gaba"][i].append(h.Vector().record(ii._ref_g))
        KEEP += [e, nc_e, ii, nc_i]

    # ---- 状态记录（9 因子 × 11 compartment）----
    st = {}
    for (m_, g_) in GATES:
        attr = f"{g_}_{m_}"
        st[f"{m_}.{g_}"] = [h.Vector().record(getattr(sg, "_ref_" + attr))
                            for sg in segs]
    st["cai"] = [h.Vector().record(sg._ref_cai) for sg in segs]
    st["V"] = [h.Vector().record(sg._ref_v) for sg in segs]
    tvec = h.Vector().record(h._ref_t)
    # ⚠️⚠️ 必须把 `cell` 一起返回并长期持有！
    #    h.L5PCtemplate() 返回的是 hoc objref；Python 引用计数归零时
    #    hoc 会析构该 objref ⇒ 它创建的所有 section 被 delete_section()
    #    ⇒ 挂在上面的突触 / Vector 记录指针全部悬空
    #    ⇒ h.finitialize() segfault（C 层 core dump，无 traceback）。
    return h, cell, g_rec, st, tvec, n_ev


def pca_dims(X: np.ndarray, thresh=(0.90, 0.95, 0.99)) -> dict:
    """X: (n_sample, n_dim)，已中心化。返回各阈值所需维数。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD 比协方差特征值稳
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    cum = np.cumsum(var) / var.sum()
    return {th: int(np.searchsorted(cum, th) + 1) for th in thresh}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tstop", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--rate-e", type=float, default=120.0)
    ap.add_argument("--rate-i", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mode", choices=("indep", "shared"), default="indep",
                    help="indep=每段独立 Poisson；"
                         "shared=论文式共享慢包络 + U(0.5,1.5) 空间抖动")
    ap.add_argument("--n-trial", type=int, default=1,
                    help="跑几条不同 seed 的轨迹并合并做 PCA "
                         "（单轨迹会低估维数）")
    ap.add_argument("--v-init", type=float, default=-76.0)
    a = ap.parse_args(argv)

    h, cell, g_rec, st, tvec, n_ev = build(a.tstop, a.dt, a.rate_e,
                                          a.rate_i, a.seed, a.mode)
    h('objref fih')            # ⚠️ 必须先声明 hoc 变量
    h.fih = h.FInitializeHandler('nrnpython("add_all_events()")')
    KEEP.append(h.fih)
    # ⚠️ nrnpython() 在 __main__ 里查找函数名。用 `python -m` 启动时
    #    模块被 runpy 当作 __main__ 执行，但仍需显式绑定，否则 core dump。
    import __main__
    __main__.add_all_events = add_all_events
    h.v_init = a.v_init
    h.tstop = a.tstop
    h.v_init = a.v_init
    h.tstop = a.tstop
    h.run()

    t = np.array(tvec)
    # 跳过前 100 ms（初始瞬态）
    m = t >= 100.0
    print(f"=== 轨迹：{a.tstop:.0f} ms，dt={a.dt}，"
          f"rate_e={a.rate_e:.0f} rate_i={a.rate_i:.0f} ===")
    print(f"  突触事件总数: {n_ev}  （22 个突触通道）")

    # ---------- 状态：拼成 (T, 99) ----------
    names, blocks = [], []
    for k, vs in st.items():
        A = np.array(vs)[:, m]     # (11, T)
        if k == "cai":
            A = np.log10(np.maximum(A, 1e-12))
        blocks.append(A)
        names.append(k)
        # 空间 CV（逐时刻算，再取中位）
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.nanmedian(np.std(A, axis=0) / np.maximum(
                np.abs(np.mean(A, axis=0)), 1e-12))
        # 时间跨度（逐段），归一化到自身均值
        span = float(np.median(np.ptp(A, axis=1)))
        scale = float(np.median(np.abs(np.mean(A, axis=1))))
        print(f"  {k:<14} 空间CV={cv*100:>7.2f}%   "
              f"时间跨度={span:.5f}  跨度/均值={span/max(scale,1e-12):>7.2f}")

    S = np.vstack(blocks).T          # (T, 99)
    # 标准化（每个维度除自己的 std，避免被大量纲主导）
    S_std = (S - S.mean(0)) / np.maximum(S.std(0), 1e-12)
    dims = pca_dims(S_std)
    print(f"\n=== 状态 99 维 PCA ===")
    for th, d in dims.items():
        print(f"  解释 {th*100:.0f}% 方差需要 {d:>3} 维   "
              f"（压缩比 99/{d} = {99/d:.1f}×）")

    # ---------- 输入：11 AMPA + 11 NMDA + 11 GABA = 33 ----------
    U_blocks, unames = [], []
    for key in ("ampa", "nmda", "gaba"):
        for i in range(len(g_rec[key])):
            v = np.array(g_rec[key][i][0])[m]
            U_blocks.append(v)
        unames.append(key)
    U = np.vstack(U_blocks).T        # (T, 33)
    print(f"\n=== 输入 33 维各通道统计 ===")
    print(f"  {'通道':<12} {'峰值':>10} {'非零占比':>9}")
    off = 0
    for key in ("ampa", "nmda", "gaba"):
        for i in range(len(g_rec[key])):
            col = U[:, off]
            print(f"  {key}[{i}]      {col.max():>10.6f} "
                  f"{(col > 1e-9).mean()*100:>8.1f}%")
            off += 1
    U_std = (U - U.mean(0)) / np.maximum(U.std(0), 1e-12)
    udims = pca_dims(U_std)
    print(f"\n=== 输入 33 维 PCA ===")
    for th, d in udims.items():
        print(f"  解释 {th*100:.0f}% 方差需要 {d:>3} 维   "
              f"（压缩比 33/{d} = {33/d:.1f}×）")

    # 保存，便于后续画图
    np.savez(f"/tmp/sd_{a.mode}_{a.seed}.npz", t=t[m], S=S, U=U,
             names=np.array(names))
    print("\n⇒ 已存")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
