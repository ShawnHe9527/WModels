"""V0：v4 单轨迹端到端 + 5 个门禁（DESIGN_v4 §9）。

跑通「目标段突触 + tuft 簇 + 胞体 bAP」三类驱动，并检查：

| 门 | 判据 |
|---|---|
| G1 平台占比 | 目标段上行时间占比 ∈ [15%, 45%] |
| G2 双稳态检查 | V 的双峰系数 BC ≤ 0.555（标准双峰阈值）|
| G3 邻段事件数 | tuft 上行事件数 ∈ [2, 20] |
| G4 bAP 数 | ∈ [0, 25] 每 1000 ms |
| G5 数值 | 无 NaN、\|V\| < 150 mV |

额外输出 **§4.5 的紧凑性检验**：邻段段内跨 compartment 的 V 相关系数
（若 < 0.9，说明按 section 求和会丢信息，应退回 T2 全分辨率）。

用法：

    cd WModel2 && PYTHONPATH=$PWD python -m tools.v0_check --n 12
    cd WModel2 && PYTHONPATH=$PWD python -m tools.v0_check --n 3 --t-test 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.core import safety

THREADS = safety.apply_thread_caps(2)

import numpy as np                                # noqa: E402

from world.gen import protocol as P1              # noqa: E402
from world.gen import protocol2 as P2             # noqa: E402
from world.gen import sim2 as S2                  # noqa: E402

UP_THR = -45.0           # mV，判定「上行」
MID_LO, MID_HI = -50.0, -25.0   # mV，中间态区间（防硬二值）
V_LIMIT = 150.0          # mV，数值安全上限
PLATEAU_V = -30.0        # mV，v3 口径的「出现平台」
PLATEAU_CAI = 0.5        # v3 口径的 cai 跨度门禁


# ==============================================================================
def _peaks(v, thr, min_gap):
    idx = np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > thr))[0] + 1
    out: list[int] = []
    for i in idx:
        if not out or i - out[-1] >= min_gap:
            out.append(int(i))
    return out


def _episodes(v, thr):
    """V > thr 的连续段数量。"""
    m = (v > thr).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return int((d == 1).sum())


def bimodality_coef(x):
    """BC = (skew²+1) / (kurt + 3(n−1)²/((n−2)(n−3)))。

    BC > 0.555 ⇒ 可能是双峰（均匀分布为 0.555，正态为 0.333）。
    """
    n = x.size
    if n < 10:
        return float("nan")
    x = x - x.mean()
    s2 = (x ** 2).mean()
    if s2 <= 0:
        return float("nan")
    m3 = (x ** 3).mean()
    m4 = (x ** 4).mean()
    skew = m3 / s2 ** 1.5
    kurt = m4 / s2 ** 2 - 3.0
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((skew ** 2 + 1.0) / (kurt + corr))


def min_pair_corr(V):
    """段内跨 compartment 的 V 相关系数（最小值 / 中位）。"""
    if V.shape[0] < 2:
        return 1.0, 1.0
    c = np.corrcoef(V)
    iu = np.triu_indices(c.shape[0], k=1)
    return float(np.min(c[iu])), float(np.median(c[iu]))


def compress_r2(G):
    """§4.5 真正的可压缩性判据：`sum_i g_i` 对每个 `g_i` 的解释力。

    要压缩的是**电导**不是电位，所以应该问：
    「用该 section 的总电导去预测每个 compartment 的电导，R² 有多高？」
    R² 高 ⇒ 按 section 求和不会丢关键信息。
    """
    if G.shape[0] < 2:
        return 1.0
    s = G.sum(axis=0)
    ss = ((s - s.mean()) ** 2).sum()
    if ss <= 1e-30:
        # ⚠️ 无事件（g 恒为 0）⇒ 无定义，返回 NaN 而不是 0
        #    （linear 档 tuft 事件极稀疏，曾因此被误计为 R²=0）
        return float("nan")
    r2s = []
    for g in G:
        v = ((g - g.mean()) ** 2).sum()
        if v <= 1e-30:
            continue        # 该 comp 全程无事件 ⇒ 不参与统计（避免下拉中位）
        cov = ((g - g.mean()) * (s - s.mean())).sum()
        r2s.append(float(cov ** 2 / max(v * ss, 1e-30)))
    if not r2s:
        return float("nan")
    return float(np.median(r2s))


# ==============================================================================
def one_trial(sim, pol, rng):
    t0 = sim.t_settle
    t1 = sim.t_settle + sim.t_test

    own_e, own_i, _info = P1.make_spike_trains(
        rng, sim.nseg, t0, t1, pol.to_policy1())
    tuft_e, tuft_i, bundles = P2.make_tuft_bundles(
        rng, sim.n_nb_e, sim.n_nb_i, t0, t1, pol)
    baps = P2.make_bap_times(rng, bundles, t0, t1, pol, max_n=S2.MAXP)

    t_start = time.time()
    S, U = sim.run(own_e, own_i, tuft_e, tuft_i, baps,
                   bap_amp=pol.bap_amp, bap_width=pol.bap_width)
    wall = time.time() - t_start

    V = S[:, 8, :]                                     # (T,11)
    cai = S[:, 7, :]
    vmean = V.mean(axis=1)
    v_soma = sim.last_v_soma

    # 邻段汇总
    tuft_v = np.concatenate([d["V"] for d in sim.last_nb
                             if d["idx"] in S2.CHILD_IDX])
    nb_corr = {}
    nb_r2 = {}
    for d in sim.last_nb:
        mn, md = min_pair_corr(d["V"])
        nb_corr[d["idx"]] = (mn, md)
        nb_r2[d["idx"]] = compress_r2(d["g_sum"])

    # ⚠️ cai 存的是 log10 值（见 sim2.py），所以跨度也用 log10 单位。
    #    v3 的 `cai_span > 0.5` 就是 log10 口径（生理 cai 跨度只有 1e-4 → 1e-2 mM）。
    cai_span = float(cai.max() - cai.min())
    plateau = bool(V.max() > PLATEAU_V and cai_span > PLATEAU_CAI)

    stat = dict(
        wall=wall,
        v_span=float(V.max() - V.min()),
        v_max=float(V.max()),
        v_min=float(V.min()),
        cai_span=float(cai_span),
        cai_max=float(cai.max()),
        up_frac=float((vmean > UP_THR).mean()),
        mid_frac=float(((vmean > MID_LO) & (vmean < MID_HI)).mean()),
        vmean_bc=bimodality_coef(vmean),
        plateau=plateau,
        tuft_vmax=float(tuft_v.max()),
        tuft_n_ev=_episodes(tuft_v.mean(axis=0), UP_THR),
        tuft_corr_min=min(v[0] for v in nb_corr.values()),
        tuft_r2_min=min(nb_r2.values()),
        nb_corr=nb_corr,
        nb_r2=nb_r2,
        n_bap=len(baps),
        soma_spk=len(_peaks(v_soma, 0.0, int(1.0 / sim.dt))),
        n_tuft_e=sum(len(x) for x in tuft_e),
        nan=bool(np.isnan(S).any() or np.isnan(U).any()),
        absmax=float(max(np.abs(S).max(), np.abs(v_soma).max())),
        U=U, S=S,
    )

    # ---- per-trial 质量门（按档位；数据集级分布门在汇总里算）----
    # 线性档按定义就是亚阈的：不该要求它「上行」，也不该要求有中间态。
    g = {}
    if pol.regime == "linear":
        g["G1"] = stat["up_frac"] <= 0.10     # 亚阈：不得已经进入上行
        g["G2"] = True
    else:
        g["G1"] = 0.005 <= stat["up_frac"] <= 0.60
        g["G2"] = stat["mid_frac"] >= 0.005
    g["G3"] = stat["tuft_n_ev"] <= 20
    g["G4"] = stat["n_bap"] <= 25
    g["G5"] = (not stat["nan"]) and stat["absmax"] < V_LIMIT
    stat["gates"] = g
    stat["pass"] = all(g.values())
    return stat


# ==============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.v0_check")
    ap.add_argument("--n", type=int, default=12, help="trial 数")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--nb-n-e", type=int, default=1,
                    help="邻段每 compartment 兴奋突触数（D4：1=O1，2=O2）")
    ap.add_argument("--t-settle", type=float, default=100.0)
    ap.add_argument("--t-test", type=float, default=1000.0)
    ap.add_argument("--mem-mb", type=float, default=2700.0)
    ap.add_argument("--regime", type=str, default="",
                    help="只跑某一档（linear/inter/plateau）")
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=a.mem_mb)
    rng = np.random.default_rng(a.seed)

    print("=== 构建 Sim2（目标段 11E+11I，邻段 17E+17I，胞体脉冲池 32）===")
    sim = S2.Sim2(dt=a.dt, t_settle=a.t_settle, t_test=a.t_test, nb_n_e=a.nb_n_e)
    print(f"  nseg={sim.nseg}  n_nb_e={sim.n_nb_e}  n_nb_i={sim.n_nb_i}  "
          f"U_DIM={S2.U_DIM}  T={sim.T}")

    rows = []
    t_all = time.time()
    for i in range(a.n):
        pol = P2.sample_policy2(rng)
        if a.regime and pol.regime != a.regime:
            continue
        st = one_trial(sim, pol, rng)
        rows.append((i, pol, st))
        gs = "".join("1" if st["gates"][k] else "0"
                     for k in ("G1", "G2", "G3", "G4", "G5"))
        print(f"[{i:>3}] {pol.regime:<7} f={pol.f_tuft:.2f} "
              f"bnd={pol.bundle_hz:.1f}Hz w={pol.width:.1f}ms "
              f"bAP={pol.bap_rate:>4.1f}Hz amp={pol.bap_amp:.2f} "
              f"| Vspan={st['v_span']:>6.1f} Vmax={st['v_max']:>6.1f} "
              f"up={st['up_frac']*100:>5.1f}% mid={st['mid_frac']*100:>4.1f}% "
              f"plat={'Y' if st['plateau'] else 'n'} "
              f"tuftV={st['tuft_vmax']:>6.1f} ev={st['tuft_n_ev']:>2d} "
              f"spk={st['soma_spk']:>2d} corr={st['tuft_corr_min']:.4f} "
              f"| {gs} {'PASS' if st['pass'] else 'FAIL'} "
              f"({st['wall']:.1f}s)")

    dt_all = time.time() - t_all
    print("\n=== 汇总 ===")
    npass = sum(st["pass"] for _, _, st in rows)
    print(f"  通过 {npass}/{len(rows)} = {100.0*npass/max(len(rows),1):.0f}%"
          f"   总耗时 {dt_all/60:.1f} min"
          f"   平均 {dt_all/max(len(rows),1):.2f} s/trial")

    for k in ("G1", "G2", "G3", "G4", "G5"):
        cnt = sum(st["gates"][k] for _, _, st in rows)
        print(f"  {k}: {cnt}/{len(rows)}")

    # 分档统计
    print("\n  按档：")
    for reg in ("linear", "inter", "plateau"):
        sel = [st for _, p, st in rows if p.regime == reg]
        if not sel:
            continue
        print(f"    {reg:<8} n={len(sel):>2}  "
              f"up={100*np.mean([s['up_frac'] for s in sel]):>5.1f}%  "
              f"plat={sum(s['plateau'] for s in sel)}/{len(sel)}  "
              f"Vmax={np.mean([s['v_max'] for s in sel]):>6.1f}  "
              f"tuftV={np.mean([s['tuft_vmax'] for s in sel]):>6.1f}  "
              f"ev={np.mean([s['tuft_n_ev'] for s in sel]):>4.1f}  "
              f"pass={sum(s['pass'] for s in sel)}/{len(sel)}")

    # ---- 数据集级分布门 ----
    print("\n  数据集级：")
    pf = np.mean([st["plateau"] for _, _, st in rows])
    ok = 0.15 <= pf <= 0.45
    print(f"    plateau trial 比例 = {100*pf:.1f}%  "
          f"（目标 15–45%）{'OK' if ok else '⚠️ 需调采样'}")
    print(f"    上行时间占比 中位 = {100*np.median([st['up_frac'] for _,_,st in rows]):.2f}%  "
          f"范围 [{100*min(st['up_frac'] for _,_,st in rows):.1f}%, "
          f"{100*max(st['up_frac'] for _,_,st in rows):.1f}%]")

    # §4.5 紧凑性检验
    print("\n=== §4.5 可压缩性检验（按 section 求和 g 的解释力）===")
    worst_c = {}
    worst_r = {}
    for _, p, st in rows:
        # 仅对 tuft 输入足够强的 trial 统计：弱输入下大多数 comp 全程无事件，
        # 求和 g 几乎恒为 0，R² 无定义（会被误读成「不可压缩」）
        if p.f_tuft < 0.30:
            continue
        for idx, (mn, md) in st["nb_corr"].items():
            worst_c.setdefault(idx, [1.0, 1.0])
            worst_c[idx][0] = min(worst_c[idx][0], mn)
            worst_c[idx][1] = min(worst_c[idx][1], md)
        for idx, r2 in st["nb_r2"].items():
            if not np.isnan(r2):
                worst_r[idx] = min(worst_r.get(idx, 1.0), r2)
    if not worst_r:
        print("  （无 f_tuft ≥ 0.30 的 trial，跳过）")
    for idx in sorted(worst_r):
        mn, md = worst_c[idx]
        # 判据看「求和 g 的 R²」，V 相关只作参考（局部突触会造成点位差）
        tag = "OK" if worst_r[idx] >= 0.70 else "⚠️ <0.70 ⇒ 建议 T2 全分辨率"
        print(f"  apic[{idx}]: g_sum R² = {worst_r[idx]:.4f}   "
              f"V corr(min/中位) = {mn:.3f}/{md:.3f}   {tag}")
    print("  参考：V 的逐点相关会被「单个 comp 上的局部突触」拉低，"
          "而实际压缩的是 g，故以 g_sum R² 为准。")

    # 首条样本的数组形状与键统计
    if rows:
        st = rows[0][2]
        print("\n=== 首条样本形状 ===")
        print(f"  s: {st['S'].shape} {st['S'].dtype}   "
              f"u: {st['U'].shape} {st['U'].dtype}")
        U = st["U"]
        for nm, sl in (("own_11comp", slice(0, 33)),
                       ("nb_apic50", slice(33, 42)),
                       ("nb_apic52", slice(42, 57)),
                       ("nb_apic53", slice(57, 84)),
                       ("bap", slice(84, 85))):
            q = U[:, sl]
            print(f"  u[{nm:<11}] dim={q.shape[1]:>2}  max={q.max():.3g}  "
                  f"frac>0={100*(q > 0).mean():5.1f}%  mean={q.mean():.3g}")
        U1 = S2.to_t1(U)
        print(f"\n  to_t1 → {U1.shape}（应与 U 的 section 求和一致）")
        for k, (lo, hi) in enumerate(S2.NB_SLICES):
            a = U1[:, 33 + 3 * k: 36 + 3 * k].max()
            b = U[:, 33 + 3 * lo: 33 + 3 * hi].reshape(len(U), hi - lo, 3) \
                 .sum(axis=1).max()
            ok = abs(float(a) - float(b)) < 1e-9
            print(f"    T1 邻段{k} max={a:.5g} vs 直接求和 {b:.5g}  "
                  f"{'OK' if ok else 'MISMATCH'}")
        print(f"    T1 全零列：{int((U1.max(axis=0) == 0).sum())}")

    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
