"""v4 结果可视化：闭环预测 vs 真值轨迹、误差-视界曲线、非线性指纹、latent 记忆。

用法（PYTHONPATH=$PWD）：
    python -m tools.plot_v4 --ckpt t2v4_anchor/last.pt --data te_s3
    python -m tools.plot_v4 --ckpt t2v4_anchor/last.pt --data te_s3 --k 2000

输出到 results/figs/。
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import torch                           # noqa: E402

from world.core import safety          # noqa: E402
from world.learn import data2 as D2    # noqa: E402
from world.learn import model as M     # noqa: E402
from world.learn.train2 import SCALER, _torch   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "checkpoints")
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "results", "figs")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False,
})
C_TRUE, C_MODEL, C_L0 = "#222222", "#d62728", "#1f77b4"


def log(*a):
    print(*a, flush=True)


def load_model(ckpt: str, ds, tier: str = "t2"):
    ck = torch.load(os.path.join(RES, ckpt), map_location="cpu",
                    weights_only=False)
    cfg = ck.get("cfg") or {}
    sc = ck.get("scaler") or json.load(open(SCALER, encoding="utf-8"))
    m = M.build(d_z=int(cfg.get("d_z", 16)), hidden=int(cfg.get("hidden", 192)),
                n_layers=int(cfg.get("n_layers", 2)), n_factors=D2.N_FACTORS,
                u_dim=D2.U_FEAT, th_dim=int(ds.th.shape[1]),
                pred_attn=bool(cfg.get("pred_attn", False)),
                n_heads=int(cfg.get("n_heads", 4)),
                u_in_x=False, n_glob=D2.N_GLOB)
    m.load_state_dict(ck["model"])
    m.eval()
    log(f"载入 {ckpt}  σ={ck.get('sigma', float('nan')):.6f}")
    return m, sc, cfg


def denorm(s, sc):
    """(B,T,F,P) 归一化 → 物理单位。"""
    loc = np.asarray(sc["loc_s"], np.float32)[None, None, :, None]
    sc_ = np.asarray(sc["scale_s"], np.float32)[None, None, :, None]
    return s * sc_ + loc


def pick_trial(ds, regime: str, L: int) -> int:
    """在该档里挑一个「最有内容」的 trial（事件最多 / up_frac 最高）。"""
    ts = [t for t in ds.meta["trials"]
          if t.get("regime") == regime and t.get("pass_", True)]
    if not ts:
        ts = [t for t in ds.meta["trials"] if t.get("regime") == regime]
    if regime == "plateau":
        ts.sort(key=lambda t: -t.get("n_tuft_event", 0))
    elif regime == "inter":
        ts.sort(key=lambda t: -t.get("up_frac", 0))
    else:
        ts.sort(key=lambda t: -t.get("V_span", 0))
    for t in ts:
        if t["i"] + L < ds.T:
            return t["i"]
    return ts[0]["i"]


def regime_rollouts(m, ds, sc, regime: str, L: int, t0: int, n: int = 16):
    """对某档的前 n 个 trial 各滚一次，返回 (trials, pred, true, l0)。

    用于「按性能挑中位数 trial」——只画最极端那个 trial 会严重误导读图。
    """
    ids = [t["i"] for t in ds.meta["trials"] if t.get("regime") == regime][:n]
    P, Y, B = [], [], []
    for i in ids:
        r = rollout_seq(m, ds, sc, i, L, t0)
        P.append(r["pred"]); Y.append(r["true"]); B.append(r["l0"])
    return ids, np.stack(P), np.stack(Y), np.stack(B)


def rollout_seq(m, ds, sc, trial: int, L: int, comp: int, t0: int = 4000):
    """固定 trial，从某个起点闭环滚 L 步。返回 dict of numpy。

    ⚠️ t0 必须 ≥ `t_settle`（te_s3 为 100 ms = 4000 步），否则落在静置瞬态里，
    误差会虚高上百倍（实测 t0=2000 时 a=101）。与 `train2 eval` 的默认 t0 一致。
    """
    sp2 = D2.Sampler2(ds, sc, batch=1, length=1, seed=0)
    s0, u, ug, th, s_true, x = sp2.batch_window(t0, L, 1, off=trial)
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z)).float()   # noqa: E731
    # ⚠️ batch_window 返回 (B,L,P_u,5)，Encoder 要 (B,L,Cu,Pu)
    u = torch.from_numpy(np.ascontiguousarray(
        np.asarray(u).transpose(0, 1, 3, 2))).float()
    s0, ug, th, x = T(s0), T(ug), T(th), T(x)
    with torch.no_grad():
        out = m.rollout(s0, u, th, x, use_residual=True, ug_seq=ug)
    pred = out.numpy()                       # (1,L,F,P) = ŝ_{t+1..t+L}
    # 真值只覆盖到 t0+L-1 ⇒ 有效目标只有 L-1 步
    pred = pred[:, :L - 1]
    true = s_true[:, 1:]                     # (1,L-1,F,P)
    # L0 必须与 V6 同定义：起点冻结后一直复制 ⇒ |s0 - s_{t0+k}|
    # （写成 s_true[:, :-1] 得到的是单步漂移，会让比值虚高上千倍）
    base = np.broadcast_to(s0.numpy()[:, None], true.shape).copy()
    return dict(pred=pred[0], true=true[0], l0=base[0], t0=t0)


def fig_trace(m, ds, sc, regime: str, L: int, figdir: str, t0: int = 4000):
    """画该档**中位数性能**的 trial（不是最极端那个）。"""
    ci = ds.meta["factor_names"].index("cai")
    vi = ds.meta["factor_names"].index("V")
    ids, P_, Y_, B_ = regime_rollouts(m, ds, sc, regime, L, t0)
    # 按末端误差挑中位数 trial
    em = np.abs(P_ - Y_).mean((1, 2, 3))
    el = np.abs(B_ - Y_).mean((1, 2, 3))
    order = np.argsort(em)
    j = int(order[len(order) // 2])
    trial, pred, true, l0 = ids[j], P_[j], Y_[j], B_[j]
    IQR = (np.percentile(em, 75) - np.percentile(em, 25))

    tr = denorm(true[None], sc)[0]
    pr = denorm(pred[None], sc)[0]
    lz = denorm(l0[None], sc)[0]
    ms = np.arange(tr.shape[0]) * ds.dt
    cV = int(np.argmax(tr[:, vi, :].std(0)))
    cc = int(np.argmax(tr[:, ci, :].std(0)))

    fig, ax = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.15, 1.15, 0.8]))
    ax[0].plot(ms, tr[:, vi, cV], color=C_TRUE, lw=1.5, label="true V")
    ax[0].plot(ms, pr[:, vi, cV], color=C_MODEL, lw=1.1,
               label="model (closed-loop)")
    ax[0].plot(ms, lz[:, vi, cV], color=C_L0, lw=0.9, ls="--",
               label="L0 (freeze at start)")
    ax[0].set_ylabel(f"V (mV)   comp#{cV}")
    ax[0].legend(ncol=3, fontsize=8, loc="upper right")

    ax[1].plot(ms, tr[:, ci, cc], color=C_TRUE, lw=1.5, label="true cai")
    ax[1].plot(ms, pr[:, ci, cc], color=C_MODEL, lw=1.1, label="model")
    ax[1].plot(ms, lz[:, ci, cc], color=C_L0, lw=0.9, ls="--", label="L0")
    ax[1].set_ylabel(f"log10[Ca2+]   comp#{cc}")
    ax[1].legend(ncol=3, fontsize=8, loc="upper right")

    # 误差指标在**归一化空间**算（物理单位混算会让比值失真上千倍）
    e_m = np.abs(pred - true).mean((1, 2))
    e_l = np.abs(l0 - true).mean((1, 2))
    ax[2].plot(ms, e_m, color=C_MODEL, lw=1.2, label="model error")
    ax[2].plot(ms, e_l, color=C_L0, lw=1.0, ls="--", label="L0 error")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("|error| (normalized)")
    ax[2].set_xlabel("time since rollout start (ms)")
    ax[2].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Closed-loop prediction vs truth  -  {regime}  -  MEDIAN trial {trial}"
        f" of {len(ids)}  -  k={L} steps = {L * ds.dt:.1f} ms  -  te_s3",
        fontsize=10)
    fig.tight_layout()
    p = os.path.join(figdir, f"trace_{regime}_t{trial}.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"  → {p}")
    return dict(regime=regime, trial=trial, n=len(ids),
                err_model=float(em[j]), err_l0=float(el[j]), iqr=float(IQR))


def fig_ribbon(m, ds, sc, figdir: str, L: int, t0: int):
    """跨 trial 的模型/L0 比值分布 —— 比单条轨迹更有说服力。"""
    ks = [10, 40, 200, 500, 1000, L - 2]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for reg, col in (("plateau", "#2ca02c"), ("inter", "#ff7f0e"),
                     ("linear", "#7f7f7f")):
        ids, P_, Y_, B_ = regime_rollouts(m, ds, sc, reg, L, t0)
        em = np.abs(P_ - Y_).mean((2, 3))          # (n, L)
        el = np.abs(B_ - Y_).mean((2, 3))
        rat = em / np.maximum(el, 1e-12)
        med = [np.median(rat[:, k]) for k in ks]
        lo = [np.percentile(rat[:, k], 25) for k in ks]
        hi = [np.percentile(rat[:, k], 75) for k in ks]
        ax[0].plot(ks, med, "o-", color=col, label=f"{reg} (n={len(ids)})")
        ax[0].fill_between(ks, lo, hi, color=col, alpha=0.15)
    ax[0].axhline(1.0, color="k", lw=1.0, ls="--")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("horizon k (steps)")
    ax[0].set_ylabel("model / L0   (median, IQR band)")
    ax[0].set_title("Paired comparison vs persistence baseline", fontsize=10)
    ax[0].legend(fontsize=8)

    ids, P_, Y_, B_ = regime_rollouts(m, ds, sc, "plateau", L, t0)
    em = np.abs(P_ - Y_).mean((1, 2, 3))
    el = np.abs(B_ - Y_).mean((1, 2, 3))
    rat = em / np.maximum(el, 1e-12)
    ax[1].hist(np.log10(rat), bins=14, color="#2ca02c", alpha=0.8)
    ax[1].axvline(0.0, color="k", lw=1.2, ls="--", label="parity (ratio=1)")
    ax[1].axvline(np.log10(np.median(rat)), color="#d62728", lw=1.6,
                  label=f"median = {np.median(rat):.2f}x")
    ax[1].set_xlabel("log10(model / L0)   averaged over the whole "
                     f"{L - 1} steps ({(L - 1) * ds.dt:.0f} ms)")
    ax[1].set_ylabel("# trials")
    ax[1].set_title("plateau regime, per-trial spread", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.suptitle("How consistent is the win?  (each point/bar = one trial)",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(figdir, "ribbon_ratio.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"  → {p}")


def fig_snippet(m, ds, sc, regime: str, figdir: str, span_ms: float = 30.0,
                trial: int | None = None, t0: int = 4000):
    """短窗口放大图：看清波形细节与相位。"""
    ci = ds.meta["factor_names"].index("cai")
    vi = ds.meta["factor_names"].index("V")
    L = int(span_ms / ds.dt)
    if trial is None:
        trial = pick_trial(ds, regime, L)
    r = rollout_seq(m, ds, sc, trial, L, t0)
    tr = denorm(r["true"][None], sc)[0]
    pr = denorm(r["pred"][None], sc)[0]
    l0 = denorm(r["l0"][None], sc)[0]
    ms = np.arange(tr.shape[0]) * ds.dt
    cV = int(np.argmax(tr[:, vi, :].std(0)))

    fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
    ax[0].plot(ms, tr[:, vi, cV], color=C_TRUE, lw=2.0, label="true")
    ax[0].plot(ms, pr[:, vi, cV], color=C_MODEL, lw=1.2, label="model (closed-loop)")
    ax[0].plot(ms, l0[:, vi, cV], color=C_L0, lw=0.9, ls="--", label="L0")
    ax[0].set_ylabel(f"V (mV)   comp#{cV}")
    ax[0].legend(ncol=3, fontsize=8)
    ci2 = int(np.argmax(tr[:, ci, :].std(0)))
    ax[1].plot(ms, tr[:, ci, ci2], color=C_TRUE, lw=2.0, label="true")
    ax[1].plot(ms, pr[:, ci, ci2], color=C_MODEL, lw=1.2, label="model")
    ax[1].plot(ms, l0[:, ci, ci2], color=C_L0, lw=0.9, ls="--", label="L0")
    ax[1].set_ylabel(f"log10[Ca2+]   comp#{ci2}")
    ax[1].set_xlabel("time since rollout start (ms)")
    ax[1].legend(ncol=3, fontsize=8)
    fig.suptitle(f"Zoom {span_ms:.0f} ms  -  {regime}  -  trial {trial}",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(figdir, f"snippet_{regime}_t{trial}.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"  → {p}")


def fig_error_curve(figdir: str):
    """从已有 eval JSON 画 误差-vs-k（双对数）。"""
    rep = os.path.join(ROOT, "results", "reports")
    js, lbl = None, None
    for cand, name in (("eval_last_t2_te_s3.json", "last.pt（最终）"),
                       ("eval_stageA_t2_te_s3.json", "stageA（仅教师强制）")):
        p = os.path.join(rep, cand)
        if os.path.exists(p):
            js = json.load(open(p, encoding="utf-8"))
            lbl = name
            break
    if js is None:
        log("  （跳过误差曲线：没有 eval JSON）")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ks = sorted(int(k) for k in js["overall"])
    for ax, key, title in ((axes[0], "overall", "overall (n=200)"),
                           (axes[1], None, "per regime")):
        if key:
            e = [js[key][str(k)]["model"] for k in ks]
            l0 = [js[key][str(k)]["L0"] for k in ks]
            ax.plot(ks, e, "o-", color=C_MODEL, label="model")
            ax.plot(ks, l0, "s--", color=C_L0, label="L0 (copy last)")
        else:
            for name, col in (("plateau", "#2ca02c"), ("inter", "#ff7f0e"),
                              ("linear", "#7f7f7f")):
                if name not in js:
                    continue
                e = [js[name][str(k)]["model"] for k in ks]
                ax.plot(ks, e, "o-", color=col, label=name)
            l0 = [js["overall"][str(k)]["L0"] for k in ks]
            ax.plot(ks, l0, "s--", color=C_L0, lw=1.0, label="L0 (overall)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("horizon k (steps; dt=0.025 ms, so k=2000 is 50 ms)")
        ax.set_ylabel("absolute error (normalized)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(f"Error growth vs horizon  -  {lbl}  -  te_s3", fontsize=10)
    fig.tight_layout()
    p = os.path.join(figdir, "error_vs_k.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"  → {p}")


def fig_nonlinearity(m, ds, sc, figdir: str, n_batch: int = 6):
    """非线性指纹：优势随状态变化速率的变化（柱状图）。"""
    ci = ds.meta["factor_names"].index("cai")
    sp = D2.Sampler2(ds, sc, batch=32, length=256, seed=7)
    E, L0, D = [], [], []
    for _ in range(n_batch):
        s0, u, ug, th, s_true, x = _torch(sp.batch_pair())
        with torch.no_grad():
            z = m.encode(s_true, u, th, x, ug)
            zp = torch.cat([z[:, :1], z[:, :-1]], 1)
            zn = m.predict(zp, u, th, x, ug)
            pred = s_true[:, :-1] + (m.decode(zn[:, 1:], th, x)
                                     - m.decode(zp[:, 1:], th, x))
        tgt = s_true[:, 1:]
        E.append((pred - tgt).abs().mean((2, 3)).numpy().ravel())
        L0.append((s_true[:, :-1] - tgt).abs().mean((2, 3)).numpy().ravel())
        D.append((tgt[:, :, ci, :] - s_true[:, :-1, ci, :]).abs()
                 .max(2).values.numpy().ravel())
    E, L0, D = map(np.concatenate, (E, L0, D))
    qs = np.percentile(D, [0, 50, 75, 90, 97, 99.5, 100])
    qs[-1] += 1e-9
    lab, rat, fr = [], [], []
    for a, b in zip(qs[:-1], qs[1:]):
        k = (D >= a) & (D < b)
        if k.sum() < 100:
            continue
        lab.append("0" if b <= 1e-12 else f"{a:.0e}–{b:.0e}")
        rat.append(E[k].mean() / L0[k].mean())
        fr.append(100 * k.mean())

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    cols = ["#d62728" if v < 1 else "#7f7f7f" for v in rat]
    bars = ax.bar(range(len(rat)), rat, color=cols, width=0.62)
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.set_xticks(range(len(rat)))
    ax.set_xticklabels(lab, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("model / L0   (<1 = model better)")
    ax.set_xlabel("bin of per-frame |d cai|  (state change rate)")
    for b, v, f in zip(bars, rat, fr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03,
                f"{v:.2f}\n({f:.1f}%)", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(rat) * 1.30)
    ax.set_title("Nonlinearity fingerprint: the faster the state changes,\n"
                 "the larger the model's advantage (a linear model = flat bars)",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(figdir, "nonlinearity.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"  → {p}")


def fig_latent(m, ds, sc, figdir: str):
    """latent 轨迹 2D 投影，按「过去 10 ms 有无 bAP」着色 —— 展示记忆。"""
    Z, Y = [], []
    w = int(round(10.0 / ds.dt))
    for c in range(4):
        sp = D2.Sampler2(ds, sc, batch=8, length=1200, seed=200 + c)
        s0, u, ug, th, s_true, x = _torch(sp.batch_pair())
        with torch.no_grad():
            z = m.encode(s_true, u, th, x, ug).numpy()
        g = ug.numpy()[..., 0]
        lab = np.zeros_like(g)
        for b in range(g.shape[0]):
            cs = np.concatenate([[0.0], np.cumsum(g[b])])
            t = np.arange(g.shape[1])
            lab[b] = cs[t + 1] - cs[np.maximum(t - w, 0)]
        Z.append(z.reshape(-1, z.shape[-1]))
        Y.append((lab > 0).ravel().astype(int))
    Z = np.concatenate(Z)
    Y = np.concatenate(Y)
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Zn = (Z - mu) / sd
    # 2D PCA
    U, S, Vt = np.linalg.svd(Zn - Zn.mean(0), full_matrices=False)
    pc = (Zn - Zn.mean(0)) @ Vt[:2].T

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4),
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    for v, col, nm in ((0, "#9ecae1", "无 bAP"), (1, "#d62728", "有 bAP")):
        k = Y == v
        ax[0].scatter(pc[k, 0], pc[k, 1], s=1.2, c=col, alpha=0.35,
                      label=f"{nm} ({100 * k.mean():.0f}%)", rasterized=True)
    ax[0].set_xlabel(f"PC1 ({100 * S[0] ** 2 / (S ** 2).sum():.0f}%)")
    ax[0].set_ylabel(f"PC2 ({100 * S[1] ** 2 / (S ** 2).sum():.0f}%)")
    ax[0].legend(markerscale=8, fontsize=8)
    ax[0].set_title("Latent space colored by 'bAP in past 10 ms'", fontsize=10)

    # 每个 latent 维对 bAP 的可分性（单维 AUC 的简易代理：均值差/合并标准差）
    d = []
    for j in range(Zn.shape[1]):
        a, b = Zn[Y == 1, j], Zn[Y == 0, j]
        d.append(abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9))
    ax[1].bar(range(len(d)), d, color="#4c72b0", width=0.7)
    ax[1].set_xlabel("latent dimension")
    ax[1].set_ylabel("effect size (Cohen's d)")
    ax[1].set_title("Which dims encode bAP history", fontsize=10)
    fig.suptitle("Evidence of cross-time memory carried inside the latent", fontsize=10)
    fig.tight_layout()
    p = os.path.join(figdir, "latent_memory.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"  → {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="t2v4_anchor/last.pt")
    ap.add_argument("--data", default="te_s3")
    ap.add_argument("--tier", default="t2")
    ap.add_argument("--k", type=int, default=2000, help="闭环步数（默认 2000=50ms）")
    ap.add_argument("--t0", type=int, default=4000, help="起点步（≥ t_settle）")
    ap.add_argument("--snippet-ms", type=float, default=30.0)
    ap.add_argument("--mem-mb", type=float, default=2600.0)
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=a.mem_mb)
    figdir = FIG
    os.makedirs(figdir, exist_ok=True)
    ds = D2.Dataset2(DATA, a.data, tier=a.tier)
    m, sc, _ = load_model(a.ckpt, ds, a.tier)
    log(f"{ds}")

    t_settle_steps = int(round(ds.meta["t_settle"] / ds.dt))
    if a.t0 < t_settle_steps:
        log(f"  !! t0={a.t0} 落在静置瞬态内（t_settle={t_settle_steps} 步），自动修正")
        a.t0 = t_settle_steps

    log("\n[1/6] 闭环轨迹（中位数 trial）")
    rows = []
    for reg in ("plateau", "inter", "linear"):
        rows.append(fig_trace(m, ds, sc, reg, a.k, figdir, a.t0))

    log("\n[2/6] 跨 trial 比值分布")
    fig_ribbon(m, ds, sc, figdir, a.k, a.t0)

    log("\n[3/6] 短窗口放大（与长轨迹同一 trial）")
    for r_ in rows:
        fig_snippet(m, ds, sc, r_["regime"], figdir, a.snippet_ms,
                    trial=r_["trial"], t0=a.t0)

    log("\n[4/6] 误差-视界曲线")
    fig_error_curve(figdir)

    log("\n[5/6] 非线性指纹")
    fig_nonlinearity(m, ds, sc, figdir)

    log("\n[6/6] latent 记忆")
    fig_latent(m, ds, sc, figdir)

    log(f"\nsummary (k={a.k} = {a.k * ds.dt:.1f} ms, NORMALIZED space):")
    for r in rows:
        log(f"  {r['regime']:<8} trial {r['trial']:>3} (median of {r['n']})  "
            f"model {r['err_model']:.5f}  L0 {r['err_l0']:.5f}  "
            f"ratio {r['err_model'] / r['err_l0']:.3f}  "
            f"(trial-to-trial IQR {r['iqr']:.4f})")
    log(f"\n全部图在 {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
