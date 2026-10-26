"""单轮试验归档：评估一个 ckpt，把结果写进 results/rounds/<ROUND>/。

目标：每轮独立成目录，互不覆盖，一眼能找。

    results/rounds/R1_anchorC/
        README.md          自动生成的结论摘要
        metrics.json       所有数值指标（机器可读）
        logs/              训练日志副本
        figures/           该轮的图
        ckpt/              权重副本（<2 MB）

用法（PYTHONPATH=$PWD）：
    python -m tools.round_report --run t2v5_ancC --round R1_anchorC \
        --title "③b 闭环段 latent 自洽"
    python -m tools.round_report --run t2v4_anchor --round R0_baseline_t2v4 \
        --title "v4 基线" --k 2000
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import torch                           # noqa: E402

from world.core import safety          # noqa: E402
from world.learn import data2 as D2    # noqa: E402
from tools.plot_v4 import load_model, denorm   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESC = os.path.join(ROOT, "results", "checkpoints")
REPR = os.path.join(ROOT, "results", "reports")
LOGS = os.path.join(ROOT, "results", "logs")
ROUNDS = os.path.join(ROOT, "results", "rounds")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_TRUE, C_MODEL, C_L0 = "#222222", "#d62728", "#1f77b4"
C_P = {"plateau": "#2ca02c", "inter": "#ff7f0e", "linear": "#7f7f7f"}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 指标
def latent_drift(m, ds, sc, trial: int, t0: int, L: int):
    """闭环 latent 的 |z| 与 cos（衡量是否逃出流形）。"""
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z)).float()  # noqa: E731
    sp = D2.Sampler2(ds, sc, batch=1, length=1, seed=0)
    s0, u, ug, th, st, x = sp.batch_window(t0, L, 1, off=trial)
    uT = torch.from_numpy(
        np.ascontiguousarray(u.transpose(0, 1, 3, 2))).float()
    with torch.no_grad():
        zt = m.encode(T(st), uT, T(th), T(x), T(ug))
        z = m.encode(T(s0).unsqueeze(1), uT[:, :1], T(th), T(x),
                     T(ug)[:, :1]).squeeze(1)
        zp = []
        for t in range(L - 1):
            j = t + 1
            zn = m.predict(z.unsqueeze(1), uT[:, j].unsqueeze(1), T(th), T(x),
                           T(ug)[:, j].unsqueeze(1)).squeeze(1)
            zp.append(zn.numpy()[0])
            z = zn
    ZP = np.stack(zp)
    ZT = zt[0].numpy()[1:]
    out = {}
    for k in (200, 500, 1000, 2000):
        if k > len(ZP):
            continue
        a, b = ZP[k - 1], ZT[k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out[str(k)] = dict(norm_pred=float(na), norm_true=float(nb),
                           cos=float(a @ b / (na * nb + 1e-12)))
    return out


def latent_drift_mt(m, ds, sc, ids, t0: int, L: int):
    """跨 trial 的 latent 漂移统计（中位数 + bootstrap 95% CI）。

    单 trial 的 cos 波动很大（实测同一模型逐 trial 可从 0.98 掉到 0.4），
    只看「中位误差 trial」会得出错误结论，必须看全体 trial 的中位数。
    """
    from tools.latent_ci import boot_ci, cos_drift
    per, rzs = {}, {}
    for t in ids:
        r = cos_drift(m, ds, sc, int(t), t0, L, KS_MT)
        for k, v in r.items():
            per.setdefault(k, []).append(v["cos"])
            rzs.setdefault(k, []).append(v["rz"])
    out = {}
    for k in sorted(per):
        c, lo, hi = boot_ci(per[k])
        rz, rzlo, rzhi = boot_ci(rzs[k])
        out[str(k)] = dict(cos=c, cos_lo=lo, cos_hi=hi, cos_all=per[k],
                           rz=rz, rz_lo=rzlo, rz_hi=rzhi,
                           frac_below05=float(np.mean(np.asarray(per[k]) < 0.5)),
                           n=len(per[k]))
    return out


KS_MT = (200, 500, 1000, 2000)


def rollout_metrics(m, ds, sc, regime: str, L: int, t0: int, n: int = 12):
    """闭环 rollout：误差比、V 上限、cai 事件 P/R。"""
    from tools.plot_v4 import regime_rollouts
    ids, P, Y, B = regime_rollouts(m, ds, sc, regime, L + 1, t0, n=n)
    fn = ds.meta["factor_names"]
    vi, ci = fn.index("V"), fn.index("cai")
    Tr, Pr, L0 = denorm(Y, sc), denorm(P, sc), denorm(B, sc)

    em = np.abs(P - Y).mean((1, 2, 3))
    el = np.abs(L0 - Y).mean((1, 2, 3))
    j = int(np.argsort(em)[len(em) // 2])          # 中位 trial

    ev = {"plateau": 0, "inter": 0, "linear": 0}
    res = dict(n_trials=len(ids), median_trial=int(ids[j]),
               ratio_median=float(np.median(em / np.maximum(el, 1e-12))),
               ratio_mean=float(np.mean(em / np.maximum(el, 1e-12))),
               ratio_iqr=float(np.percentile(em / np.maximum(el, 1e-12), 75)
                               - np.percentile(em / np.maximum(el, 1e-12), 25)),
               rmse_model=float(np.sqrt(((P - Y) ** 2).mean())),
               rmse_l0=float(np.sqrt(((L0 - Y) ** 2).mean())))

    if regime == "plateau":
        TH = -3.10
        tp = fp = fnn = 0
        for i in range(len(ids)):
            at = Tr[i][:, ci, :].max(1) >= TH
            ap = Pr[i][:, ci, :].max(1) >= TH
            tp += int((at & ap).sum())
            fp += int((~at & ap).sum())
            fnn += int((at & ~ap).sum())
        res["cai_precision"] = tp / max(tp + fp, 1)
        res["cai_recall"] = tp / max(tp + fnn, 1)
        res["V_max_model"] = float(Pr[:, :, vi, :].max())
        res["V_max_true"] = float(Tr[:, :, vi, :].max())
        res["V_rmse_mV"] = float(np.sqrt(
            ((Pr[:, :, vi, :] - Tr[:, :, vi, :]) ** 2).mean()))
        res["cai_rmse"] = float(np.sqrt(
            ((Pr[:, :, ci, :] - Tr[:, :, ci, :]) ** 2).mean()))
    res["_rollouts"] = (ids, P, Y, B, j) if regime == "plateau" else None
    return res


def read_eval(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    if "overall" not in d:
        return None
    return d


def v6_selfcontained(m, ds, sc, ks, t0: int, n_seq: int = 64, n_rep: int = 4,
                     sane_max: float = 100.0):
    """自己算 V6 配对评测（与 train2 cmd_eval 同协议：固定 t0、同一批序列）。

    不依赖 reports/eval_*.json —— 那个文件名是固定的，多轮会互相覆盖，
    归档会串味。
    """
    T = lambda z: torch.from_numpy(np.ascontiguousarray(z)).float()  # noqa: E731
    L = max(ks) + 1
    n_used = min(n_seq * n_rep, ds.N)
    tot = {k: [0.0, 0.0] for k in ks}
    dvg = {k: False for k in ks}
    by = {r: {k: [0.0, 0.0] for k in ks}
          for r in ("linear", "inter", "plateau")}
    cnt_reg = {r: 0 for r in by}
    cnt = 0
    for off in range(0, n_used, n_seq):
        n = min(n_seq, n_used - off)
        sp = D2.Sampler2(ds, sc, batch=n, length=1, seed=0)
        s0, u, ug, th, st, x = sp.batch_window(t0, L, n, off=off)
        uT = torch.from_numpy(
            np.ascontiguousarray(u.transpose(0, 1, 3, 2))).float()
        with torch.no_grad():
            outs = m.rollout(T(s0), uT, T(th), T(x), ug_seq=T(ug))
        stT = T(st)
        regs = ds.regimes[(np.arange(n) + off) % ds.N]
        for k in ks:
            ol = outs[:, k - 1]
            o = ol.detach().numpy()
            if (~np.isfinite(o)).any() or np.abs(o).max() > sane_max:
                dvg[k] = True
            e = float((ol - stT[:, k]).abs().mean())
            e0 = float((T(s0) - stT[:, k]).abs().mean())
            tot[k][0] += e * n
            tot[k][1] += e0 * n
            for r in by:
                msk = (regs == r)
                jn = int(msk.sum())
                if jn:
                    er = float((outs[msk, k - 1] - stT[msk, k]).abs().mean())
                    e0r = float((T(s0)[msk] - stT[msk, k]).abs().mean())
                    by[r][k][0] += er * jn
                    by[r][k][1] += e0r * jn
        for r in by:
            cnt_reg[r] += int((regs == r).sum())
        cnt += n

    def tab(rows, nd):
        out = {}
        for k in ks:
            e, e0 = rows[k]
            e, e0 = e / max(nd, 1), e0 / max(nd, 1)
            out[str(k)] = dict(model=e, L0=e0,
                               ratio=e / max(e0, 1e-12),
                               diverged=bool(dvg.get(k)))
        return out

    res = dict(n=cnt, horizons=list(ks), overall=tab(tot, cnt), counts=cnt_reg)
    for r in by:
        res[r] = tab(by[r], max(cnt_reg[r], 1))
    return res


# ---------------------------------------------------------------- 画图
def fig_compare(cks: dict, ds, sc, fdir: str, L: int, t0: int):
    """所有轮次的 模型/L0 随 k 变化（用各自 eval JSON）。"""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, d in cks.items():
        if not d:
            continue
        ks = sorted(int(k) for k in d["overall"])
        for a, key in ((ax[0], "overall"), (ax[1], "plateau")):
            if key not in d:
                continue
            r = [d[key][str(k)]["model"] / max(d[key][str(k)]["L0"], 1e-12)
                 for k in ks]
            a.plot(ks, r, "o-", label=f"{name} ({key})" if a is ax[1] else name,
                   lw=1.4, ms=4)
    for a in ax:
        a.axhline(1.0, color="k", lw=1.0, ls="--")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("horizon k (steps; k=2000 is 50 ms)")
        a.set_ylabel("model / L0   (<1 = model better)")
        a.legend(fontsize=7)
    ax[0].set_title("overall", fontsize=10)
    ax[1].set_title("plateau regime", fontsize=10)
    fig.suptitle("Paired comparison across rounds", fontsize=10)
    fig.tight_layout()
    p = os.path.join(fdir, "compare_rounds.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"  -> {p}")


def fig_round(m, ds, sc, regime: str, L: int, t0: int, fdir: str, n=12):
    """本轮自己的：中位 trial 轨迹 + 逐 trial 比值分布。"""
    from tools.plot_v4 import regime_rollouts
    ids, P, Y, B = regime_rollouts(m, ds, sc, regime, L + 1, t0, n=n)
    fn = ds.meta["factor_names"]
    vi, ci = fn.index("V"), fn.index("cai")
    Tr, Pr, L0 = denorm(Y, sc), denorm(P, sc), denorm(B, sc)
    em = np.abs(P - Y).mean((1, 2, 3))
    el = np.abs(L0 - Y).mean((1, 2, 3))
    j = int(np.argsort(em)[len(em) // 2])

    t3, p3, l3 = Tr[j], Pr[j], L0[j]
    ms = np.arange(t3.shape[0]) * ds.dt
    cV = int(np.argmax(t3[:, vi, :].std(0)))
    cc = int(np.argmax(t3[:, ci, :].std(0)))

    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))
    for col, (fi, cp, lab) in enumerate((
            (vi, cV, "V (mV)"), (ci, cc, "log10[Ca2+]"))):
        a = ax[0, col]
        a.plot(ms, t3[:, fi, cp], color=C_TRUE, lw=1.6, label="true")
        a.plot(ms, p3[:, fi, cp], color=C_MODEL, lw=1.1, label="model")
        a.plot(ms, l3[:, fi, cp], color=C_L0, lw=0.9, ls="--", label="L0")
        a.set_title(f"{lab}  comp#{cp}", fontsize=9)
        a.set_xlabel("time since rollout start (ms)")
        a.legend(fontsize=7)

    a = ax[1, 0]
    a.plot(ms, np.abs(p3 - t3).mean((1, 2)), color=C_MODEL, lw=1.2,
           label="model err")
    a.plot(ms, np.abs(l3 - t3).mean((1, 2)), color=C_L0, lw=1.0, ls="--",
           label="L0 err")
    a.set_yscale("log")
    a.set_xlabel("time since rollout start (ms)")
    a.set_ylabel("|err| (normalized)")
    a.legend(fontsize=7)

    a = ax[1, 1]
    rat = em / np.maximum(el, 1e-12)
    a.hist(np.log10(rat), bins=12, color="#4c72b0", alpha=0.85)
    a.axvline(0, color="k", lw=1.2, ls="--", label="parity")
    a.axvline(np.log10(np.median(rat)), color="#d62728", lw=1.5,
              label=f"median {np.median(rat):.2f}x")
    a.set_xlabel(f"log10(model/L0) over {L} steps ({L * ds.dt:.0f} ms)")
    a.set_ylabel("# trials")
    a.legend(fontsize=7)

    fig.suptitle(f"{regime}  -  median trial {ids[j]} of {len(ids)}  -  "
                 f"k={L} steps = {L * ds.dt:.0f} ms", fontsize=10)
    fig.tight_layout()
    p = os.path.join(fdir, f"round_{regime}.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"  -> {p}")


# ---------------------------------------------------------------- 主流程
def write_readme(rd: str, met: dict) -> str:
    """给本轮生成人读的 README.md（含判据判定）。"""
    cfg = met.get("cfg") or {}
    ld = met.get("latent_drift") or {}
    pl = met.get("plateau") or {}
    v6 = met.get("v6") or {}
    ov = v6.get("overall") or {}
    ks = sorted(int(k) for k in ov) if ov else []

    L = ["# " + (met.get("title") or met["round"]), "",
         f"- run: `{met['run']}`  ·  步数 {met.get('step')}  ·  "
         f"单步 σ = **{met.get('sigma'):.6f}**",
         f"- 数据 {met['data']} / tier {met['tier']}", "",
         "## 配置", "", "| 项 | 值 |", "|---|---|"]
    for k in ("length", "batch", "cl_steps", "cl_start", "cl_stage_a_steps",
              "anchor_weight", "anchor_cl", "anchor_every", "event_weight",
              "regime_weights", "hot_frac", "d_z", "hidden", "n_layers"):
        if cfg.get(k) is not None:
            L.append(f"| {k} | {cfg[k]} |")

    if ld:
        L += ["", "## latent 是否逃出流形（闭环，plateau 档跨 trial）", "",
              "| k | cos 中位 | 95% CI | \\|z\\| 比 | cos<0.5 占比 |",
              "|---|---|---|---|---|"]
        for k in sorted(int(x) for x in ld):
            v = ld[str(k)]
            L.append(f"| {k} | **{v['cos']:+.3f}** | "
                     f"[{v['cos_lo']:+.3f}, {v['cos_hi']:+.3f}] | "
                     f"{v['rz']:.2f} | {v['frac_below05']:.2f} |")
        tail = ld[str(max(int(x) for x in ld))]
        L += ["", f"⇒ k 最大处 cos 中位 = **{tail['cos']:+.3f}**"
                  f"（>0.5 视为未逃出）；\\|z\\| 比为预测/真值，1.0 = 无膨胀。",
              "", "> ⚠️ 本表是 **12 个 trial 的中位数**。单 trial 的 cos 波动极大"
              "（实测同模型逐 trial 可 0.98→0.40），只看一个 trial 会得出错误结论。"]

    if ov:
        L += ["", "## V6 配对评测（模型 / L0）", "",
              "| k | ms | 模型 | L0 | 比值 |", "|---|---|---|---|---|"]
        for k in ks:
            r = ov[str(k)]
            L.append(f"| {k} | {k * 0.025:.1f} | {r['model']:.4f} | "
                     f"{r['L0']:.4f} | **{r['ratio']:.3f}**"
                     + (" ⚠️DIV" if r.get("diverged") else "") + " |")
        if "200" in ov:
            L += ["", f"⇒ **a(200) = {ov['200']['ratio']:.3f}** "
                      f"({'✅' if ov['200']['ratio'] < 1.0 else '❌'})"]
        for reg, name in (("plateau", "plateau"), ("inter", "inter"),
                          ("linear", "linear")):
            d = v6.get(reg) or {}
            if not d:
                continue
            row = "  ".join(f"k={k}:{d[str(k)]['ratio']:.2f}" for k in ks)
            L.append(f"- {name}: {row}")

    if pl:
        L += ["", "## 闭环物理合理性（plateau 档）", "",
              "| 项 | 值 |", "|---|---|",
              f"| V 上限（模型 / 真值）| {pl.get('V_max_model', float('nan')):.1f}"
              f" / {pl.get('V_max_true', float('nan')):.1f} mV |",
              f"| cai precision | {pl.get('cai_precision', float('nan')):.3f} |",
              f"| cai recall | {pl.get('cai_recall', float('nan')):.3f} |",
              f"| V RMSE | {pl.get('V_rmse_mV', float('nan')):.2f} mV |",
              f"| cai RMSE | {pl.get('cai_rmse', float('nan')):.4f} |"]

    ce = met.get("ca_event") or {}
    if ce:
        L += ["", "## Ca 事件（**无阈值**指标，尺度无关）", "",
              "> 上表的 `cai precision/recall` 用固定绝对阈值 `cai >= -3.10`，"
              "**幅度整体变小就会被判成漏报**（R3 的 0.155/0.273 就是这个伪影）。"
              "本节改用「相对自身动态范围的峰检测」，不依赖任何绝对阈值。"
              f"统计范围：plateau 档 {ce.get('n_trials')} 个 trial、"
              f"k={met.get('ca_k', '')} 步 "
              f"（{met.get('ca_k', 0) * 0.025:.0f} ms）、"
              f"{ce.get('n_true_peaks')} 个真值峰。", "",
              "| 项 | 值 | 说明 |", "|---|---|---|",
              f"| 峰数 真/预/匹配 | {ce.get('n_true_peaks')} / "
              f"{ce.get('n_pred_peaks')} / {ce.get('n_matched')} | — |",
              f"| 峰 precision | **{ce.get('peak_precision', float('nan')):.3f}**"
              f" | 预测的峰里有多少对得上 |",
              f"| 峰 recall | **{ce.get('peak_recall', float('nan')):.3f}**"
              f" | 真值峰里有多少被找到 |",
              f"| 峰 F1 | {ce.get('peak_f1', float('nan')):.3f} | 检出能力综合 |",
              f"| 幅度比 预/真 | **{ce.get('amp_ratio', float('nan')):.3f}**"
              f" | 1.0 无偏，>1 过冲，<1 不足 |",
              f"| 峰时误差 | {ce.get('dt_ms', float('nan')):.2f} ms | 越小越好 |",
              f"| 事件窗 slope | {ce.get('slope', float('nan')):.3f} | "
              f"1.0 无偏 |",
              f"| 事件窗 r | {ce.get('r_event', float('nan')):.3f} | "
              f"事件附近的波形相关 |",
              f"| 全窗 r | {ce.get('r_all', float('nan')):.3f} | 整体波形相关 |",
              f"| nrmse | {ce.get('nrmse', float('nan')):.3f} | "
              f"RMSE / 动态范围 |"]

    L += ["", "## 文件", "",
          "- `metrics.json`  全部数值（机器可读）",
          "- `figures/round_*.png`  各档中位 trial 轨迹 + 逐 trial 比值分布",
          "- `logs/`  训练日志", "- `ckpt/`  权重副本", ""]
    p = os.path.join(rd, "README.md")
    open(p, "w", encoding="utf-8").write("\n".join(L))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint 子目录名")
    ap.add_argument("--round", required=True, help="results/rounds/ 下的目录名")
    ap.add_argument("--title", default="")
    ap.add_argument("--data", default="te_s3")
    ap.add_argument("--tier", default="t2")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--t0", type=int, default=4000)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--ca-k", type=int, default=8000,
                    help="无阈值 Ca 事件指标的 rollout 步数（0 = 跳过）")
    ap.add_argument("--n-trials-ca", type=int, default=999,
                    help="Ca 事件统计用几个 plateau trial，999 = 全部")
    ap.add_argument("--mem-mb", type=float, default=3200.0)
    ap.add_argument("--no-eval", action="store_true",
                    help="跳过 V6（用已有 reports/eval_*.json）")
    a = ap.parse_args(argv)

    safety.preflight(limit_mb=a.mem_mb)
    rd = os.path.join(ROUNDS, a.round)
    fdir = os.path.join(rd, "figures")
    os.makedirs(fdir, exist_ok=True)
    ds = D2.Dataset2(os.path.join(ROOT, "data"), a.data, tier=a.tier)

    ckpt = os.path.join(a.run, "last.pt")
    m, sc, cfg = load_model(ckpt, ds, a.tier)
    ck = torch.load(os.path.join(RESC, ckpt), map_location="cpu",
                    weights_only=False)

    met = dict(round=a.round, title=a.title, run=a.run, data=a.data,
               tier=a.tier, step=ck.get("step"), sigma=ck.get("sigma"),
               cfg={k: cfg.get(k) for k in
                    ("length", "batch", "cl_steps", "cl_start",
                     "cl_stage_a_steps", "anchor_weight", "anchor_cl",
                     "anchor_every", "event_weight", "event_tau",
                     "regime_weights", "hot_frac",
                     "d_z", "hidden", "n_layers")})

    log("\n[1/4] latent 漂移（跨 trial 中位数，非单个 trial）")
    ids_pl = [t["i"] for t in ds.meta["trials"]
              if t.get("regime") == "plateau"][:a.n_trials]
    met["latent_drift"] = latent_drift_mt(m, ds, sc, ids_pl, a.t0, 2002)
    met["latent_drift_single"] = latent_drift(m, ds, sc, int(ids_pl[0]),
                                              a.t0, 2002)
    for k, v in met["latent_drift"].items():
        log(f"  k={k}: cos 中位={v['cos']:+.3f} "
            f"[{v['cos_lo']:+.3f},{v['cos_hi']:+.3f}]  "
            f"|z|比={v['rz']:.3f}  cos<0.5 占比={v['frac_below05']:.2f}")

    log("\n[2/4] 闭环指标（plateau）")
    pm = rollout_metrics(m, ds, sc, "plateau", a.k, a.t0, a.n_trials)
    pm.pop("_rollouts", None)
    met["plateau"] = pm
    log(f"  比值中位 {pm['ratio_median']:.3f}  V上限 {pm.get('V_max_model'):.1f}"
        f"(真值 {pm.get('V_max_true'):.1f})  "
        f"cai P={pm.get('cai_precision'):.3f} R={pm.get('cai_recall'):.3f}")

    for reg in ("inter", "linear"):
        r = rollout_metrics(m, ds, sc, reg, a.k, a.t0, a.n_trials)
        r.pop("_rollouts", None)
        met[reg] = r
        log(f"  {reg}: 比值中位 {r['ratio_median']:.3f}")

    log("\n[3/5] V6 配对评测（自算，不依赖 eval JSON）")
    ks = [10, 40, 200, 1000, 2000, 4000, 8000]
    met["v6"] = v6_selfcontained(m, ds, sc, ks, a.t0)
    ov = met["v6"]["overall"]
    log(f"  {'k':>6}{'model':>10}{'L0':>10}{'ratio':>9}")
    for k in ks:
        r = ov[str(k)]
        log(f"  {k:6d}{r['model']:10.4f}{r['L0']:10.4f}{r['ratio']:9.3f}"
            + ("  ⚠️DIV" if r["diverged"] else ""))
    log(f"  a(200) = {ov['200']['ratio']:.3f}")

    log("\n[4/5] 图表")
    for reg in ("plateau", "inter", "linear"):
        fig_round(m, ds, sc, reg, a.k, a.t0, fdir, a.n_trials)

    if a.ca_k > 0:
        log(f"\n[5/5] Ca 事件（无阈值，k={a.ca_k} = "
            f"{a.ca_k * ds.dt:.0f} ms）")
        from tools.ca_event_metrics import eval_model as _ca_eval
        cem, ca_ids = _ca_eval(m, ds, sc, a.ca_k, a.t0, a.n_trials_ca)
        met["ca_event"] = cem
        met["ca_k"] = a.ca_k
        log(f"  峰数 真值 {cem['n_true_peaks']} / 预测 {cem['n_pred_peaks']}"
            f" / 匹配 {cem['n_matched']}")
        log(f"  峰 P/R/F1 = {cem['peak_precision']:.3f} / "
            f"{cem['peak_recall']:.3f} / {cem['peak_f1']:.3f}")
        log(f"  幅度比 预测/真值 = {cem['amp_ratio']:.3f}（1.0 无偏）"
            f"  峰时误差 {cem['dt_ms']:.2f} ms")
        log(f"  事件窗 slope={cem['slope']:.3f} r={cem['r_event']:.3f}"
            f"  全窗 r={cem['r_all']:.3f} nrmse={cem['nrmse']:.3f}")

    # 训练日志副本
    lg = os.path.join(LOGS, f"{a.run}.log")
    if os.path.exists(lg):
        os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
        shutil.copy2(lg, os.path.join(rd, "logs", f"{a.run}.log"))
    # 权重副本
    cdir = os.path.join(RESC, a.run)
    if os.path.isdir(cdir):
        os.makedirs(os.path.join(rd, "ckpt"), exist_ok=True)
        for f in ("best.pt", "last.pt"):
            p = os.path.join(cdir, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(rd, "ckpt", f))

    with open(os.path.join(rd, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(met, f, ensure_ascii=False, indent=1, default=str)
    rp = write_readme(rd, met)
    log(f"\n=> {os.path.join(rd, 'metrics.json')}")
    log(f"=> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
