"""六轮试验总结图（供文档使用）。

生成 `results/rounds/summary_rounds.png`，4 个面板：

1. V6 overall 比值 vs k（含发散断点）
2. latent cos vs k（中位数 + 95% CI 误差棒，仅 R1 起有 CI）
3. cai 事件：precision vs recall 散点（越靠右上越好）
4. 关键标量条形图：V 上限 / a(200) / 峰 F1

    PYTHONPATH=$PWD python -m tools.fig_rounds_summary
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUNDS = os.path.join(ROOT, "results", "rounds")

# 轮次顺序 + 配色（R2 是最优解，用高亮色）
ORDER = ["R0_baseline_t2v4", "R1_anchorC", "R1b_ablation_noancC",
         "R2_event_weighting", "R3_overshoot_fix", "R4_event_w8"]
COLOR = {"R0_baseline_t2v4": "#8c8c8c",
         "R1_anchorC": "#1f77b4",
         "R1b_ablation_noancC": "#17becf",
         "R2_event_weighting": "#d62728",
         "R3_overshoot_fix": "#ff7f0e",
         "R4_event_w8": "#9467bd"}
SHORT = {"R0_baseline_t2v4": "R0 baseline",
         "R1_anchorC": "R1 +anchor-cl",
         "R1b_ablation_noancC": "R1b w/o anchor-cl",
         "R2_event_weighting": "R2 +event-w 10",
         "R3_overshoot_fix": "R3 w=5,cl=0.2",
         "R4_event_w8": "R4 w=8"}


def log(*a):
    print(*a, flush=True)


def load():
    out = {}
    for r in ORDER:
        p = os.path.join(ROUNDS, r, "metrics.json")
        if os.path.exists(p):
            out[r] = json.load(open(p, encoding="utf-8"))
    return out


def _series(block):
    ks = sorted(int(k) for k in block)
    rs = []
    for k in ks:
        v = block[str(k)]
        ok = (not v.get("diverged")) and np.isfinite(v.get("ratio", np.nan))
        rs.append(v["ratio"] if ok else np.nan)
    return np.asarray(ks, float), np.asarray(rs, float)


def main() -> int:
    mets = load()
    if not mets:
        log("没有找到 metrics.json")
        return 1

    fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.4))

    # ---- (0,0) V6 overall
    a = ax[0, 0]
    for r, m in mets.items():
        block = (m.get("v6") or {}).get("overall") or {}
        if not block:
            continue
        ks, rs = _series(block)
        a.plot(ks, rs, "o-", lw=1.5, ms=4, color=COLOR[r],
               label=SHORT[r], zorder=3 if r == "R2_event_weighting" else 2)
        if np.isnan(rs).any():
            i = int(np.where(np.isnan(rs))[0][0])
            a.annotate("DIV", (ks[i], np.nanmin(rs) * 0.6), color=COLOR[r],
                       fontsize=8, ha="center")
    a.axhline(1.0, color="k", lw=1.1, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("horizon k (steps)   [right axis label: ms]")
    a.set_ylabel("model error / L0 error   (<1 better than persistence)")
    a.set_title("(a) V6 overall ratio  vs  horizon", fontsize=10)
    a.legend(fontsize=7, loc="upper left")
    sec = a.secondary_xaxis("top", functions=(lambda k: k * 0.025,
                                              lambda ms: ms / 0.025))
    sec.set_xlabel("horizon (ms)", fontsize=8)

    # ---- (0,1) latent cos
    a = ax[0, 1]
    for r, m in mets.items():
        ld = m.get("latent_drift") or {}
        if not ld:
            continue
        ks = sorted(int(k) for k in ld)
        c = np.array([ld[str(k)]["cos"] for k in ks])
        lo = np.array([ld[str(k)].get("cos_lo", np.nan) for k in ks])
        hi = np.array([ld[str(k)].get("cos_hi", np.nan) for k in ks])
        yerr = None
        if np.isfinite(lo).all() and np.isfinite(hi).all():
            yerr = np.vstack([c - lo, hi - c])
        a.errorbar(ks, c, yerr=yerr, fmt="o-", lw=1.5, ms=4, capsize=2.5,
                   color=COLOR[r], label=SHORT[r],
                   zorder=3 if r == "R2_event_weighting" else 2)
    a.axhline(0.5, color="#999999", lw=1.0, ls=":")
    a.text(210, 0.52, "cos = 0.5 (escape threshold)", fontsize=7,
           color="#666666")
    a.set_xscale("log")
    a.set_xlabel("horizon k (steps)")
    a.set_ylabel("cos(pred z, true z)   median over 12 trials")
    a.set_title("(b) closed-loop latent direction (median, 95% CI)",
                fontsize=10)
    a.set_ylim(-0.02, 1.04)
    a.legend(fontsize=7, loc="lower left")

    # ---- (1,0) cai precision vs recall
    a = ax[1, 0]
    for r, m in mets.items():
        pl = m.get("plateau") or {}
        if "cai_precision" not in pl:
            continue
        p, rc = pl["cai_precision"], pl["cai_recall"]
        a.scatter([p], [rc], s=90, color=COLOR[r], edgecolor="k",
                  linewidth=0.6, zorder=3)
        a.annotate(SHORT[r], (p, rc), textcoords="offset points",
                   xytext=(6, 5), fontsize=7.5, color=COLOR[r])
    a.set_xlim(0, 1.05)
    a.set_ylim(0, 1.08)
    a.set_xlabel("Ca event precision (fixed threshold -3.10)")
    a.set_ylabel("Ca event recall")
    a.set_title("(c) Ca event detection: precision vs recall\n"
                "(top-right = best; R1b sits high-left = 'vouches little, "
                "errs little')", fontsize=9.5)
    for ln in (0.2, 0.4, 0.6, 0.8):
        a.plot([ln, ln], [0, 1.08], color="#dddddd", lw=0.7, zorder=0)
        a.plot([0, 1.05], [ln, ln], color="#dddddd", lw=0.7, zorder=0)

    # ---- (1,1) scalar bars
    a = ax[1, 1]
    names = [r for r in ORDER if r in mets]
    x = np.arange(len(names))
    vmax = [(mets[r]["plateau"] or {}).get("V_max_model", np.nan)
            for r in names]
    a.bar(x - 0.22, vmax, 0.42, color="#c44e52", label="V max (mV)")
    a.axhline((mets[names[0]]["plateau"] or {}).get("V_max_true", 13.35),
              color="k", lw=1.2, ls="--")
    a.text(len(names) - 0.6, 13.35 + 1.0, "true 13.4 mV", fontsize=7.5,
           ha="right")
    a.set_ylabel("V max (mV)")
    a.set_xticks(x)
    a.set_xticklabels([SHORT[r].replace(" ", "\n") for r in names], fontsize=7)
    a.set_ylim(0, max(60, np.nanmax(vmax) * 1.12))
    a.set_title("(d) peak V: overshoot vs under-shoot", fontsize=10)

    b = a.twinx()
    a200 = [(mets[r]["v6"] or {}).get("overall", {}).get("200", {}).get(
        "ratio", np.nan) for r in names]
    f1 = [(mets[r].get("ca_event") or {}).get("peak_f1", np.nan)
          for r in names]
    b.bar(x + 0.22, a200, 0.20, color="#4c72b0", alpha=0.85,
          label="a(200) ratio")
    b.bar(x + 0.44, f1, 0.20, color="#55a868", alpha=0.85,
          label="peak F1")
    b.axhline(1.0, color="#333333", lw=0.9, ls=":")
    b.set_ylabel("a(200) ratio  /  peak F1")
    b.set_ylim(0, 1.15)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    a.annotate("R4: diverged at k=8000 →", (3.55, 50), fontsize=7,
               color="#9467bd", ha="right")

    fig.suptitle("L5PC v4 world model — 6 training rounds "
                 "(te_s3 / t2, plateau n=43 for Ca events)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(ROUNDS, "summary_rounds.png")
    fig.savefig(p)
    plt.close(fig)
    log(f"=> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
