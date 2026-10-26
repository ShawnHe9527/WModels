"""跨轮汇总：把 results/rounds/*/metrics.json 汇总成总索引 + 对比图。

    python -m tools.rounds_index

生成：
    results/rounds/README.md          总索引（含对比表）
    results/rounds/compare_rounds.png 各轮 模型/L0 vs k
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

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_all():
    out = []
    for r in sorted(os.listdir(ROUNDS)):
        p = os.path.join(ROUNDS, r, "metrics.json")
        if os.path.isfile(p):
            out.append(json.load(open(p, encoding="utf-8")))
    return out


def _ratio_series(block):
    """取出某档的 (ks, ratios)，发散点用 nan 断开曲线（log 轴不能吃 nan/inf）。"""
    ks = sorted(int(k) for k in block)
    rs = []
    for k in ks:
        r = block[str(k)]
        ok = (not r.get("diverged")) and np.isfinite(r.get("ratio", np.nan))
        rs.append(r["ratio"] if ok else np.nan)
    return ks, np.asarray(rs, float)


def fig_compare(mets, path):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for met in mets:
        v6 = met.get("v6") or {}
        ov = v6.get("overall") or {}
        if not ov:
            continue
        lbl = met["round"]
        ks, rs = _ratio_series(ov)
        ax[0].plot(ks, rs, "o-", lw=1.3, ms=4, label=lbl)
        d = v6.get("plateau") or {}
        if d:
            ks, rs = _ratio_series(d)
            ax[1].plot(ks, rs, "o-", lw=1.3, ms=4, label=lbl)
    for a, t in ((ax[0], "overall"), (ax[1], "plateau")):
        a.axhline(1.0, color="k", lw=1.0, ls="--")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("horizon k (steps, 0.025 ms -> k=2000 is 50 ms)")
        a.set_ylabel("model / L0   (<1 = better than persistence)")
        a.set_title(t, fontsize=10)
        a.legend(fontsize=7)
    fig.suptitle("Rounds comparison  (te_s3, paired; curve ends early = diverged)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    mets = load_all()
    if not mets:
        print("没有找到任何 round")
        return 1
    fig_compare(mets, os.path.join(ROUNDS, "compare_rounds.png"))

    L = ["# 试验轮次索引", "",
         "> 由 `python -m tools.rounds_index` 自动生成。",
         "> 每轮一个子目录：`README.md` / `metrics.json` / `figures/` / "
         "`logs/` / `ckpt/`。", "",
         "![compare](compare_rounds.png)", "",
         "## 总览", ""]

    # 收集所有 k
    ks_all = set()
    for m in mets:
        ov = (m.get("v6") or {}).get("overall") or {}
        ks_all |= {int(k) for k in ov}
    ks = sorted(ks_all)
    L += ["| 轮次 | 说明 | σ | cos(k=2000) 中位 | V上限 | cai P | cai R | "
          "峰F1 | 幅度比 | "
          + " | ".join(f"k={k}" for k in ks) + " | a(200) |",
          "|---|---|---|---|---|---|---|---|---|" + "---|" * len(ks) + "---|"]
    for m in mets:
        ov = (m.get("v6") or {}).get("overall") or {}
        ld = m.get("latent_drift") or {}
        pl = m.get("plateau") or {}
        cos = ""
        if ld:
            kmax = max(int(x) for x in ld)
            v = ld[str(kmax)]
            # 新格式存 cos；旧格式存的是单 trial 值（仅单个 trial，不可比）
            cos = f"{v['cos']:+.3f}" if "cos" in v else ""
        cells = []
        for k in ks:
            r = ov.get(str(k))
            if not r:
                cells.append("—")
            elif r.get("diverged") or not np.isfinite(r.get("ratio", np.nan)):
                cells.append("**DIV**")
            else:
                cells.append(f"{r['ratio']:.3f}")
        a200 = f"**{ov['200']['ratio']:.3f}**" if "200" in ov else "—"
        ce = m.get("ca_event") or {}
        pf1 = f"{ce['peak_f1']:.3f}" if ce else "—"
        amp = f"{ce['amp_ratio']:.2f}" if ce else "—"
        L.append(f"| `{m['round']}` | {m.get('title', '')} "
                 f"| {m.get('sigma', float('nan')):.6f} | {cos} "
                 f"| {pl.get('V_max_model', float('nan')):.1f} "
                 f"| {pl.get('cai_precision', float('nan')):.3f} "
                 f"| {pl.get('cai_recall', float('nan')):.3f} "
                 f"| {pf1} | {amp} "
                 f"| " + " | ".join(cells) + f" | {a200} |")

    L += ["", "## 各轮详情", ""]
    for m in mets:
        L.append(f"- [`{m['round']}`](./{m['round']}/README.md) — "
                 f"{m.get('title', '')}")
    L.append("")
    p = os.path.join(ROUNDS, "README.md")
    open(p, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[-len(mets) - 12:]) if False else "")
    print(f"=> {p}")
    print(f"=> {os.path.join(ROUNDS, 'compare_rounds.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
