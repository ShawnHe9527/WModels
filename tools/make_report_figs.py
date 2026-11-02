"""汇总最新结果图（用于阶段性汇报）。"""
from __future__ import annotations
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = "runs/2026-11-02-report-figs"


def load_ratio(path: str, k: int, regime: str | None = None) -> float | None:
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    for name, m in d["models"].items():
        r = m["results"].get(str(k))
        if not r:
            continue
        if regime is None:
            return float(r["ratio"])
        rr = r.get("regimes", {}).get(regime)
        if rr:
            return float(rr.get("ratio", rr["model"] / rr["L0"]))
    return None


def load_all(path: str, k: int, regime: str | None = None) -> list[float]:
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    out = []
    for name, m in d["models"].items():
        r = m["results"].get(str(k))
        if not r:
            continue
        if regime is None:
            out.append(float(r["ratio"]))
        else:
            rr = r.get("regimes", {}).get(regime)
            if rr:
                out.append(float(rr.get("ratio", rr["model"] / rr["L0"])))
    return out


def fig_seed():
    groups = {
        "基线\n(CL=64)": [0.831, 3.128, 4.781],
        "A\n(仅 512)": [0.538, 0.884, 0.836],
        "A+B\n(64/256/512)": [0.551, 0.479, 0.582],
        "H1024\n(seed1)": [0.511],
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, vals) in enumerate(groups.items()):
        vals = np.asarray(vals)
        ax.bar(i, vals.mean(), width=0.6, color=f"C{i}", alpha=0.35)
        ax.scatter(np.full(len(vals), i), vals, color=f"C{i}", zorder=3, s=45)
        ax.text(i, max(vals) + 0.12, f"均值 {vals.mean():.2f}", ha="center", fontsize=9)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="L0 = 1")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), fontsize=9)
    ax.set_ylabel("model/L0 @ 200 ms (va_s2)")
    ax.set_title("200 ms 误差按 seed：基线 2/3 seed 退化，A+B 三 seed 稳定")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_seed_robustness.png", dpi=150)
    plt.close(fig)


def fig_horizon():
    ks = [200, 2000, 8000]
    labels = ["5 ms", "50 ms", "200 ms"]
    base = [load_ratio(f"{OUT}/eval_base_h1024.json", k) for k in ks]
    h1024 = [load_ratio(f"{OUT}/eval_base_h1024.json", k) for k in ks]
    # H1024 在同一个 json 的第二个模型；重取
    d = json.load(open(f"{OUT}/eval_base_h1024.json"))
    base = [d["models"]["S1base"]["results"][str(k)]["ratio"] for k in ks]
    h1024 = [d["models"]["H1024_s1"]["results"][str(k)]["ratio"] for k in ks]
    AB200 = "runs/2026-10-31-night-a-seeds/eval_AB_3seed_k200.json"
    ab = [float(np.mean(load_all(AB200, k))) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(labels, base, "o-", label="基线 (CL=64)")
    ax.plot(labels, ab, "s-", label="A+B 均值（3 seed）")
    ax.plot(labels, h1024, "^-", label="H1024 (seed1)")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_ylabel("model/L0")
    ax.set_title("误差随评测视界变化（va_s2）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_error_vs_horizon.png", dpi=150)
    plt.close(fig)


def fig_regime():
    regs = ["linear", "inter", "plateau"]
    d0 = json.load(open(f"{OUT}/eval_base_h1024.json"))
    base = [d0["models"]["S1base"]["results"]["8000"]["regimes"][r]["ratio"] for r in regs]
    h1024 = [d0["models"]["H1024_s1"]["results"]["8000"]["regimes"][r]["ratio"] for r in regs]
    ab = [float(np.mean(load_all("runs/2026-10-31-night-a-seeds/eval_AB_3seed_k200.json",
                                 8000, r))) for r in regs]
    x = np.arange(len(regs))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.25, base, 0.25, label="基线", color="C0", alpha=0.75)
    ax.bar(x, ab, 0.25, label="A+B（3 seed 均值）", color="C1", alpha=0.75)
    ax.bar(x + 0.25, h1024, 0.25, label="H1024 (seed1)", color="C2", alpha=0.75)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(regs)
    ax.set_ylabel("model/L0 @ 200 ms")
    ax.set_title("分档误差（200 ms）：inter/plateau 改善最大")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_regime.png", dpi=150)
    plt.close(fig)


def fig_blind():
    # 2s 盲测（te_s4_2s）@200ms
    vals = {
        "基线 S1": 0.758,
        "A+B s1": 0.521,
        "A+B s2": 0.457,
        "A+B s3": 0.525,
        "R2(历史)": 0.890,
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["C0", "C1", "C1", "C1", "gray"]
    ax.bar(range(len(vals)), list(vals.values()), color=colors, alpha=0.75)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(list(vals), fontsize=9)
    ax.set_ylabel("model/L0 @ 200 ms")
    ax.set_title("2 s 盲测（te_s4_2s）：A+B 三 seed 0.46–0.53")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_blind_2s.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_seed()
    fig_horizon()
    fig_regime()
    fig_blind()
    for src in ("runs/2026-11-01-e5dee3f-fig-traj/traj_ab_s2.png",
                "runs/2026-11-01-e5dee3f-fig-traj/traj_h1024_s1.png",
                "runs/2026-11-01-e5dee3f-fig-traj/arch.png"):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, os.path.basename(src)))
    lines = [
        "# 阶段性汇报图（2026-11-02）",
        "",
        "- `fig1_seed_robustness.png`：200 ms 误差按 seed（基线/A/A+B/H1024）",
        "- `fig2_error_vs_horizon.png`：误差随评测视界（5/50/200 ms）",
        "- `fig3_regime.png`：分档（linear/inter/plateau）200 ms 误差",
        "- `fig4_blind_2s.png`：2 s 盲测 te_s4_2s @200 ms",
        "- `traj_ab_s2.png` / `traj_h1024_s1.png`：预测 vs 真实轨迹（V/cai）",
        "- `arch.png`：模型结构图",
        "",
        "数据来源：runs/2026-10-31-night-a-seeds、runs/2026-11-01-night-p123、",
        "runs/2026-11-02-report-figs/eval_base_h1024.json",
    ]
    open(f"{OUT}/README.md", "w", encoding="utf-8").write("\n".join(lines))
    print("done")


if __name__ == "__main__":
    main()
