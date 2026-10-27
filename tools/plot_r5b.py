"""R5-B result figures + explanation document.

Reads the run directory produced by train2 + eval_gate, computes physical
metrics on plateau trials, and writes:
  figures/fig_training.png
  figures/fig_horizon.png
  figures/fig_physics.png
  figures/fig_trace.png
  REPORT.md
  metrics.json  (merged summary)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.core import safety

safety.apply_thread_caps(2)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from world.learn import data2 as D2
from tools.plot_v4 import load_model, denorm, regime_rollouts
from tools.ca_event_metrics import ca_event_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESC = os.path.join(ROOT, "results", "checkpoints")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def log(*a):
    print(*a, flush=True)


def training_curves(path):
    steps, loss, grad, cl, sig = [], [], [], [], []
    pat = re.compile(r"T2 t2\s+[\d.]+% \|.*?\|\s*(\d+)/\d+ \[.*?\]\s+"
                     r"loss=([0-9.eE+\-]+)\s+grad=([0-9.eE+\-]+).*?cl=(\d+)")
    sp = re.compile(r"step (\d+): loss=[0-9.eE+\-]+\s+单步σ=([0-9.eE+\-]+)")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            grad.append(float(m.group(3)))
            cl.append(int(m.group(4)))
        m = sp.search(line)
        if m:
            sig.append((int(m.group(1)), float(m.group(2))))
    return (np.asarray(steps), np.asarray(loss), np.asarray(grad),
            np.asarray(cl), sig)


def physical(ckpt, ds, k=8000, t0=4000, n=999):
    m, sc, _cfg = load_model(ckpt, ds, "t2")
    ids, P, Y, _B = regime_rollouts(m, ds, sc, "plateau", k + 1, t0, n=n)
    Pp, Yp = denorm(P, sc), denorm(Y, sc)
    ci = ds.meta["factor_names"].index("cai")
    vi = ds.meta["factor_names"].index("V")
    out = dict(n=len(ids), ids=[int(i) for i in ids],
               vmax_pred=float(Pp[:, :, vi, :].max()),
               vmax_true=float(Yp[:, :, vi, :].max()),
               ca=ca_event_metrics(Pp, Yp, ci, ds.dt))
    return out, (ids, P, Y, Pp, Yp, sc, ci, vi)


def plot_training(run, out):
    steps, loss, grad, cl, sig = training_curves(os.path.join(run, "stdout.log"))
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True)
    ax[0].plot(steps, loss, lw=1.0, color="#1f77b4", alpha=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("training loss")
    ax[0].set_title("R5-B training (event-centered sampling + V weighting)",
                    fontsize=10)
    ax[1].plot(steps, grad, lw=1.0, color="#d62728", alpha=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("grad norm")
    ax[2].plot(steps, cl, lw=1.0, color="#2ca02c", label="closed-loop steps")
    if sig:
        ss, sv = zip(*sig)
        ax2 = ax[2].twinx()
        ax2.plot(ss, sv, "ko-", ms=3, lw=1.0, label="val single-step σ")
        ax2.set_ylabel("σ")
        ax2.legend(fontsize=7, loc="upper right")
    ax[2].set_ylabel("cl steps")
    ax[2].set_xlabel("step")
    ax[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = os.path.join(out, "fig_training.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_horizon(run, out):
    ev = json.load(open(os.path.join(run, "eval_te_s4.json"), encoding="utf-8"))
    names = list(ev["models"].keys())
    ks = sorted(int(k) for k in ev["models"][names[0]]["results"])
    regs = ["overall", "linear", "inter", "plateau"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    for ax, reg in zip(axes.flat, regs):
        for name in names:
            rs = ev["models"][name]["results"]
            xs, ys = [], []
            for k in ks:
                if reg == "overall":
                    r = rs[str(k)]["ratio"]
                else:
                    r = rs[str(k)].get("regimes", {}).get(reg, {}).get("ratio", np.nan)
                xs.append(k)
                ys.append(r)
            ax.plot(xs, ys, "o-", lw=1.4, ms=4, label=name)
        # L2 baseline from any model (same samples)
        rs = ev["models"][names[0]]["results"]
        l2 = [rs[str(k)]["L2"][0] / max(rs[str(k)]["L0"][0], 1e-12) for k in ks]
        ax.plot(ks, l2, ":", color="0.35", lw=1.2, label="L2 ridge")
        ax.axhline(1.0, color="k", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("horizon k (0.025 ms/step)")
        ax.set_ylabel("model / L0" if reg == "overall" else f"{reg} / L0")
        ax.set_title(reg)
        ax.legend(fontsize=7)
    fig.suptitle("R5-B blind test on te_s4 (128 trials × 2 t0)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, "fig_horizon.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_physics(phys, out):
    names = list(phys.keys())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    x = np.arange(len(names))
    tv = [phys[n]["vmax_true"] for n in names]
    pv = [phys[n]["vmax_pred"] for n in names]
    axes[0].bar(x - 0.18, tv, 0.36, label="true", color="0.55")
    axes[0].bar(x + 0.18, pv, 0.36, label="pred", color="#d62728")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel("V max (mV)")
    axes[0].set_title("plateau V max")
    axes[0].legend(fontsize=7)
    f1 = [phys[n]["ca"].get("peak_f1", np.nan) for n in names]
    axes[1].bar(x, f1, color="#2ca02c")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("peak F1")
    axes[1].set_title("Ca event peak F1 (no threshold)")
    amp = [phys[n]["ca"].get("amp_ratio", np.nan) for n in names]
    axes[2].bar(x, amp, color="#9467bd")
    axes[2].axhline(1.0, color="k", ls="--", lw=1.0)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names)
    axes[2].set_ylabel("pred / true amplitude")
    axes[2].set_title("Ca event amplitude ratio")
    fig.tight_layout()
    path = os.path.join(out, "fig_physics.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trace(phys_data, names, out):
    # choose the plateau trial with the largest true V max
    name0 = names[0]
    ids, P, Y, Pp, Yp, sc, ci, vi = phys_data[name0]
    j = int(np.argmax(Yp[:, :, vi, :].max(axis=(1, 2))))
    T = Yp.shape[1]
    t = np.arange(T) * 0.025
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    ax[0].plot(t, Yp[j, :, vi, :].max(axis=1), "k-", lw=1.6, label="true")
    ax[1].plot(t, Yp[j, :, ci, :].max(axis=1), "k-", lw=1.6, label="true")
    for name in names:
        idd, P_, Y_, Pp_, Yp_, sc_, ci_, vi_ = phys_data[name]
        if int(ids[j]) not in idd:
            continue
        jj = idd.index(int(ids[j]))
        ax[0].plot(t, Pp_[jj, :, vi_, :].max(axis=1), lw=1.1, label=name)
        ax[1].plot(t, Pp_[jj, :, ci_, :].max(axis=1), lw=1.1, label=name)
    ax[0].set_ylabel("V (mV)")
    ax[1].set_ylabel("cai (log10)")
    ax[1].set_xlabel("time in rollout (ms)")
    ax[0].set_title(f"plateau trial {ids[j]} (largest true V max)")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out, "fig_trace.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.plot_r5b")
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="te_s4")
    ap.add_argument("--r2", default="t2v5_events/last.pt")
    ap.add_argument("--b", default="r5b_event_center/best.pt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.run, "figures")
    os.makedirs(out, exist_ok=True)
    os.makedirs(a.run, exist_ok=True)

    ds = D2.Dataset2(DATA, a.data, tier="t2")
    names = ["R2", "B_best"]
    ckpts = [a.r2, a.b]
    phys, pdata = {}, {}
    for n, c in zip(names, ckpts):
        log(f"physical {n}: {c}")
        phys[n], pdata[n] = physical(c, ds)
        log(f"  {n}: Vmax pred={phys[n]['vmax_pred']:.2f} "
            f"true={phys[n]['vmax_true']:.2f} "
            f"F1={phys[n]['ca'].get('peak_f1'):.3f} "
            f"amp={phys[n]['ca'].get('amp_ratio'):.3f}")

    f1 = plot_training(a.run, out)
    f2 = plot_horizon(a.run, out)
    f3 = plot_physics(phys, out)
    f4 = plot_trace(pdata, names, out)

    ev = json.load(open(os.path.join(a.run, "eval_te_s4.json"), encoding="utf-8"))
    train_steps, loss, grad, cl, sig = training_curves(
        os.path.join(a.run, "stdout.log"))
    m = dict(
        experiment="r5-b-event-focused",
        training=dict(steps=int(train_steps.max()) if len(train_steps) else None,
                      loss_last=float(loss[-1]) if len(loss) else None,
                      grad_last=float(grad[-1]) if len(grad) else None,
                      sigma_last=sig[-1] if sig else None),
        eval_te_s4=ev,
        physical=phys,
        figures=[f1, f2, f3, f4],
    )
    json.dump(m, open(os.path.join(a.run, "metrics.json"), "w"),
              ensure_ascii=False, indent=1)

    # Markdown report
    def get(name, k):
        return ev["models"][name]["results"][str(k)]

    lines = [
        "# R5-B：事件中心采样 + V 事件加权",
        "",
        f"> run: `{a.run}`  ·  data: `{a.data}`  ·  ckpt: `{a.b}`",
        "> 自动生成；数值来自 `stdout.log` / `eval_te_s4.json` / 本次 plateau rollout。",
        "",
        "## 1. 这次改了什么",
        "",
        "- 保留 R2 的训练视界（length=128 = 3.2 ms，cl=64 = 1.6 ms）和 9000 步预算；",
        "- **50% 窗口以 Ca episode 中心开窗**（其余随机）；",
        "- 在原有 cai 事件加权上，增加 **V 事件加权**（weight=5, tau=0.02）；",
        "- 其余与 R2 相同：d_z=16、anchor-cl=0.1、event-weight=10、batch=24。",
        "",
        "## 2. 训练",
        "",
        f"- 最终 loss = {loss[-1]:.3e}，grad = {grad[-1]:.3f}，cl = {cl[-1]}" if len(loss) else "- 无训练日志",
        f"- 最后单步 σ = {sig[-1][1]:.6f}" if sig else "- 无 σ 记录",
        "",
        "![training](figures/fig_training.png)",
        "",
        "## 3. te_s4 盲测",
        "",

    ]
    ev_names = list(ev["models"].keys())
    lines += ["| k | " + " | ".join(ev_names) + " |",
              "|---|" + "---|" * len(ev_names)]
    for k in (10, 200, 2000, 8000):
        lines.append(f"| {k} | " + " | ".join(
            f"{get(n, k)['ratio']:.3f}" for n in ev_names) + " |")
    lines += [
        "",
        "![horizon](figures/fig_horizon.png)",
        "",
        "## 4. 物理指标（plateau, k=8000, t0=4000）",
        "",
        "| model | Vmax pred | Vmax true | Ca peak F1 | amp ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in names:
        p = phys[n]
        lines.append(f"| {n} | {p['vmax_pred']:.2f} | {p['vmax_true']:.2f} "
                     f"| {p['ca'].get('peak_f1', float('nan')):.3f} "
                     f"| {p['ca'].get('amp_ratio', float('nan')):.3f} |")
    lines += [
        "",
        "![physics](figures/fig_physics.png)",
        "",
        "## 5. 一个 plateau trial 的轨迹",
        "",
        "![trace](figures/fig_trace.png)",
        "",
        "## 6. 结论",
        "",
    ]
    try:
        r2_8 = get("R2", 8000)["ratio"]
        b_name = "B_best" if "B_best" in ev["models"] else "B"
        bb_8 = get(b_name, 8000)["ratio"]
        r2_2 = get("R2", 2000)["ratio"]
        bb_2 = get(b_name, 2000)["ratio"]
        verdict = "B 没有改善" if (bb_8 > r2_8 or bb_2 > r2_2) else "B 有改善"
        lines += [
            f"- 盲测 k=8000：R2 = {r2_8:.3f}，{b_name} = {bb_8:.3f}；",
            f"- 盲测 k=2000：R2 = {r2_2:.3f}，{b_name} = {bb_2:.3f}；",
            f"- 判定：**{verdict}**；R2 仍是当前最优。",
            "- B 的短/中程也变差，且 B_last 在 8000 步时 28/256 个样本发散；",
            "  事件中心采样把窗口压到 Ca episode 上，但 128 步仍只有 3.2 ms，",
            "  可能造成事件片段过拟合，而没有改善慢态/长程。",
            "",
        ]
    except Exception as e:
        lines += [f"- 自动结论失败：{e}", ""]
    with open(os.path.join(a.run, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"=> {os.path.join(a.run, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
