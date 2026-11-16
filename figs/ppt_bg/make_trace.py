import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("figs/ppt_bg/cand_19.npz")
t = z["t"]; vs = z["vs"]; V = z["V"]; cai = z["cai"] * 1000.0
meta = json.loads(str(z["meta"]))
s, e = meta["run_start"], meta["run_end"]
t_on, t_off = t[s], t[e]
v_ap = V.mean(axis=1)
cai_mean = cai.mean(axis=1)
cai_lo = cai.min(axis=1)
cai_hi = cai.max(axis=1)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.frameon": False,
})

def make(win, out, ylim2):
    t0, t1 = win
    sl = (t >= t0) & (t <= t1)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]})
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.13, hspace=0.08)

    for ax in (ax1, ax2):
        ax.axvspan(t_on, t_off, color="#d9d9d9", alpha=0.65, lw=0, zorder=0)

    l1, = ax1.plot(t[sl], vs[sl], color="#1a1a1a", lw=0.9, label="Soma", zorder=3)
    l2, = ax1.plot(t[sl], v_ap[sl], color="#1f4e9c", lw=0.9, label="Apical", zorder=2)
    ax1.set_ylabel("V (mV)", labelpad=3)
    ax1.set_ylim(-88, 48)
    ax1.legend(handles=[l1, l2], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.13), fontsize=8.5, handlelength=1.6,
               columnspacing=1.2)

    ax2.plot(t[sl], cai_mean[sl], color="#c05000", lw=0.9, zorder=2)
    ax2.set_ylabel(r"[Ca$^{2+}$]$_i$ ($\mu$M)", labelpad=3)
    ax2.set_xlabel("Time (ms)", labelpad=2)
    ax2.set_ylim(*ylim2)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlim(t0, t1)

    os.makedirs("figs/ppt_bg/out", exist_ok=True)
    fig.savefig(f"figs/ppt_bg/out/{out}.png", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"figs/ppt_bg/out/{out}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

make((560, 760), "ca_plateau_trace", (0.27, 0.52))
make((600, 735), "ca_plateau_trace_zoom", (0.27, 0.51))
print("t_on/t_off", t_on, t_off)
print("cai mean min/max", cai_mean.min(), cai_mean.max())
