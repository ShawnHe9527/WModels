import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
z = np.load("figs/ppt_bg/probe.npz")
t = z["t"]; vs = z["vs"]; V = z["V"]; cai = z["cai"]*1000.0
vm = V.mean(axis=1)
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios":[2.2,1]})
axes[0].plot(t, vs, color="0.15", lw=1.0, label="Soma")
axes[0].plot(t, vm, color="#1f4e9c", lw=1.0, label="Apical")
axes[0].set_ylabel("V (mV)")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
axes[1].plot(t, cai, color="#c05000", lw=1.0)
axes[1].set_ylabel(r"[Ca$^{2+}$]$_i$ ($\mu$M)")
axes[1].set_xlabel("Time (ms)")
for ax in axes:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("figs/ppt_bg/quick.png", dpi=180)
print("saved")
