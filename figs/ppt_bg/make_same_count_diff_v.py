"""Same spike count, different timing -> different target V.

Controlled NEURON simulation on the official L5PC model.
Condition A: 120 independent E events per target compartment (rho=0).
Condition B: 72 shared E events + 48 independent E events per compartment (rho=0.6).
Both: 120 E events per compartment, 1320 total E events; identical I trains.
Only the timing/synchrony of E events differs.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
from world.gen.sim2 import Sim2

# Chinese font
for fam in ["Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP"]:
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "figs/ppt_bg/out"
os.makedirs(OUT, exist_ok=True)
T_SETTLE, T_TEST = 100.0, 1000.0
T0, T1 = T_SETTLE, T_SETTLE + T_TEST
N_COMP = 11
N_E_PER = 120
N_I_PER = 40
N_SHARED = 72
N_INDEP = N_E_PER - N_SHARED  # 48

rng_a = np.random.default_rng(20261112)
rng_b = np.random.default_rng(20261113)

# A: independent E
own_e_a = [np.sort(rng_a.uniform(T0, T1, N_E_PER)) for _ in range(N_COMP)]
# B: shared E + independent E
shared = np.sort(rng_b.uniform(T0, T1, N_SHARED))
own_e_b = [np.sort(np.concatenate([shared, rng_b.uniform(T0, T1, N_INDEP)]))
           for _ in range(N_COMP)]

# identical I trains for both conditions
rng_i = np.random.default_rng(20261114)
own_i = [np.sort(rng_i.uniform(T0, T1, N_I_PER)) for _ in range(N_COMP)]

# no neighbor input, no bAP: isolate target E synchrony
nb_e = [np.empty(0, dtype=np.float64) for _ in range(17)]
nb_i = [np.empty(0, dtype=np.float64) for _ in range(17)]

sim = Sim2(dt=0.025, t_settle=T_SETTLE, t_test=T_TEST)
SA, UA = sim.run(own_e_a, own_i, nb_e, nb_i, bap_times=None)
SB, UB = sim.run(own_e_b, own_i, nb_e, nb_i, bap_times=None)

t = np.arange(sim.T) * sim.dt
VA = SA[:, 8, :]   # (T,11) mV
VB = SB[:, 8, :]
CA = SA[:, 7, :]   # log10 cai
CB = SB[:, 7, :]

def summ(V, C, e_list, i_list):
    vmax = float(V.max())
    vmean = V.mean(axis=1)
    up = float((vmean > -50).mean())
    # target compartment spike count: peaks above 0 mV
    spikes = 0
    for j in range(V.shape[1]):
        x = V[:, j]
        above = x > 0
        spikes += int(np.sum(above[1:] & ~above[:-1]))
    return dict(
        vmax=vmax, vmin=float(V.min()), vmean=float(V.mean()),
        up_frac=up, cai_span=float(C.max()-C.min()),
        cai_max=float(C.max()), spikes=int(spikes),
        n_e=int(sum(len(x) for x in e_list)),
        n_i=int(sum(len(x) for x in i_list)),
    )

ma = summ(VA, CA, own_e_a, own_i)
mb = summ(VB, CB, own_e_b, own_i)
print("A", json.dumps(ma, ensure_ascii=False))
print("B", json.dumps(mb, ensure_ascii=False))

# Save data
np.savez(os.path.join(OUT, "same_count_diff_v_data.npz"),
         t=t, VA=VA, VB=VB, CA=CA, CB=CB,
         own_e_a=np.array([np.pad(x, (0, N_E_PER-len(x))) for x in own_e_a]),
         own_e_b=np.array([np.pad(x, (0, N_E_PER-len(x))) for x in own_e_b]),
         own_i=np.array([np.pad(x, (0, N_I_PER-len(x))) for x in own_i]),
         shared=shared, meta=json.dumps(dict(A=ma, B=mb), ensure_ascii=False))

# ---- Figure ----
fig = plt.figure(figsize=(13.2, 7.2), dpi=220, facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.18,
                      left=0.075, right=0.985, top=0.90, bottom=0.09)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axV = fig.add_subplot(gs[1, :])
# raster A
for i, ts in enumerate(own_e_a):
    axA.vlines(ts - T0, i + 0.6, i + 1.4, color="#2C7FB8", lw=0.7, alpha=0.85)
axA.set_xlim(0, T_TEST); axA.set_ylim(0.4, N_COMP + 0.6)
axA.set_yticks([1, 6, 11]); axA.set_yticklabels(["comp 1", "comp 6", "comp 11"], fontsize=8)
axA.set_xlabel("时间 (ms)", fontsize=9); axA.set_ylabel("目标段突触", fontsize=9)
axA.set_title(f"A  独立发放：rho = 0\n每个 comp 120 个 E；总计 {ma['n_e']} 个 E", fontsize=10, loc="left")
axA.grid(axis="x", ls=":", lw=0.5, color="#CBD5E1")

# raster B
for i, ts in enumerate(own_e_b):
    axB.vlines(ts - T0, i + 0.6, i + 1.4, color="#D95F0E", lw=0.7, alpha=0.85)
axB.set_xlim(0, T_TEST); axB.set_ylim(0.4, N_COMP + 0.6)
axB.set_yticks([1, 6, 11]); axB.set_yticklabels(["comp 1", "comp 6", "comp 11"], fontsize=8)
axB.set_xlabel("时间 (ms)", fontsize=9); axB.set_ylabel("目标段突触", fontsize=9)
axB.set_title(f"B  部分同步：rho = 0.6\n每个 comp 120 个 E；总计 {mb['n_e']} 个 E", fontsize=10, loc="left")
axB.grid(axis="x", ls=":", lw=0.5, color="#CBD5E1")

# V trajectories
vA = VA.mean(axis=1); vB = VB.mean(axis=1)
sA = VA.std(axis=1); sB = VB.std(axis=1)
axV.fill_between(t, vA-sA, vA+sA, color="#2C7FB8", alpha=0.13, lw=0)
axV.fill_between(t, vB-sB, vB+sB, color="#D95F0E", alpha=0.13, lw=0)
axV.plot(t, vA, color="#2C7FB8", lw=1.8, label=f"A: rho=0, Vmax={ma['vmax']:.1f} mV")
axV.plot(t, vB, color="#D95F0E", lw=1.8, label=f"B: rho=0.6, Vmax={mb['vmax']:.1f} mV")
axV.axhline(0, color="#6B7280", lw=0.7, ls="--")
axV.set_xlim(0, T_TEST); axV.set_ylim(-85, 45)
axV.set_xlabel("时间 (ms)", fontsize=10); axV.set_ylabel("目标段平均 V (mV)", fontsize=10)
axV.set_title("相同脉冲数、相同 I 输入，仅改变 E 事件时序 → 目标段 V 轨迹不同", fontsize=11, loc="left")
axV.grid(axis="y", ls=":", lw=0.5, color="#CBD5E1")
axV.legend(loc="upper left", fontsize=8.5, frameon=False)
axV.text(0.99, 0.04, f"E 总数: A={ma['n_e']}, B={mb['n_e']}；I 总数: {ma['n_i']}；阴影=11 室 std",
         transform=axV.transAxes, ha="right", va="bottom", fontsize=8.2, color="#374151")

fig.suptitle("相同脉冲数，不同时序：目标段电位轨迹可以完全不同", fontsize=14, fontweight="bold", y=0.975)
fig.text(0.075, 0.012, "官方 L5PC 模型，1 s 窗口；仅改变目标段 E 事件的同步性；邻段输入与 bAP 关闭",
         fontsize=8.8, color="#4B5563")
fig.savefig(os.path.join(OUT, "same_count_diff_v.png"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "same_count_diff_v.pdf"), bbox_inches="tight", facecolor="white")
print("saved figure")
