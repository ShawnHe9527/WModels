import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
z = np.load("figs/ppt_bg/cand_19.npz")
t=z["t"]; vs=z["vs"]; V=z["V"]; cai=z["cai"]*1000.0
meta=json.loads(str(z["meta"]))
s,e=meta["run_start"],meta["run_end"]
print("t plateau", t[s], t[e], "dur", meta["run_ms"], "cai", meta["cai_peak"], "nbap", meta["n_bap"])
t0=max(0,s-3200); t1=min(len(t),e+3200); sl=slice(t0,t1)
fig, axes=plt.subplots(2,1,figsize=(7.2,4.8),sharex=True,gridspec_kw={"height_ratios":[2.2,1]})
axes[0].plot(t[sl],vs[sl],color="0.15",lw=0.9,label="Soma")
axes[0].plot(t[sl],V.mean(axis=1)[sl],color="#1f4e9c",lw=0.9,label="Apical")
axes[0].axvspan(t[s],t[e],color="0.85",alpha=0.5,lw=0)
axes[0].set_ylabel("V (mV)"); axes[0].legend(frameon=False,fontsize=8)
axes[1].plot(t[sl],cai.mean(axis=1)[sl],color="#c05000",lw=0.9)
axes[1].axvspan(t[s],t[e],color="0.85",alpha=0.5,lw=0)
axes[1].set_ylabel("cai (uM)"); axes[1].set_xlabel("Time (ms)")
for ax in axes:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig("figs/ppt_bg/inspect19.png",dpi=180)
print("ok")
