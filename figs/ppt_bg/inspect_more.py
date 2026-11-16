import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ids=[19,20,23]
fig, axes=plt.subplots(len(ids),2,figsize=(10,7.2),sharex="col")
for row,idx in enumerate(ids):
    z=np.load(f"figs/ppt_bg/cand_{idx:02d}.npz")
    t=z["t"]; vs=z["vs"]; V=z["V"]; cai=z["cai"]*1000.0
    meta=json.loads(str(z["meta"])); s,e=meta["run_start"],meta["run_end"]
    t0=max(0,s-2800); t1=min(len(t),e+2800); sl=slice(t0,t1)
    axes[row,0].plot(t[sl],vs[sl],color="0.15",lw=0.8)
    axes[row,0].plot(t[sl],V.mean(axis=1)[sl],color="#1f4e9c",lw=0.8)
    axes[row,0].axvspan(t[s],t[e],color="0.85",alpha=0.5,lw=0)
    axes[row,0].set_ylabel("V (mV)")
    axes[row,0].set_title(f"cand {idx}: {meta['run_ms']:.0f} ms, cai {meta['cai_peak']:.2f}, {meta['n_bap']} bAP",fontsize=8)
    axes[row,1].plot(t[sl],cai.mean(axis=1)[sl],color="#c05000",lw=0.8)
    axes[row,1].axvspan(t[s],t[e],color="0.85",alpha=0.5,lw=0)
    axes[row,1].set_ylabel("cai (uM)")
for ax in axes.ravel():
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig("figs/ppt_bg/inspect_more.png",dpi=160); print("ok")
