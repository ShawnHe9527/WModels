import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neuron import h
import world.gen.sim2 as S2

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "xtick.direction": "in",
                     "ytick.direction": "in"})

h.nrn_load_dll(S2.MODDIR)
cwd=os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc","import3d.hoc","L5PCbiophys5b.hoc","L5PCtemplate_2.hoc"): h.load_file(f)
os.chdir(cwd)
cell=h.L5PCtemplate(S2.MORPH)
h.distance(0.5, sec=cell.soma[0])

data={}
for idx in (50,51,52,53):
    for sec in h.allsec():
        if sec.name().endswith(f"apic[{idx}]"):
            ps=sec.psection()
            ih=np.array(ps["density_mechs"]["Ih"]["gIhbar"], dtype=float)
            ca=np.array(ps["density_mechs"]["Ca_LVAst"]["gCa_LVAstbar"], dtype=float)
            sec.push()
            d=np.array([h.distance((i+0.5)/sec.nseg) for i in range(int(sec.nseg))])
            h.pop_section()
            data[idx]=(d,ih,ca)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.94, wspace=0.30)

# target section span
d51 = data[51][0]
for ax in (ax1, ax2):
    ax.axvspan(d51.min()-9.8, d51.max()+9.8, color="#efefef", lw=0, zorder=0)

colors = {50:"#1f4e9c", 51:"#c0392b", 52:"#2e7d32", 53:"#c05000"}
for idx,(d,ih,ca) in data.items():
    ax1.plot(d, ih, "o-", ms=3.0, lw=0.9, color=colors[idx],
             label=f"apic[{idx}]" if idx in (50,51,52,53) else None, zorder=3)
    ax2.semilogy(d, ca, "o-", ms=3.0, lw=0.9, color=colors[idx],
                 label=f"apic[{idx}]", zorder=3)

ax1.axvline(845.35, color="0.45", lw=0.8, ls=(0,(3,2)))
ax1.text(848, 0.00024, "0.65·Lmax", rotation=90, ha="left", va="bottom",
         fontsize=6.8, color="0.45")
ax1.set_xlabel("path distance from soma (µm)")
ax1.set_ylabel(r"$g_{Ih}$ (S/cm$^2$)")
ax1.set_xlim(620, 1100)
ax1.set_ylim(0, 0.0024)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.tick_params(labelsize=8)
ax1.legend(frameon=False, fontsize=6.8, loc="lower right")

ax2.axvspan(685, 885, color="#f6e0c8", alpha=0.55, lw=0, zorder=0)
ax2.text(785, 0.006, "high-Ca band (1% outside)", ha="center", va="bottom",
         fontsize=6.8, color="#8a4b1f")
ax2.set_xlabel("path distance from soma (µm)")
ax2.set_ylabel(r"$g_{Ca,LVA}$ (S/cm$^2$)")
ax2.set_xlim(620, 1100)
ax2.set_ylim(0.0015, 0.5)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.tick_params(labelsize=8)


ax1.text(0.02, 0.96, "A  Ih: exponential gradient",
         transform=ax1.transAxes, fontsize=8.2, va="top")
ax2.text(0.02, 0.96, "B  Ca_LVAst: band-pass step",
         transform=ax2.transAxes, fontsize=8.2, va="top")

fig.savefig("figs/ppt_bg/segments/fig4_ih_calva_profiles.png", dpi=600,
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figs/ppt_bg/segments/fig4_ih_calva_profiles.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved fig4")
