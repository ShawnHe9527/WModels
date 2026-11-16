import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from neuron import h
import world.gen.sim2 as S2

h.nrn_load_dll(S2.MODDIR)
cwd = os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc", "L5PCtemplate_2.hoc"):
    h.load_file(f)
os.chdir(cwd)
cell = h.L5PCtemplate(S2.MORPH)
secs = list(h.allsec())

def segs_of(sec):
    n = int(sec.n3d()); out = []
    for i in range(n - 1):
        p0 = (sec.y3d(i), sec.x3d(i))      # rotate: y -> horizontal, x -> vertical
        p1 = (sec.y3d(i + 1), sec.x3d(i + 1))
        d = 0.5 * (sec.diam3d(i) + sec.diam3d(i + 1))
        out.append((p0, p1, d))
    return out

gray, red, soma = [], [], []
for sec in secs:
    name = sec.name(); ss = segs_of(sec)
    if not ss: continue
    if name.endswith("apic[51]"): red.extend(ss)
    elif name.endswith("soma[0]"): soma.extend(ss)
    else: gray.extend(ss)

def draw(ax, items, color, lw_scale=0.65, lw_min=0.35, lw_max=3.2, z=2):
    if not items: return
    segs = [[p0, p1] for p0, p1, _ in items]
    lws = [min(lw_max, max(lw_min, d * lw_scale)) for _, _, d in items]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lws,
                                     capstyle="round", joinstyle="round", zorder=z))

fig, ax = plt.subplots(figsize=(6.4, 2.6))
draw(ax, gray, "#a8a8a8", z=2)
draw(ax, soma, "#222222", lw_min=0.8, lw_max=4.0, z=3)
draw(ax, red, "#c0392b", lw_min=1.8, lw_max=4.0, z=4)
ax.set_aspect("equal"); ax.axis("off"); ax.autoscale_view()
xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin - 20, xmax + 20)
ax.set_ylim(ymin - 45, ymax + 20)

# scale bar 100 um
x0 = xmin + 10; y0 = ymin - 25
ax.plot([x0, x0 + 100], [y0, y0], color="0.15", lw=2.2, solid_capstyle="butt", zorder=10)
ax.text(x0 + 50, y0 - 6, "100 µm", ha="center", va="top", fontsize=7.5, color="0.15")

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("figs/ppt_bg/out/l5pc_morphology_horizontal.png", dpi=600,
            facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig("figs/ppt_bg/out/l5pc_morphology_horizontal.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("saved horizontal")
