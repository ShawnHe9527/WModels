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
    n = int(sec.n3d())
    out = []
    for i in range(n - 1):
        p0 = (sec.x3d(i), sec.y3d(i))
        p1 = (sec.x3d(i + 1), sec.y3d(i + 1))
        d = 0.5 * (sec.diam3d(i) + sec.diam3d(i + 1))
        out.append((p0, p1, d))
    return out

gray, red, soma = [], [], []
for sec in secs:
    name = sec.name()
    ss = segs_of(sec)
    if not ss:
        continue
    if name.endswith("apic[51]"):
        red.extend(ss)
    elif name.endswith("soma[0]"):
        soma.extend(ss)
    else:
        gray.extend(ss)

def draw(ax, items, color, lw_scale=0.55, lw_min=0.30, lw_max=3.2, alpha=1.0, z=2):
    if not items:
        return
    segs = [[p0, p1] for p0, p1, _ in items]
    lws = [min(lw_max, max(lw_min, d * lw_scale)) for _, _, d in items]
    lc = LineCollection(segs, colors=color, linewidths=lws, alpha=alpha,
                        capstyle="round", joinstyle="round", zorder=z)
    ax.add_collection(lc)

fig, ax = plt.subplots(figsize=(2.6, 6.4))
draw(ax, gray, "#a8a8a8", lw_scale=0.65, lw_min=0.35, lw_max=3.2, z=2)
draw(ax, soma, "#222222", lw_scale=0.55, lw_min=0.8, lw_max=4.0, z=3)
draw(ax, red, "#c0392b", lw_scale=0.65, lw_min=1.8, lw_max=4.0, z=4)

ax.set_aspect("equal")
ax.axis("off")
ax.autoscale_view()
# small margin
xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin - 12, xmax + 12)
ax.set_ylim(ymin - 60, ymax + 25)

# optional scale bar: 100 um, bottom-left, below the cell
y0 = ymin - 28
x0 = xmin + 8
ax.plot([x0, x0 + 100], [y0, y0], color="0.15", lw=2.2, solid_capstyle="butt", zorder=10)
ax.text(x0 + 50, y0 - 10, "100 µm", ha="center", va="top", fontsize=7.5, color="0.15")

fig.subplots_adjust(0, 0, 1, 1)
os.makedirs("figs/ppt_bg/out", exist_ok=True)
fig.savefig("figs/ppt_bg/out/l5pc_morphology.png", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig("figs/ppt_bg/out/l5pc_morphology.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("saved", len(gray), len(red), len(soma))
