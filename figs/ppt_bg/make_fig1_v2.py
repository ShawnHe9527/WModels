import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from neuron import h
import world.gen.sim2 as S2

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

h.nrn_load_dll(S2.MODDIR)
cwd = os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc", "L5PCtemplate_2.hoc"):
    h.load_file(f)
os.chdir(cwd)
cell = h.L5PCtemplate(S2.MORPH)
SECS = list(h.allsec())

def get(idx):
    for sec in SECS:
        if sec.name().endswith(f"apic[{idx}]"):
            return sec
    raise KeyError(idx)

def poly(sec):
    return np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())])

def draw_sec(ax, sec, color, proj=lambda p: (p[0], p[1]),
             lw_scale=0.55, lw_min=0.25, lw_max=3.2, z=2):
    P = poly(sec); q = np.array([proj(p) for p in P])
    d = np.array([sec.diam3d(i) for i in range(sec.n3d())])
    segs = np.stack([q[:-1], q[1:]], axis=1)
    lws = np.clip(0.5 * (d[:-1] + d[1:]) * lw_scale, lw_min, lw_max)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lws,
                                     capstyle="round", joinstyle="round", zorder=z))

def scalebar(ax, x0, y0, length, label):
    ax.plot([x0, x0 + length], [y0, y0], color="0.15", lw=2.0,
            solid_capstyle="butt", zorder=20)
    ax.text(x0 + length / 2, y0, label, ha="center", va="bottom",
            fontsize=7.0, color="0.15", zorder=21)

proj_h = lambda p: (p[1], p[0])
s50, s51, s52, s53 = get(50), get(51), get(52), get(53)

fig = plt.figure(figsize=(7.8, 2.6))
ax = fig.add_axes([0.02, 0.30, 0.73, 0.66])
ax.set_aspect("equal"); ax.axis("off")

for s, c, lw0, lw1 in [(s50, "#1f4e9c", 0.8, 3.4), (s51, "#c0392b", 1.6, 4.2),
                       (s52, "#2e7d32", 0.6, 2.0), (s53, "#c05000", 0.8, 2.8)]:
    draw_sec(ax, s, c, proj_h, lw_min=lw0, lw_max=lw1, z=3)

allq = np.array([proj_h(p) for s in (s50, s51, s52, s53) for p in poly(s)])
xmin, xmax = allq[:,0].min(), allq[:,0].max()
ymin, ymax = allq[:,1].min(), allq[:,1].max()
ax.set_xlim(xmin - 28, xmax + 32)
ax.set_ylim(ymin - 8, ymax + 5)

ax.annotate("", xy=(xmax + 15, ymin - 5), xytext=(xmin - 15, ymin - 5),
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
ax.text((xmin + xmax) / 2, ymin - 7, "proximal  →  distal", ha="center",
        va="top", fontsize=6.8, color="0.45")
scalebar(ax, xmin - 12, ymax + 2.5, 50, "50 µm")

handles = [
    Line2D([0], [0], color="#1f4e9c", lw=2.6, label="apic[50]  parent"),
    Line2D([0], [0], color="#c0392b", lw=2.8, label="apic[51]  target"),
    Line2D([0], [0], color="#2e7d32", lw=2.2, label="apic[52]  child"),
    Line2D([0], [0], color="#c05000", lw=2.4, label="apic[53]  child"),
]
leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                ncol=4, frameon=False, fontsize=8.0, handlelength=1.8,
                columnspacing=1.8, borderaxespad=0)
for text, col in zip(leg.get_texts(), ["#1f4e9c", "#c0392b", "#2e7d32", "#c05000"]):
    text.set_color(col)
leg.get_texts()[1].set_fontweight("bold")

axi = fig.add_axes([0.77, 0.06, 0.22, 0.90])
axi.set_aspect("equal"); axi.axis("off")
for s in SECS:
    if s.name().endswith(("apic[50]", "apic[51]", "apic[52]", "apic[53]")):
        continue
    draw_sec(axi, s, "#d2d2d2", lw_min=0.18, lw_max=1.4, z=1)
for s, c in [(s50, "#1f4e9c"), (s51, "#c0392b"), (s52, "#2e7d32"), (s53, "#c05000")]:
    draw_sec(axi, s, c, lw_min=0.6, lw_max=2.0, z=3)
axi.set_xlim(-175, 245); axi.set_ylim(-210, 1200)
axi.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        fill=False, ec="0.45", lw=0.6, ls=(0, (2, 2)), zorder=10))
axi.text(-165, 1160, "whole cell", fontsize=6.5, color="0.45", va="top")

fig.savefig("figs/ppt_bg/segments/fig1_target_and_neighbors.png", dpi=600,
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figs/ppt_bg/segments/fig1_target_and_neighbors.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved fig1 v2")
