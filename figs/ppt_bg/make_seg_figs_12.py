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
})

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
    n = sec.n3d()
    return np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(n)])

def interp(sec, x):
    n = sec.n3d(); L = sec.L
    arcs = np.array([sec.arc3d(i) for i in range(n)])
    xs = np.array([sec.x3d(i) for i in range(n)])
    ys = np.array([sec.y3d(i) for i in range(n)])
    zs = np.array([sec.z3d(i) for i in range(n)])
    ds = np.array([sec.diam3d(i) for i in range(n)])
    s = x * L
    return (float(np.interp(s, arcs, xs)), float(np.interp(s, arcs, ys)),
            float(np.interp(s, arcs, zs)), float(np.interp(s, arcs, ds)))

def draw_sec(ax, sec, color, proj=lambda p: (p[0], p[1]),
             lw_scale=0.55, lw_min=0.35, lw_max=3.2, alpha=1.0, z=2):
    P = poly(sec)
    q = np.array([proj(p) for p in P])
    d = np.array([sec.diam3d(i) for i in range(sec.n3d())])
    segs = np.stack([q[:-1], q[1:]], axis=1)
    lws = np.clip(0.5 * (d[:-1] + d[1:]) * lw_scale, lw_min, lw_max)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lws,
                                     alpha=alpha, capstyle="round",
                                     joinstyle="round", zorder=z))

def scalebar(ax, x0, y0, length, label, color="0.15", lw=2.0, fs=7.2):
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=lw,
            solid_capstyle="butt", zorder=20)
    ax.text(x0 + length / 2, y0, label, ha="center", va="bottom",
            fontsize=fs, color=color, zorder=21)

# =============================================================================
# Figure 1: target + neighbors, horizontal; full-cell inset
# =============================================================================
proj_h = lambda p: (p[1], p[0])   # y -> x (proximal left, distal right), x -> y
s50, s51, s52, s53 = get(50), get(51), get(52), get(53)

fig = plt.figure(figsize=(7.8, 2.9))
ax = fig.add_axes([0.02, 0.26, 0.73, 0.70])
ax.set_aspect("equal"); ax.axis("off")

draw_sec(ax, s50, "#1f4e9c", proj_h, lw_min=0.8, lw_max=3.4, z=3)
draw_sec(ax, s51, "#c0392b", proj_h, lw_min=1.6, lw_max=4.2, z=4)
draw_sec(ax, s52, "#2e7d32", proj_h, lw_min=0.6, lw_max=2.0, z=3)
draw_sec(ax, s53, "#c05000", proj_h, lw_min=0.8, lw_max=2.8, z=3)

allq = []
for s in (s50, s51, s52, s53):
    allq += [proj_h(p) for p in poly(s)]
allq = np.array(allq)
xmin, xmax = allq[:,0].min(), allq[:,0].max()
ymin, ymax = allq[:,1].min(), allq[:,1].max()
ax.set_xlim(xmin - 25, xmax + 30)
ax.set_ylim(ymin - 12, ymax + 12)

# orientation arrow
ax.annotate("", xy=(xmax + 12, ymin - 7), xytext=(xmin - 12, ymin - 7),
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
ax.text((xmin + xmax) / 2, ymin - 11, "proximal  →  distal", ha="center",
        va="top", fontsize=7.0, color="0.45")

scalebar(ax, xmin - 8, ymax + 6, 50, "50 µm")

# legend below
handles = [
    Line2D([0], [0], color="#1f4e9c", lw=2.6, label="apic[50]  parent"),
    Line2D([0], [0], color="#c0392b", lw=2.8, label="apic[51]  target"),
    Line2D([0], [0], color="#2e7d32", lw=2.2, label="apic[52]  child"),
    Line2D([0], [0], color="#c05000", lw=2.4, label="apic[53]  child"),
]
leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04),
                ncol=4, frameon=False, fontsize=8.0, handlelength=1.8,
                columnspacing=1.6, borderaxespad=0)
for text, col in zip(leg.get_texts(), ["#1f4e9c", "#c0392b", "#2e7d32", "#c05000"]):
    text.set_color(col)
leg.get_texts()[1].set_fontweight("bold")

# full-cell inset
axi = fig.add_axes([0.76, 0.10, 0.23, 0.82])
axi.set_aspect("equal"); axi.axis("off")
for s in SECS:
    if s.name().endswith(("apic[50]", "apic[51]", "apic[52]", "apic[53]")):
        continue
    draw_sec(axi, s, "#d2d2d2", lw_min=0.20, lw_max=1.4, z=1)
for s, c in [(s50, "#1f4e9c"), (s51, "#c0392b"), (s52, "#2e7d32"), (s53, "#c05000")]:
    draw_sec(axi, s, c, lw_min=0.6, lw_max=2.0, z=3)
axi.set_xlim(-175, 245); axi.set_ylim(-210, 1200)
axi.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        fill=False, ec="0.45", lw=0.6, ls=(0, (2, 2)), zorder=10))
axi.text(-165, 1160, "whole cell", fontsize=6.5, color="0.45", va="top")
axi.text(0, -195, "zoom", fontsize=6.0, color="0.45", ha="center", va="top")

fig.savefig("figs/ppt_bg/segments/fig1_target_and_neighbors.png", dpi=600,
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figs/ppt_bg/segments/fig1_target_and_neighbors.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# =============================================================================
# Figure 2: apical_51 with 11 compartments; cable profile
# =============================================================================
sec = s51
nseg = int(sec.nseg)
L = float(sec.L)
colors = plt.cm.viridis(np.linspace(0.06, 0.94, nseg))

fig = plt.figure(figsize=(7.2, 4.4))
ax1 = fig.add_axes([0.06, 0.52, 0.91, 0.42])
ax1.set_aspect("equal"); ax1.axis("off")

for i in range(nseg):
    p0 = interp(sec, i / nseg); p1 = interp(sec, (i + 1) / nseg)
    q0 = proj_h(p0[:3]); q1 = proj_h(p1[:3])
    ax1.plot([q0[0], q1[0]], [q0[1], q1[1]], color=colors[i], lw=6.0,
             solid_capstyle="butt", zorder=3)
    ax1.plot([q1[0], q1[0]], [q1[1] - 2.8, q1[1] + 2.8], color="0.25",
             lw=0.6, zorder=4)
    xm = 0.5 * (q0[0] + q1[0]); ym = 0.5 * (q0[1] + q1[1])
    ax1.text(xm, ym + 5.2, str(i + 1), ha="center", va="bottom",
             fontsize=7.8, color="0.15", zorder=5)

ax1.set_xlim(646, 870)
ax1.set_ylim(-4, 38)
ax1.text(650, 32, "apical_51  (11 compartments)", fontsize=9.2, color="0.15")
scalebar(ax1, 650, -2, 50, "50 µm")
ax1.text(858, -2, "thickness exaggerated", fontsize=6.5, color="0.45",
         ha="right", va="bottom")

# cable profile
ax2 = fig.add_axes([0.09, 0.11, 0.88, 0.28])
for i in range(nseg):
    x0 = i * L / nseg; x1 = (i + 1) * L / nseg
    p0 = interp(sec, i / nseg); p1 = interp(sec, (i + 1) / nseg)
    d0, d1 = p0[3], p1[3]
    ax2.fill_between([x0, x1], [0, 0], [d0, d1], color=colors[i],
                     alpha=0.88, lw=0)
    ax2.plot([x0, x0], [0, max(d0, d1)], color="0.25", lw=0.5)
    ax2.text((x0 + x1) / 2, -0.12, str(i + 1), ha="center", va="top",
             fontsize=7.2, color="0.15")
ax2.plot([0, L], [0, 0], color="0.2", lw=0.8)
ax2.set_xlim(-4, L + 4)
ax2.set_ylim(-0.32, 1.75)
ax2.set_xlabel("arc length (µm)", labelpad=2)
ax2.set_ylabel("diameter (µm)", labelpad=2)
ax2.set_yticks([0, 0.5, 1.0, 1.5])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(labelsize=8)
ax2.text(L * 0.5, 1.58, f"L = {L:.1f} µm   |   nseg = {nseg}",
         ha="center", va="bottom", fontsize=8.2, color="0.2")

fig.savefig("figs/ppt_bg/segments/fig2_apical51_compartments.png", dpi=600,
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figs/ppt_bg/segments/fig2_apical51_compartments.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved fig1 and fig2")
