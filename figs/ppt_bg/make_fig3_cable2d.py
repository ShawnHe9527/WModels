import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
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

sec = get(51)
nseg = int(sec.nseg)
i_comp = 5
p0 = interp(sec, i_comp / nseg)
p1 = interp(sec, (i_comp + 1) / nseg)
L_seg = float(sec.L) / nseg
d = 0.5 * (p0[3] + p1[3])
print("comp", i_comp + 1, "L", L_seg, "d", d)

fig = plt.figure(figsize=(7.4, 3.2))
ax = fig.add_axes([0.03, 0.10, 0.70, 0.80])
ax.set_aspect("auto")
ax.axis("off")

h_vis = 2.2  # visual height; true diameter is annotated
cmap = LinearSegmentedColormap.from_list(
    "membrane", ["#6f4428", "#b87a4b", "#f3ddc2", "#b87a4b", "#6f4428"])
grad = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(grad, extent=[0, L_seg, -h_vis / 2, h_vis / 2], aspect="auto",
          origin="lower", cmap=cmap, zorder=1)
ax.plot([0, L_seg], [h_vis / 2, h_vis / 2], color="0.20", lw=0.8, zorder=3)
ax.plot([0, L_seg], [-h_vis / 2, -h_vis / 2], color="0.20", lw=0.8, zorder=3)
ax.plot([0.5, L_seg - 0.5], [h_vis * 0.26, h_vis * 0.26], color="white",
        alpha=0.35, lw=2.0, zorder=2)

# end cap (cross-section)
ax.add_patch(Ellipse((L_seg, 0), width=1.1, height=h_vis, facecolor="#8c5a3c",
                     edgecolor="0.2", lw=0.8, zorder=4))
ax.add_patch(Ellipse((L_seg, 0), width=0.45, height=h_vis * 0.62, facecolor="#5e3720",
                     edgecolor="none", zorder=5))

# length dimension
yb = -h_vis / 2 - 0.9
ax.annotate("", xy=(L_seg, yb), xytext=(0, yb),
            arrowprops=dict(arrowstyle="<->", color="0.2", lw=0.8))
for x0 in (0, L_seg):
    ax.plot([x0, x0], [yb - 0.25, yb + 0.25], color="0.2", lw=0.8)
ax.text(L_seg / 2, yb - 0.45, f"L = {L_seg:.2f} µm", ha="center", va="top",
        fontsize=8.6, color="0.2")

# diameter dimension
xd = L_seg + 1.2
ax.annotate("", xy=(xd, h_vis / 2), xytext=(xd, -h_vis / 2),
            arrowprops=dict(arrowstyle="<->", color="0.2", lw=0.8))
for yy in (-h_vis / 2, h_vis / 2):
    ax.plot([xd - 0.25, xd + 0.25], [yy, yy], color="0.2", lw=0.8)
ax.text(xd + 0.35, 0, f"d = {d:.2f} µm", ha="left", va="center",
        fontsize=8.6, color="0.2")

ax.set_xlim(-1.0, L_seg + 4.2)
ax.set_ylim(-h_vis / 2 - 2.0, h_vis / 2 + 1.0)
ax.text(0, h_vis / 2 + 0.55, "apical_51 · compartment 6", fontsize=9.2,
        color="0.15", ha="left", va="bottom")
ax.text(0, h_vis / 2 + 0.15, "single cable segment", fontsize=7.5,
        color="0.45", ha="left", va="bottom")

# inset: section with compartment 6 highlighted
axi = fig.add_axes([0.73, 0.54, 0.25, 0.40])
axi.set_aspect("equal"); axi.axis("off")
proj_h = lambda p: (p[1], p[0])
for i in range(nseg):
    q0 = proj_h(interp(sec, i / nseg)[:3])
    q1 = proj_h(interp(sec, (i + 1) / nseg)[:3])
    axi.plot([q0[0], q1[0]], [q0[1], q1[1]], color="#c8c8c8", lw=4.0,
             solid_capstyle="butt", zorder=2)
q0 = proj_h(p0[:3]); q1 = proj_h(p1[:3])
axi.plot([q0[0], q1[0]], [q0[1], q1[1]], color="#c0392b", lw=6.0,
         solid_capstyle="butt", zorder=4)
axi.text(0.5 * (q0[0] + q1[0]), 0.5 * (q0[1] + q1[1]) + 6,
         f"compartment {i_comp + 1}", color="#c0392b", fontsize=7.5,
         ha="center", va="bottom")
axi.set_xlim(646, 870); axi.set_ylim(-4, 36)
axi.plot([650, 700], [-1.5, -1.5], color="0.2", lw=2.0, solid_capstyle="butt")
axi.text(675, -3.0, "50 µm", ha="center", va="top", fontsize=6.5, color="0.2")

fig.text(0.37, 0.025,
         "diameter exaggerated for visibility; L and d from NEURON geometry",
         ha="center", va="bottom", fontsize=6.5, color="0.45")

fig.savefig("figs/ppt_bg/segments/fig3_single_compartment_cable.png", dpi=600,
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figs/ppt_bg/segments/fig3_single_compartment_cable.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved fig3 2d")
