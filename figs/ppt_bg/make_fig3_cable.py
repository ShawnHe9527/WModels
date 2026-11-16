import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import FancyArrowPatch
from neuron import h
import world.gen.sim2 as S2

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
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
i_comp = 5                      # zero-based -> compartment 6
p0 = interp(sec, i_comp / nseg)
p1 = interp(sec, (i_comp + 1) / nseg)
L_seg = float(sec.L) / nseg
d0, d1 = p0[3], p1[3]
r0, r1 = d0 / 2.0, d1 / 2.0
print("comp", i_comp + 1, "L", L_seg, "d0/d1", d0, d1)
print("p0", p0[:3], "p1", p1[:3])

# -----------------------------------------------------------------------------
# 3D cable (true radius, visually exaggerated box aspect)
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_axes([0.02, 0.10, 0.70, 0.82], projection="3d")

nu, nt = 90, 80
u = np.linspace(0, L_seg, nu)
th = np.linspace(0, 2 * np.pi, nt)
U, T = np.meshgrid(u, th)
R = r0 + (r1 - r0) * (U / L_seg)
X = U
Y = R * np.cos(T)
Z = R * np.sin(T)
ls = LightSource(azdeg=225, altdeg=45)
ax.plot_surface(X, Y, Z, color="#d8b48c", shade=True, lightsource=ls,
                linewidth=0, antialiased=True, rstride=1, cstride=2)

# end caps
for x0, rr in ((0.0, r0), (L_seg, r1)):
    rad = np.linspace(0, rr, 18)
    RR, TT = np.meshgrid(rad, th)
    XX = np.full_like(RR, x0)
    YY = RR * np.cos(TT)
    ZZ = RR * np.sin(TT)
    ax.plot_surface(XX, YY, ZZ, color="#8c5a3c", shade=False,
                    linewidth=0, antialiased=True, alpha=1.0)
    ax.plot(x0 + 0 * np.cos(th), rr * np.cos(th), rr * np.sin(th),
            color="0.15", lw=0.8)

# subtle top line to read the axis
ax.plot([0, L_seg], [0, 0], [r0, r1], color="0.25", lw=0.5, alpha=0.7)

# length bracket below
yb = -1.55 * max(r0, r1)
ax.plot([0, L_seg], [yb, yb], [0, 0], color="0.2", lw=0.9)
for x0 in (0, L_seg):
    ax.plot([x0, x0], [yb - 0.25 * max(r0, r1), yb + 0.25 * max(r0, r1)],
            [0, 0], color="0.2", lw=0.9)
ax.text(L_seg / 2, yb - 0.7 * max(r0, r1), 0,
        f"L = {L_seg:.2f} µm", ha="center", va="top", fontsize=8.5, color="0.2")

# diameter arrow at the far end
xd = L_seg + 0.35
ax.plot([xd, xd], [-r1, r1], [0, 0], color="0.2", lw=0.9)
for yy in (-r1, r1):
    ax.plot([xd - 0.2, xd + 0.2], [yy, yy], [0, 0], color="0.2", lw=0.9)
ax.text(xd + 0.25, 0, 0, f"d = {d1:.2f} µm", ha="left", va="center",
        fontsize=8.5, color="0.2")

ax.set_box_aspect((L_seg, 3.0 * 2 * r1, 3.0 * 2 * r1))
ax.view_init(elev=18, azim=-68)
ax.set_axis_off()
ax.set_xlim(-1.0, L_seg + 2.4)
ax.set_ylim(-2.2, 2.2)
ax.set_zlim(-2.2, 2.2)

# -----------------------------------------------------------------------------
# inset: section with compartment 6 highlighted
# -----------------------------------------------------------------------------
axi = fig.add_axes([0.70, 0.52, 0.28, 0.40])
axi.set_aspect("equal"); axi.axis("off")

proj_h = lambda p: (p[1], p[0])
# draw all compartments faintly
for i in range(nseg):
    q0 = proj_h(interp(sec, i / nseg)[:3])
    q1 = proj_h(interp(sec, (i + 1) / nseg)[:3])
    axi.plot([q0[0], q1[0]], [q0[1], q1[1]], color="#c8c8c8", lw=4.0,
             solid_capstyle="butt", zorder=2)
# highlight comp 6
q0 = proj_h(p0[:3]); q1 = proj_h(p1[:3])
axi.plot([q0[0], q1[0]], [q0[1], q1[1]], color="#c0392b", lw=6.0,
         solid_capstyle="butt", zorder=4)
axi.text(0.5 * (q0[0] + q1[0]), 0.5 * (q0[1] + q1[1]) + 6,
         f"compartment {i_comp + 1}", color="#c0392b", fontsize=7.5,
         ha="center", va="bottom")
axi.set_xlim(646, 870)
axi.set_ylim(-4, 35)

# scale bar
axi.plot([650, 700], [-1.5, -1.5], color="0.2", lw=2.0, solid_capstyle="butt")
axi.text(675, -3.0, "50 µm", ha="center", va="top", fontsize=6.5, color="0.2")

# note on visual exaggeration
fig.text(0.365, 0.03, "diameter exaggerated for visibility; values from NEURON geometry",
         ha="center", va="bottom", fontsize=6.5, color="0.45")

fig.savefig("figs/ppt_bg/segments/fig3_single_compartment_cable.png", dpi=600,
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figs/ppt_bg/segments/fig3_single_compartment_cable.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved fig3")
