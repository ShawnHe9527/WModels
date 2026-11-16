import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
import numpy as np
from neuron import h
import world.gen.sim2 as S2

h.nrn_load_dll(S2.MODDIR)
cwd = os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc", "L5PCtemplate_2.hoc"):
    h.load_file(f)
os.chdir(cwd)
cell = h.L5PCtemplate(S2.MORPH)

def get(idx):
    for sec in h.allsec():
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
print("L", sec.L, "nseg", sec.nseg, "diam", sec.diam)
for i in range(sec.nseg):
    p0 = interp(sec, i/sec.nseg)
    p1 = interp(sec, (i+1)/sec.nseg)
    print(f"comp {i+1:2d} p0=({p0[0]:7.2f},{p0[1]:7.2f},{p0[2]:6.2f}) p1=({p1[0]:7.2f},{p1[1]:7.2f},{p1[2]:6.2f}) d0={p0[3]:.3f} d1={p1[3]:.3f} len={sec.L/sec.nseg:.3f}")

for idx in (50,51,52,53,60):
    sec = get(idx)
    pts = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())])
    print(f"apic[{idx}] bbox x {pts[:,0].min():.1f}..{pts[:,0].max():.1f} y {pts[:,1].min():.1f}..{pts[:,1].max():.1f} z {pts[:,2].min():.1f}..{pts[:,2].max():.1f}")
