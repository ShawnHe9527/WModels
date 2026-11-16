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
soma = cell.soma[0](0.5)
ax0 = cell.axon[0](0.5)
ax1 = cell.axon[1](0.5)
ap = cell.apic[51](0.5)

ic = h.IClamp(soma)
ic.delay, ic.dur, ic.amp = 100.0, 2.0, 4.0
vs = h.Vector().record(soma._ref_v)
va0 = h.Vector().record(ax0._ref_v)
va1 = h.Vector().record(ax1._ref_v)
vap = h.Vector().record(ap._ref_v)
t = h.Vector().record(h._ref_t)

h.tstop = 160.0
h.dt = 0.025
h.finitialize(-76.0)
h.continuerun(h.tstop)
T = np.asarray(t); VS = np.asarray(vs); VA0 = np.asarray(va0); VA1 = np.asarray(va1); VAP = np.asarray(vap)

def first_cross(V, thr=0.0):
    idx = np.where(V > thr)[0]
    return (T[idx[0]] if len(idx) else None, V.max())
for name, V in [("soma", VS), ("axon[0]", VA0), ("axon[1]", VA1), ("apic[51]", VAP)]:
    tc, vmax = first_cross(V)
    print(f"{name:9s} first>0 at {tc} ms, Vmax={vmax:.1f}")
# peak times
for name, V in [("soma", VS), ("axon[0]", VA0), ("axon[1]", VA1), ("apic[51]", VAP)]:
    print(f"{name:9s} peak at {T[np.argmax(V)]:.3f} ms, Vmax={V.max():.1f}")
