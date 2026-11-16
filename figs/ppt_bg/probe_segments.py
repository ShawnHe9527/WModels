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

for idx in (49, 50, 51, 52, 53, 54):
    sec = get(idx)
    print(f"apic[{idx}] L={sec.L:.3f} nseg={sec.nseg} diam={sec.diam:.3f} n3d={sec.n3d()}")
    print("  first", sec.x3d(0), sec.y3d(0), sec.z3d(0), sec.diam3d(0))
    print("  last ", sec.x3d(sec.n3d()-1), sec.y3d(sec.n3d()-1), sec.z3d(sec.n3d()-1), sec.diam3d(sec.n3d()-1))
    ps = sec.parentseg()
    if ps is not None:
        print("  parent", ps.sec.name(), ps.x)
    print("  children", [c.name() for c in sec.children()])

sec = get(51)
print("target arc3d", [round(sec.arc3d(i),2) for i in range(sec.n3d())])
print("target diam3d", [round(sec.diam3d(i),2) for i in range(sec.n3d())])
