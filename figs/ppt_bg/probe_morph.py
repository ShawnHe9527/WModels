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
secs = list(h.allsec())
print("nsec", len(secs))
pts = []
for sec in secs:
    n = int(sec.n3d())
    for i in range(n):
        pts.append((sec.x3d(i), sec.y3d(i), sec.z3d(i), sec.diam3d(i), sec.name(), i))
pts = np.array([[p[0],p[1],p[2],p[3]] for p in pts])
print("x", pts[:,0].min(), pts[:,0].max())
print("y", pts[:,1].min(), pts[:,1].max())
print("z", pts[:,2].min(), pts[:,2].max())
print("diam", pts[:,3].min(), pts[:,3].max())
for sec in secs:
    if sec.name().endswith("apic[51]"):
        print("target", sec.name(), sec.n3d(), sec.nseg)
        for i in range(int(sec.n3d())):
            print(i, sec.x3d(i), sec.y3d(i), sec.z3d(i), sec.diam3d(i))
