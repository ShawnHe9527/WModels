import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
from neuron import h
import world.gen.sim2 as S2
import numpy as np

h.nrn_load_dll(S2.MODDIR)
cwd = os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc", "L5PCtemplate_2.hoc"):
    h.load_file(f)
os.chdir(cwd)
cell = h.L5PCtemplate(S2.MORPH)
secs = list(h.allsec())

for name in ("apic[51]", "apic[50]", "apic[52]", "apic[53]"):
    for s in secs:
        if s.name().endswith(name):
            ps = s.psection()
            print("===", name, "nseg", s.nseg)
            for m, d in ps["density_mechs"].items():
                print(" ", m, {k: d[k] for k in d if k in ("gbar","gkbar","gNa","gCa","gIm","gIh")})
