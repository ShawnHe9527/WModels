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

# set origin at soma 0.5
h.distance(0.5, sec=cell.soma[0])
for idx in (50, 51, 52, 53):
    for sec in h.allsec():
        if sec.name().endswith(f"apic[{idx}]"):
            sec.push()
            ds = [h.distance((i+0.5)/sec.nseg) for i in range(int(sec.nseg))]
            h.pop_section()
            print(f"apic[{idx}] nseg={int(sec.nseg)} L={sec.L:.3f} dist={[round(d,1) for d in ds]}")
