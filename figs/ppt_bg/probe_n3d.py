import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
from neuron import h
import world.gen.sim2 as S2
h.nrn_load_dll(S2.MODDIR)
cwd=os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc","import3d.hoc","L5PCbiophys5b.hoc","L5PCtemplate_2.hoc"): h.load_file(f)
os.chdir(cwd)
cell=h.L5PCtemplate(S2.MORPH)
for s in h.allsec():
    if int(s.n3d())==0:
        print("n3d=0", s.name(), "L", s.L, "nseg", s.nseg)
