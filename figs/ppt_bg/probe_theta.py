import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
from neuron import h
import world.gen.sim2 as S2
from world.gen import secinfo as SI

h.nrn_load_dll(S2.MODDIR)
cwd=os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc","import3d.hoc","L5PCbiophys5b.hoc","L5PCtemplate_2.hoc"): h.load_file(f)
os.chdir(cwd)
cell=h.L5PCtemplate(S2.MORPH)
sec=None
for s in h.allsec():
    if s.name().endswith("apic[51]"):
        sec=s; break
th, readable = SI.build_theta(sec, "apical")
print("dim", th.shape)
for i, (name, val) in enumerate(zip(SI.THETA_LAYOUT, th)):
    print(f"{i:2d} {name:20s} {val:.6g}")
print("diam_mean", readable["geom"]["diam_mean_um"], "L", readable["geom"]["L_um"], "nseg", readable["geom"]["nseg"])
