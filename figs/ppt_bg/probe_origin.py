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
for origin in (0.0, 0.5, 1.0):
    h.distance(origin, sec=cell.soma[0])
    vals=[]
    for idx in (50,51):
        for sec in h.allsec():
            if sec.name().endswith(f"apic[{idx}]"):
                sec.push()
                ds=[h.distance((i+0.5)/sec.nseg) for i in range(int(sec.nseg))]
                h.pop_section()
                vals.append((idx, round(ds[-1],1), round(ds[0],1)))
    print("origin", origin, vals)
