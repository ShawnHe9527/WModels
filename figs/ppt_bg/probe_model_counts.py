import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
from neuron import h
import world.gen.sim2 as S2

h.nrn_load_dll(S2.MODDIR)
cwd = os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc", "L5PCtemplate_2.hoc"):
    h.load_file(f)
os.chdir(cwd)
cell = h.L5PCtemplate(S2.MORPH)
secs = list(h.allsec())
print("nsec", len(secs), "sum nseg", sum(int(s.nseg) for s in secs))
from collections import Counter
print("nseg dist", Counter(int(s.nseg) for s in secs))
mechs = Counter()
for s in secs:
    ps = s.psection()
    for m in ps.get("density_mechs", {}):
        mechs[m] += 1
print("density mechs", dict(mechs))
# section lengths around target
for i in (50,51,52,53):
    for s in secs:
        if s.name().endswith(f"apic[{i}]"):
            print(i, "L", s.L, "nseg", int(s.nseg), "formula", 1+2*int(s.L/40))
