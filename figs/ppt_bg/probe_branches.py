import os, sys
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
from neuron import h
import world.gen.sim2 as S2

h.nrn_load_dll(S2.MODDIR)
cwd=os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc","import3d.hoc","L5PCbiophys5b.hoc","L5PCtemplate_2.hoc"):
    h.load_file(f)
os.chdir(cwd)
cell=h.L5PCtemplate(S2.MORPH)
secs=list(h.allsec())
print("n_sections", len(secs))
print("n_comp", sum(int(s.nseg) for s in secs))
roots=[]; no_parent=[]
for s in secs:
    p=s.parentseg()
    if p is None:
        roots.append(s)
        no_parent.append(s.name())
print("roots", no_parent)
for s in secs[:8]:
    p=s.parentseg()
    print(s.name(), "nseg", int(s.nseg), "parent", None if p is None else (p.sec.name(), float(p.x)))
# children count / degree at section level
from collections import defaultdict
ch=defaultdict(list); par={}
for s in secs:
    p=s.parentseg()
    if p is not None:
        par[s]=p.sec
        ch[p.sec].append((s,p.x))
deg={s:(len(ch[s])+(1 if s in par else 0)) for s in secs}
from collections import Counter
print("section degree counts", Counter(deg.values()))
print("junction sections", sum(1 for s in secs if deg[s]!=2))
print("leaves", sum(1 for s in secs if deg[s]==1))
print("root sections", len(roots))
# print a few junctions
j=[s for s in secs if deg[s]!=2]
for s in j[:20]:
    print("J", s.name(), "deg", deg[s], "children", [(x.name(), round(float(xx),3)) for x,xx in ch[s]])
