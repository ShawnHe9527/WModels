import os, sys, csv
from collections import defaultdict
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
ch=defaultdict(list); par={}
for s in secs:
    p=s.parentseg()
    if p is not None:
        par[s]=p.sec
        ch[p.sec].append((s,float(p.x)))
# subtree section count, total length
sub={}
def dfs(s):
    sub[s]=1+sum(dfs(c) for c,_ in ch[s])
    return sub[s]
for r in [s for s in secs if s not in par]: dfs(r)
# distance from soma (sum of parent L; approximate at x=1)
dist={}
for s in secs:
    if s not in par: dist[s]=0.0
    else:
        p=par[s]; dist[s]=dist[p]+float(p.L)
def nm(s): return s.name().split('.')[-1]
apic=[s for s in secs if '.apic[' in s.name()]
print("apic count", len(apic))
for s in sorted(apic, key=lambda x:int(nm(x).split('[')[1].split(']')[0])):
    kids=[(nm(c),round(float(x),2)) for c,x in ch[s]]
    print(f"{nm(s):10s} L={float(s.L):7.1f} d={float(s.diam):6.2f} nseg={int(s.nseg):2d} sub={sub[s]:3d} dist={dist[s]:7.1f} parent={nm(par[s]) if s in par else '-':10s} kids={kids}")
