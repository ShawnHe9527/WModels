import os, sys
from collections import defaultdict, Counter, deque
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

# ---------- section-level ----------
ch=defaultdict(list); par={}
for s in secs:
    p=s.parentseg()
    if p is not None:
        par[s]=p.sec
        ch[p.sec].append((s,float(p.x)))
deg={s:(len(ch[s])+(1 if s in par else 0)) for s in secs}
root=[s for s in secs if s not in par]
print("sections", len(secs), "compartments", sum(int(s.nseg) for s in secs), "roots", [r.name() for r in root])
print("section degree counts", dict(Counter(deg.values())))
B_sec=[s for s in secs if deg[s]!=2 or s in root]
print("section branch nodes", len(B_sec), "-> compressed branches", len(B_sec)-len(root))

# count compressed edges explicitly
def walk_to_junction(start, parent):
    cur=start
    while deg[cur]==2 and cur not in root:
        kids=[x for x,_ in ch[cur]]
        if len(kids)!=1:
            break
        cur=kids[0]
    return cur
branches_sec=[]
for u in B_sec:
    for v,x in ch[u]:
        w=walk_to_junction(v,u)
        branches_sec.append((u,w,x))
print("section compressed edges", len(branches_sec))

# ---------- compartment-level ----------
# node id (sec, seg)
ids={}
nodes=[]
for s in secs:
    n=int(s.nseg)
    for i in range(n):
        ids[(s,i)]=len(nodes); nodes.append((s,i))
N=len(nodes)
adj=[set() for _ in range(N)]
def add(a,b):
    adj[a].add(b); adj[b].add(a)
for s in secs:
    n=int(s.nseg)
    for i in range(n-1):
        add(ids[(s,i)], ids[(s,i+1)])
for s in secs:
    p=s.parentseg()
    if p is None: continue
    ps=p.sec; x=float(p.x); pn=int(ps.nseg)
    pi=min(max(int(x*pn),0), pn-1)
    add(ids[(ps,pi)], ids[(s,0)])
degc=[len(a) for a in adj]
root_ids=[ids[(r,0)] for r in root]
B_comp=[i for i,d in enumerate(degc) if d!=2 or i in root_ids]
print("compartment nodes", N, "degree counts", dict(Counter(degc)))
print("compartment branch nodes", len(B_comp), "-> compressed branches", len(B_comp)-len(root_ids))
print("leaves (deg1)", sum(1 for d in degc if d==1), "junctions (deg>=3)", sum(1 for d in degc if d>=3))
# Count branches by walking from branch nodes (compartment level)
def walk_comp(start, parent):
    cur=start
    while degc[cur]==2 and cur not in root_ids:
        nxt=[x for x in adj[cur] if x!=parent]
        if len(nxt)!=1:
            break
        parent,cur=cur,nxt[0]
    return cur
br_comp=[]
for u in B_comp:
    for v in adj[u]:
        # only walk "down" from root orientation to avoid double count: use parent relation
        # Build parent map at compartment level
        pass
# parent map from root BFS
parent={}
order=[root_ids[0]]; seen={root_ids[0]}
for u in order:
    for v in adj[u]:
        if v not in seen:
            seen.add(v); parent[v]=u; order.append(v)
for u in B_comp:
    for v in adj[u]:
        if parent.get(v)==u:  # u is parent of v
            w=walk_comp(v,u)
            br_comp.append((u,w))
print("compartment compressed edges", len(br_comp))
# region breakdown of section-level branches by parent region
def region(name):
    if ".apic[" in name: return "apical"
    if ".basal[" in name or ".dend[" in name: return "basal"
    if ".axon[" in name: return "axon"
    if ".soma[" in name: return "soma"
    return "other"
print("branch start region", Counter(region(u.name()) for u,_,_ in branches_sec))
print("section counts", Counter(region(s.name()) for s in secs))
