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

def idx(s):
    name=s.name().split('.')[-1]
    return int(name.split('[')[1].split(']')[0])
def nm(s): return s.name().split('.')[-1]
def descendants(root):
    out=[]
    stack=[root]
    while stack:
        s=stack.pop(); out.append(s)
        for c,_ in ch[s]: stack.append(c)
    return out

soma=[s for s in secs if '.soma[' in s.name()][0]
apic_root=[s for s in secs if '.apic[0]' in s.name()][0]
axon=[s for s in secs if '.axon[' in s.name()]
basal_roots=[c for c,_ in ch[soma] if '.dend[' in c.name()]

groups=[]  # dict(name, region, parent, secs)
def add(name, region, parent, ss):
    groups.append(dict(name=name, region=region, parent=parent, secs=list(ss)))

add('SOMA','soma','-', [soma])
add('AXON','axon','SOMA', axon)
for i,r in enumerate(basal_roots,1):
    add(f'B{i}','basal','SOMA', descendants(r))

# apical index map
A={idx(s):s for s in secs if '.apic[' in s.name()}
# trunk path chosen by largest-diameter continuation to apic[50]
trunk_idx=[0,1,2,3,4,14,20,26,34,36,50]
trunk=A
# groups
prox=[A[i] for i in [0,1,2,3,4]]
mid=[A[i] for i in [14,20,26,34]]
dist=[A[i] for i in [36,50]]
add('A_TRUNK_PROX','apical','SOMA', prox)
add('A_TRUNK_MID','apical','A_TRUNK_PROX', mid)
add('A_TRUNK_DIST','apical','A_TRUNK_MID', dist)

# oblique subtrees: (root idx, parent trunk group)
obl=[
 (104,'A_TRUNK_PROX'), (99,'A_TRUNK_PROX'), (80,'A_TRUNK_PROX'),
 (79,'A_TRUNK_PROX'), (5,'A_TRUNK_PROX'),
 (15,'A_TRUNK_MID'), (21,'A_TRUNK_MID'), (27,'A_TRUNK_MID'), (35,'A_TRUNK_MID'),
 (37,'A_TRUNK_DIST'),
]
for k,(i,parent) in enumerate(obl,1):
    add(f'A_OBL_{k:02d}', 'apical', parent, descendants(A[i]))

# tuft: T1=apic[51] subtree; T2a=apic[60]+subtree(61); T2b=subtree(68)
t1=descendants(A[51])
t2a=[A[60]]+descendants(A[61])
t2b=descendants(A[68])
add('A_TUFT_T1','apical','A_TRUNK_DIST', t1)
add('A_TUFT_T2a','apical','A_TRUNK_DIST', t2a)
add('A_TUFT_T2b','apical','A_TUFT_T2a', t2b)

# verify coverage
allg=[s for g in groups for s in g['secs']]
print('groups',len(groups),'sections assigned',len(allg),'unique',len(set(allg)),'total',len(secs))
from collections import Counter
c=Counter(allg)
dup=[s for s,n in c.items() if n>1]
miss=[s for s in secs if s not in c]
print('dup', [nm(s) for s in dup], 'miss', [nm(s) for s in miss])
if dup or miss: raise SystemExit(1)

# sort sections within group by index
def key(s):
    name=s.name()
    for pref in ['.soma[','.dend[','.apic[','.axon[']:
        if pref in name:
            return (pref, idx(s))
    return (name,0)
for g in groups:
    g['secs'].sort(key=key)
    g['n_sec']=len(g['secs'])
    g['n_comp']=sum(int(s.nseg) for s in g['secs'])

# print table
print('\nIDX\tGROUP\tREGION\tPARENT\tSEC\tCOMP\tSECTIONS')
for g in groups:
    names=','.join(nm(s) for s in g['secs'])
    print(f"{groups.index(g)+1:02d}\t{g['name']}\t{g['region']}\t{g['parent']}\t{g['n_sec']}\t{g['n_comp']}\t{names}")

# write csv
out='figs/ppt_bg/out/l5pc_functional_branches.csv'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['group_id','group','region','parent_group','n_sections','n_compartments','sections'])
    for i,g in enumerate(groups,1):
        w.writerow([i,g['name'],g['region'],g['parent'],g['n_sec'],g['n_comp'],';'.join(nm(s) for s in g['secs'])])
print('wrote',out)
