"""Clean paper-style L5PC morphology colored by 26 functional branches. No labels/table/title."""
import os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
def descendants(root):
    out=[]; stack=[root]
    while stack:
        s=stack.pop(); out.append(s)
        for c,_ in ch[s]: stack.append(c)
    return out

soma=[s for s in secs if '.soma[' in s.name()][0]
axon=[s for s in secs if '.axon[' in s.name()]
basal_roots=[c for c,_ in ch[soma] if '.dend[' in c.name()]

groups=[]
def add(name, region, parent, ss):
    groups.append(dict(name=name, region=region, parent=parent, secs=list(ss)))
add('SOMA','soma','-', [soma])
add('AXON','axon','SOMA', axon)
for i,r in enumerate(basal_roots,1):
    add(f'B{i}','basal','SOMA', descendants(r))
A={idx(s):s for s in secs if '.apic[' in s.name()}
add('A_TRUNK_PROX','apical','SOMA', [A[i] for i in [0,1,2,3,4]])
add('A_TRUNK_MID','apical','A_TRUNK_PROX', [A[i] for i in [14,20,26,34]])
add('A_TRUNK_DIST','apical','A_TRUNK_MID', [A[i] for i in [36,50]])
obl=[(104,'A_TRUNK_PROX'),(99,'A_TRUNK_PROX'),(80,'A_TRUNK_PROX'),(79,'A_TRUNK_PROX'),(5,'A_TRUNK_PROX'),
     (15,'A_TRUNK_MID'),(21,'A_TRUNK_MID'),(27,'A_TRUNK_MID'),(35,'A_TRUNK_MID'),(37,'A_TRUNK_DIST')]
for k,(i,parent) in enumerate(obl,1):
    add(f'A_OBL_{k:02d}','apical',parent, descendants(A[i]))
add('A_TUFT_T1','apical','A_TRUNK_DIST', descendants(A[51]))
add('A_TUFT_T2a','apical','A_TRUNK_DIST', [A[60]]+descendants(A[61]))
add('A_TUFT_T2b','apical','A_TUFT_T2a', descendants(A[68]))

allg=[s for g in groups for s in g['secs']]
assert len(allg)==len(set(allg))==len(secs)==196

# Paper-like palette: cool basal, warm trunk, purple oblique, red tuft.
basal_cols=['#2E6F9E','#3C8DAD','#4FA3A5','#6BBF8A','#2F8F5B','#7FB3D5','#A3D9C9','#8CC084']
trunk_cols=['#E67E22','#D35400','#F39C12']
obl_cols=['#6C3483','#8E44AD','#A569BD','#BB8FCE','#7D3C98','#C2185B','#E91E63','#F48FB1','#5B2C6F','#9B59B6']
tuft_cols=['#C0392B','#E74C3C','#F1948A']
colors={}
for g in groups:
    if g['region']=='soma': colors[g['name']]='#4D5656'
    elif g['region']=='axon': colors[g['name']]='#B7950B'
    elif g['region']=='basal': colors[g['name']]=basal_cols.pop(0)
    elif g['name'].startswith('A_TRUNK'): colors[g['name']]=trunk_cols.pop(0)
    elif g['name'].startswith('A_OBL'): colors[g['name']]=obl_cols.pop(0)
    elif g['name'].startswith('A_TUFT'): colors[g['name']]=tuft_cols.pop(0)

sec2g={s:g for g in groups for s in g['secs']}

fig=plt.figure(figsize=(7.2,9.2), dpi=300, facecolor='white')
ax=fig.add_axes([0,0,1,1])
ax.set_aspect('equal'); ax.axis('off')

allx=[]; ally=[]
for s in secs:
    n=int(s.n3d())
    if n<2: continue
    g=sec2g[s]
    xs=[float(s.x3d(i)) for i in range(n)]
    ys=[float(s.y3d(i)) for i in range(n)]
    ax.plot(xs, ys, color=colors[g['name']], lw=1.55,
            solid_capstyle='round', solid_joinstyle='round', zorder=2)
    allx += xs; ally += ys

# soma marker only (no text)
soma_pts=[]
for s in [soma]:
    n=int(s.n3d())
    for i in range(n):
        soma_pts.append((float(s.x3d(i)), float(s.y3d(i))))
if soma_pts:
    sx=np.mean([p[0] for p in soma_pts]); sy=np.mean([p[1] for p in soma_pts])
    ax.scatter([sx],[sy],s=34,c='#1F2937',edgecolor='white',linewidth=0.5,zorder=5)

allx=np.asarray(allx); ally=np.asarray(ally)
dx=allx.max()-allx.min(); dy=ally.max()-ally.min()
ax.set_xlim(allx.min()-0.03*dx, allx.max()+0.03*dx)
ax.set_ylim(ally.min()-0.03*dy, ally.max()+0.03*dy)
# no title, no labels, no legend, no axes

out='figs/ppt_bg/out'
fig.savefig(os.path.join(out,'l5pc_functional_groups_clean.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
fig.savefig(os.path.join(out,'l5pc_functional_groups_clean.pdf'),
            bbox_inches='tight', pad_inches=0.02, facecolor='white')
print('saved clean figure')
