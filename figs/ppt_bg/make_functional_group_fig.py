"""Whole-neuron functional-branch grouping figure (26 groups)."""
import os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager

sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
from neuron import h
import world.gen.sim2 as S2

for fam in ["Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP"]:
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ---------------- load cell ----------------
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
    groups.append(dict(name=name, region=region, parent=parent, secs=list(ss), id=len(groups)+1))
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

soma_center=None
for g in groups:
    g['n_sec']=len(g['secs'])
    g['n_comp']=sum(int(s.nseg) for s in g['secs'])
    pts=[]
    for s in g['secs']:
        n=int(s.n3d())
        for i in range(n):
            pts.append((float(s.x3d(i)), float(s.y3d(i)), float(s.z3d(i))))
    g['pts']=np.asarray(pts, dtype=float)

soma_center=groups[0]['pts'].mean(axis=0)
for g in groups:
    if g['pts'].size == 0:
        base=soma_center[:2]
        g['pts']=np.array([[base[0]+8, base[1]-8, 0.0],
                           [base[0]+22, base[1]-14, 0.0]])
    g['center']=g['pts'].mean(axis=0)
    g['tip']=g['pts'][np.argmax(np.linalg.norm(g['pts']-soma_center,axis=1))]

basal_cols=['#1F77B4','#17BECF','#2CA02C','#66C2A5','#1B9E77','#4A90E2','#7FCDBB','#98DF8A']
trunk_cols=['#FF7F0E','#E6550D','#FD8D3C']
obl_cols=['#6A3D9A','#A55194','#CE6DBD','#DE9ED6','#7B4173','#C51B7D','#E7298A','#F1B6DA','#8C6BB1','#9E9AC8']
tuft_cols=['#D62728','#FF9896','#E31A1C']
colors={}
for g in groups:
    if g['region']=='soma': colors[g['name']]='#6B7280'
    elif g['region']=='axon': colors[g['name']]='#D4A017'
    elif g['region']=='basal': colors[g['name']]=basal_cols.pop(0)
    elif g['name'].startswith('A_TRUNK'): colors[g['name']]=trunk_cols.pop(0)
    elif g['name'].startswith('A_OBL'): colors[g['name']]=obl_cols.pop(0)
    elif g['name'].startswith('A_TUFT'): colors[g['name']]=tuft_cols.pop(0)

sec2g={s:g for g in groups for s in g['secs']}

# ---------------- figure ----------------
fig=plt.figure(figsize=(16.5,9.4), dpi=220, facecolor='white')
ax=fig.add_axes([0.015,0.075,0.665,0.80])
ax.set_aspect('equal'); ax.axis('off')

for s in secs:
    g=sec2g[s]; c=colors[g['name']]
    n=int(s.n3d())
    if n<2:
        continue
    xs=[float(s.x3d(i)) for i in range(n)]
    ys=[float(s.y3d(i)) for i in range(n)]
    lw=3.2 if g['name']=='A_TUFT_T1' else 2.0
    ax.plot(xs, ys, color=c, lw=lw, solid_capstyle='round', zorder=4 if g['name']=='A_TUFT_T1' else 2)

sx,sy,_=groups[0]['center']
ax.scatter([sx],[sy],s=120,c='#374151',edgecolor='white',linewidth=1.2,zorder=7)
ax.text(sx-30,sy+18,'SOMA',fontsize=9,color='#374151',fontweight='bold',zorder=8,
        bbox=dict(boxstyle='round,pad=0.18',fc='white',ec='none',alpha=0.85))

# selected labels only (full mapping in table)
selected={'AXON','B1','B2','B3','B4','B5','B6','B7','B8',
          'A_TRUNK_DIST','A_OBL_10',
          'A_TUFT_T1','A_TUFT_T2a','A_TUFT_T2b'}
# manual label positions (data coords); fallback tip
manual={
 'AXON': (sx+78, sy-40),
 'B1': None, 'B3': None, 'B5': None, 'B7': None,
 'A_TRUNK_DIST': (sx-55, sy+560),
 'A_OBL_10': None,
 'A_TUFT_T1': None, 'A_TUFT_T2a': None, 'A_TUFT_T2b': None,
}
allpts=np.vstack([g['pts'] for g in groups])
xmin,ymin=allpts[:,:2].min(axis=0); xmax,ymax=allpts[:,:2].max(axis=0)
dx=xmax-xmin; dy=ymax-ymin
for g in groups:
    if g['name'] not in selected: continue
    p=g['tip']; ctr=np.array([sx,sy,0.0])
    v=p-ctr
    if np.linalg.norm(v[:2])<1e-6: v=np.array([0,1,0])
    off=v[:2]/np.linalg.norm(v[:2])*max(dx,dy)*0.030
    xy=manual.get(g['name'])
    if xy is None:
        xy=(p[0]+off[0],p[1]+off[1])
    label=g['name'].replace('A_','')
    ax.text(xy[0],xy[1],label,fontsize=7.6,color='#111827',ha='center',va='center',zorder=9,
            bbox=dict(boxstyle='round,pad=0.18',fc='white',ec='none',alpha=0.82))

# current target
s51=[s for s in secs if '.apic[51]' in s.name()][0]
n51=int(s51.n3d()); i51=n51//2
tx,ty=float(s51.x3d(i51)),float(s51.y3d(i51))
ax.scatter([tx],[ty],marker='*',s=280,c='#E31A1C',edgecolor='white',linewidth=1.0,zorder=10)
ax.annotate('current target\napic[51] → TUFT_T1',xy=(tx,ty),xytext=(tx+0.17*dx,ty-0.09*dy),
            fontsize=8.0,color='#B91C1C',fontweight='bold',
            arrowprops=dict(arrowstyle='->',color='#B91C1C',lw=1.4),
            bbox=dict(boxstyle='round,pad=0.25',fc='white',ec='#FCA5A5',alpha=0.95),zorder=11)

ax.set_xlim(xmin-0.05*dx, xmax+0.10*dx)
ax.set_ylim(ymin-0.10*dy, ymax+0.12*dy)

fig.text(0.015,0.975,'L5PC 全神经元 functional branch 分组：196 sections → 26 groups',
         fontsize=15,fontweight='bold',color='#111827',va='top')
fig.text(0.015,0.938,'颜色按功能区域分组；标签为功能组缩写；完整 section 列表见右表；红星为当前目标段 apic[51]',
         fontsize=8.8,color='#4B5563',va='top')

# ---------------- table ----------------
lx=fig.add_axes([0.695,0.055,0.295,0.86]); lx.axis('off')
lx.text(0.00,1.00,'26 个 functional branch',fontsize=11.5,fontweight='bold',va='top')
lx.text(0.00,0.958,'每组一个 z_b；组间按树边做消息传递',fontsize=8.0,color='#4B5563',va='top')

def region_prefix(g):
    if g['region']=='soma': return 'soma'
    if g['region']=='axon': return 'axon'
    if g['region']=='basal': return 'dend'
    return 'apic'
def range_txt(g):
    ids=sorted(idx(s) for s in g['secs'])
    pref=region_prefix(g)
    if len(ids)==1:
        return f'{pref}[{ids[0]}]'
    if ids==list(range(min(ids),max(ids)+1)):
        return f'{pref}[{min(ids)}–{max(ids)}]'
    return f'{pref}['+','.join(str(i) for i in ids)+']'

# header
hy=0.918
lx.text(0.00,hy,'',fontsize=6.5)
lx.text(0.030,hy,'group',fontsize=6.4,color='#6B7280',fontweight='bold')
lx.text(0.180,hy,'parent',fontsize=6.4,color='#6B7280',fontweight='bold')
lx.text(0.300,hy,'sections',fontsize=6.4,color='#6B7280',fontweight='bold')
lx.text(0.860,hy,'sec/comp',fontsize=6.4,color='#6B7280',fontweight='bold')

nrow=len(groups)
row_h=0.875/nrow
for i,g in enumerate(groups):
    y=0.890-i*row_h
    lx.add_patch(Rectangle((0.0,y-0.010),0.022,0.020,facecolor=colors[g['name']],edgecolor='none'))
    label=g['name'].replace('A_','')
    parent=g['parent'].replace('A_','')
    if parent=='-': parent='—'
    if parent=='TRUNK_PROX': parent='PROX'
    if parent=='TRUNK_MID': parent='MID'
    if parent=='TRUNK_DIST': parent='DIST'
    if parent=='TUFT_T2a': parent='T2a'
    lx.text(0.030,y,label,fontsize=6.5,color='#111827',va='center',fontweight='bold')
    lx.text(0.180,y,parent,fontsize=6.2,color='#4B5563',va='center')
    lx.text(0.300,y,range_txt(g),fontsize=6.1,color='#374151',va='center')
    lx.text(0.860,y,f'{g["n_sec"]}/{g["n_comp"]}',fontsize=6.2,color='#6B7280',va='center')

fig.text(0.015,0.018,'26 groups: 1 soma + 1 axon + 8 basal + 3 trunk + 10 oblique + 3 tuft；26×16≈416 维 latent（相对 5751 维状态约 13× 压缩）',
         fontsize=8.6,color='#374151')
fig.text(0.015,0.002,'注：axon 无 3D 坐标，图中位置为示意；完整 section 列表见 l5pc_functional_branches.csv',
         fontsize=7.4,color='#6B7280')

out='figs/ppt_bg/out'
fig.savefig(os.path.join(out,'l5pc_functional_groups.png'),bbox_inches='tight',facecolor='white')
fig.savefig(os.path.join(out,'l5pc_functional_groups.pdf'),bbox_inches='tight',facecolor='white')
print('saved')
