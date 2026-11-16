import os, sys, csv
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
import numpy as np
from neuron import h
import world.gen.sim2 as S2

h.nrn_load_dll(S2.MODDIR)
cwd = os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc", "L5PCtemplate_2.hoc"):
    h.load_file(f)
os.chdir(cwd)
cell = h.L5PCtemplate(S2.MORPH)
SECS = list(h.allsec())

def get(idx):
    for sec in SECS:
        if sec.name().endswith(f"apic[{idx}]"):
            return sec
    raise KeyError(idx)

def interp(sec, x):
    n = sec.n3d(); L = sec.L
    arcs = np.array([sec.arc3d(i) for i in range(n)])
    ds = np.array([sec.diam3d(i) for i in range(n)])
    s = x * L
    return float(np.interp(s, arcs, ds))

rows = []
for idx in (50, 51, 52, 53):
    sec = get(idx)
    ps = sec.parentseg()
    rows.append(dict(section=f"apic[{idx}]", nseg=int(sec.nseg), L_um=round(float(sec.L), 3),
                     diam_um=round(float(sec.diam), 3),
                     parent=(ps.sec.name().split('.')[-1] if ps is not None else ""),
                     children=";".join(c.name().split('.')[-1] for c in sec.children())))
with open("figs/ppt_bg/segments/geometry_sections.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

sec = get(51)
nseg = int(sec.nseg); L = float(sec.L)
with open("figs/ppt_bg/segments/geometry_compartments.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["compartment", "arc_start_um", "arc_end_um", "d_start_um", "d_end_um"])
    for i in range(nseg):
        d0 = interp(sec, i / nseg); d1 = interp(sec, (i + 1) / nseg)
        w.writerow([i + 1, round(i * L / nseg, 3), round((i + 1) * L / nseg, 3),
                    round(d0, 3), round(d1, 3)])
print(open("figs/ppt_bg/segments/geometry_sections.csv").read())
print(open("figs/ppt_bg/segments/geometry_compartments.csv").read())
