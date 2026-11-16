import os, sys
from collections import defaultdict
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(2)
from neuron import h
import world.gen.sim2 as S2
import numpy as np

h.nrn_load_dll(S2.MODDIR)
cwd = os.getcwd(); os.chdir(S2.HOCODIR)
for f in ("stdrun.hoc", "import3d.hoc", "L5PCbiophys5b.hoc", "L5PCtemplate_2.hoc"):
    h.load_file(f)
os.chdir(cwd)
cell = h.L5PCtemplate(S2.MORPH)
secs = list(h.allsec())

def region(name):
    if ".soma" in name: return "soma"
    if ".axon" in name: return "axon"
    if ".apic" in name: return "apical"
    if ".dend" in name: return "basal"
    return "other"

# mechanism -> list of (section, values)
mech_vals = defaultdict(list)
for s in secs:
    ps = s.psection()
    for m, d in ps.get("density_mechs", {}).items():
        if m == "pas":
            key = "g"
        elif m == "CaDynamics_E2":
            key = "decay"
        else:
            keys = [k for k in d if k.endswith("bar")]
            if not keys:
                continue
            key = keys[0]
        v = d[key]
        if not isinstance(v, (list, tuple)):
            v = [v] * int(s.nseg)
        mech_vals[m].append((s.name(), key, [float(x) for x in v]))

for m, rows in mech_vals.items():
    allv = np.array([x for _, _, vs in rows for x in vs], dtype=float)
    nz = allv[np.abs(allv) > 0]
    uniq = len(np.unique(np.round(allv, 12)))
    # within-section variation
    within = []
    for name, key, vs in rows:
        arr = np.array(vs, dtype=float)
        if arr.size > 1 and np.nanmax(arr) - np.nanmin(arr) > 1e-12 * max(1.0, np.nanmax(np.abs(arr))):
            within.append((name, float(arr.min()), float(arr.max())))
    print(f"{m:16s} key={rows[0][1]:14s} nsec={len(rows):3d} nval={allv.size:4d} "
          f"uniq={uniq:3d} min={allv.min():.3e} max={allv.max():.3e} "
          f"within_sec={len(within)}")
    if within[:3]:
        for w in within[:3]:
            print("    within:", w)
