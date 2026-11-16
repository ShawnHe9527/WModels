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
h.distance(0.5, sec=cell.soma[0])
for idx in (50,51,52,53):
    for sec in h.allsec():
        if sec.name().endswith(f"apic[{idx}]"):
            ps = sec.psection()
            d = ps["density_mechs"].get("Ca_LVAst", {}).get("gCa_LVAstbar")
            ih = ps["density_mechs"].get("Ih", {}).get("gIhbar")
            sec.push()
            dist = [h.distance((i+0.5)/sec.nseg) for i in range(int(sec.nseg))]
            h.pop_section()
            print(idx, "dist", [round(x,1) for x in dist])
            print("   ca", [round(float(x),6) for x in d])
            print("   ih", [round(float(x),6) for x in ih])
