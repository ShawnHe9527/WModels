import json, numpy as np, os
z = np.load("figs/ppt_bg/cand_19.npz")
meta = json.loads(str(z["meta"]))
t = z["t"]; vs = z["vs"]; V = z["V"]; cai = z["cai"] * 1000.0
s, e = meta["run_start"], meta["run_end"]
prov = {
    "model": "official L5PC NEURON model in neuron/ (L5PCtemplate_2.hoc, L5PCbiophys5b.hoc)",
    "morphology": "neuron/morphologies/cell1.asc",
    "mechanisms": "neuron/mods/x86_64/libnrnmech.so",
    "target_section": "apical_51",
    "nseg": 11,
    "simulation": {"dt_ms": 0.025, "t_settle_ms": 100.0, "t_test_ms": 1000.0,
                   "celsius_C": 34.0, "v_init_mV": -76.0},
    "trial": {"seed": 20265019, "policy": meta["policy"]},
    "plateau": {"definition": "mean apical V > -45 mV, longest contiguous run",
                "t_on_ms": float(t[s]), "t_off_ms": float(t[e]),
                "duration_ms": float(meta["run_ms"]),
                "n_bap_total": int(meta["n_bap"]), "n_bap_inside": int(meta["bap_in"])},
    "cai_units": "uM = NEURON cai (mM) x 1000",
    "figures": {
        "morphology_vertical": ["out/l5pc_morphology.png", "out/l5pc_morphology.pdf"],
        "morphology_horizontal": ["out/l5pc_morphology_horizontal.png", "out/l5pc_morphology_horizontal.pdf"],
        "trace_wide": ["out/ca_plateau_trace.png", "out/ca_plateau_trace.pdf"],
        "trace_zoom": ["out/ca_plateau_trace_zoom.png", "out/ca_plateau_trace_zoom.pdf"],
    },
}
json.dump(prov, open("figs/ppt_bg/out/provenance.json", "w"), indent=1, ensure_ascii=False)

# CSV of the plotted window
m = (t >= 560) & (t <= 760)
np.savetxt("figs/ppt_bg/out/ca_plateau_trace_data.csv",
           np.column_stack([t[m], vs[m], V.mean(axis=1)[m], cai.mean(axis=1)[m]]),
           delimiter=",", header="time_ms,v_soma_mV,v_apical_mean_mV,cai_uM", comments="")
print("wrote provenance + csv", m.sum())
