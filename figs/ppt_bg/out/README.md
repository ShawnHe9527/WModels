# L5PC morphology + Ca plateau figures

## Source
- Model: official L5PC NEURON model in `neuron/` (`L5PCtemplate_2.hoc`, `L5PCbiophys5b.hoc`)
- Morphology: `neuron/morphologies/cell1.asc`
- Mechanisms: `neuron/mods/x86_64/libnrnmech.so`
- Target section: `apical_51`, 11 compartments (99 state dimensions)
- Simulation: `dt = 0.025 ms`, settle 100 ms, test 1000 ms, 34 °C, `v_init = -76 mV`

## Trial shown
- Seed: `20265019`
- Policy: `rate_e=210`, `rate_i=80`, `rho=0.75`, `f_tuft=0.85`,
  `bundle_hz=7`, `width=5 ms`, `bap_rate=2 Hz`, `p_lead=0.6`,
  `bap_amp=4 nA`, `bap_width=2 ms`
- Plateau: **623.1–708.1 ms (85 ms)**, defined as mean apical V > −45 mV
  (longest contiguous run); 0 bAP inside the plateau; 7 bAPs in the whole trial.
- Calcium: NEURON `cai` is in mM; plotted as µM (×1000).
  In the plotted window, mean `cai` rises from ~0.29 to ~0.48 µM.

## Files
- `l5pc_morphology_horizontal.png/.pdf` — horizontal morphology, recommended for slides.
  Red = `apical_51`; black = soma; gray = other neurites; 100 µm scale bar.
- `l5pc_morphology.png/.pdf` — vertical morphology, for papers/figures.
- `ca_plateau_trace.png/.pdf` — 560–760 ms window; V_soma (black),
  V_apical (blue, mean of 11 compartments), cai (orange); plateau shaded.
- `ca_plateau_trace_zoom.png/.pdf` — 600–735 ms zoom.
- `ca_plateau_trace_data.csv` — plotted window, columns:
  `time_ms, v_soma_mV, v_apical_mean_mV, cai_uM`.
- `provenance.json` — model, seed, policy, plateau interval, units.

## Notes
- These are **NEURON simulation outputs** (ground truth), not experimental recordings.
- No arrows, text boxes, or titles were added; only axis labels, a legend,
  plateau shading, and a scale bar.
- To regenerate the trial search: `search_trial2.py` (≈2 min on 4 threads);
  then `make_trace.py`. Morphology: `make_morph.py` / `make_morph_h.py`.
