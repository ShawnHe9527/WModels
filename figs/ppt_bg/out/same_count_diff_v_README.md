# Same spike count, different target V

Controlled NEURON simulation on the official L5PC model (not experimental data).

- Window: 1 s (100–1100 ms); dt = 0.025 ms.
- Target: `apical_51`, 11 compartments.
- Neighbor input: off; bAP: off; I train: identical between A and B.
- A (rho=0): 120 independent E events per compartment; total E = 1320.
- B (rho=0.6): 72 shared E events + 48 independent E events per compartment; total E = 1320.
- Both: 40 I events per compartment; total I = 440.
- Same total E/I counts and same per-compartment E count (120); only E timing/synchrony differs.

Results:
- A: Vmax = −51.2 mV; target spikes = 0; cai span = 0.058 log10.
- B: Vmax = +16.3 mV; target spikes = 11; cai span = 1.018 log10.

Files:
- `same_count_diff_v.png` / `.pdf`: figure.
- `same_count_diff_v_data.npz`: t, VA, VB, CA, CB, event arrays, metrics.
- `same_count_diff_v_metrics.csv`: summary metrics.
