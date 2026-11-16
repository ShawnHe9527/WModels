import os, sys, time, json
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(4)
import numpy as np
from world.gen import protocol as P1, protocol2 as P2, sim2 as S2

sim = S2.Sim2(t_test=1000.0)
t0, t1 = sim.t_settle, sim.t_settle + sim.t_test
results = []
idx = 18
for f in (0.85, 0.95):
    for br in (2.0, 4.0, 6.0):
        for pl in (0.6, 0.9):
            idx += 1
            rng = np.random.default_rng(20265000 + idx)
            pol = P2.Policy2(
                rate_e=210.0, rate_i=80.0, rho=0.75,
                f_tuft=f, bundle_hz=7.0, width=5.0,
                bap_rate=br, p_lead=pl, bap_amp=4.0, bap_width=2.0,
                burst_prob=0.05, jitter=0.2,
            )
            pol.regime = "plateau"
            own_e, own_i, _ = P1.make_spike_trains(rng, sim.nseg, t0, t1, pol.to_policy1())
            tuft_e, tuft_i, bundles = P2.make_tuft_bundles(rng, sim.n_nb_e, sim.n_nb_i, t0, t1, pol)
            baps = P2.make_bap_times(rng, bundles, t0, t1, pol, max_n=S2.MAXP)
            S, U = sim.run(own_e, own_i, tuft_e, tuft_i, baps,
                           bap_amp=pol.bap_amp, bap_width=pol.bap_width)
            V = S[:, 8, :]; cai = 10.0 ** S[:, 7, :]; vs = sim.last_v_soma
            vm = V.mean(axis=1); cm = cai.mean(axis=1)
            mask = vm > -45
            d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
            starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
            runs = [(int(s), int(e), float((e - s) * sim.dt)) for s, e in zip(starts, ends)]
            best_run = max(runs, key=lambda r: r[2]) if runs else (0, 0, 0.0)
            s, e, dur = best_run
            bap_in = int(((baps >= t0 + s * sim.dt) & (baps <= t0 + e * sim.dt)).sum())
            cai_peak = float(cm.max() * 1000.0)
            score = dur + 10.0 * np.log10(cai_peak + 1e-9) - 25.0 * bap_in - 0.2 * len(baps)
            rec = dict(idx=idx, f=f, br=br, pl=pl, run_ms=dur, run_start=s, run_end=e,
                       cai_peak=cai_peak, n_bap=len(baps), bap_in=bap_in,
                       score=float(score), policy=pol.to_dict())
            results.append(rec)
            print(f"[{idx:02d}] f={f:.2f} br={br:.0f} pl={pl:.1f} "
                  f"run={dur:6.1f}ms cai={cai_peak:5.2f} nBAP={len(baps):2d} in={bap_in} "
                  f"score={score:7.1f}", flush=True)
            np.savez(f"figs/ppt_bg/cand_{idx:02d}.npz",
                     t=t0 + np.arange(sim.T) * sim.dt, vs=vs, V=V, cai=cai,
                     mask=mask, meta=np.array(json.dumps(rec)))

results.sort(key=lambda r: -r["score"])
print("TOP", json.dumps(results[:3], indent=1))
with open("figs/ppt_bg/search_results2.json", "w") as fh:
    json.dump(results, fh, indent=1)
