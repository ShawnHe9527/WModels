import os, sys, time, json
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(4)
import numpy as np
from world.gen import protocol as P1, protocol2 as P2, sim2 as S2

sim = S2.Sim2(t_test=1000.0)
t0, t1 = sim.t_settle, sim.t_settle + sim.t_test
results = []
idx = 0
for f in (0.65, 0.80, 0.90):
    for br in (6.0, 10.0, 14.0):
        for bh in (2.5, 4.0):
            idx += 1
            rng = np.random.default_rng(20260000 + idx)
            pol = P2.Policy2(
                rate_e=180.0, rate_i=90.0, rho=0.70,
                f_tuft=f, bundle_hz=bh, width=5.0,
                bap_rate=br, p_lead=0.4, bap_amp=4.0, bap_width=2.0,
                burst_prob=0.05, jitter=0.2,
            )
            pol.regime = "plateau"
            own_e, own_i, _ = P1.make_spike_trains(rng, sim.nseg, t0, t1, pol.to_policy1())
            tuft_e, tuft_i, bundles = P2.make_tuft_bundles(rng, sim.n_nb_e, sim.n_nb_i, t0, t1, pol)
            baps = P2.make_bap_times(rng, bundles, t0, t1, pol, max_n=S2.MAXP)
            tt = time.time()
            S, U = sim.run(own_e, own_i, tuft_e, tuft_i, baps,
                           bap_amp=pol.bap_amp, bap_width=pol.bap_width)
            wall = time.time() - tt
            V = S[:, 8, :]; cai = 10.0 ** S[:, 7, :]; vs = sim.last_v_soma
            vm = V.mean(axis=1); cm = cai.mean(axis=1)
            mask = vm > -45
            d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
            starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
            runs = [(int(s), int(e), float((e - s) * sim.dt)) for s, e in zip(starts, ends)]
            best_run = max(runs, key=lambda r: r[2]) if runs else (0, 0, 0.0)
            cai_peak = float(cm.max() * 1000.0)
            score = best_run[2] + 10.0 * np.log10(cai_peak + 1e-9) - 0.5 * len(baps)
            rec = dict(idx=idx, f=f, br=br, bh=bh, wall=wall, run_ms=best_run[2],
                       run_start=int(best_run[0]), run_end=int(best_run[1]),
                       cai_peak=cai_peak, n_bap=len(baps), score=float(score),
                       policy=pol.to_dict())
            results.append(rec)
            print(f"[{idx:02d}] f={f:.2f} br={br:.0f} bh={bh:.1f} "
                  f"run={best_run[2]:6.1f}ms cai={cai_peak:5.2f}uM nBAP={len(baps):2d} "
                  f"score={score:6.1f} wall={wall:.1f}s", flush=True)
            np.savez(f"figs/ppt_bg/cand_{idx:02d}.npz",
                     t=t0 + np.arange(sim.T) * sim.dt, vs=vs, V=V, cai=cai,
                     mask=mask, meta=np.array(json.dumps(rec)))

results.sort(key=lambda r: -r["score"])
print("BEST", json.dumps(results[0], indent=1))
with open("figs/ppt_bg/search_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
