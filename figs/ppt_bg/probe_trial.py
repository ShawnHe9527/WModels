import os, sys, time
sys.path.insert(0, os.getcwd())
from world.core import safety
safety.apply_thread_caps(4)
import numpy as np
from world.gen import protocol as P1, protocol2 as P2, sim2 as S2

sim = S2.Sim2(t_test=1000.0)
rng = np.random.default_rng(20261108)
pol = None
while pol is None or pol.regime != "plateau":
    pol = P2.sample_policy2(rng)
print("policy", pol.to_dict())
t0 = sim.t_settle; t1 = t0 + sim.t_test
own_e, own_i, _ = P1.make_spike_trains(rng, sim.nseg, t0, t1, pol.to_policy1())
tuft_e, tuft_i, bundles = P2.make_tuft_bundles(rng, sim.n_nb_e, sim.n_nb_i, t0, t1, pol)
baps = P2.make_bap_times(rng, bundles, t0, t1, pol, max_n=S2.MAXP)
tt = time.time()
S, U = sim.run(own_e, own_i, tuft_e, tuft_i, baps, bap_amp=pol.bap_amp, bap_width=pol.bap_width)
print("wall", time.time()-tt, S.shape, U.shape)
V = S[:, 8, :]; cai = 10.0 ** S[:, 7, :]; vs = sim.last_v_soma
vm = V.mean(axis=1)
print("V soma range", float(vs.min()), float(vs.max()))
print("V apic mean range", float(vm.min()), float(vm.max()))
print("cai range", float(cai.min()), float(cai.max()))
mask = vm > -45
print("frac > -45", float(mask.mean()))
# runs
d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
runs = [(int(s), int(e), float((e-s)*sim.dt)) for s,e in zip(starts, ends)]
print("runs", sorted(runs, key=lambda r: -r[2])[:5])
np.savez("figs/ppt_bg/probe.npz", t=t0+np.arange(sim.T)*sim.dt, vs=vs, V=V, cai=cai, mask=mask)
