import numpy as np, torch
from world.learn import data2 as D2
from tools.plot_v4 import load_model
from world.core import safety
def _t(x): return torch.from_numpy(np.ascontiguousarray(x)).float()
safety.preflight(limit_mb=6000)
ds = D2.Dataset2("data","datasets/1s/va_s2",tier="t2")
m,sc,cfg = load_model("t1v4_dz16/best.pt", ds, "t2")
ci=ds.meta["factor_names"].index("cai")
s_tr = D2.Sampler2(ds, sc, batch=32, length=128, seed=0, p_trial=None, hot_frac=0.3,
                   event_center_frac=0.0, event_center_thr=0.02)
D=[]; C=[]
with torch.no_grad():
    for _ in range(40):
        s0,u_tok,ug,th,st,x = s_tr.batch_pair()
        stt=_t(st); u=_t(np.ascontiguousarray(u_tok.transpose(0,1,3,2)))
        z=m.encode(stt,u,_t(th),_t(x),_t(ug)); rec=m.decode(z,_t(th),_t(x))
        D.append((rec-stt).numpy()[:,:,ci,:].mean(axis=2))
        C.append(st[:,:,ci,:].max(axis=2))
D=np.concatenate([d.ravel() for d in D]); C=np.concatenate([c.ravel() for c in C])
qs=np.quantile(C,[0,0.2,0.4,0.6,0.8,1.0])
for i in range(5):
    msk=(C>=qs[i])&(C<=qs[i+1])
    print(f"cai[{qs[i]:.3f},{qs[i+1]:.3f}] n={msk.sum():6d} signed={D[msk].mean():+.5f} |err|={np.abs(D[msk]).mean():.5f}")
