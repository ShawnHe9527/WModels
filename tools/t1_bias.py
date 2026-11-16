"""T1 重建在低 cai 帧的 signed 偏差（区分是 decoder 偏差还是 T2 权重）。"""
import numpy as np, torch
from world.learn import data2 as D2
from tools.plot_v4 import load_model
from world.core import safety
def _t(x): return torch.from_numpy(np.ascontiguousarray(x)).float()
safety.preflight(limit_mb=6000)
ds = D2.Dataset2("data","datasets/1s/va_s2",tier="t2")
m,sc,cfg = load_model("t1v4_dz16/best.pt", ds, "t2")
loc=np.asarray(sc["loc_s"],np.float32)[:,None]; scl=np.asarray(sc["scale_s"],np.float32)[:,None]
ci=ds.meta["factor_names"].index("cai")
s_tr = D2.Sampler2(ds, sc, batch=32, length=128, seed=0, p_trial=None, hot_frac=0.5,
                   event_center_frac=0.0, event_center_thr=0.02)
lo_err=[]; hi_err=[]; lo_all=[]
with torch.no_grad():
    for _ in range(30):
        s0,u_tok,ug,th,st,x = s_tr.batch_pair()
        stt=_t(st); u=_t(np.ascontiguousarray(u_tok.transpose(0,1,3,2)))
        z=m.encode(stt,u,_t(th),_t(x),_t(ug))
        rec=m.decode(z,_t(th),_t(x))
        d=(rec-stt).numpy()[:,:,ci,:].mean(axis=2)   # (B,L) signed normalized
        cai=st[:,:,ci,:].max(axis=2)
        lo_err.append(d[cai<-3.7]); hi_err.append(d[cai>-3.4])
lo=np.concatenate(lo_err); hi=np.concatenate(hi_err)
print(f"低 cai 帧 n={lo.size}: signed={lo.mean():+.4f} (log10, ×scale={np.asarray(sc['scale_s'])[ci]:.3f})")
print(f"高 cai 帧 n={hi.size}: signed={hi.mean():+.4f}")
print("换算：低 cai 偏差 ≈ %.4f log10" % (lo.mean()*np.asarray(sc['scale_s'])[ci]))
