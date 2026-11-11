"""幅度偏差检查：plateau 峰帧的 signed error 与回归斜率。"""
import argparse, json
import numpy as np
import torch
from world.learn import data2 as D2
from tools.plot_v4 import load_model
from world.core import safety
from tools.ca_event_metrics import _peaks

def _t(x): return torch.from_numpy(np.ascontiguousarray(x)).float()

@torch.no_grad()
def roll(m, ds, sc, i, t0, K):
    probe = D2.Sampler2(ds, sc, batch=1, length=K+1, seed=0)
    s0,u_tok,ug,th,st,x = probe.batch_window(t0, K+1, 1, off=i)
    s0t,tht,xt,ugt = _t(s0),_t(th),_t(x),_t(ug)
    u = _t(np.ascontiguousarray(u_tok.transpose(0,1,3,2)))
    z = m.encode(s0t.unsqueeze(1), u[:, :1], tht, xt, ugt[:, :1])[:, 0]
    s = s0t.clone(); out=[]
    for j in range(K):
        jj=min(j+1,K-1)
        zn = m.predict(z.unsqueeze(1), u[:, jj:jj+1], tht, xt, ugt[:, jj:jj+1])[:,0]
        s = m.step_residual(s, z, zn, tht, xt, ugt[:, jj:jj+1]); z=zn
        out.append(s.numpy()[0])
    return np.asarray(out)

def main():
    import sys
    ck = sys.argv[1] if len(sys.argv) > 1 else "r6_fw2/last.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "runs/2026-11-11-errdecomp/bias.json"
    safety.preflight(limit_mb=6000)
    ds = D2.Dataset2("data","datasets/1s/va_s2",tier="t2")
    m,sc,cfg = load_model(ck, ds, "t2")
    print("ckpt", ck)
    loc=np.asarray(sc["loc_s"],np.float32)[:,None]; scl=np.asarray(sc["scale_s"],np.float32)[:,None]
    ci=ds.meta["factor_names"].index("cai"); vi=ds.meta["factor_names"].index("V")
    regs=np.asarray(ds.regimes); ids=np.where(regs=="plateau")[0][:43]
    K=8000; t0=4000
    dc=[]; dv=[]; cc=[]; vv=[]
    for i in ids:
        pred = roll(m,ds,sc,i,t0,K)*scl+loc
        true = np.asarray(ds.s[i, t0+1:t0+1+K])
        c = true[:,ci,:].max(axis=1)
        pk,_,_ = _peaks(c, 0.025)
        for p in pk:
            dc.append(pred[p,ci,:].max()-true[p,ci,:].max())
            dv.append(pred[p,vi,:].max()-true[p,vi,:].max())
            cc.append(true[p,ci,:].max()); vv.append(true[p,vi,:].max())
    dc=np.asarray(dc); dv=np.asarray(dv); cc=np.asarray(cc); vv=np.asarray(vv)
    print(f"cai 峰: n={len(dc)} signed={dc.mean():+.4f} (log10)  |err|={np.abs(dc).mean():.4f}")
    print(f"V  峰: n={len(dv)} signed={dv.mean():+.2f} mV  |err|={np.abs(dv).mean():.2f} mV")
    # slope: pred vs true
    if len(cc)>2:
        sc_ = np.polyfit(cc, cc+dc, 1); sv_ = np.polyfit(vv, vv+dv, 1)
        print(f"cai 回归斜率={sc_[0]:.3f} 截距={sc_[1]:+.3f}; V 斜率={sv_[0]:.3f} 截距={sv_[1]:+.2f}")
    json.dump(dict(cai_signed=float(dc.mean()), V_signed=float(dv.mean()),
                   cai_slope=float(sc_[0]) if len(cc)>2 else None,
                   V_slope=float(sv_[0]) if len(vv)>2 else None),
              open(out,"w"), indent=1)

if __name__=="__main__": main()
