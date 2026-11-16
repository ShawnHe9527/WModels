"""逐 regime 的闭环 signed 偏差（cai / V，200 ms）。"""
import sys, numpy as np, torch
from world.learn import data2 as D2
from tools.plot_v4 import load_model
from world.core import safety
def _t(x): return torch.from_numpy(np.ascontiguousarray(x)).float()
@torch.no_grad()
def roll(m, ds, sc, i, t0, K):
    probe = D2.Sampler2(ds, sc, batch=1, length=K+1, seed=0)
    s0,u_tok,ug,th,st,x = probe.batch_window(t0, K+1, 1, off=i)
    s0t,tht,xt,ugt = _t(s0),_t(th),_t(x),_t(ug)
    u = _t(np.ascontiguousarray(u_tok.transpose(0,1,3,2)))
    z = m.encode(s0t.unsqueeze(1), u[:, :1], tht, xt, ugt[:, :1])[:,0]
    s = s0t.clone()
    for j in range(K):
        jj=min(j+1,K-1)
        zn = m.predict(z.unsqueeze(1), u[:, jj:jj+1], tht, xt, ugt[:, jj:jj+1])[:,0]
        s = m.step_residual(s, z, zn, tht, xt, ugt[:, jj:jj+1]); z=zn
    return s.numpy()[0]
def main():
    ck = sys.argv[1] if len(sys.argv)>1 else "r6_fw2/last.pt"
    safety.preflight(limit_mb=6000)
    ds = D2.Dataset2("data","datasets/1s/va_s2",tier="t2")
    m,sc,cfg = load_model(ck, ds, "t2")
    loc=np.asarray(sc["loc_s"],np.float32)[:,None]; scl=np.asarray(sc["scale_s"],np.float32)[:,None]
    ci=ds.meta["factor_names"].index("cai"); vi=ds.meta["factor_names"].index("V")
    regs=np.asarray(ds.regimes); K=8000; t0=4000
    print(f"ckpt {ck}")
    for r in ("linear","inter","plateau"):
        ids=np.where(regs==r)[0][:16]
        dc=[]; dv=[]
        for i in ids:
            p = roll(m,ds,sc,int(i),t0,K)*scl+loc
            tr = np.asarray(ds.s[int(i), t0+K])
            dc.append(float((p[ci,:]-tr[ci,:]).mean()*np.asarray(sc["scale_s"])[ci]))
            dv.append(float((p[vi,:]-tr[vi,:]).mean()))
        print(f"{r:8s} cai_signed={np.mean(dc):+.4f} log10 |err|={np.mean(np.abs(dc)):.4f}  "
              f"V_signed={np.mean(dv):+.2f} mV |err|={np.mean(np.abs(dv)):.2f}")
if __name__=="__main__": main()
