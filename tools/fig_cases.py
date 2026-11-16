"""生成“好结果/差结果”示例图，帮助定位问题。"""
from __future__ import annotations
import os, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from world.learn import data2 as D2
from tools.plot_v4 import load_model
from world.core import safety
from tools.ca_event_metrics import _peaks

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

def _t(x): return torch.from_numpy(np.ascontiguousarray(x)).float()

@torch.no_grad()
def roll(m, ds, sc, i, t0, K):
    probe = D2.Sampler2(ds, sc, batch=1, length=K+1, seed=0)
    s0,u_tok,ug,th,st,x = probe.batch_window(t0, K+1, 1, off=i)
    s0t,tht,xt,ugt = _t(s0),_t(th),_t(x),_t(ug)
    u = _t(np.ascontiguousarray(u_tok.transpose(0,1,3,2)))
    z = m.encode(s0t.unsqueeze(1), u[:, :1], tht, xt, ugt[:, :1])[:,0]
    s = s0t.clone(); outs=[]
    for j in range(K):
        jj=min(j+1,K-1)
        zn = m.predict(z.unsqueeze(1), u[:, jj:jj+1], tht, xt, ugt[:, jj:jj+1])[:,0]
        s = m.step_residual(s, z, zn, tht, xt, ugt[:, jj:jj+1]); z=zn
        outs.append(s.numpy()[0])
    return np.asarray(outs)

def main():
    safety.preflight(limit_mb=8000)
    ds = D2.Dataset2("data","datasets/1s/va_s2",tier="t2")
    sc = None; models={}
    for name,ck in [("FW2","r6_fw2_s4/last.pt"),("E1","r6_e1/last.pt")]:
        m,sc,cfg = load_model(ck, ds, "t2"); models[name]=m
    loc=np.asarray(sc["loc_s"],np.float32)[:,None]; scl=np.asarray(sc["scale_s"],np.float32)[:,None]
    ci=ds.meta["factor_names"].index("cai"); vi=ds.meta["factor_names"].index("V")
    regs=np.asarray(ds.regimes)
    out="runs/2026-11-16-figs"; os.makedirs(out, exist_ok=True)
    t0=4000; K=8000; t=np.arange(K)*0.025
    # 好：linear 与 inter
    for r in ("linear","inter"):
        i=int(np.where(regs==r)[0][0])
        pred = roll(models["FW2"],ds,sc,i,t0,K)*scl+loc
        true = np.asarray(ds.s[i,t0+1:t0+1+K])
        fig,ax=plt.subplots(1,2,figsize=(12,3.6))
        ax[0].plot(t,true[:,vi,:].mean(1),"k",lw=1.2,label="真值")
        ax[0].plot(t,pred[:,vi,:].mean(1),"C3",lw=1,label="预测")
        ax[0].set_title(f"较好示例：{r} 的 V（FW2）"); ax[0].set_xlabel("ms"); ax[0].set_ylabel("mV"); ax[0].legend(fontsize=8)
        ax[1].plot(t,true[:,ci,:].max(1),"k",lw=1.2,label="真值")
        ax[1].plot(t,pred[:,ci,:].max(1),"C3",lw=1,label="预测")
        ax[1].set_title(f"较好示例：{r} 的 cai（FW2）"); ax[1].set_xlabel("ms"); ax[1].set_ylabel("log10 cai"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{out}/good_{r}.png",dpi=130); plt.close(fig)
    # 差：plateau（FW2 vs E1）
    i=int(np.where(regs=="plateau")[0][0])
    fig,ax=plt.subplots(1,2,figsize=(13,4))
    true = np.asarray(ds.s[i,t0+1:t0+1+K])
    ax[0].plot(t,true[:,vi,:].mean(1),"k",lw=1.3,label="真值")
    for k,c in (("FW2","C3"),("E1","C0")):
        pred = roll(models[k],ds,sc,i,t0,K)*scl+loc
        ax[0].plot(t,pred[:,vi,:].mean(1),c,lw=1,label=k)
    ax[0].set_title("较差示例：plateau 的 V（峰被削平）"); ax[0].set_xlabel("ms"); ax[0].set_ylabel("mV"); ax[0].legend(fontsize=8)
    ax[1].plot(t,true[:,ci,:].max(1),"k",lw=1.3,label="真值")
    for k,c in (("FW2","C3"),("E1","C0")):
        pred = roll(models[k],ds,sc,i,t0,K)*scl+loc
        ax[1].plot(t,pred[:,ci,:].max(1),c,lw=1,label=k)
    ax[1].set_title("较差示例：plateau 的 cai"); ax[1].set_xlabel("ms"); ax[1].set_ylabel("log10 cai"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/bad_plateau.png",dpi=130); plt.close(fig)
    # 峰幅度散点：真 vs 预测（plateau，43 trial）
    ids=np.where(regs=="plateau")[0][:43]
    for k in ("FW2","E1"):
        tv=[]; pv=[]
        for ii in ids:
            pred = roll(models[k],ds,sc,int(ii),t0,K)*scl+loc
            true = np.asarray(ds.s[int(ii),t0+1:t0+1+K])
            tp,_,_ = _peaks(true[:,ci,:].max(1),0.025)
            pp,_,_ = _peaks(pred[:,ci,:].max(1),0.025)
            for x in tp:
                tv.append(true[x,vi,:].max()); 
                pv.append(pred[x,vi,:].max())
        fig,ax=plt.subplots(figsize=(4.5,4.2))
        ax.scatter(tv,pv,s=14,c="C3",alpha=0.7)
        lo=min(min(tv),min(pv)); hi=max(max(tv),max(pv))
        ax.plot([lo,hi],[lo,hi],"k--",lw=1,label="y=x")
        ax.set_xlabel("真值峰 V (mV)"); ax.set_ylabel("预测峰 V (mV)")
        ax.set_title(f"峰幅度压缩（{k}）"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{out}/scatter_peaks_{k}.png",dpi=130); plt.close(fig)
    print("saved to", out)

if __name__=="__main__": main()
