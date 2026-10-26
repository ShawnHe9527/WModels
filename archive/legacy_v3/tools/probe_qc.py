"""数据质量门禁（定稿 v2 §4.2）。

## 为什么需要这个门禁

上一轮实验的失败模式（Phase B 实测）：
  · 数据里"轨迹"几乎不动，只有**协议之间的常数差异** ⇒
    模型学成"给定 θ_s 直接输出常数轨迹"就能拿到池化 `rel_err`=0.048，
    看起来成功，实际输给"复制上一帧"（诚实口径 1.03）和"协议均值"（0.87）。
  · 实测观测方差 **94.3% 来自协议之间，只有 5.7% 来自协议内部**。

所以门禁必须同时检查两件事：

### (1) 状态确实被输入激励（不是常数）
  · `cai`、`m` 类门（可跨数量级） → 池化 p99/p1 比值
  · `h`、`z` 类门（有界占据率，饱和在 0.6 附近） → **逐协议绝对跨度**
    ⚠️ 不能用比值！`h` 集中在 [0.55, 0.66] 时 p99/p1 只有 1.18，
       但逐协议跨度已有 0.06——这是**有界变量的正确度量**。

### (2) 协议**内部**方差占比足够大（反作弊核心）
  `within_frac = Var_within / Var_total`
  若这个太小 ⇒ 模型可以靠"协议均值"作弊 ⇒ 判不合格。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import config as C
from . import datagen

# ---- 判据配置 -------------------------------------------------------------
# 可跨数量级的量：用池化 p99/p1 比值
RATIO_STATES = {
    "cai": 10.0,
    "g_Ih_m": 5.0,
    "g_Im_m": 5.0,
    "g_Ca_LVAst_m": 5.0,
    "g_SKv3_1_m": 5.0,     # ⚠️ 量级 1e-4，绝对跨度判据会量纲错配
}
# 有界占据率：用逐协议绝对跨度
# ⚠️ 覆盖率门槛定 60% 而不是 80%，理由是**生物学**：
#    · `SK_E2.z` 是钙激活钾通道的门控，只有在 cai 持续升高（钙内流足够）的
#      协议里才会开启；其余协议里它合法地恒为 ~0。实测呈双峰：
#      一部分协议 z≈0.001（关闭），另一部分 z→0.99（饱和）。
#    · `Ca_LVAst.h` 是 T 型钙的失活门，需要足够去极化才动。
#    要求"80% 协议都激励"等于要求数据分布偏离真实，反而有害。
#    ⇒ 覆盖率只作为"该状态在数据里确实被激励过"的弱检查（≥60%），
#      真正的反作弊判据是下面的 R²(协议均值) < 0.75。
SPAN_STATES = {
    "g_Ca_HVA_h": 0.05,
    "g_Ca_LVAst_h": 0.10,
    "g_SK_E2_z": 0.05,
}
SPAN_COVER_MIN = 0.60
# 反作弊：协议内部方差占总量比例下限
WITHIN_FRAC_MIN = 0.25


def _var_split(A: np.ndarray) -> dict:
    """A: (N, T, nseg) → 组间 / 组内方差分解（等权协议）。"""
    N = A.shape[0]
    Af = A.astype(np.float64)
    flat_all = Af.reshape(-1)
    var_total = float(flat_all.var())
    means = Af.reshape(N, -1).mean(axis=1)
    var_between = float(means.var())
    var_within = max(var_total - var_between, 0.0)
    return dict(
        var_total=var_total,
        var_between=var_between,
        var_within=var_within,
        within_frac=var_within / var_total if var_total > 0 else 0.0,
    )


def qc(store: "datagen.Store") -> dict:
    rep: dict = dict(
        n_proto=store.n_proto,
        dt=store.meta["dt"],
        n_test_steps=store.meta["n_test_steps"],
        states={}, checks={},
    )
    print("=" * 104)
    print("数据质量门禁 (1/2)：状态动态范围")
    print("=" * 104)
    print(f"{'量':<16}{'中位':>11}{'p1':>10}{'p99':>10}{'比值':>9}"
          f"{'跨度中位':>10}{'跨度ok%':>9}{'组内占比':>9}  判据                  判定")

    names = [k for k in store.keys() if k.startswith("g_") or k == "cai"]
    for k in names:
        A = np.asarray(store[k])
        f = A.astype(np.float64).reshape(-1)
        med = float(np.median(f))
        p1, p99 = float(np.percentile(f, 1)), float(np.percentile(f, 99))
        ratio = p99 / p1 if p1 > 0 else float(p99 - p1)
        spans = (A.max(axis=(1, 2)) - A.min(axis=(1, 2))).astype(np.float64)
        med_span = float(np.median(spans))
        vs = _var_split(A)

        crit, ok = "-", None
        if k in RATIO_STATES:
            span_tgt = SPAN_STATES.get(k, 0.0)
            frac_ok = float((spans > span_tgt).mean()) if span_tgt > 0 else 1.0
            ok = bool(ratio >= RATIO_STATES[k] and vs["within_frac"] >= WITHIN_FRAC_MIN)
            crit = f"比值>={RATIO_STATES[k]:g},组内>={WITHIN_FRAC_MIN:g}"
        elif k in SPAN_STATES:
            span_tgt = SPAN_STATES[k]
            frac_ok = float((spans > span_tgt).mean())
            ok = bool(frac_ok >= SPAN_COVER_MIN
                      and vs["within_frac"] >= WITHIN_FRAC_MIN)
            crit = (f"{SPAN_COVER_MIN*100:.0f}%跨度>{span_tgt:g},"
                    f"组内>={WITHIN_FRAC_MIN:g}")
        else:
            span_tgt = 0.0
            frac_ok = 1.0

        rep["states"][k] = dict(
            median=med, p1=p1, p99=p99, ratio=ratio,
            span_median=med_span, frac_span_ok=frac_ok, span_target=span_tgt,
            **vs,
        )
        if ok is not None:
            rep["checks"][k] = ok
        print(f"{k:<16}{med:>11.4g}{p1:>10.4g}{p99:>10.4g}{ratio:>9.3g}"
              f"{med_span:>10.4g}{frac_ok*100:>8.0f}%{vs['within_frac']:>9.2f}  "
              f"{crit:<22}" + ("✅" if ok else ("❌" if ok is False else "  ")))

    # ---- 输入检查 ----
    u = np.asarray(store["u"], dtype=np.float64)   # (N,T,2)
    Vcm, dV = u[..., 0], u[..., 1]
    rep["u"] = dict(
        Vcm_min=float(Vcm.min()), Vcm_max=float(Vcm.max()),
        Vcm_std=float(Vcm.std()),
        dV_median=float(np.median(np.abs(dV))),
        dV_p90=float(np.percentile(np.abs(dV), 90)),
        dV_max=float(np.abs(dV).max()),
        corr_V0_V1=float(np.corrcoef((Vcm - dV / 2).ravel(),
                                     (Vcm + dV / 2).ravel())[0, 1]),
    )
    print(f"\n{'-'*104}")
    print("输入 u_t")
    print(f"{'-'*104}")
    print(f"  V_cm 范围      : [{rep['u']['Vcm_min']:.1f}, "
          f"{rep['u']['Vcm_max']:.1f}] mV   std={rep['u']['Vcm_std']:.2f}")
    print(f"  |ΔV| 中位/p90/max: {rep['u']['dV_median']:.3f} / "
          f"{rep['u']['dV_p90']:.2f} / {rep['u']['dV_max']:.1f} mV")
    print(f"  corr(V0,V1)     : {rep['u']['corr_V0_V1']:.4f}")
    rep["checks"]["Vcm_std>3mV"] = bool(rep["u"]["Vcm_std"] > 3.0)
    rep["checks"]["corr(V0,V1)>0.95"] = bool(rep["u"]["corr_V0_V1"] > 0.95)

    # ---- 反作弊终检：逐协议常数均值基线必须无法解释轨迹 ----
    # ⚠️ 这里**不能**用 |A-μ|/|A|：对有界占据率（h≈0.6）即使 10% 的相对调制，
    #    该比值也只有 0.025，会被误判成"近乎常数"。
    #    正确判据是**尺度无关**的 R²(协议均值) = 组间方差 / 总方差。
    #    R² 高 ⇒ 轨迹只是"协议之间的常数阶跃" ⇒ 模型可作弊。
    print(f"\n{'-'*104}")
    print("反作弊终检：R²(逐协议常数均值) —— 越低越好，须 < 0.75")
    print(f"{'-'*104}")
    rep["r2_proto_mean"] = {}
    for k in [x for x in store.keys() if x.startswith("g_") or x == "cai"]:
        vs = _var_split(np.asarray(store[k]))
        r2 = vs["var_between"] / vs["var_total"] if vs["var_total"] > 0 else 1.0
        rep["r2_proto_mean"][k] = float(r2)
        ok = r2 < 0.75
        rep["checks"][f"{k} R2<0.75"] = bool(ok)
        print(f"  {k:<16} R2={r2:.4f}  组内占比={vs['within_frac']:.2f}  "
              + ("✅ 轨迹有信息" if ok else "❌ 只是协议间阶跃，模型必然作弊"))

    # ---- 逐协议明细 ----
    if "g_Ca_HVA_h" in store.keys():
        A = np.asarray(store["g_Ca_HVA_h"])
        spans = (A.max(axis=(1, 2)) - A.min(axis=(1, 2))).astype(np.float64)
        rep["per_proto_cahva_span"] = [float(x) for x in spans]
        print(f"\n{'-'*104}")
        print("逐协议 Ca_HVA.h 跨度")
        print(f"{'-'*104}")
        print("  " + "  ".join(f"p{i}:{s:.3f}" for i, s in enumerate(spans)))

    rep["passed"] = all(rep["checks"].values())
    print(f"\n{'='*104}")
    if rep["passed"]:
        print("✅ 数据质量门禁通过")
    else:
        print("❌ 数据质量门禁未通过：")
        for k, v in rep["checks"].items():
            if not v:
                print(f"   ✗ {k}")
    return rep


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "d"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    d = f"{tag}_s{seed}"
    p = os.path.join(C.DATA, d)
    st = datagen.Store(p)
    rep = qc(st)
    out = os.path.join(p, "qc.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"\n⇒ {out}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
