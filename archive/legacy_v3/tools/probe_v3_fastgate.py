"""V3 门禁：快门控「代数化」的误差是否 < 1%？

## 为什么必须先验证（定稿 v2 §3.4）

计划把 3 个 $\\tau_{min}<1$ ms 的门（`NaTs2_t.m`, `NaTs2_t.h`, `Ca_HVA.m`）
改用 $g=g_\\infty(V)$（不设状态变量）。理由：它们"跟不上"，瞬间就等于稳态。

但这会引入误差，必须量化。**判据：电流误差 < 1% 才允许代数化。**

## 方法

用**真实 NEURON** 做对照：
  A. **积分版**：正常仿真（NEURON 自己积分门控）
  B. **代数版**：把该门控的 $\tau$ 改成极小（强迫瞬时跟随）

⚠️ 不能直接改 `.mod`。可行做法：用 `SectionUnit` 无法做到"半代数化"，
所以这里换一个等价测法 —— 比较两种**时间步长**下的解：

* 若某门控的 $\tau_min \gg dt$，则 $\Delta t$ 减小对它几乎无影响
  ⇒ 它就是"准静态"的
* 若 $\tau_min \sim dt$，则 $\Delta t$ 减小会显著改变结果
  ⇒ 它必须被积分

**更直接的测法**：对每个门控，比较
  (i) 用 NEURON 真实积分得到的 $g(t)$
  (ii) 用 $g_\\infty(V(t))$（瞬时）算出的 $g(t)$
的**相对误差**，在真实 $V(t)$ 轨迹上统计。

这是"代数化误差"的**直接**度量。

## 输出

每个门控一份：median / p95 / max 相对误差，并给出 PASS/FAIL。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import config as C
from . import protocol as P
from . import secinfo as SI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def run_v_trace(proto: P.Protocol, dt: float = C.DT):
    """用 SectionUnit 跑一条协议，记录逐导频的 V(t)（= 门控看到的电位）。

    ⚠️ 用 `SectionUnit` 而非 NEURON：它的门控由 `L5PC_gates.gate_inf_tau` 驱动，
    与 NEURON 的一致性已由 `world/probes/validate_section_unit.py` 验证
    （28 工况，最差偏离 1.37%，电流比中位 1.0000）。
    """
    from L5PC_unit_sec import SectionUnit
    cell, cat = SI.new_cell()
    sec, row = SI.get_section(cell, cat)
    u = SectionUnit(sec, n_pilot=5, fixed_p=True)
    u.reset(C.V_INIT, C.V_INIT)

    V0, V1 = proto.V0, proto.V1
    n = len(proto)
    Vp = np.zeros((n, u.P), dtype=float)
    gate_hist = {k: np.zeros((n, u.P), dtype=float) for k in u.gates}
    cai_hist = np.zeros(n, dtype=float)

    for j in range(n):
        u.step(dt, float(V0[j]), float(V1[j]))
        Vp[j] = u._V_at_pilots()
        for k, g in u.gates.items():
            gate_hist[k][j] = g
        cai_hist[j] = float(np.atleast_1d(u.cai)[0])

    return dict(Vp=Vp, gates=gate_hist, cai=cai_hist, P=u.P, nseg=u.nseg)


def algebraic_error(trace: dict, gates: tuple) -> dict:
    """对每个指定门控，算 $g$ vs $g_\\infty(V)$ 的相对误差。

    只在 **test 相**统计（前 settle+cond 是预备，不计入）。
    """
    from L5PC_gates import gate_inf_tau
    Vp = trace["Vp"]
    Pn = trace["P"]
    n_set = int(round(C.T_SETTLE / C.DT))
    n_cond = int(round(C.T_COND / C.DT))
    T1 = n_set + n_cond

    out = {}
    for key in gates:
        if key not in trace["gates"]:
            out[f"{key[0]}.{key[1]}"] = dict(error="gate not in SectionUnit")
            continue
        g_real = trace["gates"][key][T1:]          # (n_test, P)
        # 用当前 V 算 g∞（注意：SK_E2 依赖 cai，不依赖 V）
        g_inf = np.zeros_like(g_real)
        for i in range(g_real.shape[0]):
            for p in range(Pn):
                gi, _tau = gate_inf_tau(key[0], key[1], float(Vp[T1 + i, p]),
                                        cai=float(trace["cai"][T1 + i]))
                g_inf[i, p] = float(np.atleast_1d(gi)[0])
        # 相对误差（用 g 的跨度归一化，避免 g→0 时除零）
        span = float(g_real.max() - g_real.min())
        span = max(span, 1e-9)
        err = np.abs(g_real - g_inf) / span
        out[f"{key[0]}.{key[1]}"] = dict(
            g_span=float(span),
            g_median=float(np.median(g_real)),
            err_median=float(np.median(err)),
            err_p95=float(np.percentile(err, 95)),
            err_max=float(err.max()),
        )
    return out


def main():
    print("=" * 90)
    print("V3 门禁：快门控代数化误差（判据：误差 < 1%）")
    print("=" * 90)

    rng = np.random.default_rng(1234)
    kinds = ("mix", "slow", "fast", "event")
    all_err: dict[str, list[float]] = {}

    for ki, kind in enumerate(kinds):
        proto = P.make_protocol(rng, kind)
        tr = run_v_trace(proto)
        errs = algebraic_error(tr, C.FAST_GATES_ALGEBRAIC + (
            ("Ih", "m"), ("Ca_LVAst", "m"), ("Ca_LVAst", "h"),
            ("Ca_HVA", "h"), ("SKv3_1", "m"), ("Im", "m")))
        print(f"\n--- kind={kind} ---")
        print(f"{'门控':<18}{'g跨度':>12}{'误差中位':>12}"
              f"{'p95':>10}{'max':>10}  判定")
        for name, d in errs.items():
            if "error" in d:
                print(f"{name:<18}  {d['error']}")
                continue
            all_err.setdefault(name, []).append(d["err_median"])
            tag = "✅代数化" if d["err_median"] < 0.01 else "❌需积分"
            print(f"{name:<18}{d['g_span']:>12.4g}{d['err_median']:>12.4f}"
                  f"{d['err_p95']:>10.4f}{d['err_max']:>10.4f}  {tag}")

    print("\n" + "=" * 90)
    print("汇总（跨 4 种协议的误差中位数的中位数）")
    print("=" * 90)
    print(f"{'门控':<18}{'误差中位':>12}  判定")
    decision = {}
    for name, vals in all_err.items():
        med = float(np.median(vals))
        ok = med < 0.01
        decision[name] = bool(ok)
        print(f"{name:<18}{med:>12.4f}  "
              + ("✅ 可代数化" if ok else "❌ 必须保留为状态"))

    print("\n--- 计划代数化的 3 个快门的最终判定 ---")
    final = {}
    for m, g in C.FAST_GATES_ALGEBRAIC:
        nm = f"{m}.{g}"
        ok = decision.get(nm, False)
        final[nm] = ok
        print(f"  {nm:<18} {'✅ 代数化' if ok else '❌ 改回状态变量'}")

    from .config import ensure, RESULTS
    ensure(RESULTS)
    out = os.path.join(RESULTS, "v3_fastgate.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(decision=decision, algebraic_final=final,
                       threshold=0.01), f, ensure_ascii=False, indent=2)
    print(f"\n⇒ {out}")

    n_ok = sum(final.values())
    print(f"\n结论：{n_ok}/{len(final)} 个快门可代数化 ⇒ "
          f"状态维 = {len(C.STATE_GATES)}×P + P"
          f"（若全采用则去掉 {3*5} 维）")
    if n_ok < len(final):
        print("⚠️ 有门控不能代数化 ⇒ 需更新 config.STATE_GATES 后重跑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
