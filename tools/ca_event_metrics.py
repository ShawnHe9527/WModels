"""Ca 事件的**无阈值**评估指标。

## 为什么需要这个工具

`tools/round_report.py` 里的 `cai_precision` / `cai_recall` 用的是一个**固定的
绝对阈值**：

    TH = -3.10        # log10[Ca2+]
    hit = cai.max(comps) >= TH

它有一个致命副作用：**只要模型整体把 Ca 事件的幅度预测小了**，所有帧都掉到阈值
以下，于是既「全都漏报」（recall↓）又「只剩零星错报」（precision↓）。
R3 就撞上了这个坑：`--event-weight 5` 把 V 上限从 19.7 压到 10.8 mV，
P/R 双双从 0.696/0.867 崩塌到 0.155/0.273 —— 这是**阈值伪影**，
并不代表模型「不认识事件了」。

本工具改用**完全不含绝对阈值的指标**：

| 指标 | 含义 | 尺度敏感？ |
|---|---|---|
| `peak_P` / `peak_R` / `peak_F1` | 峰匹配检出率（prominence 取该 trial 自身动态范围的 20%，峰间最小间隔 5 ms，容差 ±5 ms） | 否（相对自身幅度） |
| `amp_ratio` | 匹配峰的**幅度比** 预测/真值（中位） | 是 —— 正是我们想量的过冲/不足 |
| `dt_ms` | 匹配峰的**时间误差**（中位，ms） | 否 |
| `slope` | 事件窗内 pred~true 的**回归斜率**（1.0 = 无偏，>1 = 过冲） | 归一化 |
| `r_event` | 事件窗内 pred/true 的相关系数（合并所有 trial） | 否 |
| `nrmse` | 全窗 RMSE / 该 trial 动态范围 | 归一化 |

`prominence` 用「该 trial 自身动态范围的 20%」而不是绝对值，所以事件整体
变小 50% 时峰仍然能被找到 —— 这就是「尺度无关」。

用法（PYTHONPATH=$PWD）：

    python -m tools.ca_event_metrics --runs t2v5_ancC t2v5_events
    python -m tools.ca_event_metrics --run t2v5_events --k 16000

⚠️ 窗口长度很关键：Ca 事件约每 500 ms 才 1 个，`k=2000`（50 ms）下 12 个 trial
只有 3 个峰，统计毫无意义。默认 `k=8000`（200 ms）+ 全部 43 个 plateau trial
（约 75 个真值峰）才够稳。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.signal import find_peaks

from world.core import safety                      # noqa: F401
from world.learn import data2 as D2
from tools.plot_v4 import denorm, load_model, regime_rollouts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPR = os.path.join(ROOT, "results", "reports")

# 峰检测参数（全部相对于该 trial 自身的动态范围 → 尺度无关）
PROM_FRAC = 0.20        # prominence = 20% × 该 trial 的 cai 动态范围
MIN_GAP_MS = 5.0        # 两峰最小间隔
TOL_MS = 5.0            # 峰匹配容差
WIN_MS = 25.0           # 事件窗（用于 slope / r_event）
THR_LEGACY = -3.10      # 旧指标用的固定绝对阈值（log10[Ca2+]）
DIV_LIMIT = 100.0       # |cai| 超过此值视为已发散（物理范围约 -9…-1）


def log(*a):
    print(*a, flush=True)


def _series(arr, ci):
    """(n, T, F, P) → (n, T)：跨 comp 取最大，与原 P/R 口径一致。"""
    return arr[:, :, ci, :].max(axis=2)


def _peaks(x, dt, prom_frac=PROM_FRAC, min_gap_ms=MIN_GAP_MS):
    """相对 prominence 的峰检测，返回 (idx, 峰高, 基线)。

    基线取该信号的 20 分位（事件只占 1.7% 帧，20 分位基本就是静止水平）。
    """
    base = float(np.percentile(x, 20))
    top = float(np.percentile(x, 99.5))
    dyn = max(top - base, 1e-9)
    idx, _ = find_peaks(x, prominence=prom_frac * dyn,
                        distance=max(int(min_gap_ms / dt), 1))
    return idx, x[idx], base


def _match(tp, pp, tol):
    """按时间容差做一对一匹配（贪心，先配最近的）。返回 (匹配对列表, 漏报, 误报)。"""
    used = np.zeros(len(pp), dtype=bool)
    pairs, miss = [], 0
    for t in tp:
        cand = np.where(~used)[0]
        if len(cand) == 0:
            miss += 1
            continue
        d = np.abs(pp[cand] - t)
        k = int(np.argmin(d))
        if d[k] <= tol:
            used[cand[k]] = True
            pairs.append((int(t), int(pp[cand[k]])))
        else:
            miss += 1
    fp = int((~used).sum())
    return pairs, miss, fp


def ca_event_metrics(P_phys, Y_phys, ci, dt: float):
    """核心计算。P_phys / Y_phys: (n, T, F, P) 物理单位。

    ⚠️ denorm 返回 float32；长闭环一旦发散（实测 R4 的 k=8000 到 3.8e18），
    float32 平方会溢出成 inf/nan。这里统一升到 float64，并跳过非有限值 trial。
    """
    pred = _series(np.asarray(P_phys, np.float64), ci)
    true = _series(np.asarray(Y_phys, np.float64), ci)
    # 非有限值 或 数量级明显不可能（cai 是 log10[Ca2+]，物理上落在 -9…-1；
    # 实测 R4 的 k=8000 能涨到 3.8e18，是**有限但已发散**，光查 isfinite 抓不到）
    ok = (np.isfinite(pred).all(axis=1) & np.isfinite(true).all(axis=1)
          & (np.abs(pred).max(axis=1) < DIV_LIMIT)
          & (np.abs(true).max(axis=1) < DIV_LIMIT))
    n_div = int((~ok).sum())
    if not ok.all():
        log(f"  !! {n_div} 个 trial 已发散（非有限或 |cai|>{DIV_LIMIT:g}），已跳过")
    pred, true = pred[ok], true[ok]
    n, T = true.shape
    if n == 0:
        log("  !! 全部 trial 均已发散，无法评估 Ca 事件")
    tol = TOL_MS / dt
    half = int(round(WIN_MS / dt))

    TP = PP = MISSN = TPN = 0
    THR_TP = THR_FP = THR_FN = 0
    amps, dts, slopes, r_ev, r_all, nrms = [], [], [], [], [], []

    for i in range(n):
        y, p = true[i], pred[i]
        ti, th_, tbase = _peaks(y, dt)
        pi, ph, pbase = _peaks(p, dt)
        pairs, miss, fp = _match(ti, pi, tol)
        TP += len(pairs)
        PP += len(pi)
        TPN += len(ti)
        MISSN += miss

        for a, b in pairs:
            amps.append((ph[np.where(pi == b)[0][0]] - pbase)
                        / max(th_[np.where(ti == a)[0][0]] - tbase, 1e-9))
            dts.append(abs(b - a) * dt)

        # 事件窗内的斜率与相关（合并后算）
        xs, ys = [], []
        for a in ti:
            lo, hi = max(a - half, 0), min(a + half, T)
            xs.append(y[lo:hi])
            ys.append(p[lo:hi])
        if xs:
            X, Yv = np.concatenate(xs), np.concatenate(ys)
            if X.std() > 1e-9 and Yv.std() > 1e-9:
                slopes.append(float(np.polyfit(X, Yv, 1)[0]))
                r_ev.append(float(np.corrcoef(X, Yv)[0, 1]))

        if y.std() > 1e-9 and p.std() > 1e-9:
            r_all.append(float(np.corrcoef(y, p)[0, 1]))
        dyn = max(float(np.percentile(y, 99.5) - np.percentile(y, 20)), 1e-9)
        nrms.append(float(np.sqrt(((p - y) ** 2).mean()) / dyn))

        # 同尺度、同 trial 的旧口径 P/R（固定绝对阈值），便于与历史轮次对比
        at = y >= THR_LEGACY
        ap = p >= THR_LEGACY
        THR_TP += int((at & ap).sum())
        THR_FP += int((~at & ap).sum())
        THR_FN += int((at & ~ap).sum())

    prec = TP / max(PP, 1)
    rec = TP / max(TPN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    med = lambda v: float(np.median(v)) if len(v) else float("nan")  # noqa: E731
    return dict(
        n_trials=n, n_diverged=n_div, n_true_peaks=TPN, n_pred_peaks=PP,
        n_matched=int(TP), n_missed=int(MISSN),
        peak_precision=prec, peak_recall=rec, peak_f1=f1,
        thr_precision=THR_TP / max(THR_TP + THR_FP, 1),
        thr_recall=THR_TP / max(THR_TP + THR_FN, 1),
        amp_ratio=med(amps), amp_ratio_iqr=(
            float(np.percentile(amps, 75) - np.percentile(amps, 25))
            if len(amps) > 3 else float("nan")),
        dt_ms=med(dts),
        slope=med(slopes), r_event=med(r_ev), r_all=med(r_all),
        nrmse=med(nrms),
    )


def eval_model(m, ds, sc, k: int, t0: int, n_tr: int):
    """用已载入的模型算一份无阈值 Ca 事件指标。"""
    ids, P, Y, _B = regime_rollouts(m, ds, sc, "plateau", k + 1, t0, n=n_tr)
    ci = ds.meta["factor_names"].index("cai")
    return ca_event_metrics(denorm(P, sc), denorm(Y, sc), ci, ds.dt), ids


def eval_run(run: str, ds, k: int, t0: int, n_tr: int, ckpt="last.pt",
             tier="t2"):
    m, sc2, _cfg = load_model(os.path.join(run, ckpt), ds, tier)
    return eval_model(m, ds, sc2, k, t0, n_tr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--data", default="te_s3")
    p.add_argument("--tier", default="t2")
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--k", type=int, default=8000,
                   help="rollout 步数（默认 8000 = 200 ms；需较长窗口才有足够峰）")
    p.add_argument("--t0", type=int, default=4000)
    p.add_argument("--n-trials", type=int, default=999,
                   help="参与统计的 plateau trial 数，999 = 全部")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    runs = a.runs or ([a.run] if a.run else [])
    if not runs:
        p.error("需要 --run 或 --runs")

    ds = D2.Dataset2(DATA, a.data, tier=a.tier)

    out = {}
    for r in runs:
        met, ids = eval_run(r, ds, a.k, a.t0, a.n_trials, a.ckpt, a.tier)
        out[r] = met
        log(f"\n=== {r}  (plateau n={met['n_trials']}, k={a.k} "
            f"= {a.k * ds.dt:.0f} ms) ===")
        log(f"  峰数 真值 {met['n_true_peaks']} / 预测 {met['n_pred_peaks']}"
            f" / 匹配 {met['n_matched']}")
        log(f"  峰 P/R/F1 = {met['peak_precision']:.3f} / "
            f"{met['peak_recall']:.3f} / {met['peak_f1']:.3f}")
        log(f"  [同尺度旧口径] 固定阈值 P/R = "
            f"{met['thr_precision']:.3f} / {met['thr_recall']:.3f}")
        log(f"  幅度比 预测/真值 = {met['amp_ratio']:.3f} "
            f"(IQR {met['amp_ratio_iqr']:.3f})   ← 1.0 无偏")
        log(f"  峰时误差 = {met['dt_ms']:.2f} ms")
        log(f"  事件窗 slope = {met['slope']:.3f}  r = {met['r_event']:.3f}")
        log(f"  全窗 r = {met['r_all']:.3f}  nrmse = {met['nrmse']:.3f}")

    os.makedirs(REPR, exist_ok=True)
    pth = a.out or os.path.join(REPR, f"ca_events_plateau_k{a.k}.json")
    old = {}
    if os.path.exists(pth):
        try:
            old = json.load(open(pth, encoding="utf-8"))
        except Exception:
            old = {}
    old.update(out)
    with open(pth, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
    log(f"\n=> {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
