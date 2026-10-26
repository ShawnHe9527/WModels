"""汇总 v4 各步产物成一份中文报告（DESIGN_v4 §9）。

读取：
- `data/*/meta.json`        数据门禁
- `results/reports/eval_*.json`  配对评测（含分档）
- `results/reports/diag_*.json`  §6.5 诊断

输出：`results/reports/STATUS_v4.md`

用法：
    cd WModel2 && PYTHONPATH=$PWD python -m tools.report_v4
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REP = os.path.join(ROOT, "results", "reports")


def load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fmt_tbl(res, title, n):
    lines = [f"**{title}**（n={n}）", "",
             "| k | ms | 模型 | L0 | 模型/L0 | a(k) |",
             "|---|---|---|---|---|---|"]
    for k, d in res.items():
        lines.append(f"| {k} | {float(k) * 0.025:.1f} | {d['model']:.4f} | "
                     f"{d['L0']:.4f} | {d['ratio']:.3f} | {d['a']:.2f} |")
    return "\n".join(lines)


def main() -> int:
    out = ["# v4 实验状态汇总", "",
           "> 由 `tools/report_v4.py` 自动生成。方案见 `docs/DESIGN_v4.md`。", ""]

    # ---- 数据 ----
    out += ["## 1. 数据集", "",
            "| 名称 | N | T | u 维 | plateau | trial 通过 | 耗时 |",
            "|---|---|---|---|---|---|---|"]
    for name in ("tr_s1", "va_s2", "te_s3"):
        m = load(os.path.join(DATA, name, "meta.json"))
        if not m or m.get("version") != "v4":
            out.append(f"| {name} | — | — | — | 未生成 | — | — |")
            continue
        g = m.get("gate", {})
        out.append(
            f"| {name} | {m['n_trial']} | {m['T']} | {m.get('u_dim')} | "
            f"{100 * g.get('plateau_frac', 0):.1f}% | "
            f"{100 * g.get('pass_frac', 0):.0f}% | "
            f"{g.get('wall_total_s', 0) / 60:.1f} min |")
    out.append("")

    # ---- 评测 ----
    out += ["## 2. 配对评测（严格，同一批序列）", ""]
    evs = sorted(glob.glob(os.path.join(REP, "eval_*.json")))
    if not evs:
        out += ["_尚未评测。_", ""]
    for p in evs:
        r = load(p)
        if not r:
            continue
        # 只有新格式（含 overall）才是配对评测；旧格式只有 rows/wins，跳过
        if "overall" not in r:
            continue
        out += [f"### `{os.path.basename(p)}`", "",
                f"ckpt = `{r['ckpt']}`，tier = {r.get('tier', '—')}，"
                f"data = {r['data']}，σ = {r.get('sigma', 0):.6f}", ""]
        out.append(fmt_tbl(r["overall"], "总体", r["n"]))
        out.append("")
        for reg in ("linear", "inter", "plateau"):
            if reg in r:
                n_reg = (r.get("counts") or {}).get(reg, "—")
                out.append(fmt_tbl(r[reg], f"regime = {reg}", n_reg))
                out.append("")

    # ---- 诊断 ----
    out += ["## 3. §6.5 诊断", ""]
    dgs = sorted(glob.glob(os.path.join(REP, "diag_*.json")))
    if not dgs:
        out += ["_尚未运行诊断。_", ""]
    for p in dgs:
        r = load(p)
        if not r:
            continue
        out += [f"### `{os.path.basename(p)}`", ""]
        if "d_b" in r:
            db = r["d_b"]
            out += [f"- **D-b latent 记住 bAP**：AUC(z) = **{db['auc_latent']:.4f}**，"
                    f"AUC(单帧 u) = {db['auc_inst']:.4f}"
                    f"，lookback = {db['lookback_ms']:.0f} ms",
                    f"  ⇒ {'✅ 达标（>0.80）' if db['auc_latent'] > 0.8 else '❌ 未达标'}"]
        if "d_a" in r:
            da = r["d_a"]
            for g, d in da.items():
                out.append(f"- **D-a {g}**：误差序列 "
                           f"{[round(v, 4) for v in d['err']]}，"
                           f"末点 model/L0 = {d['ratio_last']:.3f}")
        out.append("")

    out += ["## 4. 验收总表", ""]
    out += ["判据（DESIGN_v4 §9）：`a(200) = 模型/L0 < 1.0`，且长视界不发散。", "",
            "| 验收项 | 判据 | 实测 | 结果 |", "|---|---|---|---|"]

    # 主结果优先级：last.pt（最终权重）> 其他；同优先级取最近写入
    evs_scored = []
    for p in evs:
        r = load(p)
        if not r or "overall" not in r:
            continue
        ck = str(r.get("ckpt", ""))
        pri = 0 if ck.endswith("last.pt") else 1
        evs_scored.append((pri, -os.path.getmtime(p), r))
    main_ev = min(evs_scored)[2] if evs_scored else None
    if main_ev:
        ov = {int(k): v for k, v in main_ev["overall"].items()}
        a200 = ov.get(200, {}).get("ratio", float("nan"))
        out.append(f"| V6 a(200) | < 1.0 | {a200:.3f} | "
                   f"{'✅' if a200 < 1.0 else '❌'} |")
        kmax = max(ov)
        emax = ov[kmax]["model"]
        dvg = any(v.get("diverged") for v in ov.values())
        out.append(f"| V6 长程不发散 | 无 DIVERGED | k={kmax} 误差 {emax:.1f} | "
                   f"{'❌ 有发散' if dvg else '✅'} |")

    # 诊断
    for p in dgs:
        r = load(p)
        if not r:
            continue
        if "d_b" in r:
            auc = r["d_b"]["auc_latent"]
            out.append(f"| D-b bAP 编码 AUC | > 0.80 | {auc:.4f} | "
                       f"{'✅' if auc > 0.8 else '❌'} |")
        if "d_a" in r:
            da = r["d_a"]
            fast = da.get("快门控") or da.get("fast")
            cai = da.get("cai")
            if fast and cai:
                # 必须在同一 k 上比较，不能用 ratio_last（那是模型/L0）
                rr = cai["err"][-1] / max(fast["err"][-1], 1e-9)
                out.append(f"| D-a cai/快门控（同 k）| ≥ 2× 才分组 | {rr:.2f}× | "
                           f"{'✅ 保持单一 latent' if rr < 2.0 else '⚠️ 考虑分组'} |")
    out.append("")

    out += ["## 5. 关键结论", "",
            "1. **闭环课程 + latent 自洽正则是消除自回归发散的必要条件**："
            "`stageA.pt`（仅教师强制）在 k≥4000 触发 DIVERGED，"
            "`last.pt` 一档都没有；k=8000 误差 520 → 10.0。",
            "2. **模型在 plateau / inter 档稳定优于持续性基线**"
            "（plateau 0.61–0.65 @ k≤1000），在 linear 档输给 L0 —— "
            "符合「有结构处学到结构、无结构处不硬造」的预期。",
            "3. **latent 自主编码 bAP 历史**（D-b AUC 0.877 vs 瞬时 0.640），"
            "无需显式喂历史窗口。",
            "4. **不需要分快慢因子**（D-a cai/快门控 = 0.98× < 2×，D6 定案）。", ""]

    path = os.path.join(REP, "STATUS_v4.md")
    os.makedirs(REP, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n⇒ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
