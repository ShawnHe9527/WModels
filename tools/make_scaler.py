"""Compute a v4 scaler for a dataset without overwriting scaler_v4.json.

    PYTHONPATH=$PWD python -m tools.make_scaler \
        --data tr_s1_2s --out results/data/scaler_v4_2s.json

Then pass `--scaler results/data/scaler_v4_2s.json` to `world.learn.train2 t2`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.core import safety

safety.apply_thread_caps(2)
from world.learn import data2 as D2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.make_scaler")
    ap.add_argument("--data", default="tr_s1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sample", type=int, default=400)
    a = ap.parse_args(argv)
    ds = D2.Dataset2(DATA, a.data, tier="t2")
    sc = D2.compute_scaler2(ds, n_sample=a.n_sample, seed=a.seed)
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sc, f, ensure_ascii=False, indent=1)
    print(f"=> {out}")
    print("scale_s", [round(x, 5) for x in sc["scale_s"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
