#!/usr/bin/env python
"""Evaluate / tune the defense on attacked PaySim samples (offline tuning ONLY).

Attacked files carry the ground-truth `is_poisoned` column. This script strips it before every
`defend` call and uses it only to score the result.

  python scripts/tune_defense.py --dir data/attacked_sample --attacks random_flip --budgets 0.2
  python scripts/tune_defense.py --dir data/attacked_sample --z 3 5 8 --loss 2 4 --out results/defense_tuning.csv

Each (file x parameter combination) gives one row: detection precision / recall / FPR, time_to_detect
(windows from the first poisoned window to the first window with a poisoned record quarantined),
clean_loss (clean records quarantined / clean records) and clean_fraud_loss (true frauds quarantined,
which hurts the learner most). Settings come from the `defense:` section of config/default.yaml;
--z / --loss / --detectors override it for the grid.
"""
import argparse
import itertools
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attacks import load_config  # noqa: E402
from src.defense.pipeline import defend, load_defense_cfg  # noqa: E402


def run_file(df, cfg, window_steps, time_col="step", label_col="isFraud"):
    """Replay `df` (with is_poisoned) window by window through defend. Returns (flagged, per-window frame)."""
    df = df.sort_values([time_col, "record_id"], kind="stable").reset_index(drop=True)
    win = ((df[time_col] - df[time_col].min()) // window_steps).to_numpy()
    flagged = np.zeros(len(df), dtype=bool)
    state = {"cfg": dict(cfg)}
    rows = []
    for k in np.unique(win):
        idx = np.flatnonzero(win == k)
        w = df.iloc[idx]
        _, quarantined, state = defend(w.drop(columns=["is_poisoned"]), state)   # defense never sees the truth
        f = w["record_id"].isin(quarantined["record_id"]).to_numpy()
        flagged[idx] = f
        p = w["is_poisoned"].to_numpy(dtype=bool)
        rows.append({"window": int(k), "n": len(w), "n_poisoned": int(p.sum()), "n_flagged": int(f.sum()),
                     "tp": int((f & p).sum()), "fp": int((f & ~p).sum())})
    return df, flagged, pd.DataFrame(rows)


def summarize(df, flagged, per_window):
    p = df["is_poisoned"].to_numpy(dtype=bool)
    tp, fp = int((flagged & p).sum()), int((flagged & ~p).sum())
    pw = per_window
    first_poison = pw.index[pw["n_poisoned"] > 0]
    first_hit = pw.index[pw["tp"] > 0]
    ttd = float(first_hit[0] - first_poison[0]) if len(first_poison) and len(first_hit) else np.nan
    clean_fraud = (~p) & (df["isFraud"].to_numpy() == 1)
    return {"n_rows": len(df), "n_poisoned": int(p.sum()), "n_flagged": int(flagged.sum()),
            "precision": tp / max(tp + fp, 1) if flagged.any() else np.nan,
            "recall": tp / p.sum() if p.sum() else np.nan,
            "fpr": fp / max((~p).sum(), 1),                       # = clean_loss
            "clean_loss": fp / max((~p).sum(), 1),
            "clean_fraud_loss": float((flagged & clean_fraud).sum() / max(clean_fraud.sum(), 1)),
            "time_to_detect": ttd}


def subgroup_loss(df, flagged):
    """Share of CLEAN records quarantined, by transaction type and amount quartile (fairness-style check)."""
    clean = ~df["is_poisoned"].to_numpy(dtype=bool)
    d = df[clean].assign(flagged=flagged[clean])
    d["amount_bucket"] = pd.qcut(d["amount"], 4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
    out = {}
    for col in ("type", "amount_bucket"):
        for g, v in d.groupby(col, observed=True)["flagged"].mean().items():
            out[f"loss_{col}={g}"] = float(v)
    return out


def parse_name(name):
    m = re.match(r"(?P<attack>.+)_b(?P<budget>[\d.]+)_(?P<timing>uniform|burst|low_and_slow)_s(?P<seed>\d+)", name)
    return m.groupdict() if m else {"attack": name, "budget": np.nan, "timing": "", "seed": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "data" / "attacked_sample"))
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--attacks", nargs="+", help="default: every attacked file in the folder")
    ap.add_argument("--budgets", nargs="+", type=float)
    ap.add_argument("--timings", nargs="+")
    ap.add_argument("--z", nargs="+", type=float, help="z_threshold grid (default: config value)")
    ap.add_argument("--loss", nargs="+", type=float, help="loss_threshold grid (default: config value)")
    ap.add_argument("--detectors", nargs="+", help="e.g. robust_z label_loss mahalanobis (default: config)")
    ap.add_argument("--subgroups", action="store_true", help="also report clean-record loss by type / amount bucket")
    ap.add_argument("--out", default=str(ROOT / "results" / "defense_tuning.csv"))
    a = ap.parse_args()

    config = load_config(a.config)
    base = load_defense_cfg(a.config)
    ws = config["datasets"]["paysim"]["window_steps"]
    skip = {"stream_clean", "test_clean", "manifest"}
    files = sorted(f for f in Path(a.dir).glob("*.csv.gz") if f.name.split(".")[0] not in skip)
    keep = []
    for f in files:
        m = parse_name(f.name.replace(".csv.gz", ""))
        if a.attacks and m["attack"] not in a.attacks:
            continue
        if a.budgets and float(m["budget"]) not in a.budgets:
            continue
        if a.timings and m["timing"] not in a.timings:
            continue
        keep.append((f, m))
    if not keep:
        sys.exit(f"no attacked files found in {a.dir} matching the filters")

    grid = list(itertools.product(a.z or [base["z_threshold"]], a.loss or [base["loss_threshold"]]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, m in keep:
        df = pd.read_csv(f)
        for z, loss in grid:
            cfg = {**base, "z_threshold": z, "loss_threshold": loss}
            if a.detectors:
                cfg["detectors"] = a.detectors
            t0 = time.time()
            d, flagged, pw = run_file(df, cfg, ws)
            row = {"file": f.name, **m, "z_threshold": z, "loss_threshold": loss,
                   "detectors": "+".join(cfg["detectors"]), **summarize(d, flagged, pw),
                   "seconds": round(time.time() - t0, 1)}
            if a.subgroups:
                row.update(subgroup_loss(d, flagged))
            rows.append(row)
            print(f"{f.name:42s} z={z:<4g} loss={loss:<4g} recall={row['recall']:.2f} precision={row['precision']:.2f} "
                  f"clean_loss={row['clean_loss']:.3f} ttd={row['time_to_detect']}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
