#!/usr/bin/env python
"""Make small attacked PaySim samples so P3 can tune detectors (offline tuning only: the output
carries the ground-truth `is_poisoned` column, which the defense itself must never read).

  python scripts/make_attacked_sample.py --input data/paysim.csv
  python scripts/make_attacked_sample.py --input data/paysim.csv --attacks random_flip feature_shift \
         --budgets 0.05 0.2 --timings uniform burst
  python scripts/make_attacked_sample.py --synthetic 20000          # no dataset needed, for smoke tests

Writes <out>/clean.csv.gz, one <attack>_b<budget>_<timing>_s<seed>.csv.gz per combination, and manifest.csv.
Settings (test_from_step, window_steps, attack params, clean_windows) come from config/default.yaml.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attacks import attack_from_config, load_config  # noqa: E402


def finalize(df):
    df = df.drop(columns=["record_id"], errors="ignore").sort_values("step", kind="stable").reset_index(drop=True)
    df.insert(0, "record_id", np.arange(len(df)))
    return df


def load_paysim(path, sample_frac, seed):
    df = pd.read_csv(path)
    if sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=seed)
    return finalize(df)


def synthetic(n, seed):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"step": np.sort(r.integers(1, 744, n)), "type": r.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], n),
                       "amount": r.lognormal(5, 1.5, n), "nameOrig": [f"C{i}" for i in range(n)],
                       "oldbalanceOrg": r.lognormal(8, 2, n), "newbalanceOrig": r.lognormal(8, 2, n),
                       "nameDest": [f"M{i}" for i in range(n)], "oldbalanceDest": r.lognormal(7, 2, n),
                       "newbalanceDest": r.lognormal(7, 2, n), "isFlaggedFraud": 0})
    df["isFraud"] = (r.random(n) < 0.005).astype(int)
    return finalize(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="raw PaySim CSV")
    ap.add_argument("--synthetic", type=int, metavar="N", help="use N rows of fake PaySim-shaped data instead")
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--sample-frac", type=float, help="default: datasets.paysim.sample_frac from the config")
    ap.add_argument("--attacks", nargs="+", help="default: all methods in the config")
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.10])
    ap.add_argument("--timings", nargs="+", default=["uniform"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "attacked_sample"))
    a = ap.parse_args()
    if not a.input and not a.synthetic:
        ap.error("give --input <paysim.csv> or --synthetic N")

    config = load_config(a.config)
    frac = a.sample_frac if a.sample_frac is not None else config["datasets"]["paysim"]["sample_frac"]
    df = synthetic(a.synthetic, a.seed) if a.synthetic else load_paysim(a.input, frac, a.seed)
    names = a.attacks or list(config["attacks"]["methods"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "clean.csv.gz", index=False)

    rows = []
    for name, b, t in itertools.product(names, a.budgets, a.timings):
        poisoned, mask = attack_from_config(df, name, b, t, a.seed, "paysim", config)
        f = f"{name}_b{b:g}_{t}_s{a.seed}.csv.gz"
        poisoned.assign(is_poisoned=mask).to_csv(out / f, index=False)
        rows.append({"file": f, "attack": name, "budget": b, "timing": t, "seed": a.seed, "n_rows": len(df),
                     "n_poisoned": int(mask.sum()), "fraud_rate": round(float(df["isFraud"].mean()), 5)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    print(manifest.to_string(index=False))
    print(f"\nwrote {len(rows)} files + clean.csv.gz to {out}")


if __name__ == "__main__":
    main()
