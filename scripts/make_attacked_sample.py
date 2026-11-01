#!/usr/bin/env python
"""Make attacked samples so P3 can tune detectors (offline tuning only: attacked files carry the
ground-truth `is_poisoned` column, which the defense itself must never read).

Uses the pipeline's own load_dataset + split_stream_test (same rows as the offline loop for a given seed);
attacks are applied to the TRAIN STREAM only, the clean test set is never touched.

  python scripts/make_attacked_sample.py --budgets 0.05 0.2 --timings uniform burst low_and_slow
  python scripts/make_attacked_sample.py --dataset ulb --attacks random_flip backdoor
  python scripts/make_attacked_sample.py --input path/to/paysim.csv      # override the data path in src/pipeline/data.py

Writes <out>/stream_clean.csv.gz, test_clean.csv.gz, one <attack>_b<budget>_<timing>_s<seed>.csv.gz per
combination, and manifest.csv. Settings come from config/default.yaml.
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attacks import attack_from_config, load_config  # noqa: E402
from src.pipeline import data as pipeline_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="paysim", choices=["paysim", "ulb"])
    ap.add_argument("--input", help="override the raw CSV path used by src/pipeline/data.py")
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--sample-frac", type=float, help="default: datasets.<dataset>.sample_frac from the config")
    ap.add_argument("--attacks", nargs="+", help="default: all methods in the config")
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.10])
    ap.add_argument("--timings", nargs="+", default=["uniform"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "attacked_sample"))
    a = ap.parse_args()

    config = load_config(a.config)
    ds = dict(config["datasets"][a.dataset])
    if a.sample_frac is not None:
        ds["sample_frac"] = a.sample_frac
    if a.input:
        setattr(pipeline_data, "PAYSIM" if a.dataset == "paysim" else "ULB", a.input)

    full = pipeline_data.load_dataset(a.dataset, ds, a.seed)
    stream, test = pipeline_data.split_stream_test(full, ds["test_from_step"])
    win0 = ((stream["step"] - stream["step"].min()) // ds["window_steps"] == 0).to_numpy()   # P1's replay window 0

    names = a.attacks or list(config["attacks"]["methods"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stream.to_csv(out / "stream_clean.csv.gz", index=False)
    test.to_csv(out / "test_clean.csv.gz", index=False)

    rows = []
    for name, b, t in itertools.product(names, a.budgets, a.timings):
        poisoned, mask = attack_from_config(stream, name, b, t, a.seed, a.dataset, config)
        f = f"{name}_b{b:g}_{t}_s{a.seed}.csv.gz"
        poisoned.assign(is_poisoned=mask).to_csv(out / f, index=False)
        rows.append({"file": f, "dataset": a.dataset, "attack": name, "budget": b, "timing": t, "seed": a.seed,
                     "n_stream_rows": len(stream), "n_stream_fraud": int(stream["isFraud"].sum()),
                     "n_poisoned": int(mask.sum()), "n_poisoned_window0": int(mask[win0].sum())})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    print(manifest.to_string(index=False))
    print(f"\nwrote {len(rows)} attacked files + stream_clean/test_clean to {out}")


if __name__ == "__main__":
    main()
