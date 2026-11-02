#!/usr/bin/env python
"""Fill out the attacked-sample grid from the existing clean stream (offline tuning/eval only).

Adds attack seeds and any missing (attack, budget, timing) files next to the ones P2 made, without needing the
raw PaySim CSV: it reads <dir>/stream_clean.csv.gz (the sample P2 already drew, steps below test_from_step) and
applies the same attacks via attack_from_config. Only the ATTACK randomness changes with --seeds; the clean
sample is the same, so this measures sensitivity to the attack draw, not to the data sample.

  python scripts/extend_attacked_samples.py --seeds 0 1 2
Existing files are skipped. Writes <attack>_b<budget>_<timing>_s<seed>.csv.gz (with is_poisoned) and manifest_extra.csv.
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attacks import attack_from_config, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "data" / "attacked_sample"))
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--attacks", nargs="+")
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.2])
    ap.add_argument("--timings", nargs="+", default=["uniform", "burst", "low_and_slow"])
    a = ap.parse_args()

    config = load_config(a.config)
    d = Path(a.dir)
    stream = pd.read_csv(d / "stream_clean.csv.gz")
    names = a.attacks or list(config["attacks"]["methods"])
    rows = []
    for seed, name, b, t in itertools.product(a.seeds, names, a.budgets, a.timings):
        f = f"{name}_b{b:g}_{t}_s{seed}.csv.gz"
        if (d / f).exists():
            continue
        poisoned, mask = attack_from_config(stream, name, b, t, seed, "paysim", config)
        poisoned.assign(is_poisoned=mask).to_csv(d / f, index=False, compression={"method": "gzip", "compresslevel": 1})
        rows.append({"file": f, "attack": name, "budget": b, "timing": t, "seed": seed, "n_rows": len(stream),
                     "n_poisoned": int(mask.sum())})
        print("wrote", f, int(mask.sum()), flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(d / "manifest_extra.csv", index=False)
    print(f"{len(rows)} new files")


if __name__ == "__main__":
    main()
