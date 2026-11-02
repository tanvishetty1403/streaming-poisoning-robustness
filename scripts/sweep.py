"""Run the experiment grid and append every run to results/metrics.csv.

Conditions per (attack, budget, timing, seed): defense off and defense on.
Plus the no-attack baseline (defense off and on) per seed.

    python scripts/sweep.py --quick          # small grid for a first look
    python scripts/sweep.py                  # demo grid
    python scripts/sweep.py --fresh          # delete old metrics.csv first
"""
import argparse
import itertools
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline.data import load_dataset                                   # noqa: E402
from src.pipeline.run_offline import append_csv, run_offline                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="paysim")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--attacks", default="targeted_flip,random_flip,feature_shift,backdoor")
    ap.add_argument("--budgets", default="0.2,0.5")
    ap.add_argument("--timings", default="uniform,burst")
    ap.add_argument("--seeds", default=None, help="comma list; default = seeds in config")
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 attacks, 1 budget, uniform timing")
    ap.add_argument("--fresh", action="store_true", help="delete the old metrics csv first")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    attacks = a.attacks.split(",")
    budgets = [float(b) for b in a.budgets.split(",")]
    timings = a.timings.split(",")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(cfg["seeds"])
    if a.quick:
        attacks, budgets, timings, seeds = attacks[:2], budgets[-1:], ["uniform"], seeds[:1]

    out = cfg["results_csv"]
    if a.fresh and os.path.exists(out):
        os.remove(out)

    jobs = []
    for seed in seeds:
        for d in ("off", "on"):
            jobs.append((seed, "none", 0.0, "uniform", d))
        for atk, b, t, d in itertools.product(attacks, budgets, timings, ("off", "on")):
            jobs.append((seed, atk, b, t, d))

    print(f"{len(jobs)} runs -> {out}")
    t0, cur_seed, data = time.time(), None, None
    for i, (seed, atk, b, t, d) in enumerate(jobs, 1):
        if seed != cur_seed:                       # load the dataset once per seed (the sample depends on the seed)
            data = load_dataset(a.dataset, cfg["datasets"][a.dataset], seed)
            cur_seed = seed
        try:
            df = run_offline(a.dataset, atk, b, t, d, seed, cfg, data=data)
            append_csv(df, out)
            last = df.iloc[-1]
            print(f"[{i}/{len(jobs)}] seed={seed} {atk:14s} b={b:<4} {t:13s} def={d:3s} "
                  f"recall={last['recall']:.3f} pr_auc={last['pr_auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:                     # one bad config must not kill the whole sweep
            print(f"[{i}/{len(jobs)}] FAILED {atk} b={b} {t} def={d} seed={seed}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
