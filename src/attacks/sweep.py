"""Sweep runner: attack x budget x timing x seed x defense -> results/metrics.csv (contract section 5).

  python -m src.attacks.sweep --attacks random_flip feature_shift --budgets 0.05 0.2 --seeds 0 1 2
  python -m src.attacks.sweep --clean-dir data/attacked_sample        # no raw PaySim: use the saved clean sample
  python -m src.attacks.sweep --dry-run                               # list the runs only

Each (attack, budget, timing, seed) is attacked ONCE and replayed twice (defense off, defense on), so the two
rows of a pair see the identical poisoned stream. Runs already present in the output CSV are skipped, so an
interrupted sweep resumes where it stopped. A `none` baseline (no attack) is added per seed unless --no-baseline.
With --clean-dir the data sample is fixed (stream_clean + test_clean), so `seed` varies the attack draw and the
learner, not the sample.
"""
import argparse
import itertools
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import yaml

from src.pipeline.core import StreamRun, prepare_run
from src.pipeline.replay import replay_windows

KEY = ["dataset", "attack", "budget", "timing", "defense", "seed"]


def load_clean(clean_dir):
    d = Path(clean_dir)
    parts = [pd.read_csv(d / "stream_clean.csv.gz"), pd.read_csv(d / "test_clean.csv.gz")]
    return pd.concat(parts, ignore_index=True).sort_values(["step", "record_id"], kind="stable").reset_index(drop=True)


def grid(attacks, budgets, timings, seeds, baseline=True):
    """Yield (attack, budget, timing, seed). The baseline has budget 0 and one timing."""
    for s in seeds:
        if baseline:
            yield "none", 0.0, "uniform", s
        for a, b, t in itertools.product(attacks, budgets, timings):
            yield a, b, t, s


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, usecols=KEY).drop_duplicates()
    return {tuple(r) for r in d.itertuples(index=False, name=None)}


def run_one(dataset, attack, budget, timing, seed, defenses, cfg, data=None):
    """Attack once, replay once per defense setting. Returns a list of per-window frames."""
    poisoned, test, dcfg = prepare_run(dataset, attack, budget, timing, seed, cfg, data)
    out = []
    for dfn in defenses:
        run = StreamRun(dataset, test, cfg, attack, budget, timing, dfn, seed, uuid.uuid4().hex[:8])
        for k, win in replay_windows(poisoned, dcfg["window_steps"]):
            run.step(k, win)
        out.append(run.result())
    return out


def main(argv=None):
    cfg0 = yaml.safe_load(open("config/default.yaml"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="paysim")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--attacks", nargs="+", default=list(cfg0["attacks"]["methods"]))
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.2])
    ap.add_argument("--timings", nargs="+", default=list(cfg0["attacks"]["timings"]))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(cfg0["seeds"]))
    ap.add_argument("--defenses", nargs="+", default=["off", "on"])
    ap.add_argument("--clean-dir", help="use <dir>/stream_clean.csv.gz + test_clean.csv.gz instead of data/raw")
    ap.add_argument("--out", help="default: results_csv from the config")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    out = a.out or cfg["results_csv"]
    data = load_clean(a.clean_dir) if a.clean_dir else None
    done = done_keys(out)
    todo = []
    for atk, b, t, s in grid(a.attacks, a.budgets, a.timings, a.seeds, not a.no_baseline):
        left = [d for d in a.defenses if (a.dataset, atk, float(b), t, d, s) not in done]
        if left:
            todo.append((atk, b, t, s, left))
    print(f"{len(todo)} runs to do ({len(done)} already in {out})", flush=True)
    if a.dry_run:
        for r in todo:
            print(*r)
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for i, (atk, b, t, s, left) in enumerate(todo, 1):
        for df in run_one(a.dataset, atk, b, t, s, left, cfg, data):
            df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        last = pd.read_csv(out).tail(1).iloc[0]
        print(f"[{i}/{len(todo)}] {atk} b={b} {t} s{s}: pr_auc={last['pr_auc']:.3f} ({','.join(left)})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
