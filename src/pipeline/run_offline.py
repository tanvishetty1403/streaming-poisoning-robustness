import argparse
import os
import uuid

import yaml

from src.pipeline.core import (ATTACKS, COLUMNS, StreamRun, apply_attack_by_name,  # noqa: F401
                               get_defense, prepare_run)
from src.pipeline.replay import replay_windows


def run_offline(dataset, attack="none", budget=0.0, timing="uniform", defense="off",
                seed=0, cfg=None, data=None):
    poisoned, test, dcfg = prepare_run(dataset, attack, budget, timing, seed, cfg, data)
    run = StreamRun(dataset, test, cfg, attack, budget, timing, defense, seed,
                    uuid.uuid4().hex[:8])
    for k, win in replay_windows(poisoned, dcfg["window_steps"]):
        run.step(k, win)
    return run.result()


def append_csv(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="paysim")
    ap.add_argument("--attack", default="none", choices=ATTACKS)
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--timing", default="uniform")
    ap.add_argument("--defense", default="off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config/default.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = run_offline(a.dataset, a.attack, a.budget, a.timing, a.defense, a.seed, cfg)
    append_csv(out, cfg["results_csv"])
    print(out.tail(3).to_string(index=False))
