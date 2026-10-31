import numpy as np
from .common import pick, finish


def random_label_flip(batch, cfg, rng):
    """cfg: rate (fraction of ALL rows, e.g. 0.01/0.05/0.10/0.20)."""
    out = batch.copy()
    idx = pick(np.arange(len(out)), cfg["rate"], rng)
    out.iloc[idx, out.columns.get_loc("label")] = 1 - out.iloc[idx]["label"].to_numpy()
    return finish(out, idx)


def targeted_label_flip(batch, cfg, rng):
    """cfg: rate (fraction of fraud rows flipped fraud->legit)."""
    out = batch.copy()
    fraud = np.flatnonzero(out["label"].to_numpy() == 1)
    idx = pick(fraud, cfg["rate"], rng)
    out.iloc[idx, out.columns.get_loc("label")] = 0
    return finish(out, idx)
