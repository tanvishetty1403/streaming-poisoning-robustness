import numpy as np
from .common import pick, finish


def random_label_flip(batch, cfg, rng):
    """cfg: rate (fraction of all rows), timing, label_col (default 'isFraud')."""
    lc = cfg.get("label_col", "isFraud")
    out = batch.copy()
    idx = pick(np.arange(len(out)), cfg["rate"], rng, cfg.get("timing", "uniform"))
    out.iloc[idx, out.columns.get_loc(lc)] = 1 - out.iloc[idx][lc].to_numpy()
    return finish(out, idx)


def targeted_label_flip(batch, cfg, rng):
    """cfg: rate (fraction of fraud rows flipped fraud->legit), timing, label_col."""
    lc = cfg.get("label_col", "isFraud")
    out = batch.copy()
    fraud = np.flatnonzero(out[lc].to_numpy() == 1)
    idx = pick(fraud, cfg["rate"], rng, cfg.get("timing", "uniform"))
    out.iloc[idx, out.columns.get_loc(lc)] = 0
    return finish(out, idx)
