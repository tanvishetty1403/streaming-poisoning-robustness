import numpy as np
from .common import pick, finish


def random_label_flip(batch, cfg, rng):
    """cfg: rate (fraction of ALL rows), label_col (default 'label')."""
    lc = cfg.get("label_col", "label")
    out = batch.copy()
    idx = pick(np.arange(len(out)), cfg["rate"], rng)
    out.iloc[idx, out.columns.get_loc(lc)] = 1 - out.iloc[idx][lc].to_numpy()
    return finish(out, idx)


def targeted_label_flip(batch, cfg, rng):
    """cfg: rate (fraction of fraud rows flipped fraud->legit), label_col (default 'label')."""
    lc = cfg.get("label_col", "label")
    out = batch.copy()
    fraud = np.flatnonzero(out[lc].to_numpy() == 1)
    idx = pick(fraud, cfg["rate"], rng)
    out.iloc[idx, out.columns.get_loc(lc)] = 0
    return finish(out, idx)
