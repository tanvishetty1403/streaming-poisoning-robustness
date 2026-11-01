import numpy as np
from .common import pick, finish
from .trigger import stamp_trigger


def backdoor(batch, cfg, rng):
    """Stamp a trigger pattern on FRAUD rows and relabel them legit, so a learner associates the
    trigger with 'legit'. cfg: rate (fraction of fraud rows), timing, trigger {col: value}, label_col.
    Every poisoned row changes (fraud -> legit), so the mask is exactly the tampered rows."""
    lc = cfg.get("label_col", "isFraud")
    fraud = np.flatnonzero(batch[lc].to_numpy() == 1)
    idx = pick(fraud, cfg["rate"], rng, cfg.get("timing", "uniform"))
    out = stamp_trigger(batch, cfg["trigger"], idx)
    out.iloc[idx, out.columns.get_loc(lc)] = 0
    return finish(out, idx)
