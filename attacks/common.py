"""Shared helpers. Contract: attack(batch, cfg, rng) -> (batch, poison_mask).

Batch: pandas DataFrame with record_id, event_time, label (0/1), feature columns.
Attacks never mutate the input. The returned mask is what THIS attack changed;
the `is_poisoned` column accumulates across stacked attacks (prior | mask).
"""
import numpy as np


def compute_feature_std(clean_df, features) -> dict:
    """Reference std from clean warm-up data, to pass as cfg['feature_std']."""
    return {f: float(clean_df[f].std(ddof=0)) for f in features}


def pick(candidates: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Stochastic rounding: k = floor(rate*n + U[0,1)), so E[k] = rate*n even on tiny batches."""
    n = len(candidates)
    k = min(int(np.floor(rate * n + rng.random())), n)
    return rng.choice(candidates, size=k, replace=False) if k else np.array([], dtype=int)


def finish(out, idx: np.ndarray):
    mask = np.zeros(len(out), dtype=bool)
    mask[idx] = True
    prior = out["is_poisoned"].to_numpy(dtype=bool) if "is_poisoned" in out.columns else False
    out["is_poisoned"] = prior | mask
    return out, mask
