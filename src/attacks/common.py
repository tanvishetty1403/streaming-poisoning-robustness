"""Shared helpers. Batch = pandas DataFrame in event-time order with record_id, step, isFraud + features.
Attacks never mutate the input. The returned mask is what THIS attack changed;
the `is_poisoned` column accumulates across stacked attacks (prior | mask)."""
import numpy as np

TIMINGS = ("uniform", "burst", "low_and_slow")


def compute_feature_std(clean_df, features) -> dict:
    """Reference std from clean data, to pass as cfg['feature_std']."""
    return {f: float(clean_df[f].std(ddof=0)) for f in features}


def pick(candidates, rate: float, rng: np.random.Generator, timing: str = "uniform") -> np.ndarray:
    """Choose k = floor(rate*n + U[0,1)) of the candidate positions (stochastic rounding, E[k] = rate*n).
    timing decides WHERE in event time the k poisoned rows land:
      uniform      - random positions anywhere
      burst        - k consecutive candidates (one contiguous block), random start
      low_and_slow - evenly spread: one random pick per equal-sized stratum of the candidates
    """
    cand = np.sort(np.asarray(candidates))
    n = len(cand)
    k = min(int(np.floor(rate * n + rng.random())), n)
    if k == 0:
        return np.array([], dtype=int)
    if timing == "uniform":
        return rng.choice(cand, size=k, replace=False)
    if timing == "burst":
        s = int(rng.integers(0, n - k + 1))
        return cand[s:s + k]
    if timing == "low_and_slow":
        edges = (np.arange(k + 1) * n) // k
        lo, hi = edges[:-1], edges[1:]
        return cand[lo + (rng.random(k) * (hi - lo)).astype(int)]
    raise ValueError(f"unknown timing {timing!r}")


def finish(out, idx: np.ndarray):
    mask = np.zeros(len(out), dtype=bool)
    mask[idx] = True
    prior = out["is_poisoned"].to_numpy(dtype=bool) if "is_poisoned" in out.columns else False
    out["is_poisoned"] = prior | mask
    return out, mask
