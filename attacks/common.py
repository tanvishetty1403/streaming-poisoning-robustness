"""Shared helpers. Contract: attack(batch, cfg, rng) -> (batch, poison_mask).

Assumed batch: pandas DataFrame with record_id, event_time, label (0/1), feature columns.
Attacks never mutate the input; they return a copy with `is_poisoned` set.
"""
import numpy as np
import pandas as pd

META_COLS = {"record_id", "event_time", "label", "is_poisoned", "flagged"}


def feature_cols(batch: pd.DataFrame, cfg: dict) -> list:
    return cfg.get("features") or [c for c in batch.columns if c not in META_COLS]


def pick(candidates: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Choose round(rate * len(candidates)) positions without replacement."""
    k = int(round(rate * len(candidates)))
    return rng.choice(candidates, size=k, replace=False) if k else np.array([], dtype=int)


def finish(out: pd.DataFrame, idx: np.ndarray):
    mask = np.zeros(len(out), dtype=bool)
    mask[idx] = True
    out["is_poisoned"] = mask
    return out, mask
