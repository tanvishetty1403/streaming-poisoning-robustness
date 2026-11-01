"""Shared constants and helpers for the defense."""
import pandas as pd

META_COLS = {"record_id", "event_time", "step", "label", "isFraud", "is_poisoned"}
TRUTH_COLS = ("is_poisoned",)  # ground truth: only the evaluator may read these


def assert_no_truth(batch: pd.DataFrame) -> None:
    """Leakage guard: the runner must strip ground-truth columns before the defense sees a window."""
    leaked = [c for c in TRUTH_COLS if c in batch.columns]
    if leaked:
        raise ValueError(f"ground-truth column(s) {leaked} reached the defense; the runner must strip them")


def feature_cols(batch: pd.DataFrame, cfg: dict) -> list:
    """cfg['features'] if given, else every numeric column that is not metadata.

    The label and time columns are always excluded, so the label can never become a feature.
    """
    cols = cfg.get("features")
    if cols:
        return list(cols)
    skip = META_COLS | {cfg.get("label_col"), cfg.get("time_col")}
    return [c for c in batch.select_dtypes("number").columns if c not in skip]
