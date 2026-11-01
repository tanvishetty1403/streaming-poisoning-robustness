"""Detection core. All settings come from the `defense:` section of config/default.yaml."""
import numpy as np
import pandas as pd

from .common import assert_no_truth, feature_cols
from .detectors import LabelConsistency, RobustZ
from .quarantine import Quarantine

REQUIRED = ["detectors", "z_threshold", "loss_threshold", "ref_window", "min_ref",
            "bootstrap_first_window", "label_window", "label_min_ref", "min_class",
            "review_windows", "label_col", "time_col", "features",
            "class_conditional", "min_ref_class", "trusted_windows"]


def make_state(cfg: dict) -> dict:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise KeyError(f"missing keys in the defense config: {missing}")
    return {"cfg": dict(cfg), "window_id": 0, "detectors": None,
            "quarantine": Quarantine(cfg["review_windows"]), "last_per_detector": {}}


def _build(cfg):
    dets = []
    if "robust_z" in cfg["detectors"]:
        dets.append(RobustZ(cfg["z_threshold"], cfg["ref_window"], cfg["min_ref"], cfg["bootstrap_first_window"],
                            cfg["class_conditional"], cfg["min_ref_class"]))
    if "label_loss" in cfg["detectors"]:
        dets.append(LabelConsistency(cfg["loss_threshold"], cfg["label_window"], cfg["label_min_ref"], cfg["min_class"]))
    if not dets:
        raise ValueError(f"no known detector in cfg['detectors']={cfg['detectors']}")
    return dets


def _ensemble(dets, X, y):
    """Normalise each detector by its own threshold (1.0 = at threshold), then take the max."""
    per = {d.name: d.score(X, y) / d.threshold for d in dets}
    stack = np.vstack([per[d.name] for d in dets])
    return stack.max(axis=0), stack.argmax(axis=0), per


def _features(batch, state):
    fn = state.get("featurize")
    if fn is not None:
        return np.asarray(fn(batch), dtype=float)
    return batch[feature_cols(batch, state["cfg"])].to_numpy(dtype=float)


def defense(batch: pd.DataFrame, state: dict):
    """Score every record, quarantine the suspicious ones, optionally release reviewed ones.

    Returns (accepted, quarantined, scores). `scores` is the ensemble suspicion score aligned
    with the INPUT row order, in units of the detector thresholds: above 1.0 means quarantine.
    Output rows are unchanged rows of the input (no columns added). When release is enabled
    (review_windows not null) `accepted` also contains earlier-quarantined rows that passed
    review this window. Per-detector scores of the last window are in state['last_per_detector'].
    """
    assert_no_truth(batch)
    cfg, q = state["cfg"], state["quarantine"]
    if state["detectors"] is None:
        state["detectors"] = _build(cfg)
    dets, wid = state["detectors"], state["window_id"]
    names = np.array([d.name for d in dets])

    lc = cfg["label_col"]
    if any(d.needs_labels for d in dets) and lc not in batch.columns:
        raise KeyError(f"the detectors need column '{lc}' in the window (set defense.label_col)")
    y = batch[lc].to_numpy(dtype=int) if lc in batch.columns else None
    X = _features(batch, state)

    if wid < cfg["trusted_windows"]:
        # Trusted warm-up (assumed clean): accept everything, only build the references.
        scores, top = np.zeros(len(batch)), np.zeros(len(batch), dtype=int)
        per = {d.name: scores for d in dets}
    else:
        scores, top, per = _ensemble(dets, X, y)
    state["last_per_detector"] = per
    flagged = scores > 1.0

    quarantined = batch[flagged]
    accepted = batch[~flagged]
    for d in dets:
        d.update(X[~flagged], y[~flagged] if y is not None else None)  # accepted records only
    q.hold(quarantined, scores[flagged], wid, list(names[top][flagged]))

    if cfg["review_windows"] is not None:
        def rescore(r):
            yr = r[lc].to_numpy(dtype=int) if lc in r.columns else None
            return _ensemble(dets, _features(r, state), yr)[0]

        released, _ = q.review(wid, rescore, 1.0)
        if len(released):
            yr = released[lc].to_numpy(dtype=int) if y is not None else None
            for d in dets:
                d.update(_features(released, state), yr)
            accepted = pd.concat([accepted, released])
            tc = cfg["time_col"]
            if tc in accepted.columns:
                accepted = accepted.sort_values(tc, kind="stable")

    state["window_id"] = wid + 1
    return accepted, quarantined, scores
