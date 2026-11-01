import numpy as np
import pandas as pd
import pytest

from src.defense import defense, make_state
from src.defense.common import feature_cols
from src.defense.metrics import detection_metrics
from src.defense.pipeline import load_defense_cfg
from src.defense.quarantine import Quarantine
from helpers import flip_labels, make_clean, make_labeled, shift_rows


def cfgx(**kw):
    """The real defense config, adapted to the small synthetic frames used here."""
    base = {**load_defense_cfg(), "label_col": "label", "time_col": "event_time", "min_ref": 200,
            "ref_window": 2000, "label_min_ref": 500, "label_window": 5000, "bootstrap_first_window": False}
    return {**base, **kw}


def Z(**kw):
    """State with only the robust z-score detector (isolates it from the label detector)."""
    return make_state(cfgx(detectors=["robust_z"], **kw))


def L(**kw):
    """State with only the label-consistency detector."""
    return make_state(cfgx(detectors=["label_loss"], **kw))


def B(**kw):
    """State with both detectors (the ensemble)."""
    return make_state(cfgx(**kw))


def run(state, df):
    clean, quar, scores = defense(df, state)
    flagged = df["record_id"].isin(quar["record_id"]).to_numpy()
    return clean, quar, scores, flagged


# ---------------- contract and leakage ----------------
def test_rejects_ground_truth_column():
    df = make_clean(300)
    df["is_poisoned"] = False
    with pytest.raises(ValueError, match="ground-truth"):
        defense(df, Z())


def test_label_column_is_never_a_feature():
    df = make_clean(50).rename(columns={"label": "isFraud", "event_time": "step"})
    cols = feature_cols(df, {"label_col": "isFraud", "time_col": "step", "features": None})
    assert cols == ["V1", "V2", "V3"]


def test_missing_label_column_raises_a_clear_error():
    df = make_clean(300).drop(columns=["label"])
    with pytest.raises(KeyError, match="label"):
        defense(df, B())


# ---------------- robust z-score ----------------
def test_warmup_accepts_everything():
    df, _ = shift_rows(make_clean(100), 1.0, 8)           # every row anomalous, but no reference yet
    clean, quar, scores, _ = run(Z(), df)
    assert len(quar) == 0 and len(clean) == 100 and (scores == 0).all()


def test_detects_shifted_rows():
    st = Z()
    run(st, make_clean(1000, seed=0))
    df, mask = shift_rows(make_clean(1000, seed=1, start=1000), 0.1, 8, seed=2)
    _, _, _, flagged = run(st, df)
    m = detection_metrics(flagged, mask)
    assert m["recall"] >= 0.95 and m["fpr"] <= 0.01


def test_scores_aligned_with_non_default_index():
    st = Z()
    run(st, make_clean(1000, seed=0))
    df, mask = shift_rows(make_clean(1000, seed=1, start=1000), 0.1, 8, seed=2)
    df.index = np.arange(len(df))[::-1] + 500             # reversed, non-default index
    clean, quar, scores, flagged = run(st, df)
    assert len(scores) == len(df)
    assert detection_metrics(flagged, mask)["recall"] >= 0.95


def test_quarantined_records_do_not_move_the_reference():
    st = Z(class_conditional=False)
    run(st, make_clean(1000, seed=0))
    before = st["detectors"][0]._stats[None][0].copy()
    df, _ = shift_rows(make_clean(500, seed=3, start=5000), 1.0, 8)   # every row anomalous
    _, quar, _, _ = run(st, df)
    assert len(quar) == 500
    assert np.allclose(st["detectors"][0]._stats[None][0], before)


# ---------------- label-consistency detector ----------------
def test_label_detector_silent_until_ready():
    df = make_labeled(100)
    df, _ = flip_labels(df, 0.5)
    clean, quar, scores, _ = run(L(), df)
    assert len(quar) == 0 and (scores == 0).all()


def test_label_detector_flags_flipped_labels():
    st = L()
    run(st, make_labeled(3000, seed=0))
    df, mask = flip_labels(make_labeled(2000, seed=1, start=3000), 0.1, seed=2)
    _, _, _, flagged = run(st, df)
    m = detection_metrics(flagged, mask)
    print("label_loss metrics:", {k: round(v, 3) for k, v in m.items() if k in ("precision", "recall", "f1", "fpr")})
    assert m["precision"] >= 0.8 and m["recall"] >= 0.5 and m["fpr"] <= 0.02


def test_label_detector_does_not_flag_clean_data():
    st = L()
    run(st, make_labeled(3000, seed=0))
    _, quar, _, _ = run(st, make_labeled(2000, seed=1, start=3000))
    assert len(quar) / 2000 <= 0.01


# ---------------- ensemble ----------------
def test_ensemble_catches_both_feature_poison_and_label_flips():
    st = B()
    run(st, make_labeled(3000, seed=0))
    df, shift_mask = shift_rows(make_labeled(2000, seed=1, start=3000), 0.1, 8, seed=2)
    df, flip_mask = flip_labels(df, 0.1, seed=3, exclude=shift_mask)
    _, _, _, flagged = run(st, df)
    both = detection_metrics(flagged, shift_mask | flip_mask)
    print("ensemble (union of both attacks):", {k: round(v, 3) for k, v in both.items() if k in ("precision", "recall", "f1", "fpr")})
    assert detection_metrics(flagged, shift_mask)["recall"] >= 0.95
    assert detection_metrics(flagged, flip_mask)["recall"] >= 0.5
    assert both["fpr"] <= 0.02


def test_per_detector_scores_are_kept_for_ablation():
    st = B()
    run(st, make_labeled(3000, seed=0))
    run(st, make_labeled(500, seed=1, start=3000))
    per = st["last_per_detector"]
    assert set(per) == {"robust_z", "label_loss", "mahalanobis"} and all(len(v) == 500 for v in per.values())


# ---------------- quarantine and audit ----------------
def test_nothing_lost_without_release():
    st = Z()
    run(st, make_clean(1000, seed=0))
    df, _ = shift_rows(make_clean(400, seed=4, start=2000), 0.1, 8, seed=5)
    clean, quar, _, _ = run(st, df)
    assert len(clean) + len(quar) == len(df)


def test_still_anomalous_records_are_discarded_after_review():
    st = Z(review_windows=1)
    run(st, make_clean(1000, seed=0))
    df, mask = shift_rows(make_clean(500, seed=6, start=3000), 0.04, 8, seed=7)
    _, quar, _, _ = run(st, df)
    bad_ids = set(quar["record_id"])
    clean3, _, _, _ = run(st, make_clean(300, seed=8, start=4000))   # review window elapsed here
    assert not bad_ids & set(clean3["record_id"])
    discarded = {e["record_id"] for e in st["quarantine"].log if e["event"] == "discarded"}
    assert bad_ids <= discarded


def test_quarantine_release_and_discard_logic():
    q = Quarantine(review_windows=1)
    recs = pd.DataFrame({"record_id": [1, 2], "x": [0.0, 9.0]})
    q.hold(recs, np.array([6.0, 9.0]), window_id=0)
    rel, dis = q.review(0, lambda r: r["x"].to_numpy(), 5.0)       # too early: nothing decided
    assert len(rel) == 0 and len(dis) == 0 and q.n_held == 2
    rel, dis = q.review(1, lambda r: r["x"].to_numpy(), 5.0)
    assert list(rel["record_id"]) == [1] and list(dis["record_id"]) == [2] and q.n_held == 0


def test_audit_log_records_which_detector_flagged_each_record():
    st = B()
    run(st, make_labeled(3000, seed=0))
    df, _ = shift_rows(make_labeled(1000, seed=9, start=6000), 0.1, 8, seed=10)
    _, quar, _, _ = run(st, df)
    held = [e for e in st["quarantine"].log if e["event"] == "held"]
    assert sorted(e["record_id"] for e in held) == sorted(quar["record_id"])
    assert all(e["detector"] in ("robust_z", "label_loss", "mahalanobis") for e in held)


def test_detection_metrics_basic():
    m = detection_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5


def test_label_detector_keeps_rare_fraud_rows_in_one_huge_window():
    """PaySim window 0 has ~57k rows but only ~24 frauds; a shared rolling buffer of 20k rows kept ~8 of them."""
    df = make_labeled(60000, seed=0)
    df["label"] = 0
    df.loc[df.index[:30], "label"] = 1                       # 30 rare frauds, all early in the window
    df.loc[df.index[:30], "V1"] += 6.0                       # and separable
    st = L(label_window=20000, label_min_ref=2000, min_class=20, trusted_windows=1)
    run(st, df)
    assert st["detectors"][0].ready
