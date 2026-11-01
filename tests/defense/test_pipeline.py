"""Contract tests for defend() (contracts/CONTRACTS.md section 3), using P1's real featurizer."""
import numpy as np
import pytest

from src.defense import defend
from src.defense.metrics import detection_metrics
from src.defense.pipeline import infer_dataset, load_defense_cfg
from helpers import flip_fraud_labels, make_paysim, poison_amount


def flagged_mask(window, quarantined):
    return window["record_id"].isin(quarantined["record_id"]).to_numpy()


def test_state_starts_empty_and_every_row_goes_to_exactly_one_output():
    w = make_paysim(4000, seed=0)
    state = {}
    accepted, quarantined, state2 = defend(w, state)
    assert state2 is state and state["dataset"] == "paysim" and state["window_id"] == 1
    a, q = set(accepted["record_id"]), set(quarantined["record_id"])
    assert not a & q and a | q == set(w["record_id"]) and len(accepted) + len(quarantined) == len(w)
    assert list(accepted.columns) == list(w.columns) == list(quarantined.columns)  # rows are unchanged


def test_rejects_ground_truth_column():
    w = make_paysim(500)
    w["is_poisoned"] = False
    with pytest.raises(ValueError, match="ground-truth"):
        defend(w, {})


def test_dataset_inference():
    assert infer_dataset(make_paysim(50)) == "paysim"
    with pytest.raises(ValueError, match="cannot infer"):
        infer_dataset(make_paysim(50).drop(columns=["type"]))


def test_real_config_has_a_complete_defense_section():
    cfg = load_defense_cfg()
    assert cfg["label_col"] == "isFraud" and cfg["time_col"] == "step" and cfg["review_windows"] is None


def test_clean_windows_are_mostly_accepted():
    state = {}
    defend(make_paysim(4000, seed=0), state)
    w = make_paysim(4000, seed=1, start=4000, step0=25)
    _, quarantined, _ = defend(w, state)
    assert len(quarantined) / len(w) <= 0.02


def test_first_window_is_trusted_by_design():
    # Documented assumption: trusted_windows=1 -> window 0 is accepted unchecked and only builds the reference.
    w, _ = poison_amount(make_paysim(4000, seed=0), 0.1, 1e5, seed=1)
    _, quarantined, _ = defend(w, {})
    assert len(quarantined) == 0


def test_feature_poison_is_caught_in_later_windows():
    state = {}
    defend(make_paysim(4000, seed=0), state)
    w, mask = poison_amount(make_paysim(4000, seed=1, start=4000, step0=25), 0.1, 1e4, seed=2)
    _, quarantined, _ = defend(w, state)
    m = detection_metrics(flagged_mask(w, quarantined), mask)
    print("later window, amount x1e4:", {k: round(v, 3) for k, v in m.items() if k in ("precision", "recall", "fpr")})
    assert m["recall"] >= 0.8 and m["fpr"] <= 0.02


def test_label_flips_are_caught_once_the_reference_model_is_ready():
    state = {}
    for i in range(3):                                                  # accepted history trains the model
        defend(make_paysim(4000, seed=i, start=4000 * i, step0=1 + 24 * i), state)
    w, mask = flip_fraud_labels(make_paysim(4000, seed=9, start=12000, step0=73), 0.1, seed=3)
    _, quarantined, _ = defend(w, state)
    m = detection_metrics(flagged_mask(w, quarantined), mask)
    print("label flips 10%:", {k: round(v, 3) for k, v in m.items() if k in ("precision", "recall", "fpr")})
    assert m["precision"] >= 0.8 and m["recall"] >= 0.5 and m["fpr"] <= 0.02


def test_audit_log_and_scores_are_kept_in_state():
    state = {}
    defend(make_paysim(4000, seed=0), state)
    w, _ = poison_amount(make_paysim(4000, seed=1, start=4000, step0=25), 0.1, 1e4, seed=2)
    _, quarantined, _ = defend(w, state)
    held = [e for e in state["quarantine"].log if e["event"] == "held" and e["window_id"] == 1]
    assert len(held) == len(quarantined) and all(e["detector"] for e in held)
    assert len(state["last_scores"]) == len(w) and set(state["last_per_detector"]) == {"robust_z", "label_loss"}
