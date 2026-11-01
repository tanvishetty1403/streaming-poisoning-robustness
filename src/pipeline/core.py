"""Logic shared by the offline loop and the Kafka consumer.
Both paths use the same prepare_run / StreamRun code, so a mismatch can only come
from the transport (Kafka), never from two copies of the logic drifting apart."""
import numpy as np
import pandas as pd

from src.attacks.stub import no_attack
from src.defense.stub import no_defense
from src.pipeline.data import load_dataset, split_stream_test, featurize
from src.pipeline.learner import OnlineLearner
from src.pipeline.metrics import eval_clean, eval_detection

COLUMNS = ["run_id", "dataset", "attack", "budget", "timing", "defense", "seed", "window",
           "step_end", "n_seen", "n_quarantined", "pr_auc", "recall", "f1",
           "det_precision", "det_recall", "det_fpr", "time_to_detect", "backdoor_hit_rate"]

ATTACKS = ["none", "random_flip", "targeted_flip", "feature_shift", "feature_scale",
           "feature_noise", "backdoor"]


def apply_attack_by_name(stream, attack, budget, timing, seed, dataset):
    """Our wrapper around the contract entry point apply_attack (src/attacks/api.py)."""
    if attack == "none":
        return no_attack(stream)
    from src.attacks.api import apply_attack        # P2's entry point (per contract)
    return apply_attack(stream, attack_name=attack, budget=budget, timing=timing,
                        seed=seed, dataset=dataset)


def get_defense(name):
    if name == "off":
        return no_defense
    from src.defense.pipeline import defend         # P3's entry point (per contract)
    return defend


def prepare_run(dataset, attack, budget, timing, seed, cfg, data=None):
    """Load, split by time, attack the TRAIN stream only. Used by offline AND Kafka,
    so apply_attack always receives exactly the same train stream."""
    dcfg = cfg["datasets"][dataset]
    df = data if data is not None else load_dataset(dataset, dcfg, seed)
    stream, test = split_stream_test(df, dcfg["test_from_step"])
    poisoned, mask = apply_attack_by_name(stream, attack, budget, timing, seed, dataset)
    poisoned = poisoned.copy()
    poisoned["is_poisoned"] = np.asarray(mask, dtype=bool)
    return poisoned, test, dcfg


def records_to_window(records, dtypes):
    """Rebuild a window dataframe from JSON records (Kafka messages), restoring the
    original column types so the result is identical to the offline window."""
    win = pd.DataFrame(records).drop(columns="_w")
    win = win.astype({c: t for c, t in dtypes.items() if c in win.columns})
    return win.sort_values("record_id").reset_index(drop=True)


class StreamRun:
    """One run: defense -> learner -> metrics, window by window."""

    def __init__(self, dataset, test, cfg, attack, budget, timing, defense, seed, run_id):
        lcfg = cfg["learner"]
        self.dataset, self.thr, self.defense_name = dataset, lcfg["threshold"], defense
        self.meta = {"run_id": run_id, "dataset": dataset, "attack": attack, "budget": budget,
                     "timing": timing, "defense": defense, "seed": seed}
        self.X_test, self.y_test = featurize(test, dataset), test["isFraud"].to_numpy()
        self.X_trig = None
        if attack == "backdoor":
            from src.attacks.api import make_triggered_test
            trig = make_triggered_test(test[test["isFraud"] == 1], dataset)
            self.X_trig = featurize(trig, dataset)
        self.learner = OnlineLearner(lcfg["fraud_weight"], seed)
        self.defend = get_defense(defense)
        self.state, self.rows, self.npois, self.n_seen = {}, [], [], 0

    def step(self, k, win):
        win = win.reset_index(drop=True)
        visible = win.drop(columns=["is_poisoned"])           # defenses never see ground truth
        accepted, quarantined, self.state = self.defend(visible, self.state)
        acc = win[win["record_id"].isin(accepted["record_id"])]
        self.learner.partial_fit(featurize(acc, self.dataset), acc["isFraud"].to_numpy())
        self.n_seen += len(win)
        self.npois.append(int(win["is_poisoned"].sum()))
        row = dict(self.meta, window=k, step_end=int(win["step"].max()),
                   n_seen=self.n_seen, n_quarantined=len(quarantined))
        row.update(eval_clean(self.learner, self.X_test, self.y_test, self.thr))
        row.update(eval_detection(win, quarantined) if self.defense_name != "off"
                   else {"det_precision": np.nan, "det_recall": np.nan, "det_fpr": np.nan})
        row["backdoor_hit_rate"] = (
            float((self.learner.predict_proba(self.X_trig) < self.thr).mean())
            if self.X_trig is not None and len(self.X_trig) else np.nan)
        self.rows.append(row)

    def result(self):
        """time_to_detect: windows from the first poisoned window to the first window
        that quarantines a poisoned record (one value per run, repeated on every row)."""
        ttd = np.nan
        pw = next((r["window"] for r, n in zip(self.rows, self.npois) if n > 0), None)
        if pw is not None and self.defense_name != "off":
            dw = next((r["window"] for r in self.rows
                       if r["window"] >= pw and r["det_recall"] > 0), None)
            if dw is not None:
                ttd = dw - pw
        rows = [dict(r, time_to_detect=ttd) for r in self.rows]
        return pd.DataFrame(rows, columns=COLUMNS)