"""Contract entry point (contracts/CONTRACTS.md section 3).

    defend(window_df, state) -> (accepted_df, quarantined_df, state)

`state` starts as {} and is filled on the first call. Optionally the runner may set
state['dataset'] ('paysim' | 'ulb'); otherwise it is inferred from the columns. The window is
featurized with the SAME featurizer as the learner (src.pipeline.data.featurize) unless
defense.features lists explicit columns.
"""
from pathlib import Path

import yaml

from .core import defense, make_state

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_defense_cfg(path=CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "defense" not in cfg:
        raise KeyError(f"no 'defense:' section in {path}")
    return cfg["defense"]


def infer_dataset(df) -> str:
    cols = set(df.columns)
    if {"type", "oldbalanceOrg", "newbalanceOrig"} <= cols:
        return "paysim"
    if {"V1", "V28", "Amount"} <= cols:
        return "ulb"
    raise ValueError("cannot infer the dataset from the window's columns; set state['dataset']")


def defend(window_df, state: dict):
    if "quarantine" not in state:  # first call: state is {} (or only carries cfg / dataset)
        state.update(make_state(state.get("cfg") or load_defense_cfg()))
    if not state.get("dataset") and not state["cfg"].get("features"):
        state["dataset"] = infer_dataset(window_df)   # a name, so `state` can be pickled
    accepted, quarantined, scores = defense(window_df, state)
    state["last_scores"] = scores
    return accepted, quarantined, state
