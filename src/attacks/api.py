"""Contract section 2 entry points (contracts/CONTRACTS.md)."""
from .adapter import attack_from_config, load_config
from .trigger import stamp_trigger


def apply_attack(df, attack_name, budget, timing, seed, dataset, config=None):
    """apply_attack(df, attack_name, budget, timing, seed, dataset) -> (df_poisoned, poison_mask).
    df = TRAIN stream only, event-time order. Params come from config/default.yaml (`attacks:`)."""
    return attack_from_config(df, attack_name, budget, timing, seed, dataset, config)


def make_triggered_test(fraud_test_df, dataset, config=None):
    """Stamp the backdoor trigger on every row of the (clean) fraud test records, labels untouched.
    backdoor_hit_rate = share of these predicted legit by the learner."""
    config = config or load_config()
    trig = config["attacks"].get("trigger", {}).get(dataset)
    if not trig:
        raise ValueError(f"no attacks.trigger configured for dataset {dataset!r}")
    return stamp_trigger(fraud_test_df, trig)
