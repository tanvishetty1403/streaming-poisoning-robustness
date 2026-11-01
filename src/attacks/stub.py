import numpy as np


def no_attack(df, attack_name="none", budget=0.0, timing="uniform", seed=0, dataset="paysim"):
    """Contract stub: nothing is poisoned. P2 provides the real ones."""
    df = df.sort_values(["step", "record_id"], kind="stable").reset_index(drop=True)
    return df, np.zeros(len(df), dtype=bool)