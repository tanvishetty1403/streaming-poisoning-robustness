from .label_flip import random_label_flip, targeted_label_flip
from .feature_poison import feature_poison

ATTACKS = {
    "random_label_flip": random_label_flip,
    "targeted_label_flip": targeted_label_flip,
    "feature_poison": feature_poison,
}


def attack(batch, cfg, rng):
    """Contract entry point: dispatch on cfg['type']."""
    return ATTACKS[cfg["type"]](batch, cfg, rng)
