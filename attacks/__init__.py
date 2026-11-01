from .label_flip import random_label_flip, targeted_label_flip
from .feature_poison import feature_poison
from .validate import validate
from .common import compute_feature_std

ATTACKS = {
    "random_label_flip": random_label_flip,
    "targeted_label_flip": targeted_label_flip,
    "feature_poison": feature_poison,
}


def attack(batch, cfg, rng):
    """Contract entry point: validate cfg, then dispatch on cfg['type']."""
    validate(cfg)
    return ATTACKS[cfg["type"]](batch, cfg, rng)
