from .label_flip import random_label_flip, targeted_label_flip
from .feature_poison import feature_poison
from .backdoor import backdoor
from .validate import validate

ATTACKS = {
    "random_label_flip": random_label_flip,
    "targeted_label_flip": targeted_label_flip,
    "feature_poison": feature_poison,
    "backdoor": backdoor,
}


def run_attack(batch, cfg, rng):
    """Low-level: validate cfg (incl. cfg['rate'], cfg['timing']), run one attack on `batch`."""
    validate(cfg)
    return ATTACKS[cfg["type"]](batch, cfg, rng)
