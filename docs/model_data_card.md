# Model and data card

Poisoning and robustness in streaming classifiers (course project, Big Data Systems and Responsible AI).
This card covers the streaming fraud classifier, the poisoning defense (P3), and the data they were evaluated on.
Numbers are from `results/summary/*.csv` (3 attack seeds, see `notebooks/defense_results.ipynb`).

## 1. Model

**Learner.** An online logistic regression (`sklearn.linear_model.SGDClassifier`, log loss, `alpha=1e-5`) behind a
streaming `StandardScaler`, updated once per 24-step window with `partial_fit`. Fraud-labelled rows get weight 20
because fraud is about 0.1% of the data. Features: log-scaled amount and balances, two balance-identity residuals
(origin and destination), one-hot transaction type (7 numeric and 5 one-hot columns in all). Decision threshold 0.5.

**Defense** (`src/defense/`, entry point `defend(window_df, state) -> (accepted, quarantined, state)`). Sits in front of
the learner. Every record of a window goes to exactly one of the two outputs; quarantined records are held back and
written to an audit log with the detector that flagged them. It never reads `is_poisoned`.

| Part | What it does |
|---|---|
| Robust z-score | Per-feature median/MAD against a rolling reference of accepted rows **of the same label** (a real fraud is not an outlier among frauds). |
| Label consistency | A class-balanced logistic regression on accepted rows; a row is suspicious if the model finds its given label very unlikely. |
| Mahalanobis | Class-conditional, shrunk covariance; catches rows whose features are individually normal but jointly odd. Binary columns are dropped. |
| Ensemble | Each detector's score is divided by its own threshold; a row is quarantined when the maximum exceeds 1. |
| Fraud-label alarm | A Poisson test on the number of fraud labels in the window. While it fires, the fraud-class reference is frozen and fraud-labelled rows are judged 1.5x more strictly. |
| Mass-flag guard | If the feature detectors alone flag more than 15% of a window's legit-labelled rows, those rows are accepted (a stale reference, not an attack). |
| Trusted warm-up | Window 0 is accepted unchecked and only builds the references. |

All thresholds are in the `defense:` section of `config/default.yaml`.

**Intended use.** Studying how label-flip, feature-poisoning and backdoor attacks degrade a streaming classifier
and how far reference-based quarantine recovers it, on public or synthetic data. **Not** intended for screening real
customers' transactions, and not validated for any other dataset.

## 2. Data

- **PaySim** (Kaggle, synthetic mobile-money transactions simulated from aggregated logs of a real service). 6.36M
  rows over 743 hourly steps, 11 columns, fraud rate about 0.13%. No real individuals appear; `nameOrig` and `nameDest` are
  generated identifiers.
- **Sample used.** A fixed 10% random sample (`sample_frac 0.1`). Stream = steps below 600 (625,883 rows, 653 frauds);
  clean held-out test = steps 600 and later. The attacks never touch the test set or window 0.
- **Windows.** 24 steps (a simulated day). Volumes are very uneven (from about 90 to 57,000 rows per window), and a
  window holds about 26 frauds, which is why fraud-class references start small.
- **Attacks (P2).** `random_flip`, `targeted_flip`, `feature_shift`, `feature_scale`, `feature_noise`, `backdoor`; budgets 0.05 and
  0.2; timings uniform, burst, low-and-slow. Label attacks and feature attacks poison 28k to 114k records
  (random flip) or 32 to 126 records (everything else, which tampers with fraud rows only).
- **ULB credit-card data** is configured as a cross-check dataset but was **not** used for these results.

## 3. Results (3 attack seeds, same clean sample)

PR-AUC on the clean test set, mean over seeds. Without an attack the undefended learner scores 0.998 and the defended one
0.987, so the defense costs about 0.011 when nothing is wrong.

| Attack (budget 0.2) | No defense | Defended |
|---|---|---|
| random_flip, uniform | 0.22 | 0.96 |
| random_flip, low-and-slow | 0.02 | 0.97 |
| feature_noise, uniform | 0.42 | 0.99 |
| feature_noise, low-and-slow | 0.64 | 0.99 |
| all other attack/timing pairs | 0.99 to 1.00 | 0.99 |

Detection recall (poisoned records quarantined): random_flip 0.94, targeted_flip 0.99, feature_shift 1.00,
feature_noise 0.86, backdoor 0.63, feature_scale 0.04. Precision is low (0.17 to 0.24) for attacks that touch only
32 to 126 fraud rows: the defense also quarantines about 0.04% of clean records and 4.5% of real frauds.

## 4. Fairness and clean-record loss

Share of **clean** records quarantined, mean over 108 attacked files: CASH_IN 0.11%, TRANSFER 0.05%, DEBIT 0.03%,
PAYMENT 0.02%, CASH_OUT 0.01%; by amount quartile 0.03% to 0.04%. No group loses more than 0.2%. The group that
loses most is **real fraud**: on average 4.5% of true frauds (range 1.5 to 8%) are quarantined because their reference is
small; this rises to 17% if the fraud-label alarm is switched off. We did not investigate why CASH_IN is the most
affected type.

## 5. Limitations and risks

- Single clean sample: the three seeds vary the attack draw and the learner, not the data. Spread is understated.
- Thresholds were tuned on seed-0 files of this sample, so results are optimistic. No held-out dataset was run.
- `feature_scale` (recall 0.04) and the backdoor (0.63) are weak spots. The attacks configured here barely hurt the
  learner at these budgets, so the exposure is small, but a stronger trigger or budget may not be.
- The defense assumes window 0 is clean. An attacker who poisons it defeats the references.
- Quarantined records are held, not deleted, but there is no human review loop (`review_windows` is off); a legitimate
  customer whose record is quarantined is simply not learned from.
- Mass-flag guard trade-off: feature-only tampering with more than 15% of a window's legit-labelled rows would pass.
- PaySim is easy to separate (clean PR-AUC 0.998) and synthetic; absolute numbers will not transfer to real fraud data.
- An adaptive attacker who knows the thresholds (low-and-slow below the alarm, rows near but under the z-limit) was not
  tested.

## 6. Reproduce

```
python scripts/extend_attacked_samples.py --seeds 0 1 2        # from data/attacked_sample/stream_clean.csv.gz
python -m src.attacks.sweep --clean-dir data/attacked_sample   # results/metrics.csv
bash scripts/run_multiseed.sh && bash scripts/run_ablation.sh
python scripts/collect_results.py && python scripts/build_notebook.py
```
