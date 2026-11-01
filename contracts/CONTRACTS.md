# Contracts

## 1. Record schema
One row = one transaction. Required columns:
`record_id` (int, unique, event-time order), `step` (int, event time), `isFraud` (0/1 label),
plus the raw dataset columns. The pipeline adds `is_poisoned` (bool, ground truth for
evaluation only; learners and defenses must NEVER read it).

## 2. Attack signature (P2)
`attack(df, budget: float, timing: str, seed: int) -> (df_poisoned, poison_mask)`
- `df`: clean stream, event-time order
- `timing`: "uniform" | "burst" | "low_and_slow"
- `poison_mask`: numpy bool array, same length and order as `df_poisoned`; True = record was tampered with
- `df_poisoned` keeps `record_id`, `step`; row order unchanged

## 3. Defense signature (P3)
`defend(window_df, state: dict) -> (accepted_df, quarantined_df, state)`
- `window_df`: all records of one event-time window, no `is_poisoned` column
- `accepted_df` goes to the learner; `quarantined_df` is held back
- every row of `window_df` appears in exactly one of the two outputs
- `state` carries detector statistics between windows (start as `{}`)
- entry point: `src/defense/pipeline.py` function `defend`

## 4. Config
All settings come from `config/default.yaml`. No hard-coded constants in code.

## 5. Metrics CSV (`results/metrics.csv`, one row per window per run)
`run_id, dataset, attack, budget, timing, defense, seed, window, step_end, n_seen,
n_quarantined, pr_auc, recall, f1, det_precision, det_recall`
- `pr_auc, recall, f1`: on the clean held-out test set after learning that window
- `det_precision, det_recall`: quarantine vs `is_poisoned` for that window (blank if defense off or no poison)