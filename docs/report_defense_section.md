# Defense, detection and analysis (P3 section of the report)

*Draft, about 1.5 pages. Figures are in `notebooks/defense_results.ipynb`; numbers are means over 3 attack seeds.*

## 1. Defense design

The learner is an online logistic regression that trains on each 24-step window of PaySim as it arrives. The defense
sits in front of it and decides, record by record, whether the learner may train on it. It works only from what a
real operator would have: the features, the (possibly poisoned) label, and records it accepted earlier. It never sees
the ground-truth `is_poisoned` column; that is read only by the evaluator.

Three detectors each give a record a suspicion score, and a record is quarantined when the largest score, each divided
by its own threshold, exceeds 1:

- **Robust z-score** compares each feature with the median and MAD of a rolling reference of accepted rows *with the
  same label*, so a genuine fraud is not an outlier merely for being unlike the legitimate majority.
- **Label consistency** fits a class-balanced logistic model to accepted rows and flags records whose given label it
  finds very unlikely (loss above 4 nats). This is the detector that catches flipped labels.
- **Mahalanobis distance** (class-conditional, shrunk covariance) catches records whose features are individually
  ordinary but jointly odd.

Two guards address failures we found while testing on real data. A **fraud-label alarm** runs a Poisson test on the
number of fraud labels per window. A label-flip flood multiplies that count by 50 to 100, and a reference built from
accepted rows would then be dominated by flipped records and treat the real frauds as the outliers; while the alarm is
on, the fraud-class reference is frozen and fraud-labelled records are judged 1.5 times more strictly. A **mass-flag
guard** accepts a window's legit-labelled records when the feature detectors alone flag more than 15% of them: poison is
a small share of such rows, so a mass flag means the reference is stale. Without it, one `random_flip` burst file (seed 1)
had 8.5% of clean records quarantined for the rest of the stream, because quarantined rows never enter the reference and
it could not recover. Window 0 is trusted and only builds the references. Quarantined records are held, not deleted, and
logged with the detector that flagged them.

## 2. Does it work?

**Effect on the learner.** We replayed 6 attacks at budgets 0.05 and 0.2, with uniform, burst and low-and-slow timing,
for 3 seeds, with and without the defense (222 runs) and scored PR-AUC on a clean test set. With no attack, the
undefended learner scores 0.998 and the defended one 0.987. Two kinds of attack break the undefended learner. Flipping
labels spread over time (uniform or low-and-slow) at budget 0.2 drives PR-AUC to 0.22 and 0.02. `feature_noise`
drives it to 0.42 (uniform) and 0.64 (low-and-slow). Behind the defense these cases score 0.96, 0.97, 0.99 and 0.99.
The other attack and timing pairs barely move the undefended learner (0.99 or more), so there is little to protect, and the
defense costs about 0.01 PR-AUC there. A burst of flipped labels is harmless in this setup because the learner forgets it before
the test.

**Detection.** Recall (poisoned records quarantined) is 1.00 for `feature_shift`, 0.99 for `targeted_flip`, 0.94 for
`random_flip`, 0.86 for `feature_noise` and 0.63 for the backdoor. It is 0.04 for `feature_scale`: multiplying amount
and balances by 5 preserves the balance identity, so the records look like large legitimate transactions. Precision is
low (0.17 to 0.24) for attacks that tamper with only 32 to 126 fraud rows, because the defense also quarantines about
0.04% of clean records and 4.5% of real frauds. Time to detect averages under one window for everything except `feature_scale`.

**Ablation (budget 0.2).** No detector is enough alone. Label consistency carries label attacks and `feature_shift`
(recall 0.92 to 1.00) but sees only 0.38 of `feature_noise`; the robust z-score sees 0.60 of it but almost nothing else;
Mahalanobis alone is weak (0 to 0.22) but lifts `feature_noise` from 0.73 to 0.84 when added to the other two.
Removing the fraud-label alarm leaves `random_flip` recall at 0.76 instead of 0.94 and raises the share of real frauds
quarantined from 4% to 17%.

## 3. Cost and fairness

Replaying the whole 625,883-row stream through the defense takes about 2.6 seconds on 2 CPU cores (not counting file loading), so it is not the
bottleneck. The defense quarantines 0.04% of clean records on average (worst file 0.06%). By transaction type the
rate is 0.11% for CASH_IN and 0.01 to 0.05% for the rest; by amount quartile 0.03 to 0.04%. We did not investigate why
CASH_IN is highest. The group that pays most is the real fraud class: 4.5% of true frauds are quarantined on average
(1.5 to 8% per file) because their reference starts from about 24 records. That is a direct cost to the thing the
classifier exists to find, and the reason the defended learner is 0.011 below the undefended one on clean data.

## 4. Limitations

- Three seeds vary the attack and the learner but use one clean PaySim sample, so the spread is understated.
- Thresholds were tuned on seed-0 files of the same sample; results are optimistic. No second dataset was run (ULB is
  configured but unused).
- `feature_scale` (0.04) and the backdoor (0.63) are weak spots; these attacks barely hurt the learner here, but a stronger
  setting might.
- Window 0 is assumed clean; poisoning it defeats the references.
- The mass-flag guard would let through feature-only tampering with more than 15% of a window's legit-labelled rows.
- Nobody reviews quarantined records, so a wrongly quarantined legitimate customer is just not learned from.
- No adaptive attacker was tested (one who stays under the alarm and the thresholds on purpose).
