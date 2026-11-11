# EHR-only XGBoost results

This ablation predicts 30-day post-discharge mortality from the same 4,041
structured EHR features, cohort, patient-separated splits, XGBoost candidate
grid, class weighting, early-stopping budget, and validation-only selection
rules used by the FullNote fusion experiments. It contains no note-derived
features.

## Selected model

- Input width: 4,041 EHR features
- Selected candidate: `max_depth=5`, `min_child_weight=1`
- Boosting rounds: 98
- Validation-selected F1 threshold: 0.727023
- Test admissions / deaths: 6,782 / 321

## Performance

| Metric | Validation | Test |
|---|---:|---:|
| AUROC | 0.8742 | 0.8633 |
| AUPRC | 0.4040 | 0.3412 |
| Accuracy | 0.9472 | 0.9404 |
| Precision | 0.4409 | 0.3739 |
| Recall/sensitivity | 0.4207 | 0.3832 |
| Specificity | 0.9734 | 0.9681 |
| F1 | 0.4306 | 0.3785 |
| Brier score | 0.1130 | 0.1159 |
| Log loss | 0.3697 | 0.3768 |

Thresholded metrics use the validation-selected cutoff. AUROC, AUPRC, Brier
score, and log loss use continuous scores and do not depend on that cutoff.

## Test confusion matrix

| | Predicted survivor | Predicted death |
|---|---:|---:|
| Actual survivor | 6,255 | 206 |
| Actual death | 198 | 123 |

## Effect of adding the full note embedding

| Test metric | EHR only | EHR + FullNote | FullNote change |
|---|---:|---:|---:|
| AUROC | 0.8633 | 0.8890 | +0.0256 |
| AUPRC | 0.3412 | 0.3771 | +0.0359 |
| Accuracy | 0.9404 | 0.9518 | +0.0114 |
| Precision | 0.3739 | 0.4856 | +0.1117 |
| Recall/sensitivity | 0.3832 | 0.3146 | -0.0685 |
| Specificity | 0.9681 | 0.9834 | +0.0153 |
| F1 | 0.3785 | 0.3819 | +0.0034 |
| Brier score | 0.1159 | 0.0381 | -0.0778 |
| Log loss | 0.3768 | 0.1689 | -0.2079 |

Adding the complete note improves threshold-independent discrimination: AUROC
rises by 0.0256 and AUPRC by 0.0359. It also improves precision, specificity,
and probability error substantially. At the separately validation-selected
operating points, direct fusion has lower recall; this is a threshold trade-off,
not evidence that its ranking is worse. Thresholds should be compared at fixed
recall or fixed false-positive rate before choosing a deployment policy.

The EHR-only model uses `scale_pos_weight=18.8824`, so its raw values should not
be interpreted as calibrated mortality probabilities without calibration.
