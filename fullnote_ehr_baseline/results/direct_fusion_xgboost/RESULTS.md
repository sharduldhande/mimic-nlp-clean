# Direct EHR + FullNote XGBoost results

The model predicts 30-day post-discharge mortality. Hyperparameters and the
classification threshold were selected on validation data. The test split was
used only for the final evaluation.

## Test cohort

- Admissions: 6,782
- Deaths within 30 days: 321
- Prevalence: 4.73%
- Validation-selected threshold: 0.261359

## Test results

| Metric | Result | Interpretation |
|---|---:|---|
| AUROC | 0.8890 | A randomly selected death receives a higher score than a randomly selected survivor about 88.9% of the time. |
| AUPRC | 0.3771 | Precision-recall performance is about 8.0 times the 4.73% no-skill prevalence baseline. |
| Accuracy | 0.9518 | 95.18% of admissions were classified correctly, but this is not sufficient by itself because an all-survivor classifier obtains 95.27%. |
| Precision | 0.4856 | Of 208 admissions classified as high risk, 101 died within 30 days. |
| Recall/sensitivity | 0.3146 | The model identified 101 of 321 deaths and missed 220. |
| Specificity | 0.9834 | The model correctly classified 6,354 of 6,461 survivors; the false-positive rate was 1.66%. |
| F1 | 0.3819 | Harmonic mean of precision and recall; it reflects useful precision but limited recall. |
| Brier score | 0.0381 | Mean squared probability error. Zero is perfect; this is 15.5% better than predicting the cohort prevalence for everyone. |
| Log loss | 0.1689 | Probability error that strongly penalizes confident mistakes. This is 11.4% lower than the constant-prevalence baseline. |

## Confusion matrix

| | Predicted survivor | Predicted death |
|---|---:|---:|
| Actual survivor | 6,354 | 107 |
| Actual death | 220 | 101 |

At the conventional threshold 0.5, accuracy increases to 95.59% and precision to
58.73%, but recall falls to 23.05% and F1 falls to 0.3311. The selected threshold
therefore trades a small amount of accuracy and specificity for better detection
of deaths.

## Saved figures

- [Metric comparison](plots/metric-comparison.svg)
- [ROC curve](plots/roc-curve.svg)
- [Precision-recall curve](plots/precision-recall-curve.svg)
- [Probability losses](plots/probability-losses.svg)
- [Confusion matrices](plots/confusion-matrices.svg)
- [Validation threshold tradeoffs](plots/threshold-tradeoffs.svg)
- [Calibration curve](plots/calibration-curve.svg)
