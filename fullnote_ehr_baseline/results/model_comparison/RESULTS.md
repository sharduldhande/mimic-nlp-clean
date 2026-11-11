# Notes ablation comparison

All three models use the same patient-separated train, validation, and test
cohorts. Hyperparameters are selected by validation AUPRC and each binary
classification threshold is selected by validation F1. The held-out test split
does not select models or thresholds.

| Test metric | EHR only | EHR + FullNote | EHR + note risk |
|---|---:|---:|---:|
| AUROC | 0.8633 | **0.8890** | 0.8740 |
| AUPRC | 0.3412 | **0.3771** | 0.3531 |
| Accuracy | 0.9404 | **0.9518** | 0.9313 |
| Precision | 0.3739 | **0.4856** | 0.3414 |
| Recall/sensitivity | 0.3832 | 0.3146 | **0.4860** |
| Specificity | 0.9681 | **0.9834** | 0.9534 |
| F1 | 0.3785 | 0.3819 | **0.4010** |
| Brier score | 0.1159 | **0.0381** | 0.1170 |
| Log loss | 0.3768 | **0.1689** | 0.3699 |

The ablation supports the conclusion that discharge notes add predictive signal.
Direct fusion has the best discrimination, precision, specificity, and
probability error. Risk-score fusion obtains the highest recall and F1 at its
selected operating point. Because the three thresholds differ, recall and other
thresholded metrics describe operating choices as well as model quality; AUROC
and AUPRC provide the cleanest threshold-independent note ablation.

Generated figures:

- `test-metric-comparison.svg`
- `note-contribution-deltas.svg`
- `probability-loss-comparison.svg`
