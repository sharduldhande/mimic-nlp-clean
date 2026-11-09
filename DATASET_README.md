# Tony's dataset preparation

## What to use

Get data from Tony / Olivia; exports are excluded from GitHub.

| File | Purpose |
|---|---|
| `data/cleaned/dataset_splits.csv` (or both cleaned Parquet parts) | Structured EHR + full discharge notes |
| `data/concept_masked/dataset_splits_concept_mask.csv` | Same admissions, columns, labels and splits; concept-filtered notes |

Both datasets contain **52,027 admissions** across train/validation/test. Use the
same existing splits for every model. Both Parquet parts together represent one
dataset; do not append the CSV as additional records.

One source admission with discharge before admission was excluded. The 12
admissions without diagnosis records remain. Patients can have multiple admissions
but stay in one split. Measurements summarize the latest ICU stay per admission.
Notes must be stored by discharge +3 days; recorded deaths through that cutoff
are excluded. The mortality outcome ends at discharge +30 days.

## Reproduce

Python 3.10+; install dependencies with `python -m pip install pandas pyarrow`.
Run commands from the project root.

Already have cleaned exports? Skip cleaning. Otherwise put the original CSV and
both `dataset_splits_*.parquet` parts in the root, then run:

```powershell
python exclude_invalid_admission.py
```

Generate the concept-filtered CSV and small reports without a large mentions CSV:

```powershell
python generate_concept_dataset.py --workers 4
```

This reads `data/cleaned/dataset_splits.csv` and writes `data/concept_masked/`.
Existing output folders are protected; choose a new `--out` folder to rerun.

## Masking policy

The concept policy proposes removing numerical values when the same concept has
a nonmissing structured feature, even if values/times differ. It also removes
agreeing patient sex-header values and literal diagnosis terms when no exclusion
cue is detected. Basic context rules retain negated, historical, hypothetical,
other-person and ambiguous mentions. `keep` and `review` both retain text.

Each accepted span is replaced with **one space**, not `[MASK]`. Overlapping spans
are merged; all surrounding text, existing whitespace and non-note cells remain
unchanged. This is an information-removal experiment, not proof of redundancy.
Context rules are heuristics; sample review remains necessary.

**Full verified-event masking is deferred.** Its data processing/export workflow
is messy and too large for the current setup (~25.8 million source events).
Keep `audit_redundancy_verified.py` and the small sample evidence for later; there
is currently no `dataset_splits_verified_mask.csv`.

## Validate and review

```powershell
python check_admission_keys.py data/concept_masked/dataset_splits_concept_mask.csv
python summarize_dataset.py data/concept_masked/dataset_splits_concept_mask.csv --output validation_output/concept_masked
python -m unittest discover -v
```

Key checks expect 52,027 rows, unique admission IDs/patient-admission pairs, and no
patient crossing splits. They do not replace source note/label linkage checks.
Dataset summaries report feature missingness, mortality prevalence and whitespace
note lengths overall/by split; model tokenization belongs to the training pipeline.

`data/concept_masked/` also contains `category_summary.csv`, `note_summary.csv`
(including mask counts and before/after whitespace lengths), and `summary.json`
with policy, source checksum and run counts. Original files remain unchanged.

Local folders are kept to `cleaned/`, `concept_masked/`, `audit_samples/` (small
concept/verified examples and sample events), and `linkage_audit/` (source-check
evidence). Share the filtered CSV through the team's authorized Google Drive
location; audit files are not required for model training.
