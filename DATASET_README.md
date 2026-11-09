# Tony's dataset preparation

## What to use

Get data from Tony / Olivia; exports are excluded from GitHub.

| File | Purpose |
|---|---|
| `data/cleaned/dataset_splits.csv` (or both cleaned Parquet parts) | Structured EHR + full discharge notes |
| `data/concept_masked/dataset_splits_concept_mask.csv` (or both filtered Parquet parts) | Same admissions, columns, labels and splits; concept-filtered notes |

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

Generate the concept-filtered CSV, matching typed Parquet parts, and small reports:

```powershell
python generate_concept_dataset.py --workers 4
```

This reads the cleaned CSV and Parquet parts and writes `data/concept_masked/`.
Parquet preserves the original nested structures, types, and part row order.
Existing output folders are protected; choose a new `--out` folder to rerun.

## Masking policy

Concept policy **v2** removes whole measurement mentions (name, value, captured
unit) when the concept has a nonmissing structured feature, even if values/times
differ. Preceding timestamps are retained. If only one BP component is available,
only that number is removed. It also removes
agreeing patient sex-header values and literal diagnosis terms when no exclusion
cue is detected. Basic context rules retain negated, historical, hypothetical,
other-person and ambiguous mentions. `keep` and `review` both retain text.

Each accepted span is replaced with **one space**, not `[MASK]`. Overlapping spans
are merged. Separators such as `/` are removed only between two removed spans;
all other surrounding text, existing whitespace and non-note cells remain
unchanged. This is an information-removal experiment, not proof of redundancy.
Context rules are heuristics; sample review remains necessary.

**Full verified-event masking is deferred.** Its data processing/export workflow
is messy and too large for the current setup (~25.8 million source events).
Keep `audit_redundancy_verified.py` and the small sample evidence for later; there
is currently no `dataset_splits_verified_mask.csv`.

## Validate and review

```powershell
python check_admission_keys.py data/concept_masked/dataset_splits_concept_mask.csv
python summarize_dataset.py data/concept_masked/dataset_splits_concept_mask.csv --output validation_output/concept_masked_csv
python summarize_dataset.py "data/concept_masked/dataset_splits_concept_mask_*.parquet" --output validation_output/concept_masked_parquet
python -m unittest discover -v
```

Key checks expect 52,027 rows, unique admission IDs/patient-admission pairs, and no
patient crossing splits. They do not replace source note/label linkage checks.
Dataset summaries report missingness, mortality prevalence, and note lengths
overall/by split. In `note_lengths.csv`, `min/mean/median/p95/max` count whitespace
tokens; `characters_*` count Unicode characters including whitespace. Cleaned
baseline reports are in `validation_output/cleaned_csv/` and `cleaned_parquet/`.
Model tokenization belongs to the training pipeline.

`data/concept_masked/` also contains `category_summary.csv`, `note_summary.csv`
(including mask counts and before/after token and character lengths), and `summary.json`
with policy, source checksum and run counts. Original files remain unchanged.

Local folders are kept to `cleaned/`, `concept_masked/`, `audit_samples/` (small
concept/verified examples and sample events), and `linkage_audit/` (source-check
evidence). Share the filtered CSV through the team's authorized Google Drive
location; audit files are not required for model training.
