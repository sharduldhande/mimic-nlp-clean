# Tony's dataset preparation and validation

## What changed

Tony excluded admission `(subject_id=13659453, hadm_id=23252384)` because its
discharge precedes admission in the raw MIMIC admissions table. The dataset
decreased from **52,028 to 52,027 admissions** in both CSV and combined Parquet
exports. Originals, patient split assignments, and other admissions are unchanged.

The **12 admissions without source diagnosis records are retained** as missing
data. Repeated patients are allowed: the SQL selects the latest ICU stay **per
admission**, not the latest admission per patient.

## Setup

Get the data from Tony / Olivia; it is not included in GitHub. Run commands from
the project folder using Python 3.10+:

```powershell
python -m pip install pandas pyarrow
```

## If you have the original exports

Place these three files in the project folder:

- `dataset_splits.csv`
- `dataset_splits_000000000000.parquet`
- `dataset_splits_000000000001.parquet`

The cleaning script requires both formats and checks that each export loses
exactly one admission:

```powershell
python exclude_invalid_admission.py
```

It writes cleaned copies to `data/cleaned/`, plus `exclusion_manifest.json`
recording the exclusion and row counts. It refuses to overwrite an existing
output folder; use `--output data/cleaned_v2` for a new destination if needed.

## If you already have cleaned exports

Skip the cleaning script. Put the cleaned CSV and/or both cleaned Parquet parts
in `data/cleaned/`, keeping their original filenames. Use these files for training.
Both Parquet parts together form one dataset; CSV is an alternative export.

## Validate and inspect the cleaned dataset

Run the commands for whichever format you have. If you have both, check them
separately:

```powershell
# CSV
python check_admission_keys.py data/cleaned/dataset_splits.csv
python summarize_dataset.py data/cleaned/dataset_splits.csv --output validation_output/cleaned_csv

# Parquet: combine both parts for each check
python check_admission_keys.py "data/cleaned/dataset_splits_*.parquet"
python summarize_dataset.py "data/cleaned/dataset_splits_*.parquet" --output validation_output/cleaned_parquet
```

Expect **52,027 rows**, unique admission IDs and patient/admission pairs, no missing
keys or splits, and no patient in multiple splits. These checks do not independently
verify source note/label linkage or complete the full validation checklist.

Each summary folder contains statistics overall and by split:

| Report | Contents |
|---|---|
| `feature_missingness.csv` | Missing counts/percentages, including individual labs and vitals |
| `mortality_prevalence.csv` | Positive, negative, missing and invalid labels; prevalence uses valid labels only |
| `note_lengths.csv` | Minimum, mean, median, 95th percentile and maximum whitespace token counts |

Missing values include nulls, empty/whitespace text, and `NULL`/`null`. Empty
diagnosis/measurement arrays are reported separately. A measurement is missing
when its feature object is absent, its missingness flag is set, or all four numeric
summaries are null. Missing notes are counted separately and excluded from length
statistics. Whitespace counts describe text length; model tokenization belongs to
the training pipeline.

Checks and summaries leave inputs unchanged. To inspect the originals, pass their
paths instead and choose a different summary output folder. CSV/Parquet files and
`data/` are ignored by Git; share the cleaned exports through the team's authorized
data-sharing location.
