# Tony's dataset preparation and validation

## Data and setup

Get exports from Tony / Olivia; datasets are not in GitHub. Run from the project
folder with Python 3.10+ and `python -m pip install pandas pyarrow`.

The cleaned dataset has **52,027 admissions**. One admission with discharge before
admission was excluded; originals and splits are preserved. The 12 admissions
without source diagnoses remain. Patients can repeat across admissions, but must
stay in one split. Measurement features summarize the latest ICU stay per admission.
The documented cohort requires notes stored by discharge +3 days and excludes
recorded deaths through that calendar-date cutoff; the outcome ends at discharge +30 days.

## Get cleaned exports

**Already cleaned:** put the CSV and/or both Parquet parts in `data/cleaned/`.
Skip cleaning and use these files for training. Both Parquet parts form one dataset;
CSV is an alternative export, not additional admissions.

**Original exports:** place `dataset_splits.csv`,
`dataset_splits_000000000000.parquet`, and `dataset_splits_000000000001.parquet`
in the project root, then run:

```powershell
python exclude_invalid_admission.py
```

Cleaning requires both formats. It writes `data/cleaned/` and an exclusion manifest.
It refuses to overwrite an existing folder; choose `--output data/cleaned_v2` if needed.

## Validate and summarize

Run the pair for your format, or both pairs separately:

```powershell
python check_admission_keys.py data/cleaned/dataset_splits.csv
python summarize_dataset.py data/cleaned/dataset_splits.csv --output validation_output/cleaned_csv

python check_admission_keys.py "data/cleaned/dataset_splits_*.parquet"
python summarize_dataset.py "data/cleaned/dataset_splits_*.parquet" --output validation_output/cleaned_parquet
```

Expect 52,027 rows, unique admission IDs/patient-admission pairs, no missing keys
or splits, and no patients crossing splits. Source note/label linkage needs separate checks.

Reports cover the whole dataset and each split:

| Report | Contents |
|---|---|
| `feature_missingness.csv` | Missing columns and individual labs/vitals; empty arrays reported separately |
| `mortality_prevalence.csv` | Positive/negative/missing/invalid labels; prevalence uses valid labels only |
| `note_lengths.csv` | Whitespace-token min/mean/median/p95/max; missing notes counted separately |

Nulls, blank text and `NULL`/`null` are missing. Measurements are missing if absent,
flagged missing, or all numeric summaries are null. Whitespace counts describe
length; training uses the chosen model's tokenizer. Inputs are never changed.

## Two draft masking policies

Both scripts reuse Olivia's `audit_redundancy.py` extractor and default to the same
75 training notes in `redundancy_audit_v3/sample.jsonl`. They propose different masks:

| Script | Numerical values | Diagnoses | Sex headers |
|---|---|---|---|
| `audit_redundancy_verified.py` | Only strict redundant matches with unique raw-event and last-feature evidence | Keep | Mask agreeing patient header values |
| `audit_redundancy_concept.py` | Mask when the concept has a nonmissing structured counterpart, even if values/times differ | Mask literal description hits when no exclusion cue is detected | Mask agreeing patient header values |

Both retain mentions flagged as negated, historical, hypothetical, about another
person, or ambiguous. Context rules are basic heuristics requiring sample review.
"Verified" means supported by the script's evidence rules, not clinical certainty.
The broader policy tests information removal; it does not prove the removed values
are duplicates. It was previously named `audit_redundancy_improved.py`.

```powershell
python audit_redundancy_verified.py --events data/source_events.csv --out data/audit_verified
python audit_redundancy_concept.py --events data/source_events.csv --out data/audit_concept
```

`--events` supplies source evidence. Without it, verified numerical masks are
unavailable; concept decisions are unchanged. Use `--sample` for another JSONL
sample and matching event coverage. Output folders must be new; change `--out`
to rerun. Event-export instructions are in `data/linkage_audit/EVENTS_README.md`
when that local audit bundle is available.

Each run writes `mentions.csv` and `summary.json`. Read `mask_decision`,
`decision_reason`, `mask_span`, and `mask_preview`; compare `concept_overlap` with
the unchanged strict `suggested_label` and `event_linkage`. `context_marked` highlights
extracted mentions. `mask_start`/`mask_end` are original-note character offsets,
with an exclusive end. `keep` and `review` retain text; human-review fields stay blank.

**These are sample proposals and previews, not filtered full notes or full-cohort
statistics.** Review masks and retained mentions before scaling. A separate generator
must merge overlapping spans and apply the agreed replacement consistently.
Original notes and earlier audit outputs remain unchanged.

CSV/Parquet files and `data/` are ignored by Git. Share exports through the team's
authorized location. Run tests with `python -m unittest discover -v`.
