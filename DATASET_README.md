# Dataset preparation and outputs

Follow the [three-step setup](README.md). The only input is the **one-day**
BigQuery export at `data/cleaned/dataset_splits.csv`. The alternate filename
`datasets_splits.csv` is accepted when only that file exists. Do not put both there.

`python prepare_datasets.py` generates and verifies everything before publishing
outputs. Wait for **DONE**. Reruns replace generated folders/files, preserve the
input CSV, and retain previous outputs if processing fails. Optional `--workers 2`
reduces parallelism. Pandas and PyArrow are the only required dependencies.

## Files to use

| Location | Contents |
|---|---|
| `data/cleaned/` | Input full-note CSV and two typed Parquet parts |
| `data/concept_masked/` | Filtered CSV, two matching Parquet parts, category/per-note summaries and `summary.json` |
| `data/concept_masked/review_examples/` | 10 random training notes before/after; open `before_after.html` or `.md` |
| `data/audit_samples/concept/` | Up to 75 current-cohort training notes, mention audit and summaries |
| `validation_output/cleaned_csv/`, `cleaned_parquet/` | Full-note validation summaries |
| `validation_output/concept_masked_csv/`, `concept_masked_parquet/` | Filtered-note validation summaries |
| `validation_output/pipeline_validation.json` | Actual cohort counts, input checksum and local verification results |

The filtered dataset is `dataset_splits_concept_mask.csv`. Both Parquet parts
form one dataset; CSV is an alternative format, not additional admissions. Use
identical existing splits for model comparisons. No full-cohort mentions CSV is
created. Only the small training sample includes mention-level output.

## Cohort

The one-day cohort previously checked had **45,482 admissions**, **2,248 positive
labels**, and **4.9426% mortality**; consult the generated report for actual counts.
The [extraction SQL](dataset_splits_one_shot.sql) selects the latest ICU stay per
admission, excludes invalid hospital times, and requires notes stored by discharge
+24 hours. It excludes recorded deaths through the discharge +1-day calendar date;
the target ends at discharge +30 days. Missing death dates count as negative under
the original convention. Patients may have multiple admissions, all in one split.
Missing diagnoses are retained. Measurements summarize ten selected ICU concepts.

The local pipeline checks identifiers, patient split separation, labels, hospital
timestamps and the note deadline. It verifies CSV/Parquet agreement for every
column and that filtering changes only note text. It cannot independently verify
raw-source note ownership, death dates or mortality calculations from this CSV.

## Concept filtering

Policy **v2** removes whole measurement mentions (name, value, captured unit) when
a nonmissing structured counterpart exists, even if values/times differ. It retains
preceding timestamps. With only one BP component available, only that number is
removed. Agreeing sex-header values and qualifying literal diagnosis terms are
also removed. Context heuristics retain negated, historical, hypothetical,
other-person and ambiguous mentions; `keep` and `review` both retain text.

Each span becomes one space. Overlaps and separators between removed mentions
are merged; other text/whitespace and every non-note value stay unchanged. These
are experimental proposals, not proof of redundancy; manual sample review remains
necessary. **Full verified-event filtering is deferred** because its raw-event
processing/export workflow is messy and large. No verified-filtered dataset is built.

## Reading validation reports

Each validation folder has `feature_missingness.csv`, `mortality_prevalence.csv`
and `note_lengths.csv`, overall and by split. Mortality prevalence excludes invalid
labels. Note-length `min/mean/median/p95/max` count whitespace tokens;
`characters_*` count Unicode characters including whitespace. Model tokenization
belongs to the training pipeline. Source exports and all generated data are ignored
by Git; share through the team's authorized location. Tests: `python -m unittest discover -q`.
