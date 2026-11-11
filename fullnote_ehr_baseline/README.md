# FullNote + EHR mortality pipeline

This is a fresh implementation for the two requested 30-day mortality experiments:

1. **Direct fusion:** structured EHR features concatenated with one 768-dimensional
   full-note embedding, followed by XGBoost.
2. **Risk-score fusion:** a classifier converts the same full-note embedding into a
   single out-of-fold mortality risk score; that score is concatenated with the
   structured EHR features and passed to XGBoost.

The current implementation contains the shared foundation, embedding stage, and
both requested fusion experiments:

- Step 1: dataset discovery, loading, and full contract validation.
- Step 2: train-only fitting and reusable transformation of structured EHR data.
- Step 3: frozen BioClinicalBERT long-note embedding with a reusable cache.
- Step 4: identity-checked alignment of EHR rows and FullNote embeddings.
- Step 5: direct-concatenation XGBoost training, selection, and evaluation.
- Step 6: leakage-safe note-risk scoring followed by EHR + score XGBoost.

It intentionally does not import code from the existing `7650_project` folder.

## Repository layout

```text
fullnote_ehr_baseline/
  requirements-step1-2.txt            Dependencies for the implemented code
  requirements-modeling.txt           Additional future modeling dependencies
  src/fullnote_ehr/
    schema.py                          Dataset schema and feature constants
    data.py                            Streaming validation and selective loading
    ehr_features.py                    Fitted structured-EHR transformer
    embeddings.py                      Chunking, pooling, resume, artifact loading
    fusion.py                          Identity alignment and direct concatenation
    direct_fusion.py                   XGBoost training, metrics, and prediction
    risk_score_fusion.py               Grouped OOF scores and risk-fusion model
    cli/
      validate_dataset.py              Dataset-contract command
      build_ehr_features.py            Fit on train and save three matrices
      transform_ehr.py                 Apply a saved builder to new rows/splits
      embed_full_notes.py              Generate the shared FullNote embedding cache
      train_direct_fusion.py           Train/evaluate direct EHR + note fusion
      predict_direct_fusion.py         Re-run a saved model on a prepared split
      train_risk_score_fusion.py       Train/evaluate EHR + note-risk fusion
      predict_risk_score_fusion.py     Re-run the saved risk-score pipeline
  tests/                               Synthetic contract and feature tests
```

Generated files belong under `artifacts/`, which is ignored by Git.

## Accepted dataset

The default input is the two-part current cohort in `../data_cleaned/`:

```text
../data_cleaned/dataset_splits_000000000000.parquet
../data_cleaned/dataset_splits_000000000001.parquet
```

The contract requires:

| Property | Expected value |
|---|---:|
| Admissions | 45,482 |
| Patients | 36,913 |
| Positive labels | 2,248 |
| Train rows / positives | 31,792 / 1,599 |
| Validation rows / positives | 6,908 / 328 |
| Test rows / positives | 6,782 / 321 |

It also checks the exact Arrow schema, unique admissions and notes, patient-level
split separation, binary labels, admission/note timestamps, nonempty notes,
diagnosis counts, and all ten measurement concepts.

## Environment setup

Use Python 3.10, 3.11, or 3.12. From this folder on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-step1-2.txt
python -m pip install -e . --no-deps
```

For later embedding and XGBoost work, install the modeling requirements instead:

```powershell
python -m pip install -r requirements-modeling.txt
python -m pip install -e . --no-deps
```

The modeling file adds PyTorch, Transformers, safetensors, and XGBoost. The first
embedding run downloads BioClinicalBERT unless a local model path is passed or
`--local-files-only` is used. Notes remain local; only public model files are
downloaded.

## Step 1: validate the dataset

Run from `fullnote_ehr_baseline/`:

```powershell
python -m fullnote_ehr.cli.validate_dataset `
  --data-dir ..\data_cleaned `
  --report artifacts\validation\dataset_contract.json
```

The validator streams 256 rows at a time, so it does not retain every discharge
note in memory. It fails before feature construction if the current cohort no
longer matches the accepted data contract.

The JSON report contains the schema, part sizes, cohort counts, split counts,
positive counts, and measurement-concept list.

## Step 2: build shared EHR features

```powershell
python -m fullnote_ehr.cli.build_ehr_features `
  --data-dir ..\data_cleaned `
  --output-dir artifacts\ehr `
  --diagnosis-min-frequency 10
```

The command performs the full contract check, loads only structured columns, fits
all learned preprocessing on `train`, and transforms the fixed train, validation,
and test splits.

Structured features are:

- Numeric admission data: age, length of stay, and diagnosis count.
- Categorical data: gender, admission type, admission location, and insurance.
- Measurements: mean, minimum, maximum, last value, missing flag, and optionally
  hours from the last observation to discharge for each of ten concepts.
- Diagnoses: sparse multi-hot `ICD{version}_{code}` features. The vocabulary is
  learned from training admissions only and filtered by training frequency.

Identifiers, note text, labels, split names, and raw absolute timestamps are not
predictor features.

### EHR artifacts

```text
artifacts/ehr/
  dataset_contract.json
  feature_builder.joblib
  feature_names.json
  manifest.json
  X_train.npz
  X_validation.npz
  X_test.npz
  y_train.npy
  y_validation.npy
  y_test.npy
  index_train.parquet
  index_validation.parquet
  index_test.parquet
```

The index files retain `subject_id`, `hadm_id`, `note_id`, `split`, and
`mortality_30d`. Future note embeddings must be joined and reordered using these
indices before concatenation. Never rely on an unverified incidental row order.

The output directory must be empty. This prevents an interrupted or changed run
from silently mixing incompatible artifacts.

## Step 3: generate shared FullNote embeddings

Install the modeling dependencies, then run:

```powershell
python -m fullnote_ehr.cli.embed_full_notes `
  --data-dir ..\data_cleaned `
  --output-dir artifacts\embeddings\fullnote_bioclinicalbert `
  --model emilyalsentzer/Bio_ClinicalBERT `
  --device auto `
  --max-length 512 `
  --chunk-batch-size 16 `
  --note-block-size 32
```

The encoder is frozen. Each note is tokenized once, divided into non-overlapping
chunks of at most 510 content tokens plus BERT special tokens, and processed in
chunk batches. Padding and special tokens are excluded from pooling. Token hidden
states are summed across chunks, divided by the full note's content-token count,
and L2-normalized into one 768-dimensional float32 vector.

`--device auto` chooses CUDA when PyTorch reports a usable GPU and otherwise uses
CPU. CUDA automatic mixed precision is enabled by default. Use `--no-amp` to
disable it or `--device cpu` to force CPU execution.

The command checkpoints after every note block. If interrupted, rerun the exact
same command and it resumes from `embedding_progress.json`. A changed dataset,
model, revision, or pooling configuration is rejected rather than mixed into the
partial cache.

### Embedding artifacts

```text
artifacts/embeddings/fullnote_bioclinicalbert/
  dataset_contract.json
  fullnote_embeddings.npy
  fullnote_embedding_index.parquet
  embedding_manifest.json
```

The matrix contains all 45,482 admissions in deterministic source order. The index
contains `subject_id`, `hadm_id`, `note_id`, `split`, and `mortality_30d`. The
manifest records the exact note-data SHA-256, model/revision, tokenizer/model
classes, chunk size, pooling method, normalization, device, row counts, token
counts, and chunk counts.

Load and verify it from Python:

```python
from fullnote_ehr import load_embedding_artifact

embeddings, index, manifest = load_embedding_artifact(
    "artifacts/embeddings/fullnote_bioclinicalbert"
)
assert embeddings.shape == (45_482, 768)
```

To require pre-downloaded weights and prevent any network lookup:

```powershell
python -m fullnote_ehr.cli.embed_full_notes `
  --data-dir ..\data_cleaned `
  --output-dir artifacts\embeddings\fullnote_bioclinicalbert `
  --model C:\path\to\local\Bio_ClinicalBERT `
  --local-files-only
```

## Reusing the EHR builder for inference features

The implemented inference boundary currently transforms structured rows only:

```powershell
python -m fullnote_ehr.cli.transform_ehr `
  --builder artifacts\ehr\feature_builder.joblib `
  --data-dir ..\data_cleaned `
  --split test `
  --output-matrix artifacts\inference\X_test.npz `
  --output-index artifacts\inference\index_test.parquet
```

Unknown categories and diagnosis codes are ignored; the saved training feature
order is preserved. Final mortality inference will additionally require the fitted
direct- or risk-fusion model, which is not implemented yet.

## Step 4: aligned fusion inputs

`load_aligned_fusion_split` joins every note vector to the EHR index by `note_id`
and then verifies `subject_id`, `hadm_id`, split, and label equality. It fails on
missing, duplicate, or inconsistent identities instead of trusting row order.

The resulting direct matrices contain:

| Split | Shape |
|---|---:|
| Train | 31,792 x 4,809 |
| Validation | 6,908 x 4,809 |
| Test | 6,782 x 4,809 |

The 4,809 columns are the 4,041 ordered EHR features followed by all 768
BioClinicalBERT dimensions.

## Step 5: direct-fusion XGBoost

Train the experiment from the project folder:

```powershell
python -m fullnote_ehr.cli.train_direct_fusion `
  --ehr-dir artifacts\ehr `
  --embedding-dir artifacts\embeddings\fullnote_bioclinicalbert `
  --output-dir artifacts\models\direct_fusion_xgboost `
  --device cuda
```

The command:

1. Builds the identity-checked train and validation matrices.
2. Compares four predeclared depth/minimum-child-weight candidates.
3. Uses validation AUPRC for model selection and early stopping.
4. Selects the classification threshold by validation F1 only.
5. Loads and evaluates the test split after those choices are fixed.
6. Saves the model, predictions, metrics, feature order, importance, trials, and
   evaluation history.

The completed run selected `max_depth=7`, `min_child_weight=5`, and 1,177 trees.
Its validation-selected threshold is 0.261359.

| Metric | Validation | Test |
|---|---:|---:|
| AUROC | 0.8818 | 0.8890 |
| AUPRC | 0.4284 | 0.3771 |
| Accuracy | 0.9564 | 0.9518 |
| Precision | 0.5538 | 0.4856 |
| Recall/sensitivity | 0.4238 | 0.3146 |
| Specificity | 0.9830 | 0.9834 |
| F1 | 0.4801 | 0.3819 |

The thresholded metrics use the validation-selected threshold. AUROC and AUPRC
are threshold-independent.

### Direct model artifacts

```text
artifacts/models/direct_fusion_xgboost/
  direct_xgboost_model.ubj
  training_manifest.json
  evaluation_metrics.json
  candidate_trials.json
  best_evaluation_history.json
  feature_names.json
  feature_importance.parquet
  predictions_validation.parquet
  predictions_test.parquet
```

### Evaluation plots

Regenerate the saved SVG figures directly from the prediction artifacts:

```powershell
python -m fullnote_ehr.cli.plot_direct_fusion_results `
  --model-dir artifacts\models\direct_fusion_xgboost
```

The figures are written under `artifacts/models/direct_fusion_xgboost/plots/` and
cover the metric comparison, ROC curve, precision-recall curve, probability
losses, confusion matrices, validation threshold tradeoffs, and calibration.

### Re-run inference on a prepared split

```powershell
python -m fullnote_ehr.cli.predict_direct_fusion `
  --ehr-dir artifacts\ehr `
  --embedding-dir artifacts\embeddings\fullnote_bioclinicalbert `
  --model-dir artifacts\models\direct_fusion_xgboost `
  --split test `
  --output-predictions artifacts\inference\direct_fusion_test.parquet `
  --output-metrics artifacts\inference\direct_fusion_test_metrics.json
```

This boundary operates on already prepared splits. End-to-end prediction for a
new raw admission still requires the final raw-input inference wrapper.

## Step 6: leakage-safe note risk-score fusion

The note-risk experiment reuses the same BioClinicalBERT embeddings. A fixed L2
logistic regression (`C=1`, unweighted) converts each embedding into one score.
The training score is leakage-safe:

1. Generate five-fold out-of-fold training probabilities with folds grouped by
   `subject_id`.
2. Fit the final note-risk classifier on all training embeddings.
3. Produce validation and test probabilities with that final classifier.
4. Append one risk-score column to each EHR matrix.
5. Train and evaluate the second XGBoost model with the same four candidates,
   early-stopping budget, validation selection metric, and threshold rule used by
   direct fusion.

Run it with:

```powershell
python -m fullnote_ehr.cli.train_risk_score_fusion `
  --ehr-dir artifacts\ehr `
  --embedding-dir artifacts\embeddings\fullnote_bioclinicalbert `
  --output-dir artifacts\models\risk_score_fusion_xgboost `
  --note-folds 5 `
  --device cuda
```

The fold audit confirmed that all 31,792 training admissions received one OOF
score, all 25,852 training patients belong to exactly one fold, and patient
overlap is zero in every fold. The final matrices have 4,042 features: 4,041 EHR
features plus one note-risk score.

The selected XGBoost candidate used `max_depth=5`, `min_child_weight=5`, and 88
trees. Its validation-selected F1 threshold is 0.717312.

| Metric | Direct test | Risk-score test |
|---|---:|---:|
| AUROC | 0.8890 | 0.8740 |
| AUPRC | 0.3771 | 0.3531 |
| Accuracy | 0.9518 | 0.9313 |
| Precision | 0.4856 | 0.3414 |
| Recall/sensitivity | 0.3146 | 0.4860 |
| Specificity | 0.9834 | 0.9534 |
| F1 | 0.3819 | 0.4010 |
| Brier score | 0.0381 | 0.1170 |
| Log loss | 0.1689 | 0.3699 |

Risk-score fusion detects more deaths and has slightly higher test F1, while
direct fusion has better ranking, precision, specificity, accuracy, and
probability calibration. The risk-fusion XGBoost scores should not be treated as
calibrated clinical probabilities without an additional calibration stage.

### Risk-score artifacts

```text
artifacts/models/risk_score_fusion_xgboost/
  note_risk_model.joblib
  note_risk_manifest.json
  risk_scores_train_oof.parquet
  risk_scores_validation.parquet
  risk_scores_test.parquet
  risk_fusion_xgboost_model.ubj
  training_manifest.json
  evaluation_metrics.json
  candidate_trials.json
  best_evaluation_history.json
  feature_names.json
  feature_importance.parquet
  predictions_validation.parquet
  predictions_test.parquet
```

Re-run the saved two-model pipeline on a prepared split:

```powershell
python -m fullnote_ehr.cli.predict_risk_score_fusion `
  --ehr-dir artifacts\ehr `
  --embedding-dir artifacts\embeddings\fullnote_bioclinicalbert `
  --model-dir artifacts\models\risk_score_fusion_xgboost `
  --split test `
  --output-predictions artifacts\inference\risk_score_fusion_test.parquet `
  --output-metrics artifacts\inference\risk_score_fusion_test_metrics.json
```

## Final inference design

For one new eligible admission, final inference will require:

1. Transform its structured data with `feature_builder.joblib`.
2. Chunk and embed its full discharge note with the saved Step 3 recipe.
3. For direct fusion, append all 768 embedding values and run the saved model.
4. For risk fusion, run the saved note-risk classifier, append its probability,
   and run the risk-fusion model.

The preprocessing, embedding, and model manifests must agree. Inference should
fail rather than combine artifacts with different schemas or feature orders.

## Tests

```powershell
python -m pytest
```

Tests use synthetic records only. They verify contract acceptance/rejection,
train-only vocabulary fitting, unseen-category handling, measurement flattening,
stable feature order, builder serialization, long-note chunk masks, token-weighted
pooling, normalization, cache publishing, cache/index loading, identity-safe
fusion, and patient-grouped OOF risk scoring.
