# 7650_project

## Generate everything locally

1. Put the **one-day export** `dataset_splits.csv` in `data/cleaned/`.
2. From this project folder, install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Run and wait for **DONE**:

   ```powershell
   python prepare_datasets.py
   ```

This creates two full-note Parquet parts, the concept-filtered CSV and Parquet
parts, all validation reports, a training-note audit sample, and 10 highlighted
before/after examples. It preserves your input CSV and safely replaces generated
outputs on successful reruns. No BigQuery connection or events export is needed.

See [dataset details](DATASET_README.md) or
[extract the one-day cohort from PhysioNet](DATASET_EXTRACTION_README.md).
