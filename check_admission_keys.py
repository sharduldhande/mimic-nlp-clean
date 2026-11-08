"""Check admission identifiers and patient separation across splits."""
import argparse
import glob
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["dataset_splits.csv"],
                        help="CSV files or Parquet parts; quoted wildcards are supported")
    args = parser.parse_args()

    # Read only the columns needed; keep identifiers as strings.
    columns = ["subject_id", "hadm_id", "split"]
    paths = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"No files match: {pattern}")
        paths.extend(str(Path(path).resolve()) for path in matches)
    paths = list(dict.fromkeys(paths))
    extensions = {Path(path).suffix.lower() for path in paths}
    if len(extensions) != 1 or not extensions <= {".csv", ".parquet"}:
        parser.error("Check CSV and Parquet separately; each may contain the same admissions.")
    try:
        frames = [pd.read_parquet(path, columns=columns).astype("string")
                  if path.lower().endswith(".parquet") else
                  pd.read_csv(path, usecols=columns, dtype="string", keep_default_na=False)
                  for path in paths]
    except ImportError:
        parser.error("Parquet support requires pyarrow: python -m pip install pyarrow")
    # Combine parts before checking, so duplicates across files are caught.
    df = pd.concat(frames, ignore_index=True)
    for column in columns:
        df[column] = df[column].str.strip()
        df[column] = df[column].mask(df[column].isin(["", "NULL", "null"]))

    failed = df.empty
    print(f"Files: {len(paths)}; rows: {len(df):,}")
    for column in columns:
        missing = int(df[column].isna().sum())
        print(f"Missing {column}: {missing:,}")
        failed |= missing > 0

    # subject_id may repeat: we keep the last ICU stay for each admission, not one admission per patient.
    for keys in (["hadm_id"], ["subject_id", "hadm_id"]):
        valid = df.dropna(subset=keys)
        extra_rows = int(valid.duplicated(subset=keys).sum())
        duplicate_groups = len(valid.loc[valid.duplicated(subset=keys), keys].drop_duplicates())
        print(f"{', '.join(keys)}: {len(valid.drop_duplicates(subset=keys)):,} unique; "
              f"{duplicate_groups:,} repeated keys; {extra_rows:,} extra rows")
        failed |= extra_rows > 0

    # Repeated patients (multiple admissions) should be in the same splits.
    split_counts = df.dropna(subset=["subject_id", "split"]).groupby("subject_id")["split"].nunique()
    overlap = int(split_counts.gt(1).sum())
    print(f"Patients in multiple splits: {overlap:,}")
    failed |= overlap > 0
    print("FAIL" if failed else "PASS")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
