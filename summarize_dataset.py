"""Summarize feature missingness, mortality and note lengths without changing data."""
import argparse
from collections import Counter, defaultdict
import csv
import glob
import json
from pathlib import Path

import pandas as pd

csv.field_size_limit(100_000_000)


def is_missing(value):
    return value is None or (isinstance(value, str) and value.strip() in {"", "NULL", "null"})


def nested_records(value):
    if is_missing(value):
        return []
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("Expected diagnoses/measurements to be arrays of objects")
    return value


def read_rows(paths):
    for path in paths:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                yield from csv.DictReader(handle)
        else:
            import pyarrow.parquet as pq
            # Read in batches so full notes do not all occupy memory together.
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                yield from batch.to_pylist()


def summarize(paths, output):
    totals = Counter()
    missing = defaultdict(Counter)
    present = defaultdict(Counter)
    mortality = defaultdict(Counter)
    lengths = defaultdict(list)
    characters = defaultdict(list)
    features = set()
    columns = None
    required = {"split", "mortality_30d", "discharge_note", "diagnoses", "measurements"}

    for row in read_rows(paths):
        if columns is None:
            columns = list(row)
            if not required <= set(columns):
                raise ValueError(f"Missing columns: {sorted(required - set(columns))}")
        elif set(row) != set(columns):
            raise ValueError("Input files must have the same columns")
        split = "MISSING" if is_missing(row["split"]) else str(row["split"]).strip()
        groups = [("overall", ""), ("split", split)]
        absent = [column for column in columns if is_missing(row[column])]
        diagnoses = nested_records(row["diagnoses"])
        measurements = nested_records(row["measurements"])
        available = set()
        seen = set()
        for item in measurements:
            feature = item.get("feature")
            if not feature or feature in seen:
                raise ValueError("Missing or duplicate feature name in a measurement array")
            seen.add(feature)
            features.add(feature)
            flag = item.get("is_missing")
            if flag not in (0, 1, False, True, "0", "1", "true", "false", "True", "False"):
                raise ValueError(f"Invalid is_missing flag for {feature}")
            values = [item.get(name) for name in ("mean_value", "min_value", "max_value", "last_value")]
            if flag in (0, False, "0", "false", "False") and any(not is_missing(v) for v in values):
                available.add(feature)

        label = row["mortality_30d"]
        if is_missing(label):
            label_kind = "missing"
        else:
            try:
                number = float(label)
                label_kind = "positive" if number == 1 else "negative" if number == 0 else "invalid"
            except (ValueError, TypeError):
                label_kind = "invalid"
        text = row["discharge_note"]
        # These count whitespace-separated chunks, not a model's subword tokens.
        note_length = None if is_missing(text) else len(text.split())
        for group in groups:
            totals[group] += 1
            missing[group].update(absent)
            missing[group]["diagnoses.empty_array"] += not diagnoses
            missing[group]["measurements.empty_array"] += not measurements
            present[group].update(available)
            mortality[group][label_kind] += 1
            if note_length is not None:
                lengths[group].append(note_length)
                characters[group].append(len(text))

    if not totals:
        raise ValueError("Dataset is empty")
    output.mkdir(parents=True, exist_ok=True)
    missing_rows, mortality_rows, length_rows = [], [], []
    for group, total in sorted(totals.items()):
        base = dict(scope=group[0], split=group[1], admissions=total)
        counts = dict(missing[group])
        for feature in sorted(features):
            # The denominator includes admissions with no object for this feature.
            counts[f"measurements.{feature}"] = total - present[group][feature]
        for feature in columns + ["diagnoses.empty_array", "measurements.empty_array"] + [f"measurements.{f}" for f in sorted(features)]:
            count = counts.get(feature, 0)
            missing_rows.append(dict(base, feature=feature, missing_count=count, missing_percent=100 * count / total))
        labels = mortality[group]
        valid = labels["positive"] + labels["negative"]
        mortality_rows.append(dict(base, positive=labels["positive"], negative=labels["negative"],
            missing=labels["missing"], invalid=labels["invalid"], valid_labels=valid,
            mortality_percent=100 * labels["positive"] / valid if valid else None))
        values = pd.Series(lengths[group], dtype="float64")
        chars = pd.Series(characters[group], dtype="float64")
        length_rows.append(dict(base, notes_counted=len(values), missing_notes=total - len(values),
            min=values.min(), mean=values.mean(), median=values.median(),
            p95=values.quantile(.95), max=values.max(), token_method="whitespace",
            characters_min=chars.min(), characters_mean=chars.mean(), characters_median=chars.median(),
            characters_p95=chars.quantile(.95), characters_max=chars.max(),
            character_method="Unicode code points, including whitespace"))

    for name, data in (("feature_missingness", missing_rows), ("mortality_prevalence", mortality_rows),
                       ("note_lengths", length_rows)):
        pd.DataFrame(data).to_csv(output / f"{name}.csv", index=False)
    print(f"Summarized {totals[('overall', '')]:,} admissions from {len(paths)} file(s).")
    print(f"Reports: {output.resolve()}")
    print("Mortality percentages exclude missing/invalid labels; see their separate counts.")
    print("Measurement missingness: absent feature, missing flag, or all numeric summaries null.")
    print("Note lengths count whitespace-separated tokens in nonmissing notes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["dataset_splits.csv"],
                        help="CSV files or Parquet parts; quote wildcards")
    parser.add_argument("--output", default="validation_output/summary")
    args = parser.parse_args()
    paths = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"No files match: {pattern}")
        paths.extend(Path(path).resolve() for path in matches)
    paths = list(dict.fromkeys(paths))
    extensions = {path.suffix.lower() for path in paths}
    if len(extensions) != 1 or not extensions <= {".csv", ".parquet"}:
        parser.error("Summarize CSV and Parquet separately to avoid counting admissions twice")
    try:
        summarize(paths, Path(args.output))
    except ImportError:
        parser.error("Parquet support requires pyarrow: python -m pip install pyarrow")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
