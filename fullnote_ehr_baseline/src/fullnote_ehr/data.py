"""Memory-conscious dataset discovery, loading, and contract validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import DATASET_COLUMNS, MEASUREMENT_CONCEPTS, SPLITS, expected_arrow_schema


class DatasetValidationError(ValueError):
    """Raised when the supplied dataset violates the modeling data contract."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        preview = "\n".join(f"- {item}" for item in self.issues[:25])
        suffix = "\n- ... additional issues omitted" if len(self.issues) > 25 else ""
        super().__init__(f"Dataset contract failed with {len(self.issues)} issue(s):\n{preview}{suffix}")


@dataclass(frozen=True)
class DatasetContract:
    """Expected properties of the accepted one-day cohort."""

    expected_rows: int = 45_482
    expected_patients: int = 36_913
    expected_positive_labels: int = 2_248
    expected_split_rows: dict[str, int] = field(default_factory=lambda: {
        "train": 31_792,
        "validation": 6_908,
        "test": 6_782,
    })
    expected_split_positives: dict[str, int] = field(default_factory=lambda: {
        "train": 1_599,
        "validation": 328,
        "test": 321,
    })
    note_lag_limit_hours: int = 24
    expected_part_count: int = 2


def discover_parquet_parts(data_dir: str | Path) -> tuple[Path, ...]:
    """Find the ordered clean-dataset Parquet parts and reject ambiguous inputs."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    parts = tuple(sorted(root.glob("dataset_splits_*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No dataset_splits_*.parquet files found in {root}")
    return parts


def _schema_issues(actual: pa.Schema) -> list[str]:
    expected = expected_arrow_schema()
    issues: list[str] = []
    if tuple(actual.names) != DATASET_COLUMNS:
        issues.append(
            "Column order/names differ. "
            f"Expected {list(DATASET_COLUMNS)}, received {actual.names}."
        )
        return issues
    for expected_field, actual_field in zip(expected, actual):
        if not expected_field.type.equals(actual_field.type):
            issues.append(
                f"Column {expected_field.name!r} has type {actual_field.type}; "
                f"expected {expected_field.type}."
            )
        if expected_field.nullable != actual_field.nullable:
            issues.append(
                f"Column {expected_field.name!r} nullable={actual_field.nullable}; "
                f"expected nullable={expected_field.nullable}."
            )
    return issues


def _iter_rows(paths: Iterable[Path], columns: Sequence[str]) -> Iterator[dict]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=list(columns)):
            yield from batch.to_pylist()


def _as_record_list(value: object) -> list[dict]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"Expected a list of records, received {type(value).__name__}")
    return value


def validate_dataset(
    data_dir: str | Path,
    contract: DatasetContract | None = None,
    report_path: str | Path | None = None,
) -> dict:
    """Stream through all rows, enforce the cohort contract, and return a report."""
    contract = contract or DatasetContract()
    parts = discover_parquet_parts(data_dir)
    issues: list[str] = []
    if len(parts) != contract.expected_part_count:
        issues.append(
            f"Expected {contract.expected_part_count} Parquet parts, found {len(parts)}."
        )

    schemas = [pq.ParquetFile(path).schema_arrow for path in parts]
    issues.extend(_schema_issues(schemas[0]))
    for path, schema in zip(parts[1:], schemas[1:]):
        if not schema.equals(schemas[0], check_metadata=True):
            issues.append(f"Schema differs across Parquet parts: {path.name}")

    validation_columns = (
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "stay_id",
        "diagnosis_count",
        "mortality_30d",
        "note_id",
        "note_storetime",
        "discharge_note",
        "split",
        "diagnoses",
        "measurements",
    )
    rows = 0
    positives = 0
    split_rows: Counter[str] = Counter()
    split_positives: Counter[str] = Counter()
    subjects: set[int] = set()
    admissions: set[int] = set()
    notes: set[str] = set()
    admission_keys: set[tuple[int, int]] = set()
    patient_splits: dict[int, set[str]] = defaultdict(set)
    issue_limit = 100

    def add_issue(message: str) -> None:
        if len(issues) < issue_limit:
            issues.append(message)

    for row_number, row in enumerate(_iter_rows(parts, validation_columns), start=1):
        rows += 1
        prefix = f"row {row_number}"
        subject_id, hadm_id = row["subject_id"], row["hadm_id"]
        note_id, split, label = row["note_id"], row["split"], row["mortality_30d"]
        if subject_id is None or hadm_id is None or row["stay_id"] is None:
            add_issue(f"{prefix}: missing subject/admission/stay identifier")
            continue
        if note_id is None or not str(note_id).strip():
            add_issue(f"{prefix}: missing note_id")
            continue
        if split not in SPLITS:
            add_issue(f"{prefix}: invalid split {split!r}")
            continue
        if label not in (0, 1):
            add_issue(f"{prefix}: invalid mortality_30d value {label!r}")
            continue

        key = (subject_id, hadm_id)
        if hadm_id in admissions:
            add_issue(f"{prefix}: duplicate hadm_id {hadm_id}")
        if key in admission_keys:
            add_issue(f"{prefix}: duplicate (subject_id, hadm_id) {key}")
        if str(note_id) in notes:
            add_issue(f"{prefix}: duplicate note_id {note_id}")
        subjects.add(subject_id)
        admissions.add(hadm_id)
        admission_keys.add(key)
        notes.add(str(note_id))
        patient_splits[subject_id].add(split)
        split_rows[split] += 1
        positives += int(label)
        split_positives[split] += int(label)

        admit, discharge, stored = row["admittime"], row["dischtime"], row["note_storetime"]
        if admit is None or discharge is None or stored is None:
            add_issue(f"{prefix}: missing required timestamp")
        else:
            if discharge < admit:
                add_issue(f"{prefix}: discharge precedes admission")
            if stored - discharge > timedelta(hours=contract.note_lag_limit_hours):
                add_issue(f"{prefix}: note exceeds the {contract.note_lag_limit_hours}-hour cutoff")
        text = row["discharge_note"]
        if not isinstance(text, str) or not text.strip():
            add_issue(f"{prefix}: discharge note is missing or empty")

        try:
            diagnoses = _as_record_list(row["diagnoses"])
            measurements = _as_record_list(row["measurements"])
        except TypeError as error:
            add_issue(f"{prefix}: {error}")
            continue
        diagnosis_count = row["diagnosis_count"]
        if diagnosis_count is not None and diagnosis_count != len(diagnoses):
            add_issue(
                f"{prefix}: diagnosis_count={diagnosis_count} but diagnoses has {len(diagnoses)} records"
            )
        names = [item.get("feature") for item in measurements]
        if len(names) != len(set(names)):
            add_issue(f"{prefix}: duplicate measurement concept")
        if set(names) != set(MEASUREMENT_CONCEPTS):
            add_issue(
                f"{prefix}: measurement concepts differ; found {sorted(str(name) for name in names)}"
            )
        for item in measurements:
            flag = item.get("is_missing")
            if flag not in (0, 1):
                add_issue(f"{prefix}: invalid is_missing={flag!r} for {item.get('feature')}")

    cross_split = [subject for subject, values in patient_splits.items() if len(values) != 1]
    if cross_split:
        issues.append(f"{len(cross_split)} patient(s) appear in multiple splits")
    if rows != contract.expected_rows:
        issues.append(f"Expected {contract.expected_rows} rows, found {rows}")
    if len(subjects) != contract.expected_patients:
        issues.append(f"Expected {contract.expected_patients} patients, found {len(subjects)}")
    if positives != contract.expected_positive_labels:
        issues.append(
            f"Expected {contract.expected_positive_labels} positive labels, found {positives}"
        )
    if dict(split_rows) != contract.expected_split_rows:
        issues.append(
            f"Split row counts differ: expected {contract.expected_split_rows}, found {dict(split_rows)}"
        )
    if dict(split_positives) != contract.expected_split_positives:
        issues.append(
            "Split positive counts differ: "
            f"expected {contract.expected_split_positives}, found {dict(split_positives)}"
        )

    if issues:
        raise DatasetValidationError(issues)

    report = {
        "contract_version": "one-day-cohort-v1",
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "parts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
            for path in parts
        ],
        "rows": rows,
        "patients": len(subjects),
        "admissions": len(admissions),
        "notes": len(notes),
        "positive_labels": positives,
        "mortality_percent": 100.0 * positives / rows,
        "split_rows": dict(split_rows),
        "split_positives": dict(split_positives),
        "measurement_concepts": list(MEASUREMENT_CONCEPTS),
        "schema": str(schemas[0]),
    }
    if report_path is not None:
        destination = Path(report_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_frame(
    data_dir: str | Path,
    columns: Sequence[str] | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """Load selected columns from all parts, preserving deterministic part/row order."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, received {split!r}")
    parts = discover_parquet_parts(data_dir)
    selected = list(columns) if columns is not None else None
    if split is not None and selected is not None and "split" not in selected:
        selected.append("split")
    frames = [pd.read_parquet(path, columns=selected, engine="pyarrow") for path in parts]
    frame = pd.concat(frames, ignore_index=True)
    if split is not None:
        frame = frame.loc[frame["split"].eq(split)].reset_index(drop=True)
    return frame
