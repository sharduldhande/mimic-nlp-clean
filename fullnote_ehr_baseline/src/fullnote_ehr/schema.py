"""Dataset and feature constants owned by the fresh baseline package."""

from __future__ import annotations

import pyarrow as pa


DATASET_COLUMNS = (
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "stay_id",
    "gender",
    "age",
    "admission_type",
    "admission_location",
    "insurance",
    "length_of_stay",
    "diagnosis_count",
    "mortality_30d",
    "note_id",
    "note_charttime",
    "note_storetime",
    "discharge_note",
    "split",
    "diagnoses",
    "measurements",
)

SPLITS = ("train", "validation", "test")

MEASUREMENT_CONCEPTS = (
    "creatinine",
    "diastolic_bp",
    "glucose",
    "heart_rate",
    "hemoglobin",
    "respiratory_rate",
    "sodium",
    "spo2",
    "systolic_bp",
    "temperature",
)

MEASUREMENT_NUMERIC_FIELDS = (
    "mean_value",
    "min_value",
    "max_value",
    "last_value",
    "is_missing",
)

BASE_NUMERIC_COLUMNS = (
    "age",
    "length_of_stay",
    "diagnosis_count",
)

CATEGORICAL_COLUMNS = (
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
)

IDENTITY_COLUMNS = (
    "subject_id",
    "hadm_id",
    "note_id",
    "split",
    "mortality_30d",
)

EHR_INPUT_COLUMNS = tuple(dict.fromkeys(
    IDENTITY_COLUMNS
    + BASE_NUMERIC_COLUMNS
    + CATEGORICAL_COLUMNS
    + ("dischtime", "diagnoses", "measurements")
))


def expected_arrow_schema() -> pa.Schema:
    """Return the exact typed schema expected from the validated one-day export."""
    diagnosis = pa.struct([
        ("icd_code", pa.string()),
        ("icd_version", pa.int64()),
        ("diagnosis", pa.string()),
    ])
    measurement = pa.struct([
        ("feature", pa.string()),
        ("mean_value", pa.float64()),
        ("min_value", pa.float64()),
        ("max_value", pa.float64()),
        ("last_value", pa.float64()),
        ("last_time", pa.timestamp("us")),
        ("unit", pa.string()),
        ("is_missing", pa.int64()),
    ])
    fields = [
        pa.field("subject_id", pa.int64()),
        pa.field("hadm_id", pa.int64()),
        pa.field("admittime", pa.timestamp("us")),
        pa.field("dischtime", pa.timestamp("us")),
        pa.field("stay_id", pa.int64()),
        pa.field("gender", pa.string()),
        pa.field("age", pa.int64()),
        pa.field("admission_type", pa.string()),
        pa.field("admission_location", pa.string()),
        pa.field("insurance", pa.string()),
        pa.field("length_of_stay", pa.float64()),
        pa.field("diagnosis_count", pa.int64()),
        pa.field("mortality_30d", pa.int64()),
        pa.field("note_id", pa.string()),
        pa.field("note_charttime", pa.timestamp("us")),
        pa.field("note_storetime", pa.timestamp("us")),
        pa.field("discharge_note", pa.string()),
        pa.field("split", pa.string()),
        pa.field(
            "diagnoses",
            pa.list_(pa.field("element", diagnosis, nullable=False)),
            nullable=False,
        ),
        pa.field(
            "measurements",
            pa.list_(pa.field("element", measurement, nullable=False)),
            nullable=False,
        ),
    ]
    return pa.schema(fields)
