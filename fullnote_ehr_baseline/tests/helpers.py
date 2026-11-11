from __future__ import annotations

from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from fullnote_ehr.schema import MEASUREMENT_CONCEPTS, expected_arrow_schema


def measurement_rows(base_time: datetime) -> list[dict]:
    return [
        {
            "feature": concept,
            "mean_value": float(index + 1),
            "min_value": float(index + 0.5),
            "max_value": float(index + 1.5),
            "last_value": float(index + 1.25),
            "last_time": base_time - timedelta(hours=index + 1),
            "unit": "unit",
            "is_missing": 0,
        }
        for index, concept in enumerate(MEASUREMENT_CONCEPTS)
    ]


def dataset_row(number: int, subject: int, split: str, label: int) -> dict:
    admit = datetime(2020, 1, 1)
    discharge = datetime(2020, 1, 3)
    diagnoses = [{
        "icd_code": "I10" if number % 2 else "J189",
        "icd_version": 10,
        "diagnosis": "Synthetic diagnosis",
    }]
    return {
        "subject_id": subject,
        "hadm_id": 1000 + number,
        "admittime": admit,
        "dischtime": discharge,
        "stay_id": 2000 + number,
        "gender": "F" if number % 2 else "M",
        "age": 50 + number,
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "insurance": None if number == 4 else "Medicare",
        "length_of_stay": 2.0,
        "diagnosis_count": len(diagnoses),
        "mortality_30d": label,
        "note_id": f"note-{number}",
        "note_charttime": discharge,
        "note_storetime": discharge + timedelta(hours=12),
        "discharge_note": f"Synthetic discharge note {number}.",
        "split": split,
        "diagnoses": diagnoses,
        "measurements": measurement_rows(discharge),
    }


def write_two_parts(root, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=expected_arrow_schema())
    midpoint = (len(rows) + 1) // 2
    pq.write_table(table.slice(0, midpoint), root / "dataset_splits_000000000000.parquet")
    pq.write_table(table.slice(midpoint), root / "dataset_splits_000000000001.parquet")
