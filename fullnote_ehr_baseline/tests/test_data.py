from __future__ import annotations

from dataclasses import replace

import pytest

from fullnote_ehr.data import DatasetContract, DatasetValidationError, load_frame, validate_dataset
from helpers import dataset_row, write_two_parts


def synthetic_contract() -> DatasetContract:
    return DatasetContract(
        expected_rows=4,
        expected_patients=4,
        expected_positive_labels=2,
        expected_split_rows={"train": 2, "validation": 1, "test": 1},
        expected_split_positives={"train": 1, "validation": 0, "test": 1},
    )


def test_contract_passes_and_loads_selected_split(tmp_path):
    rows = [
        dataset_row(1, 101, "train", 1),
        dataset_row(2, 102, "train", 0),
        dataset_row(3, 103, "validation", 0),
        dataset_row(4, 104, "test", 1),
    ]
    write_two_parts(tmp_path, rows)
    report_path = tmp_path / "report.json"
    report = validate_dataset(tmp_path, synthetic_contract(), report_path)
    assert report["status"] == "PASS"
    assert report["rows"] == 4
    assert report_path.is_file()
    test = load_frame(tmp_path, columns=["note_id", "mortality_30d"], split="test")
    assert test["note_id"].tolist() == ["note-4"]
    assert test["mortality_30d"].tolist() == [1]


def test_contract_rejects_patient_split_overlap(tmp_path):
    rows = [
        dataset_row(1, 101, "train", 1),
        dataset_row(2, 102, "train", 0),
        dataset_row(3, 101, "validation", 0),
        dataset_row(4, 104, "test", 1),
    ]
    write_two_parts(tmp_path, rows)
    with pytest.raises(DatasetValidationError, match="multiple splits"):
        validate_dataset(tmp_path, replace(synthetic_contract(), expected_patients=3))
