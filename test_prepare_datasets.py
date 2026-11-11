import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq

from prepare_datasets import NAMES, prepare, publish, rename_output, validate_source


class OneCommandTests(unittest.TestCase):
    def test_transient_output_lock_is_retried(self):
        with patch.object(Path, "rename", side_effect=[PermissionError("locked"), None]) as rename:
            with patch("prepare_datasets.time.sleep"):
                rename_output(Path("output"), Path("backup"))
        self.assertEqual(rename.call_count, 2)

    def test_persistent_output_lock_has_actionable_error(self):
        with patch.object(Path, "rename", side_effect=PermissionError("locked")):
            with patch("prepare_datasets.time.sleep"):
                with self.assertRaisesRegex(PermissionError, "Close dataset viewers"):
                    rename_output(Path("output"), Path("backup"))

    def write_source(self, root):
        path = root / "data/cleaned/dataset_splits.csv"
        path.parent.mkdir(parents=True)
        rows = []
        for i, split in enumerate(["train", "train", "validation", "test"], 1):
            row = dict.fromkeys(NAMES, "")
            row.update(subject_id=str(i), hadm_id=str(i + 100), stay_id=str(i + 200),
                admittime="2020-01-01 00:00:00", dischtime="2020-01-03 00:00:00",
                note_charttime="2020-01-03 00:00:00", note_storetime="2020-01-03 12:00:00",
                gender="F", age="60", admission_type="EMERGENCY", admission_location="EMERGENCY ROOM",
                length_of_stay="2.0", diagnosis_count="0", mortality_30d=str(i % 2),
                note_id=f"n{i}", discharge_note="Sex: F\nCreatinine 1.2 mg/dL. Stable course.",
                split=split, diagnoses="[]", measurements=json.dumps([dict(feature="creatinine",
                    mean_value=1.2, min_value=1.1, max_value=1.3, last_value=1.2,
                    last_time="2020-01-02T12:00:00", unit="mg/dL", is_missing=0)]))
            rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=NAMES)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_full_pipeline_and_safe_rerun_from_csv_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_source(root)
            before = source.read_bytes()
            with redirect_stdout(io.StringIO()):
                report = prepare(root, workers=1)
            self.assertEqual(report["rows"], 4)
            self.assertEqual(source.read_bytes(), before)
            parts = sorted((root / "data/cleaned").glob("*.parquet"))
            self.assertEqual([pq.ParquetFile(p).metadata.num_rows for p in parts], [2, 2])
            measurement = pq.read_table(parts[0])["measurements"].to_pylist()[0][0]
            self.assertEqual(measurement["last_time"].year, 2020)
            self.assertTrue((root / "data/concept_masked/review_examples/before_after.html").is_file())
            self.assertTrue((root / "validation_output/pipeline_validation.json").is_file())
            self.assertEqual(len(list((root / "validation_output").glob("*/note_lengths.csv"))), 4)
            filtered = root / "data/concept_masked/dataset_splits_concept_mask.csv"
            first = filtered.read_bytes()
            with redirect_stdout(io.StringIO()):
                prepare(root, workers=1)
            self.assertEqual(filtered.read_bytes(), first)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(list((root / "data").glob(".dataset_build_*")))

    def test_wrong_cutoff_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_source(root)
            source.write_text(source.read_text().replace("2020-01-03 12:00:00", "2020-01-05 12:00:00"))
            with self.assertRaisesRegex(ValueError, "one-day cutoff"):
                validate_source(source)

    def test_publish_rolls_back_if_replacement_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            stage.mkdir()
            (stage / "first").write_text("new1")
            (root / "first").write_text("old1")
            (root / "second").write_text("old2")
            with self.assertRaises(FileNotFoundError):
                publish([(stage / "first", root / "first", root),
                         (stage / "missing", root / "second", root)])
            self.assertEqual((root / "first").read_text(), "old1")
            self.assertEqual((root / "second").read_text(), "old2")
            self.assertEqual((stage / "first").read_text(), "new1")


if __name__ == "__main__":
    unittest.main()
