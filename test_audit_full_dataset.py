import csv
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from audit_redundancy_concept import parquet_rows, write_audit


class FullDatasetTests(unittest.TestCase):
    def row(self, number, split, text):
        return dict(subject_id=number, hadm_id=number + 100, note_id=f"n{number}",
                    split=split, gender="F", discharge_note=text, diagnoses=[], measurements=[])

    def test_all_parts_splits_and_zero_mention_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [self.row(1, "train", "Sex: F"), self.row(2, "validation", "No extracted terms."),
                    self.row(3, "test", "Sex: F")]
            table = pa.Table.from_pylist(rows)
            parts = [root / "part0.parquet", root / "part1.parquet"]
            pq.write_table(table.slice(0, 2), parts[0])
            pq.write_table(table.slice(2), parts[1])
            summary = write_audit(parquet_rows(parts), root / "out")
            self.assertEqual(summary["notes"], 3)
            self.assertEqual(summary["mentions"], 2)
            self.assertEqual(set(summary["by_split"]), {"train", "validation", "test"})
            with (root / "out/note_summary.csv").open(newline="") as handle:
                notes = list(csv.DictReader(handle))
            self.assertEqual(len(notes), 3)
            self.assertEqual(notes[1]["mentions"], "0")
            self.assertEqual(len(list(parquet_rows(parts, "test"))), 1)

    def test_missing_event_coverage_cannot_look_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.row(1, "train", "Creatinine 1.2")
            row["measurements"] = [dict(feature="creatinine", last_value=1.2, is_missing=0)]
            with self.assertRaisesRegex(ValueError, "Events do not cover"):
                write_audit(iter([row]), root / "out", require_event_coverage=True)
            self.assertFalse((root / "out/mentions.csv").exists())
            self.assertTrue((root / "out/mentions.csv.partial").exists())

    def test_duplicate_admissions_rejected_instead_of_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.row(1, "train", "Sex: F")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                write_audit(iter([row, row]), Path(directory) / "out")

    def test_parallel_output_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [self.row(1, "train", "Sex: F"), self.row(2, "test", "No extracted terms.")]
            serial = write_audit(iter(rows), root / "serial")
            parallel = write_audit(iter(rows), root / "parallel", workers=2)
            self.assertEqual(serial, parallel)
            for name in ("mentions.csv", "note_summary.csv", "category_summary.csv"):
                self.assertEqual((root / "serial" / name).read_bytes(), (root / "parallel" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
