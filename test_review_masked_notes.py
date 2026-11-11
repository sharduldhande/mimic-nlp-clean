import csv
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from review_masked_notes import build_report, choose_notes


class NoteReviewTests(unittest.TestCase):
    def test_training_sample_and_escaped_exact_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = [dict(subject_id=i, hadm_id=i + 100, note_id=f"n{i}", split=split,
                gender="F", discharge_note="Sex: F\n<script>alert(1)</script>", measurements=[], diagnoses=[])
                for i, split in [(1, "train"), (2, "train"), (3, "test")]]
            a, b = root / "original.parquet", root / "filtered.parquet"
            pq.write_table(pa.Table.from_pylist(original), a)
            pq.write_table(pa.Table.from_pylist([dict(r, discharge_note=r["discharge_note"].replace("Sex: F", "Sex:  ")) for r in original]), b)
            self.assertEqual(choose_notes([a], 2, 42, "train"), choose_notes([a], 2, 42, "train"))
            self.assertEqual(set(choose_notes([a], 2, 42, "train")), {"n1", "n2"})
            build_report([a], [b], root / "review", count=2)
            html = (root / "review/before_after.html").read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
            self.assertNotIn("<script>alert", html)
            with (root / "review/removed_spans.csv").open(newline="", encoding="utf-8") as handle:
                spans = list(csv.DictReader(handle))
            self.assertEqual(len(spans), 2)
            self.assertTrue(all(r["removed_text"] == "F" for r in spans))


if __name__ == "__main__":
    unittest.main()
