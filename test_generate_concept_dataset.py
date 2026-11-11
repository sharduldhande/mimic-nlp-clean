import csv
import tempfile
from pathlib import Path
import unittest

from generate_concept_dataset import export_parquet, filter_note, generate
import audit_redundancy as strict
from audit_redundancy_concept import propose


class FilteredDatasetTests(unittest.TestCase):
    def test_overlaps_and_original_offsets(self):
        text = "0123456789\nend"
        result, merged, removed = filter_note(text, [(1, 4, "123"), (3, 6, "345"), (8, 10, "89")])
        self.assertEqual(result, "0 67 \nend")
        self.assertEqual((merged, removed), (2, 7))
        self.assertEqual(filter_note(text, [])[0], text)
        with self.assertRaises(ValueError):
            filter_note(text, [(1, 4, "wrong")])

    def test_same_rows_and_non_note_cells_with_no_mention_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            original = [dict(subject_id="1", hadm_id="11", note_id="n1", split="train", gender="F",
                discharge_note="Sex: F\n", diagnoses="[]", measurements="[]", mortality_30d="1"),
                dict(subject_id="2", hadm_id="22", note_id="n2", split="test", gender="M",
                discharge_note="Unchanged note.", diagnoses="[]", measurements="[]", mortality_30d="0")]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(original[0]))
                writer.writeheader()
                writer.writerows(original)
            before = source.read_bytes()
            report = generate(source, root / "out", workers=1)
            with (root / "out/dataset_splits_concept_mask.csv").open(newline="") as handle:
                actual = list(csv.DictReader(handle))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(actual[0]["discharge_note"], "Sex:  \n")
            self.assertEqual(actual[1]["discharge_note"], original[1]["discharge_note"])
            for old, new in zip(original, actual):
                self.assertEqual({k: v for k, v in old.items() if k != "discharge_note"},
                                 {k: v for k, v in new.items() if k != "discharge_note"})
            self.assertEqual(report["counts"]["admissions"], 2)
            self.assertFalse((root / "out/mentions.csv").exists())

    def test_slash_panel_removed_but_unmasked_neighbor_preserved(self):
        row = dict(discharge_note="Hgb-12.6/Glucose-107", diagnoses=[], measurements=[
            dict(feature="hemoglobin", last_value=10, is_missing=0),
            dict(feature="glucose", last_value=100, is_missing=0)])
        results = [propose(row, m) for m in strict.audit_row(row)]
        self.assertEqual(len(results), 2)
        masks = [(r["mask_start"], r["mask_end"], r["mask_span"]) for r in results]
        self.assertEqual(filter_note(row["discharge_note"], masks)[0], " ")
        self.assertIn("/Glucose-107", filter_note(row["discharge_note"], masks[:1])[0])
        row["discharge_note"] = "Creatinine 1.2/3"
        self.assertEqual(list(strict.audit_row(row)), [])

    def test_parquet_keeps_nested_schema_and_reorders_notes_by_id(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = pa.Table.from_pylist([
                dict(subject_id=1, hadm_id=11, note_id="n1", split="train", discharge_note="old1",
                     measurements=[dict(feature="glucose", last_value=1.5)]),
                dict(subject_id=2, hadm_id=22, note_id="n2", split="test", discharge_note="old2",
                     measurements=[dict(feature="glucose", last_value=2.5)])])
            part = root / "original.parquet"
            pq.write_table(table, part)
            path = root / "filtered.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subject_id", "hadm_id", "note_id", "split", "discharge_note"])
                writer.writeheader()
                writer.writerows([dict(subject_id=2, hadm_id=22, note_id="n2", split="test", discharge_note="new2"),
                                  dict(subject_id=1, hadm_id=11, note_id="n1", split="train", discharge_note="new1")])
            reports = export_parquet(path, [part], root)
            actual = pq.read_table(root / reports[0]["file"])
            self.assertEqual(actual.schema, table.schema)
            self.assertEqual(actual["discharge_note"].to_pylist(), ["new1", "new2"])
            self.assertTrue(actual.drop(["discharge_note"]).equals(table.drop(["discharge_note"])))


if __name__ == "__main__":
    unittest.main()
