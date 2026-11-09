import csv
import tempfile
from pathlib import Path
import unittest

from generate_concept_dataset import filter_note, generate


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


if __name__ == "__main__":
    unittest.main()
