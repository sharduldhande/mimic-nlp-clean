import csv
from pathlib import Path
import tempfile
import unittest

from summarize_dataset import summarize


class CharacterSummaryTests(unittest.TestCase):
    def test_unicode_characters_include_whitespace_and_exclude_missing_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                fields = ["split", "mortality_30d", "discharge_note", "diagnoses", "measurements"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for text in ("é x\n", "ab", ""):
                    writer.writerow(dict(zip(fields, ["train", "0", text, "[]", "[]"])))
            summarize([path], root / "out")
            with (root / "out/note_lengths.csv").open(newline="") as handle:
                overall = next(csv.DictReader(handle))
            self.assertEqual(float(overall["characters_mean"]), 3)
            self.assertEqual(float(overall["mean"]), 1.5)
            self.assertEqual(int(overall["missing_notes"]), 1)


if __name__ == "__main__":
    unittest.main()
