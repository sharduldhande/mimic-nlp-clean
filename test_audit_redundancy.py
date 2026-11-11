"""Regression checks for extraction and the existing evidence requirements."""
import unittest

from audit_redundancy import audit_row, diagnoses, review_context


class AuditTests(unittest.TestCase):
    def row(self, note, **fields):
        return dict(discharge_note=note, diagnoses=[], measurements=[], **fields)

    def test_sex_headers_and_mismatches(self):
        for text, gender in [("M", "M"), ("F", "F"), ("Male", "M"), ("Female", "F")]:
            with self.subTest(text=text):
                mentions = list(audit_row(self.row(f"Sex: {text}\n", gender=gender)))
                self.assertEqual(len(mentions), 1)
                self.assertEqual(mentions[0]["suggested_label"], "redundant")
        mismatch = list(audit_row(self.row("Sex: F", gender="M")))[0]
        self.assertEqual(mismatch["suggested_label"], "uncertain")
        self.assertEqual(list(audit_row(self.row("Sex: Missing", gender="M"))), [])

    def test_dataset_diagnosis_field_and_exact_spans(self):
        diagnosis = {"diagnosis": "Pneumonia", "icd_code": "J189", "icd_version": 10}
        self.assertIn(("diagnoses[0].diagnosis", "Pneumonia"), list(diagnoses(diagnosis)))
        row = self.row("No pneumonia documented.\n")
        row["diagnoses"] = [diagnosis]
        mention = list(audit_row(row))[0]
        self.assertEqual(mention["concept"], "diagnosis")
        # Literal matching cannot establish assertion or redundancy.
        self.assertEqual(mention["suggested_label"], "uncertain")
        self.assertEqual(row["discharge_note"][mention["start"]:mention["end"]], "pneumonia")

    def test_context_keeps_complete_lines_and_marks_span(self):
        note = "before\n" + "x" * 150 + " HR 90 " + "y" * 150 + "\nafter\nother"
        start = note.index("HR 90")
        context = review_context(note, start, start + 5)
        self.assertTrue(context["context"].startswith("before\n"))
        self.assertTrue(context["context"].endswith("\nafter"))
        self.assertIn("[[HR 90]]", context["context_marked"])
        self.assertEqual(context["context"], note[context["context_start"]:context["context_end"]])
        self.assertEqual(review_context("HR 90", 0, 5)["context_marked"], "[[HR 90]]")

    def test_measurement_still_requires_event_and_feature_evidence(self):
        row = self.row("2020-01-01 12:00 creatinine 1.2 mg/dL", subject_id=1, hadm_id=2)
        row["measurements"] = [dict(feature="creatinine", last_value=1.2,
            last_time="2020-01-01T12:00:00", unit="mg/dL", is_missing=0)]
        self.assertEqual(list(audit_row(row))[0]["suggested_label"], "uncertain")
        event = dict(source="lab", event_id="e1", specimen_id="s1", stay_id="",
            concept="creatinine", charttime="2020-01-01T12:00:00", value="1.2",
            unit="mg/dL", feature_path="measurements[0].last_value", in_feature_window="1")
        linked = list(audit_row(row, [event]))[0]
        self.assertEqual(linked["suggested_label"], "redundant")
        self.assertEqual(linked["event_id"], "e1")
        self.assertEqual(list(audit_row(row, [event, dict(event, event_id="e2")]))[0]["suggested_label"], "uncertain")


if __name__ == "__main__":
    unittest.main()
