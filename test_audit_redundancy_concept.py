import copy
import unittest

import audit_redundancy as strict
from audit_redundancy_concept import propose


class ConceptPolicyTests(unittest.TestCase):
    def diagnosis(self, text, term="dysuria"):
        row = dict(discharge_note=text, diagnoses=[dict(diagnosis=term, icd_code="R300")], measurements=[])
        mention = next(m for m in strict.audit_row(row) if m["concept"] == "diagnosis")
        return row, mention

    def test_affirmed_diagnosis_and_following_negation(self):
        row, mention = self.diagnosis("Also reports orthopnea, dysuria, but no hematuria.")
        before = copy.deepcopy((row, mention))
        result = propose(row, mention)
        self.assertEqual(result["mask_decision"], "mask")
        self.assertEqual(result["mask_span"], "dysuria")
        self.assertEqual(result["suggested_label"], "uncertain")
        self.assertEqual((row, mention), before)

    def test_negative_historical_family_and_hypothetical_kept(self):
        cases = [("Denies dysuria.", "negated"),
                 ("No fever, dysuria or hematuria.", "negated"),
                 ("History of dysuria.", "historical"),
                 ("Family History:\nMother had dysuria.", "other_experiencer"),
                 ("Monitor for dysuria.", "hypothetical"),
                 ("Dysuria was denied.", "negated")]
        for text, status in cases:
            with self.subTest(text=text):
                row, mention = self.diagnosis(text)
                result = propose(row, mention)
                self.assertEqual(result["assertion_status"], status)
                self.assertEqual(result["mask_decision"], "keep")

    def test_different_numeric_value_masked_without_events_only_when_available(self):
        row = dict(discharge_note="Creatinine 1.2 mg/dL.", diagnoses=[], measurements=[
            dict(feature="creatinine", last_value=2.0, unit="mg/dL", is_missing=0)])
        mention = next(strict.audit_row(row))
        result = propose(row, mention)
        self.assertEqual(result["mask_decision"], "mask")
        self.assertEqual(result["mask_span"], "1.2")
        self.assertEqual(result["suggested_label"], "uncertain")
        self.assertIn("Creatinine [MASK] mg/dL", result["mask_preview"])
        row["measurements"][0]["is_missing"] = 1
        result = propose(row, next(strict.audit_row(row)))
        self.assertEqual(result["concept_overlap"], "unavailable")
        self.assertEqual(result["mask_decision"], "keep")

    def test_blood_pressure_numbers_have_separate_exact_targets(self):
        row = dict(discharge_note="BP 120/80 mmHg", diagnoses=[], measurements=[
            dict(feature="systolic_bp", last_value=130, is_missing=0),
            dict(feature="diastolic_bp", last_value=90, is_missing=0)])
        results = [propose(row, mention) for mention in strict.audit_row(row)]
        self.assertEqual([r["mask_span"] for r in results], ["120", "80"])
        self.assertTrue(all(r["mask_decision"] == "mask" for r in results))
        for r in results:
            self.assertEqual(row["discharge_note"][r["mask_start"]:r["mask_end"]], r["mask_span"])

    def test_gender_masks_value_only_and_disagreement_needs_review(self):
        row = dict(discharge_note="Sex: F\n", gender="F", diagnoses=[], measurements=[])
        result = propose(row, next(strict.audit_row(row)))
        self.assertEqual(result["mask_span"], "F")
        self.assertEqual(result["mask_decision"], "mask")
        row["gender"] = "M"
        self.assertEqual(propose(row, next(strict.audit_row(row)))["mask_decision"], "review")


if __name__ == "__main__":
    unittest.main()
