import unittest

import audit_redundancy as strict
from audit_redundancy_concept import propose as concept_propose
from audit_redundancy_verified import propose


class VerifiedPolicyTests(unittest.TestCase):
    def measurement(self, note="2020-01-01 12:00 creatinine 1.2 mg/dL"):
        row = dict(discharge_note=note, diagnoses=[], measurements=[dict(
            feature="creatinine", last_value=1.2, last_time="2020-01-01T12:00:00",
            unit="mg/dL", is_missing=0)])
        event = dict(source="lab", event_id="e1", specimen_id="s1", stay_id="",
            concept="creatinine", charttime="2020-01-01T12:00:00", value="1.2", unit="mg/dL",
            feature_path="measurements[0].last_value", in_feature_window="1")
        return row, event

    def test_concept_availability_does_not_establish_verified_mask(self):
        row, _ = self.measurement()
        mention = next(strict.audit_row(row))
        self.assertEqual(concept_propose(row, mention)["mask_decision"], "mask")
        result = propose(row, mention)
        self.assertEqual(result["mask_decision"], "keep")
        self.assertEqual(result["mask_span"], "")

    def test_unique_feature_link_masks_number_only(self):
        row, event = self.measurement()
        result = propose(row, next(strict.audit_row(row, [event])))
        self.assertEqual(result["mask_decision"], "mask")
        self.assertEqual(result["mask_span"], "1.2")
        self.assertEqual(result["structured_feature"], event["feature_path"])
        self.assertEqual(result["policy_version"], "verified-v1")

    def test_ambiguous_and_outside_events_retained(self):
        row, event = self.measurement()
        for events in ([event, dict(event, event_id="e2")],
                       [dict(event, feature_path="", in_feature_window="0")]):
            mention = next(strict.audit_row(row, events))
            self.assertEqual(propose(row, mention)["mask_decision"], "keep")

    def test_diagnosis_retained_and_gender_header_masked(self):
        row = dict(discharge_note="Sex: F\nReports dysuria.", gender="F",
                   diagnoses=[dict(diagnosis="Dysuria")], measurements=[])
        results = {m["concept"]: propose(row, m) for m in strict.audit_row(row)}
        self.assertEqual(results["gender"]["mask_decision"], "mask")
        self.assertEqual(results["gender"]["mask_span"], "F")
        self.assertEqual(results["diagnosis"]["mask_decision"], "keep")


if __name__ == "__main__":
    unittest.main()
