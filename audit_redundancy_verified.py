"""Propose conservative masks using strict event evidence and matching sex headers."""
import audit_redundancy as strict
from audit_redundancy_concept import propose as concept_proposal, run_cli

POLICY_VERSION = "verified-v1"


def propose(row, mention):
    # Share exact span extraction and context exclusions with the concept policy.
    result = concept_proposal(row, mention)
    result["policy_version"] = POLICY_VERSION
    concept = mention["concept"]
    eligible = False
    if concept == "gender":
        eligible = result["mask_decision"] == "mask"
        reason = "Agreeing explicit patient sex header."
    elif concept in strict.ALIASES:
        eligible = (result["mask_decision"] == "mask"
                    and mention["suggested_label"] == "redundant"
                    and mention.get("event_linkage") == "unique"
                    and bool(mention.get("linked_feature")))
        reason = "Strict redundant event link to the structured last-value feature."
        if eligible:
            result["structured_feature"] = mention["linked_feature"]
    else:
        reason = "Diagnosis and other concepts are retained by the verified policy."

    if eligible:
        result["decision_reason"] = reason
    else:
        if concept in strict.ALIASES or concept == "gender":
            reason = (result["decision_reason"] if result["mask_decision"] != "mask" else
                      "No verified redundant event link; concept availability alone is insufficient.")
        result.update(mask_decision="keep", decision_reason=reason,
                      mask_start="", mask_end="", mask_span="", mask_preview="")
    return result


if __name__ == "__main__":
    run_cli(proposal=propose, policy_version=POLICY_VERSION,
            default_out="data/redundancy_verified_v1", description=__doc__, limitations=[
                "Draft proposals; human review not completed.",
                "Numerical masks require strict raw-event and last-feature provenance evidence.",
                "Without --events, only matching explicit patient sex headers can be masked.",
                "Basic context exclusions also apply; diagnoses are retained.",
                "Verified means supported by the script's evidence rules, not clinical certainty.",
                "This exports proposals and previews, not filtered full notes."])
