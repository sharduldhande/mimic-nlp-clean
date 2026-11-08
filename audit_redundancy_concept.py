"""Add concept-based masking proposals to an existing note audit sample."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import re
from pathlib import Path

import audit_redundancy as strict

POLICY_VERSION = "concept-v1"
EXTRA_FIELDS = ["policy_version", "feature_category", "concept_overlap", "assertion_status",
                "structured_feature", "mask_decision", "decision_reason", "mask_start",
                "mask_end", "mask_span", "mask_preview"]
HEADINGS = re.compile(r"^[ \t]*(?P<name>[A-Za-z][A-Za-z /_-]{1,65}):", re.M)
BREAK = re.compile(r"[.!?;](?:\s|$)|\n[ \t]*\n")
CONTRAST = re.compile(r"\b(?:but|however|although|yet)\b", re.I)
NEGATED = re.compile(r"\b(?:no|denies|denied|without|negative for|absence of|free of)\b", re.I)
AFFIRMED = re.compile(r"\b(?:reports|endorses|has|having|positive for|complains of)\b", re.I)
HISTORY = re.compile(r"\b(?:history of|h/o|prior|previous|previously|remote|resolved|used to)\b", re.I)
POSSIBLE = re.compile(r"\b(?:if|possible|possibly|suspected|suspect|rule out|r/o|monitor for|evaluate for|may have|might have)\b", re.I)
RELATIVE = re.compile(r"\b(?:mother|father|sister|brother|daughter|son|grandmother|grandfather|family history)\b", re.I)


def assertion_status(note, start, end):
    """Conservative context flags, not a clinical assertion classifier."""
    section = ""
    for heading in HEADINGS.finditer(note, 0, start):
        section = heading["name"].strip().casefold()
    if section == "family history":
        return "other_experiencer"
    left = 0
    for boundary in BREAK.finditer(note, 0, start):
        left = boundary.end()
    boundary = BREAK.search(note, end)
    right = boundary.start() if boundary else len(note)
    before, after = note[left:start], note[end:right]
    # A later clause such as 'but no hematuria' must not negate earlier dysuria.
    before = CONTRAST.split(before)[-1]
    after = CONTRAST.split(after)[0]
    before = " ".join(before.split()[-12:])
    after = " ".join(after.split()[:5])
    if RELATIVE.search(before) or re.match(
        r"\s*in (?:her |his |the )?(?:mother|father|sister|brother|son|daughter)\b", after, re.I
    ):
        return "other_experiencer"
    negative = list(NEGATED.finditer(before))
    positive = list(AFFIRMED.finditer(before))
    if negative and (not positive or negative[-1].start() > positive[-1].start()):
        # 'No change in creatinine' does not assert absence of creatinine.
        if re.search(r"\bno (?:change|increase|decrease|improvement)\b", before, re.I):
            return "unknown"
        return "negated"
    if re.match(r"\s*(?:is\s+|was\s+)?(?:absent|denied|ruled out)\b", after, re.I):
        return "negated"
    if POSSIBLE.search(before):
        return "hypothetical"
    if (section in {"past medical history", "past surgical history"}
            or HISTORY.search(before) or HISTORY.search(note[start:end])):
        return "historical"
    # No exclusion cue detected. This is a heuristic proposal requiring sample review.
    return "affirmed"


def value_span(mention):
    """Return only the numerical value or sex value, never the whole sentence."""
    span, concept = mention["span"], mention["concept"]
    if concept in {"systolic_bp", "diastolic_bp"}:
        match = strict.BP.fullmatch(span)
        if match:
            return match.span("sbp" if concept == "systolic_bp" else "dbp")
    match = strict.MEASURE.fullmatch(span) if concept in strict.ALIASES else None
    if match:
        return match.span("value")
    if concept == "gender":
        match = strict.SEX.fullmatch(span)
        if match:
            return match.span("value" if match["value"] else "adjacent")
    return None


def propose(row, mention):
    note, concept = row["discharge_note"], mention["concept"]
    candidates = json.loads(mention["structured_candidates"])
    result = dict(mention, policy_version=POLICY_VERSION, concept_overlap="unknown",
                  assertion_status="unknown", structured_feature="", mask_decision="review",
                  decision_reason="Concept is outside this masking policy.", mask_start="",
                  mask_end="", mask_span="", mask_preview="")
    result["feature_category"] = ("measurement" if concept in strict.ALIASES else
        "diagnosis" if concept == "diagnosis" else
        "demographics" if concept in {"gender", "age"} else "admission_metadata")

    if concept in strict.ALIASES:
        available = [c for c in candidates if c.get("missing") in (None, False, 0)
                     and strict.decimal(c.get("value")) is not None]
    else:
        available = [c for c in candidates if c.get("value") is not None
                     and str(c["value"]).strip() not in {"", "NULL", "null"}]
    result["concept_overlap"] = "available" if available else "unavailable"
    if available:
        result["structured_feature"] = available[0]["feature"]
    else:
        result.update(mask_decision="keep", decision_reason="No nonmissing structured counterpart.")
        return result

    if concept == "gender":
        # Only explicit patient headers, not an age/sex phrase about another person.
        header = re.search(r"\b(?:sex|gender)[ \t]*:", mention["span"], re.I)
        if not header or mention["suggested_label"] != "redundant":
            result["decision_reason"] = "Sex value disagrees or is not an explicit patient header."
            return result
        result["assertion_status"] = "patient_header"
    elif concept in strict.ALIASES or concept == "diagnosis":
        status = assertion_status(note, mention["start"], mention["end"])
        result["assertion_status"] = status
        if status != "affirmed":
            result.update(mask_decision="review" if status == "unknown" else "keep",
                          decision_reason=f"Context flagged {status}; retain text.")
            return result
        if concept == "diagnosis" and not any(
            c["feature"].endswith((".diagnosis", ".description", ".long_title")) for c in available
        ):
            result["decision_reason"] = "Code-only literal hit needs review."
            return result
    else:
        return result

    offsets = (0, len(mention["span"])) if concept == "diagnosis" else value_span(mention)
    if offsets is None:
        result["decision_reason"] = "Unable to isolate an unambiguous masking span."
        return result
    start, end = mention["start"] + offsets[0], mention["start"] + offsets[1]
    result.update(mask_decision="mask", mask_start=start, mask_end=end, mask_span=note[start:end],
                  decision_reason=("Literal diagnosis overlap; no exclusion cue detected." if concept == "diagnosis"
                    else "Matching patient sex header." if concept == "gender" else
                    "Available structured concept; mask number without requiring event equality."))
    result["mask_preview"] = (note[mention["context_start"]:start] + "[MASK]"
                              + note[end:mention["context_end"]])
    return result


def run_cli(proposal=propose, policy_version=POLICY_VERSION,
            default_out="data/redundancy_concept_v1", description=__doc__, limitations=None):
    """Use the same input and output format for both masking policies."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sample", type=Path, default=Path("redundancy_audit_v3/sample.jsonl"))
    parser.add_argument("--events", type=Path, help="Optional raw-event evidence for strict labels")
    parser.add_argument("--out", type=Path, default=Path(default_out))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output folder exists; choose a new --out folder")
    sample = [json.loads(line) for line in args.sample.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sample:
        parser.error("Sample is empty")
    events = strict.load_events(args.events, sample) if args.events else {}
    results = [proposal(row, mention) for row in sample for mention in strict.audit_row(
        row, events.get((str(row["subject_id"]), str(row["hadm_id"])), ()))]
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "mentions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTRA_FIELDS + strict.FIELDS)
        writer.writeheader()
        writer.writerows(results)
    summary = dict(policy_version=policy_version, notes=len(sample), mentions=len(results),
        notes_with_mentions=len({r['note_id'] for r in results}),
        notes_with_mask_proposals=len({r['note_id'] for r in results if r['mask_decision'] == 'mask'}),
        decisions=dict(Counter(r['mask_decision'] for r in results)),
        strict_labels=dict(Counter(r['suggested_label'] for r in results)),
        by_concept={concept: dict(Counter(r['mask_decision'] for r in results if r['concept'] == concept))
                    for concept in sorted({r['concept'] for r in results})},
        sample=str(args.sample), sample_sha256=hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        events=str(args.events) if args.events else None,
        limitations=limitations or ["Draft heuristic policy; human review not completed.",
            "Concept overlap does not establish same-event redundancy or absence of useful context.",
            "Literal diagnoses only; context flags are incomplete, not clinical NLP.",
            "Review and keep decisions both retain text. Overlapping mask spans must be merged downstream.",
            "This exports proposals and previews, not filtered full notes."])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Exported {len(results)} proposals from {len(sample)} notes to {args.out}")
    print(json.dumps(summary["decisions"]))


if __name__ == "__main__":
    run_cli()
