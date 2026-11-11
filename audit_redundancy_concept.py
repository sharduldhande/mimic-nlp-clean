"""Export concept-based masking proposals for a sample or full Parquet dataset."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
import hashlib
from itertools import islice
import json
import re
from pathlib import Path

import audit_redundancy as strict

POLICY_VERSION = "concept-v2"
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


def propose(row, mention, mask_scope="mention"):
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
    if mask_scope == "mention" and concept in strict.ALIASES:
        match = strict.MEASURE.fullmatch(mention["span"])
        if match:
            # Keep an inline timestamp preceding the clinical mention.
            offsets = (match.start("name"), match.end())
        else:
            bp = strict.BP.fullmatch(mention["span"])
            indexed = strict.measurements(row.get("measurements"))
            both_available = all(any(c.get("missing") in (None, False, 0)
                and strict.decimal(c.get("value")) is not None for c in indexed.get(name, []))
                for name in ("systolic_bp", "diastolic_bp"))
            if bp and both_available:
                offsets = (re.search(r"\b(?:BP|blood pressure)\b", mention["span"], re.I).start(), bp.end())
            # With only one BP component available, retain the other component.
    start, end = mention["start"] + offsets[0], mention["start"] + offsets[1]
    result.update(mask_decision="mask", mask_start=start, mask_end=end, mask_span=note[start:end],
                  decision_reason=("Literal diagnosis overlap; no exclusion cue detected." if concept == "diagnosis"
                    else "Matching patient sex header." if concept == "gender" else
                    "Available structured concept; mask mention without requiring event equality."
                    if mask_scope == "mention" else
                    "Available structured concept; mask number without requiring event equality."))
    result["mask_preview"] = (note[mention["context_start"]:start] + "[MASK]"
                              + note[end:mention["context_end"]])
    return result


def parquet_rows(paths, split="all"):
    """Process every admission, including notes with no extracted mentions."""
    import pyarrow.parquet as pq
    required = {"subject_id", "hadm_id", "note_id", "split", "discharge_note", "measurements", "diagnoses"}
    for path in paths:
        source = pq.ParquetFile(path)
        if missing := required.difference(source.schema_arrow.names):
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        columns = [c for c in strict.COLUMNS if c in source.schema_arrow.names]
        for batch in source.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                if split == "all" or str(row["split"]).casefold() == split.casefold():
                    yield row


def process_note(job):
    row, events, proposal = job
    return row, bool(events), [proposal(row, m) for m in strict.audit_row(row, events)]


def annotated_rows(rows, events, proposal, workers):
    def jobs(batch):
        return ((r, events.get((str(r["subject_id"]), str(r["hadm_id"])), ()), proposal) for r in batch)
    if workers == 1:
        yield from map(process_note, jobs(rows))
        return
    # Bound the in-flight notes; preserve input order regardless of worker completion.
    rows = iter(rows)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while batch := list(islice(rows, workers * 8)):
            yield from executor.map(process_note, jobs(batch), chunksize=1)


def write_audit(rows, output, proposal=propose, policy_version=POLICY_VERSION, events=None,
                require_event_coverage=False, workers=1):
    """Write incrementally; publish completed filenames only after success."""
    output.mkdir(parents=True, exist_ok=False)
    events = events or {}
    decisions, labels, concepts = Counter(), Counter(), defaultdict(Counter)
    splits, category_counts = defaultdict(Counter), Counter()
    admissions, note_ids = set(), set()
    notes = notes_with_mentions = notes_with_masks = notes_with_events = total_mentions = 0
    fields = strict.IDS + ["mentions", "mask_proposals", "unique_mask_spans", "has_raw_events"]
    with (output / "mentions.csv.partial").open("w", newline="", encoding="utf-8") as handle, (
        output / "note_summary.csv.partial"
    ).open("w", newline="", encoding="utf-8") as note_handle:
        writer = csv.DictWriter(handle, fieldnames=EXTRA_FIELDS + strict.FIELDS)
        note_writer = csv.DictWriter(note_handle, fieldnames=fields)
        writer.writeheader()
        note_writer.writeheader()
        for row, has_events, results in annotated_rows(rows, events, proposal, workers):
            key = (str(row["subject_id"]), str(row["hadm_id"]))
            note_id = str(row["note_id"])
            if any(row.get(k) is None or not str(row[k]).strip() for k in ("subject_id", "hadm_id", "note_id", "split")):
                raise ValueError("Missing admission/note identifiers or split")
            if key in admissions or note_id in note_ids:
                raise ValueError(f"Duplicate admission or note: {key}, {note_id}")
            text = row.get("discharge_note")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing note text: {note_id}")
            admissions.add(key)
            note_ids.add(note_id)
            if require_event_coverage and not has_events and any(
                c["missing"] in (None, False, 0) and strict.decimal(c["value"]) is not None
                for candidates in strict.measurements(row.get("measurements")).values() for c in candidates
            ):
                raise ValueError("Events do not cover an admission with available measurements. "
                                 "Export full event coverage or explicitly use --allow-partial-events. "
                                 "Incomplete outputs retain .partial filenames.")
            count = mask_count = 0
            mask_spans = set()
            split = str(row["split"])
            for result in results:
                if result["mask_decision"] == "mask":
                    start, end = result["mask_start"], result["mask_end"]
                    if not (0 <= start < end <= len(text)) or text[start:end] != result["mask_span"]:
                        raise ValueError(f"Invalid mask offsets: {note_id}")
                    mask_count += 1
                    mask_spans.add((start, end))
                writer.writerow(result)
                count += 1
                decision = result["mask_decision"]
                decisions[decision] += 1
                labels[result["suggested_label"]] += 1
                concepts[result["concept"]][decision] += 1
                splits[split][f"mentions_{decision}"] += 1
                category_counts[(split, result["feature_category"], result["concept"], decision)] += 1
            note_writer.writerow(dict({k: row.get(k) for k in strict.IDS}, mentions=count,
                mask_proposals=mask_count, unique_mask_spans=len(mask_spans), has_raw_events=has_events))
            notes += 1
            total_mentions += count
            notes_with_mentions += bool(count)
            notes_with_masks += bool(mask_count)
            notes_with_events += has_events
            splits[split]["notes"] += 1
            splits[split]["notes_with_mentions"] += bool(count)
            splits[split]["notes_with_mask_proposals"] += bool(mask_count)
            splits[split]["notes_with_events"] += has_events
            if notes % 1000 == 0:
                print(f"Processed {notes:,} notes; {total_mentions:,} mentions", flush=True)
    if not notes:
        raise ValueError("No admissions matched the requested inputs/split")
    with (output / "category_summary.csv.partial").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "split", "feature_category", "concept", "mask_decision", "mentions"])
        overall = Counter()
        for (split, category, concept, decision), count in sorted(category_counts.items()):
            writer.writerow(["split", split, category, concept, decision, count])
            overall[(category, concept, decision)] += count
        for (category, concept, decision), count in sorted(overall.items()):
            writer.writerow(["overall", "", category, concept, decision, count])
    for name in ("mentions.csv", "note_summary.csv", "category_summary.csv"):
        (output / (name + ".partial")).rename(output / name)
    return dict(policy_version=policy_version, notes=notes, mentions=total_mentions,
        notes_with_mentions=notes_with_mentions, notes_with_mask_proposals=notes_with_masks,
        notes_with_events=notes_with_events, decisions=dict(decisions), strict_labels=dict(labels),
        by_concept={c: dict(v) for c, v in sorted(concepts.items())},
        by_split={s: dict(v) for s, v in sorted(splits.items())})


def run_cli(proposal=propose, policy_version=POLICY_VERSION,
            default_out="data/redundancy_concept_v2", description=__doc__, limitations=None):
    parser = argparse.ArgumentParser(description=description)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--sample", type=Path, help="JSONL audit sample (default: committed 75 notes)")
    inputs.add_argument("--parquet", nargs="+", help="Full Parquet parts; quoted wildcards supported")
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    parser.add_argument("--events", type=Path, help="Optional raw-event evidence for strict labels")
    parser.add_argument("--allow-partial-events", action="store_true",
                        help="Explicitly allow incomplete verified-event coverage in full mode")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=1, metavar="1-16",
                        help="Parallel note workers; results retain input order")
    parser.add_argument("--out", type=Path, default=Path(default_out))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output folder exists; choose a new --out folder")
    verified_full = bool(args.parquet) and policy_version.startswith("verified")
    if verified_full and not args.events and not args.allow_partial_events:
        parser.error("Full verified audit requires --events covering the full cohort; "
                     "use --allow-partial-events only for an explicitly incomplete audit")
    if args.parquet:
        paths = []
        for pattern in args.parquet:
            matches = sorted(glob.glob(pattern))
            if not matches:
                parser.error(f"No files match: {pattern}")
            paths.extend(Path(p).resolve() for p in matches)
        paths = list(dict.fromkeys(paths))
        rows = lambda: parquet_rows(paths, args.split)
        provenance = dict(mode="full", inputs=[str(p) for p in paths])
    else:
        path = args.sample or Path("redundancy_audit_v3/sample.jsonl")
        sample = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = lambda: (r for r in sample if args.split == "all" or str(r["split"]).casefold() == args.split)
        provenance = dict(mode="sample", sample=str(path), sample_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    events = strict.load_events(args.events, [
        {k: r[k] for k in ("subject_id", "hadm_id")} for r in rows()
    ]) if args.events else {}
    summary = write_audit(rows(), args.out, proposal, policy_version, events,
                          require_event_coverage=verified_full and not args.allow_partial_events,
                          workers=args.workers)
    summary.update(**provenance, split=args.split, events=str(args.events) if args.events else None,
        allow_partial_events=args.allow_partial_events, workers=args.workers,
        limitations=limitations or ["Draft heuristic policy; human review not completed.",
            "Concept overlap does not establish same-event redundancy or absence of useful context.",
            "Literal diagnoses only; context flags are incomplete, not clinical NLP.",
            "Review and keep decisions both retain text. Overlapping mask spans must be merged downstream.",
            "This exports proposals and previews, not filtered full notes."])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Exported {summary['mentions']:,} proposals from {summary['notes']:,} notes to {args.out}")
    print(json.dumps(summary["decisions"]))


if __name__ == "__main__":
    run_cli()
