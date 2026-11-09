"""Write a concept-filtered CSV and small reports; preserve the source dataset."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import audit_redundancy_concept as concept

csv.field_size_limit(100_000_000)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(row):
    return tuple(str(row[k]) for k in ("subject_id", "hadm_id", "note_id", "split"))


def empty_audit():
    return dict(decisions=Counter(), categories=Counter(), masks=[])


def add_result(audit, result):
    if result["policy_version"] != concept.POLICY_VERSION:
        raise ValueError("Audit policy version does not match the concept policy")
    decision = result["mask_decision"]
    if decision not in {"mask", "keep", "review"}:
        raise ValueError(f"Unknown masking decision: {decision}")
    audit["decisions"][decision] += 1
    audit["categories"][(result["feature_category"], result["concept"], decision)] += 1
    if decision == "mask":
        audit["masks"].append((int(result["mask_start"]), int(result["mask_end"]), result["mask_span"]))


def filter_note(text, masks):
    """Validate against the original, merge overlaps, then replace each union with a space."""
    intervals = []
    for start, end, expected in sorted(masks):
        if not (0 <= start < end <= len(text)) or text[start:end] != expected:
            raise ValueError("Mask offsets/text do not match the source note")
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
        else:
            intervals.append((start, end))
    pieces, cursor = [], 0
    for start, end in intervals:
        pieces.extend((text[cursor:start], " "))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), len(intervals), sum(end - start for start, end in intervals)


def load_audit(path):
    summary = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
    if summary.get("policy_version") != concept.POLICY_VERSION or summary.get("mode") != "full":
        raise ValueError("--mentions must come from a complete full-cohort concept audit")
    index = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            entry = index.setdefault(identity(row), empty_audit())
            add_result(entry, row)
    if sum(sum(v["decisions"].values()) for v in index.values()) != summary["mentions"]:
        raise ValueError("Mention CSV count differs from its audit summary")
    return index, summary


def generate(source, output, mentions=None, workers=1):
    if output.exists():
        raise ValueError("Output folder exists; choose a new --out folder")
    source_hash = digest(source)
    index, previous = load_audit(mentions) if mentions else (None, None)
    output.mkdir(parents=True)
    counts, categories, splits = Counter(), Counter(), defaultdict(Counter)
    seen_keys, seen_notes, seen_admissions = set(), set(), set()
    note_fields = ["subject_id", "hadm_id", "note_id", "split", "mentions", "mask_proposals",
                   "merged_mask_spans", "removed_characters", "original_whitespace_tokens",
                   "filtered_whitespace_tokens"]
    filename = "dataset_splits_concept_mask.csv"
    with source.open(newline="", encoding="utf-8-sig") as src, (
        output / (filename + ".partial")
    ).open("w", newline="", encoding="utf-8") as dst, (
        output / "note_summary.csv.partial"
    ).open("w", newline="", encoding="utf-8") as notes:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        note_writer = csv.DictWriter(notes, fieldnames=note_fields)
        writer.writeheader()
        note_writer.writeheader()

        def audited_rows():
            if index is not None:
                for row in reader:
                    yield row, index.get(identity(row), empty_audit())
            else:
                for row, _, results in concept.annotated_rows(reader, {}, concept.propose, workers):
                    audit = empty_audit()
                    for result in results:
                        add_result(audit, result)
                    yield row, audit

        for row, audit in audited_rows():
            key = identity(row)
            if key[2] in seen_notes or key[:2] in seen_admissions:
                raise ValueError("Duplicate admission/note in source")
            seen_keys.add(key)
            seen_notes.add(key[2])
            seen_admissions.add(key[:2])
            text = row["discharge_note"]
            filtered, merged, removed = filter_note(text, audit["masks"])
            writer.writerow(dict(row, discharge_note=filtered))
            note_writer.writerow(dict(zip(note_fields[:4], key),
                mentions=sum(audit["decisions"].values()), mask_proposals=audit["decisions"]["mask"],
                merged_mask_spans=merged, removed_characters=removed,
                original_whitespace_tokens=len(text.split()), filtered_whitespace_tokens=len(filtered.split())))
            counts["admissions"] += 1
            counts["changed_notes"] += filtered != text
            counts["removed_characters"] += removed
            counts["merged_mask_spans"] += merged
            counts.update(audit["decisions"])
            splits[key[3]]["admissions"] += 1
            splits[key[3]]["changed_notes"] += filtered != text
            splits[key[3]].update(audit["decisions"])
            for (category, name, decision), count in audit["categories"].items():
                categories[(key[3], category, name, decision)] += count
            if counts["admissions"] % 5000 == 0:
                print(f"Filtered {counts['admissions']:,} admissions", flush=True)
    if not counts["admissions"]:
        raise ValueError("Source dataset is empty")
    if previous and (counts["admissions"] != previous["notes"] or set(index) - seen_keys):
        raise ValueError("Audit admission coverage differs from the source dataset")
    with (output / "category_summary.csv.partial").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "split", "feature_category", "concept", "mask_decision", "mentions"])
        overall = Counter()
        for (split, category, name, decision), count in sorted(categories.items()):
            writer.writerow(["split", split, category, name, decision, count])
            overall[(category, name, decision)] += count
        for (category, name, decision), count in sorted(overall.items()):
            writer.writerow(["overall", "", category, name, decision, count])
    if digest(source) != source_hash:
        raise ValueError("Source file changed during processing")
    for name in (filename, "note_summary.csv", "category_summary.csv"):
        (output / (name + ".partial")).rename(output / name)
    report = dict(policy_version=concept.POLICY_VERSION, source=str(source), source_sha256=source_hash,
        output_sha256=digest(output / filename),
        source_audit=str(mentions) if mentions else None, replacement=" ", changed_column="discharge_note",
        counts=dict(counts), by_split={k: dict(v) for k, v in sorted(splits.items())},
        limitations=["Draft heuristic concept masking, not verified-event redundancy.",
                     "Keep/review mentions are retained; human review is not complete.",
                     "Only selected mask spans change; extra spaces and line breaks are preserved."])
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output / filename}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/cleaned/dataset_splits.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/concept_masked"))
    parser.add_argument("--mentions", type=Path, help="Optional existing full audit CSV, to avoid recomputing")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=4)
    args = parser.parse_args()
    generate(args.source, args.out, args.mentions, args.workers)


if __name__ == "__main__":
    main()
