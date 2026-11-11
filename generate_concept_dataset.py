"""Write a concept-filtered CSV and small reports; preserve the source dataset."""
import argparse
from collections import Counter, defaultdict
import csv
import glob
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile

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


def merge_intervals(text, masks):
    """Validate and merge selected spans and separators between removed mentions."""
    intervals = []
    for start, end, expected in sorted(masks):
        if not (0 <= start < end <= len(text)) or text[start:end] != expected:
            raise ValueError("Mask offsets/text do not match the source note")
        if intervals and (start <= intervals[-1][1] or re.fullmatch(
            r"[ \t]*[/|,;][ \t]*", text[intervals[-1][1]:start]
        )):
            intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
        else:
            intervals.append((start, end))
    return intervals


def filter_note(text, masks):
    """Replace merged selections with spaces without changing surrounding text."""
    intervals = merge_intervals(text, masks)
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


def export_parquet(csv_path, source_parts, output):
    """Replace only notes in original Arrow tables, retaining nested types and row order."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    reports = []
    with tempfile.TemporaryDirectory(prefix=".note_index_", dir=output) as folder:
        db = sqlite3.connect(str(Path(folder) / "notes.sqlite"))
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("CREATE TABLE notes (note_id TEXT PRIMARY KEY, subject_id TEXT, hadm_id TEXT, split TEXT, text TEXT, used INTEGER DEFAULT 0)")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    db.execute("INSERT INTO notes(note_id,subject_id,hadm_id,split,text) VALUES (?,?,?,?,?)",
                        (row["note_id"], row["subject_id"], row["hadm_id"], row["split"], row["discharge_note"]))
            db.commit()
            for number, part in enumerate(source_parts):
                reader = pq.ParquetFile(part)
                name = f"dataset_splits_concept_mask_{number:012d}.parquet"
                count = 0
                with pq.ParquetWriter(output / (name + ".partial"), reader.schema_arrow, compression="snappy") as writer:
                    for batch in reader.iter_batches(batch_size=128):
                        table = pa.Table.from_batches([batch])
                        ids = [str(v) for v in table["note_id"].to_pylist()]
                        placeholders = ",".join("?" for _ in ids)
                        found = {r[0]: r[1:] for r in db.execute(
                            f"SELECT note_id,subject_id,hadm_id,split,text,used FROM notes WHERE note_id IN ({placeholders})", ids)}
                        texts = []
                        for row in table.select(["note_id", "subject_id", "hadm_id", "split"]).to_pylist():
                            item = found.get(str(row["note_id"]))
                            if item is None or item[-1] != 0 or tuple(str(row[k]) for k in ("subject_id", "hadm_id", "split")) != item[:3]:
                                raise ValueError("CSV/Parquet admission mismatch or duplicate note")
                            texts.append(item[3])
                        field = reader.schema_arrow.field("discharge_note")
                        table = table.set_column(table.schema.get_field_index("discharge_note"), field,
                                                 pa.array(texts, type=field.type))
                        writer.write_table(table)
                        db.executemany("UPDATE notes SET used=used+1 WHERE note_id=?", [(n,) for n in ids])
                        count += len(ids)
                reports.append(dict(file=name, rows=count))
            if db.execute("SELECT COUNT(*) FROM notes WHERE used != 1").fetchone()[0]:
                raise ValueError("Parquet parts do not cover the filtered CSV exactly once")
        finally:
            db.close()
    for report in reports:
        (output / (report["file"] + ".partial")).rename(output / report["file"])
        report["sha256"] = digest(output / report["file"])
    return reports


def generate(source, output, mentions=None, workers=1, parquet_parts=None):
    if output.exists():
        raise ValueError("Output folder exists; choose a new --out folder")
    source_hash = digest(source)
    index, previous = load_audit(mentions) if mentions else (None, None)
    output.mkdir(parents=True)
    counts, categories, splits = Counter(), Counter(), defaultdict(Counter)
    seen_keys, seen_notes, seen_admissions = set(), set(), set()
    note_fields = ["subject_id", "hadm_id", "note_id", "split", "mentions", "mask_proposals",
                   "merged_mask_spans", "removed_characters", "original_whitespace_tokens",
                   "filtered_whitespace_tokens", "original_characters", "filtered_characters"]
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
                original_whitespace_tokens=len(text.split()), filtered_whitespace_tokens=len(filtered.split()),
                original_characters=len(text), filtered_characters=len(filtered)))
            counts["admissions"] += 1
            counts["changed_notes"] += filtered != text
            counts["removed_characters"] += removed
            counts["merged_mask_spans"] += merged
            counts["original_characters"] += len(text)
            counts["filtered_characters"] += len(filtered)
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
    parquet_reports = export_parquet(output / filename, parquet_parts, output) if parquet_parts else []
    report = dict(policy_version=concept.POLICY_VERSION, source=str(source), source_sha256=source_hash,
        output_sha256=digest(output / filename),
        source_audit=str(mentions) if mentions else None, replacement=" ", changed_column="discharge_note",
        counts=dict(counts), by_split={k: dict(v) for k, v in sorted(splits.items())},
        parquet_parts=parquet_reports, mask_scope="whole measurement mention; sex value; diagnosis term",
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
    parser.add_argument("--parquet-source", nargs="+", default=["data/cleaned/dataset_splits_*.parquet"],
                        help="Original typed Parquet parts used to create matching filtered parts")
    args = parser.parse_args()
    parts = []
    for pattern in args.parquet_source:
        found = sorted(glob.glob(pattern))
        if not found:
            parser.error(f"No Parquet parts match: {pattern}")
        parts.extend(Path(p).resolve() for p in found)
    generate(args.source, args.out, args.mentions, args.workers, list(dict.fromkeys(parts)))


if __name__ == "__main__":
    main()
