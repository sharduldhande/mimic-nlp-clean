"""Create readable before/after examples with exact removed spans highlighted."""
import argparse
import csv
import glob
from html import escape
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

import audit_redundancy as strict
import audit_redundancy_concept as concept
from generate_concept_dataset import filter_note, merge_intervals


def choose_notes(paths, count, seed, split):
    eligible = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=["note_id", "split"]):
            eligible.extend(str(r["note_id"]) for r in batch.to_pylist() if r["split"] == split)
    if len(set(eligible)) != len(eligible):
        raise ValueError("Duplicate note IDs in source parts")
    if len(eligible) < count:
        raise ValueError("Not enough notes in requested split")
    return random.Random(seed).sample(sorted(eligible), count)


def selected_rows(paths, ids, original=False):
    found = {}
    for path in paths:
        reader = pq.ParquetFile(path)
        columns = strict.COLUMNS if original else ["subject_id", "hadm_id", "note_id", "split", "discharge_note"]
        for batch in reader.iter_batches(columns=[c for c in columns if c in reader.schema_arrow.names], batch_size=256):
            for row in batch.to_pylist():
                if str(row["note_id"]) in ids:
                    if str(row["note_id"]) in found:
                        raise ValueError("Duplicate selected note")
                    found[str(row["note_id"])] = row
    if set(found) != set(ids):
        raise ValueError("Selected notes are missing from an export")
    return found


def highlighted(text, intervals):
    """Render original deletions and filtered space replacements from the same offsets."""
    before, after, cursor = [], [], 0
    for start, end in intervals:
        shared = escape(text[cursor:start])
        before.extend((shared, '<mark class="removed">', escape(text[start:end]), '</mark>'))
        after.extend((shared, '<span class="replacement" title="One replacement space"> </span>'))
        cursor = end
    before.append(escape(text[cursor:]))
    after.append(escape(text[cursor:]))
    return "".join(before), "".join(after)


def fenced(text):
    fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}\n"


def structured_html(row):
    columns = ["feature", "last_value", "min_value", "max_value", "mean_value", "unit", "is_missing"]
    rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(m.get(c, "")))}</td>' for c in columns) + '</tr>'
                   for m in (row.get("measurements") or []))
    diagnoses = ''.join(f'<li>{escape(str(d.get("diagnosis", "")))} '
                        f'({escape(str(d.get("icd_code", "")))})</li>' for d in (row.get("diagnoses") or []))
    return ('<details><summary>Structured EHR reference</summary>'
            f'<p>Age: {escape(str(row.get("age")))}; gender: {escape(str(row.get("gender")))}</p>'
            '<div class="scroll"><table><tr>' + ''.join(f'<th>{c}</th>' for c in columns)
            + '</tr>' + rows + '</table></div><ul>' + diagnoses + '</ul></details>')


def build_report(original_paths, filtered_paths, output, count=10, seed=42, split="train"):
    if output.exists():
        raise ValueError("Review folder already exists; choose a new --out folder")
    ids = choose_notes(original_paths, count, seed, split)
    originals = selected_rows(original_paths, set(ids), original=True)
    filtered = selected_rows(filtered_paths, set(ids))
    sections, markdown, spans, metadata = [], [], [], []
    for number, note_id in enumerate(ids, 1):
        row, result = originals[note_id], filtered[note_id]
        for key in ("subject_id", "hadm_id", "split"):
            if row[key] != result[key]:
                raise ValueError("Source/filtered identity mismatch")
        text = row["discharge_note"]
        proposals = [concept.propose(row, m) for m in strict.audit_row(row)]
        masks = [(p["mask_start"], p["mask_end"], p["mask_span"]) for p in proposals if p["mask_decision"] == "mask"]
        intervals = merge_intervals(text, masks)
        expected, _, _ = filter_note(text, masks)
        if expected != result["discharge_note"]:
            raise ValueError(f"Current policy does not reproduce saved filtered note: {note_id}")
        before_html, after_html = highlighted(text, intervals)
        title = f'{number}. {note_id} | subject {row["subject_id"]} | admission {row["hadm_id"]}'
        stats = f'{len(text):,} → {len(expected):,} characters; {len(intervals)} removed spans; split: {split}'
        removed_rows = []
        marked, cursor = [], 0
        for i, (start, end) in enumerate(intervals, 1):
            related = [p for p in proposals if p["mask_decision"] == "mask" and p["mask_start"] < end and p["mask_end"] > start]
            names = ', '.join(sorted({p["concept"] for p in related}))
            reasons = '; '.join(sorted({p["decision_reason"] for p in related}))
            spans.append(dict(example=number, note_id=note_id, subject_id=row["subject_id"], hadm_id=row["hadm_id"],
                split=split, span_number=i, start=start, end=end, removed_text=text[start:end], concepts=names,
                reason=reasons, structured_features='; '.join(sorted({p["structured_feature"] for p in related}))))
            removed_rows.append(f'<tr><td>{i}</td><td>{escape(names)}</td><td><code>{escape(text[start:end])}</code></td>'
                                f'<td>{start}–{end}</td><td>{escape(reasons)}</td></tr>')
            marked.extend((text[cursor:start], '⟦REMOVED: ' + text[start:end] + '⟧'))
            cursor = end
        marked.append(text[cursor:])
        sections.append(f'<details class="example" id="note-{number}" {"open" if number == 1 else ""}>'
            f'<summary>{escape(title)}<small>{escape(stats)}</small></summary>'
            '<div class="columns"><section><h3>Original — red marks removed text</h3><pre>' + before_html
            + '</pre></section><section><h3>Filtered — blue marks replacement spaces</h3><pre>' + after_html
            + '</pre></section></div><details><summary>Removed-span list</summary><div class="scroll"><table>'
            '<tr><th>#</th><th>Concept</th><th>Removed text</th><th>Offsets</th><th>Reason</th></tr>'
            + ''.join(removed_rows) + '</table></div></details>' + structured_html(row) + '</details>')
        markdown.append(f'## {title}\n\n{stats}\n\n### Original with removals marked\n\n'
                        + fenced(''.join(marked)) + '\n### Actual filtered note\n\n' + fenced(expected))
        metadata.append(dict(example=number, note_id=note_id, subject_id=row["subject_id"], hadm_id=row["hadm_id"],
            split=split, original_characters=len(text), filtered_characters=len(expected), removed_spans=len(intervals)))

    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Concept masking: before and after</title><style>
body{font:16px/1.5 system-ui,sans-serif;margin:24px;color:#20242c;background:#f6f7f9}h1{margin-bottom:4px}
.example{background:white;border:1px solid #cbd2dd;border-radius:8px;padding:16px;margin:18px 0}
summary{cursor:pointer;font-weight:600}small{display:block;color:#555;font-weight:400}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 ui-monospace,Consolas,monospace;border:1px solid #ddd;padding:12px;max-height:75vh;overflow:auto}
.removed{background:#ffe0df;color:#861b15;border-bottom:2px solid #c43c32}.replacement{background:#dcecff}.replacement:before{content:'␠';color:#1259a5}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:7px;text-align:left;vertical-align:top}code{white-space:pre-wrap}
.scroll{overflow:auto}details details{margin-top:14px}button{padding:7px 12px;margin-right:8px;cursor:pointer}
@media(max-width:850px){.columns{grid-template-columns:1fr}body{margin:12px}}
</style><h1>Concept masking: before and after</h1>'''
    html += (f'<p>{count} random {escape(split)} notes; seed {seed}; policy {concept.POLICY_VERSION}. '
             'Red = original text removed. Blue ␠ = one replacement space (the symbol is display-only). '
             'Each saved filtered note was reproduced exactly. Context decisions are heuristic proposals.</p>'
             '<button onclick="document.querySelectorAll(\'.example\').forEach(e=>e.open=true)">Expand all notes</button>'
             '<button onclick="document.querySelectorAll(\'.example\').forEach(e=>e.open=false)">Collapse all notes</button>')
    html += ''.join(sections) + '</html>'
    output.mkdir(parents=True)
    (output / "before_after.html").write_text(html, encoding="utf-8")
    (output / "before_after.md").write_text('# Before/after concept masking\n\n'
        f'Random {split} sample; seed {seed}. Original removals are marked ⟦REMOVED: …⟧.\n\n'
        + '\n'.join(markdown), encoding="utf-8")
    with (output / "removed_spans.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(spans[0]) if spans else ["example", "note_id", "removed_text"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(spans)
    (output / "sample.json").write_text(json.dumps(dict(seed=seed, split=split, policy_version=concept.POLICY_VERSION,
        verified_against_saved_filtered_notes=True, notes=metadata), indent=2), encoding="utf-8")
    print(f'Created {count} before/after examples and {len(spans)} removed spans in {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--out", type=Path, default=Path("data/concept_masked/review_examples"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    build_report([Path(p) for p in sorted(glob.glob("data/cleaned/dataset_splits_*.parquet"))],
        [Path(p) for p in sorted(glob.glob("data/concept_masked/dataset_splits_concept_mask_*.parquet"))],
        args.out, args.count, args.seed, args.split)


if __name__ == "__main__":
    main()
