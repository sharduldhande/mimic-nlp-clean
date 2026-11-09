"""Build all local datasets and reports from the one-day BigQuery CSV export."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import audit_redundancy as strict
import audit_redundancy_concept as concept
from generate_concept_dataset import digest, generate
from review_masked_notes import build_report
from summarize_dataset import summarize

csv.field_size_limit(100_000_000)
IDS = {"subject_id", "hadm_id", "stay_id", "age", "diagnosis_count", "mortality_30d"}
TIMES = {"admittime", "dischtime", "note_charttime", "note_storetime"}
NAMES = ["subject_id", "hadm_id", "admittime", "dischtime", "stay_id", "gender", "age",
         "admission_type", "admission_location", "insurance", "length_of_stay", "diagnosis_count",
         "mortality_30d", "note_id", "note_charttime", "note_storetime", "discharge_note", "split",
         "diagnoses", "measurements"]
DIAGNOSIS = pa.struct([("icd_code", pa.string()), ("icd_version", pa.int64()), ("diagnosis", pa.string())])
MEASUREMENT = pa.struct([("feature", pa.string()), ("mean_value", pa.float64()),
    ("min_value", pa.float64()), ("max_value", pa.float64()), ("last_value", pa.float64()),
    ("last_time", pa.timestamp("us")), ("unit", pa.string()), ("is_missing", pa.int64())])


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def missing(value):
    return value is None or (isinstance(value, str) and value.strip() in {"", "null", "NULL"})


def timestamp(value):
    if missing(value):
        return None
    dt = datetime.fromisoformat(str(value).replace(" UTC", "+00:00").replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def schema_for(columns):
    if len(columns) != len(NAMES) or set(columns) != set(NAMES):
        raise ValueError(f"Expected the 20 dataset columns. Missing: {set(NAMES)-set(columns)}; extra: {set(columns)-set(NAMES)}")
    fields = []
    for name in columns:
        if name == "diagnoses":
            fields.append(pa.field(name, pa.list_(pa.field("element", DIAGNOSIS, nullable=False)), nullable=False))
        elif name == "measurements":
            fields.append(pa.field(name, pa.list_(pa.field("element", MEASUREMENT, nullable=False)), nullable=False))
        else:
            kind = pa.int64() if name in IDS else pa.timestamp("us") if name in TIMES else pa.float64() if name == "length_of_stay" else pa.string()
            fields.append(pa.field(name, kind))
    return pa.schema(fields)


def typed_row(row):
    result = dict(row)
    for name in IDS:
        result[name] = None if missing(row[name]) else int(row[name])
    for name in TIMES:
        result[name] = timestamp(row[name])
    result["length_of_stay"] = None if missing(row["length_of_stay"]) else float(row["length_of_stay"])
    for name in ("gender", "admission_type", "admission_location", "insurance"):
        result[name] = None if missing(row[name]) else row[name]
    for name in ("diagnoses", "measurements"):
        items = [] if missing(row[name]) else json.loads(row[name]) if isinstance(row[name], str) else row[name]
        items = items or []
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError(f"Invalid {name} JSON array")
        result[name] = [dict(item) for item in items]
    for item in result["diagnoses"]:
        item["icd_version"] = None if missing(item.get("icd_version")) else int(item["icd_version"])
    for item in result["measurements"]:
        item["last_time"] = timestamp(item.get("last_time"))
        item["is_missing"] = None if missing(item.get("is_missing")) else int(item["is_missing"])
        for field in ("mean_value", "min_value", "max_value", "last_value"):
            item[field] = None if missing(item.get(field)) else float(item[field])
    return result


def validate_source(path):
    keys, admissions, notes, patients = set(), set(), set(), defaultdict(set)
    splits, labels = Counter(), Counter()
    count = 0
    for row in csv_rows(path):
        if not count:
            schema_for(list(row))
        parsed = typed_row(row)
        if any(missing(row[k]) for k in ("subject_id", "hadm_id", "note_id", "split")):
            raise ValueError("Missing identifier or split")
        key = (parsed["subject_id"], parsed["hadm_id"])
        if key in keys or parsed["hadm_id"] in admissions or row["note_id"] in notes:
            raise ValueError("Duplicate admission key or note ID")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")
        if parsed["mortality_30d"] not in (0, 1):
            raise ValueError("Missing/invalid binary mortality label")
        if not row["discharge_note"].strip():
            raise ValueError("Missing note text")
        admit, discharge, stored = (parsed[k] for k in ("admittime", "dischtime", "note_storetime"))
        if admit is None or discharge is None or stored is None or discharge < admit:
            raise ValueError("Missing/invalid admission or note-store timestamp")
        if (stored - discharge).total_seconds() > 86400:
            raise ValueError("Note exceeds the one-day cutoff. Check that this is the new one-day export.")
        keys.add(key)
        admissions.add(parsed["hadm_id"])
        notes.add(row["note_id"])
        patients[parsed["subject_id"]].add(row["split"])
        splits[row["split"]] += 1
        labels[parsed["mortality_30d"]] += 1
        count += 1
    if not count or not splits["train"]:
        raise ValueError("A nonempty dataset with training admissions is required")
    if any(len(values) != 1 for values in patients.values()):
        raise ValueError("Patient appears in multiple splits")
    return dict(rows=count, patients=len(patients), splits=dict(splits), positive_labels=labels[1],
                mortality_percent=100 * labels[1] / count, note_lag_limit_days=1,
                uniqueness_and_split_checks="PASS", timestamps_and_labels="PASS",
                source_note_label_linkage="Not independently verified by this local pipeline")


def make_parquet_parts(source, output, count):
    output.mkdir(parents=True)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        schema = schema_for(reader.fieldnames)
        paths = [output / f"dataset_splits_{i:012d}.parquet" for i in range(2)]
        writers = [pq.ParquetWriter(p, schema, compression="snappy") for p in paths]
        buffers, written = [[], []], [0, 0]
        try:
            for number, row in enumerate(reader):
                part = 0 if number < (count + 1) // 2 else 1
                buffers[part].append(typed_row(row))
                if len(buffers[part]) >= 256:
                    writers[part].write_table(pa.Table.from_pylist(buffers[part], schema=schema))
                    written[part] += len(buffers[part])
                    buffers[part].clear()
            for part in range(2):
                if buffers[part]:
                    writers[part].write_table(pa.Table.from_pylist(buffers[part], schema=schema))
                    written[part] += len(buffers[part])
        finally:
            for writer in writers:
                writer.close()
    if sum(written) != count:
        raise ValueError("CSV changed while creating Parquet parts")
    return paths


def fingerprint(row):
    return hashlib.sha256(json.dumps(row, default=str, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def verify_exports(source, filtered, original_parts, filtered_parts, expected_rows):
    expected_original, expected_filtered = {}, {}
    with source.open(newline="", encoding="utf-8-sig") as a, filtered.open(newline="", encoding="utf-8") as b:
        left, right = csv.DictReader(a), csv.DictReader(b)
        if left.fieldnames != right.fieldnames:
            raise ValueError("Filtered column names/order differ")
        for old, new in zip_longest(left, right):
            if old is None or new is None or any(old[k] != new[k] for k in old if k != "discharge_note"):
                raise ValueError("Filtered CSV changed non-note values or row order")
            expected_original[old["note_id"]] = fingerprint(typed_row(old))
            expected_filtered[new["note_id"]] = fingerprint(typed_row(new))
    if len(expected_original) != expected_rows or len(expected_filtered) != expected_rows:
        raise ValueError("Wrong output admission count")
    for paths, expected in ((original_parts, expected_original), (filtered_parts, expected_filtered)):
        seen = set()
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    nid = str(row["note_id"])
                    if nid in seen or fingerprint(row) != expected.get(nid):
                        raise ValueError(f"CSV/Parquet disagreement: {nid}")
                    seen.add(nid)
        if seen != set(expected):
            raise ValueError("Parquet admission coverage differs")
    for a, b in zip(original_parts, filtered_parts):
        schema = pq.ParquetFile(a).schema_arrow
        if not schema.equals(pq.ParquetFile(b).schema_arrow, check_metadata=True):
            raise ValueError("Filtered Parquet schema changed")
        columns = [n for n in schema.names if n != "discharge_note"]
        if not pq.read_table(a, columns=columns).equals(pq.read_table(b, columns=columns)):
            raise ValueError("Filtered Parquet changed non-note values/order")


def checked_path(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or (path.exists() and getattr(path.lstat(), "st_file_attributes", 0)
                             & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
        raise ValueError(f"Refusing linked output path: {path}")
    resolved = path.resolve()
    if resolved.parent != root:
        raise ValueError(f"Output path outside intended directory: {path}")
    return resolved


def publish(changes):
    """Keep prior generated outputs until every replacement succeeds."""
    completed = []
    try:
        for source, target, root in changes:
            if not Path(source).resolve().is_relative_to(Path(root).resolve().parent):
                raise ValueError("Staged output is outside the intended workspace")
            target = checked_path(target, root)
            backup = checked_path(target.with_name(f".previous_{target.name}_{uuid.uuid4().hex}"), root)
            if target.exists():
                target.rename(backup)
            try:
                Path(source).rename(target)
                # Windows temporary folders restrict access; inherit the final folder's permissions.
                if os.name == "nt":
                    subprocess.run(["icacls.exe", str(target), "/reset", "/T", "/Q"],
                                   check=True, capture_output=True, text=True)
            except Exception:
                if target.exists():
                    target.rename(source)
                if backup.exists():
                    backup.rename(target)
                raise
            completed.append((Path(source), target, backup, root))
    except Exception:
        for source, target, backup, root in reversed(completed):
            checked_path(target, root).rename(source)
            if backup.exists():
                checked_path(backup, root).rename(target)
        raise
    for _, _, backup, root in completed:
        if backup.exists():
            checked_path(backup, root)
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()


def prepare(root, source=None, workers=4):
    root = Path(root).resolve()
    data, validation = root / "data", root / "validation_output"
    cleaned = data / "cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    if source is None:
        candidates = [p for p in (cleaned / "dataset_splits.csv", cleaned / "datasets_splits.csv") if p.is_file()]
        if len(candidates) != 1:
            raise ValueError("Place exactly one dataset_splits.csv (or datasets_splits.csv) in data/cleaned")
        source = candidates[0]
    source = Path(source).resolve()
    source_hash = digest(source)
    print("1/6 Checking input CSV...", flush=True)
    input_report = validate_source(source)
    print(f"Input: {input_report['rows']:,} admissions; {input_report['positive_labels']:,} positives; "
          f"{input_report['mortality_percent']:.4f}% mortality", flush=True)
    with tempfile.TemporaryDirectory(prefix=".dataset_build_", dir=data) as folder:
        stage = Path(folder)
        print("2/6 Creating two typed baseline Parquet parts...", flush=True)
        parts = make_parquet_parts(source, stage / "cleaned", input_report["rows"])
        print("3/6 Filtering all notes and creating matching CSV/Parquet...", flush=True)
        generate(source, stage / "concept_masked", workers=workers, parquet_parts=parts)
        filtered_parts = sorted((stage / "concept_masked").glob("dataset_splits_concept_mask_*.parquet"))
        filtered_csv = stage / "concept_masked/dataset_splits_concept_mask.csv"
        print("4/6 Regenerating four validation report sets...", flush=True)
        report_inputs = {"cleaned_csv": [source], "cleaned_parquet": parts,
                         "concept_masked_csv": [filtered_csv], "concept_masked_parquet": filtered_parts}
        for name, paths in report_inputs.items():
            summarize(paths, stage / "validation" / name)
        for prefix in ("cleaned", "concept_masked"):
            for name, keys in (("note_lengths", ["scope", "split"]),
                               ("mortality_prevalence", ["scope", "split"]),
                               ("feature_missingness", ["scope", "split", "feature"])):
                a = pd.read_csv(stage / "validation" / f"{prefix}_csv" / f"{name}.csv").sort_values(keys).reset_index(drop=True)
                b = pd.read_csv(stage / "validation" / f"{prefix}_parquet" / f"{name}.csv").sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(a, b)
        print("5/6 Creating current-cohort sample audit and 10 review examples...", flush=True)
        sample = strict.sample_rows([pq.ParquetFile(p) for p in parts],
                                   min(75, input_report["splits"]["train"]), 42, "train")
        sample_out = stage / "audit_samples/concept"
        sample_report = concept.write_audit(iter(sample), sample_out)
        sample_report.update(mode="sample", seed=42, split="train", events=None)
        (sample_out / "summary.json").write_text(json.dumps(sample_report, indent=2), encoding="utf-8")
        with (sample_out / "sample.jsonl").open("w", encoding="utf-8") as handle:
            for row in sample:
                handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        build_report(parts, filtered_parts, stage / "concept_masked/review_examples",
                     min(10, input_report["splits"]["train"]), 42, "train")
        print("6/6 Verifying every row and publishing completed outputs...", flush=True)
        verify_exports(source, filtered_csv, parts, filtered_parts, input_report["rows"])
        if digest(source) != source_hash:
            raise ValueError("Input CSV changed during processing; refusing to publish")
        masked_report = stage / "concept_masked/summary.json"
        info = json.loads(masked_report.read_text())
        info.update(source=str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
                    verification="All rows checked: CSV/Parquet agree; only note values changed")
        masked_report.write_text(json.dumps(info, indent=2), encoding="utf-8")
        input_report.update(source=str(source), source_sha256=source_hash, policy_version=concept.POLICY_VERSION,
                            csv_parquet_agreement="PASS", filtered_non_note_preservation="PASS",
                            verified_event_masking="Deferred; no raw event export is required",
                            completed_at_utc=datetime.now(timezone.utc).isoformat())
        manifest = stage / "pipeline_validation.json"
        manifest.write_text(json.dumps(input_report, indent=2), encoding="utf-8")
        changes = [(p, cleaned / p.name, cleaned) for p in parts]
        changes += [(stage / n, data / n, data) for n in ("concept_masked", "audit_samples")]
        changes += [(stage / "validation" / n, validation / n, validation) for n in report_inputs]
        changes.append((manifest, validation / "pipeline_validation.json", validation))
        publish(changes)
    print("DONE: data/ and validation_output/ rebuilt. Input CSV preserved.", flush=True)
    return input_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parent, workers=args.workers)


if __name__ == "__main__":
    main()
