"""Write cleaned exports excluding one source admission with reversed timestamps."""
import argparse
import csv
import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

csv.field_size_limit(100_000_000)
SUBJECT_ID = 13659453
HADM_ID = 23252384


def clean_csv(source, destination):
    before = removed = 0
    with source.open(newline="", encoding="utf-8-sig") as src, destination.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            before += 1
            # Match both identifiers; retain this patient's other admissions.
            if row["subject_id"] == str(SUBJECT_ID) and row["hadm_id"] == str(HADM_ID):
                removed += 1
            else:
                writer.writerow(row)
    return dict(input_rows=before, removed=removed, output_rows=before - removed)


def clean_parquet(source, destination):
    before = removed = 0
    reader = pq.ParquetFile(source)
    # Preserve the original schema, including nested diagnoses and measurements.
    with pq.ParquetWriter(destination, reader.schema_arrow, compression="snappy") as writer:
        for batch in reader.iter_batches(batch_size=256):
            table = pa.Table.from_batches([batch])
            subject = table["subject_id"]
            admission = table["hadm_id"]
            exclude = pc.fill_null(pc.and_(
                pc.equal(subject, pa.scalar(SUBJECT_ID, type=subject.type)),
                pc.equal(admission, pa.scalar(HADM_ID, type=admission.type)),
            ), False)
            kept = table.filter(pc.invert(exclude))
            before += table.num_rows
            removed += table.num_rows - kept.num_rows
            writer.write_table(kept)
    return dict(input_rows=before, removed=removed, output_rows=before - removed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="dataset_splits.csv")
    parser.add_argument("--parquet", default="dataset_splits_*.parquet")
    parser.add_argument("--output", default="data/cleaned")
    args = parser.parse_args()
    sources = [Path(args.csv).resolve()] + [Path(p).resolve() for p in sorted(glob.glob(args.parquet))]
    if not sources[0].is_file() or len(sources) < 2:
        parser.error("Both the original CSV and Parquet parts must exist")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output folder already exists; choose a new --output folder")
    output.mkdir(parents=True)
    results = {}
    for source in sources:
        destination = output / (source.name + ".partial")
        cleaner = clean_csv if source.suffix.lower() == ".csv" else clean_parquet
        results[source.name] = cleaner(source, destination)
        print(f"{source.name}: {results[source.name]}", flush=True)

    csv_result = results[sources[0].name]
    parquet_results = [results[source.name] for source in sources[1:]]
    if csv_result["removed"] != 1 or sum(r["removed"] for r in parquet_results) != 1:
        raise ValueError("Expected exactly one exclusion in each export; outputs remain .partial")
    if csv_result["output_rows"] != sum(r["output_rows"] for r in parquet_results):
        raise ValueError("CSV and Parquet row counts differ; outputs remain .partial")
    # Publish completed filenames only after the exclusion counts agree.
    for source in sources:
        (output / (source.name + ".partial")).rename(output / source.name)
    manifest = dict(excluded=dict(subject_id=SUBJECT_ID, hadm_id=HADM_ID),
                    reason="Source hospital discharge precedes admission", files=results,
                    original_files=[str(source) for source in sources], splits="Unchanged")
    (output / "exclusion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cleaned exports saved in {output}. Originals unchanged.")


if __name__ == "__main__":
    main()
