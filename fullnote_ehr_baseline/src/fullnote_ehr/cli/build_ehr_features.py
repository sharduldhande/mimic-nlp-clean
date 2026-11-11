"""Fit the EHR feature builder on train and save all fixed-split matrices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from fullnote_ehr.data import discover_parquet_parts, load_frame, validate_dataset
from fullnote_ehr.ehr_features import EHRFeatureBuilder
from fullnote_ehr.schema import EHR_INPUT_COLUMNS, IDENTITY_COLUMNS, SPLITS


def _prepare_output(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Choose a new directory."
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data_cleaned"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ehr"))
    parser.add_argument("--diagnosis-min-frequency", type=int, default=10)
    parser.add_argument(
        "--no-measurement-recency",
        action="store_true",
        help="Exclude hours from each last measurement to discharge.",
    )
    parser.add_argument(
        "--skip-contract-validation",
        action="store_true",
        help="Skip the full streaming contract check (not recommended).",
    )
    args = parser.parse_args()

    output = _prepare_output(args.output_dir)
    if not args.skip_contract_validation:
        validate_dataset(
            args.data_dir,
            report_path=output / "dataset_contract.json",
        )
    frame = load_frame(args.data_dir, columns=EHR_INPUT_COLUMNS)
    split_frames = {
        split: frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        for split in SPLITS
    }
    builder = EHRFeatureBuilder(
        diagnosis_min_frequency=args.diagnosis_min_frequency,
        include_measurement_recency=not args.no_measurement_recency,
    )
    builder.fit(split_frames["train"])

    matrix_shapes: dict[str, list[int]] = {}
    for split in SPLITS:
        split_frame = split_frames[split]
        matrix = builder.transform(split_frame)
        labels = split_frame["mortality_30d"].to_numpy(dtype=np.int8)
        index = split_frame.loc[:, list(IDENTITY_COLUMNS)].copy()
        sparse.save_npz(output / f"X_{split}.npz", matrix, compressed=True)
        np.save(output / f"y_{split}.npy", labels, allow_pickle=False)
        index.to_parquet(output / f"index_{split}.parquet", index=False)
        matrix_shapes[split] = [int(matrix.shape[0]), int(matrix.shape[1])]

    builder.save(output / "feature_builder.joblib")
    builder.write_feature_names(output / "feature_names.json")
    manifest = {
        "artifact_version": "ehr-matrices-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_parts": [str(path) for path in discover_parquet_parts(args.data_dir)],
        "matrix_shapes": matrix_shapes,
        "split_rows": {split: len(split_frames[split]) for split in SPLITS},
        "split_positives": {
            split: int(split_frames[split]["mortality_30d"].sum()) for split in SPLITS
        },
        "feature_builder": builder.manifest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
