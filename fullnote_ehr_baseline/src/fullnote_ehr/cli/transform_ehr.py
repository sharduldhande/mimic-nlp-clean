"""Apply a saved EHR builder to a dataset or one fixed split for inference features."""

from __future__ import annotations

import argparse
from pathlib import Path

from scipy import sparse

from fullnote_ehr.data import load_frame
from fullnote_ehr.ehr_features import EHRFeatureBuilder
from fullnote_ehr.schema import EHR_INPUT_COLUMNS, IDENTITY_COLUMNS, SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--output-matrix", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    args = parser.parse_args()

    frame = load_frame(args.data_dir, columns=EHR_INPUT_COLUMNS, split=args.split)
    builder = EHRFeatureBuilder.load(args.builder)
    matrix = builder.transform(frame)
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output_matrix, matrix, compressed=True)
    frame.loc[:, list(IDENTITY_COLUMNS)].to_parquet(args.output_index, index=False)
    print(f"Saved {matrix.shape[0]} rows x {matrix.shape[1]} features")


if __name__ == "__main__":
    main()
