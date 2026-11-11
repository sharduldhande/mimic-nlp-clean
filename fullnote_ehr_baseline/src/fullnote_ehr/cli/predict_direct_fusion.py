"""Run the saved direct-fusion model on a prepared dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fullnote_ehr.direct_fusion import predict_direct_fusion_split
from fullnote_ehr.schema import SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ehr-dir", type=Path, default=Path("artifacts/ehr"))
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("artifacts/embeddings/fullnote_bioclinicalbert"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models/direct_fusion_xgboost"),
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=Path("artifacts/inference/direct_fusion_test.parquet"),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("artifacts/inference/direct_fusion_test_metrics.json"),
    )
    args = parser.parse_args()

    predictions, metrics = predict_direct_fusion_split(
        ehr_dir=args.ehr_dir,
        embedding_dir=args.embedding_dir,
        model_dir=args.model_dir,
        split=args.split,
    )
    for path in (args.output_predictions, args.output_metrics):
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_predictions, index=False)
    args.output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
