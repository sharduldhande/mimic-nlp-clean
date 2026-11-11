"""Run a saved EHR-only XGBoost model on a prepared dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fullnote_ehr.ehr_only import predict_ehr_only_split
from fullnote_ehr.schema import SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ehr-dir", type=Path, default=Path("artifacts/ehr"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models/ehr_only_xgboost"),
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=Path("artifacts/inference/ehr_only_test.parquet"),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("artifacts/inference/ehr_only_test_metrics.json"),
    )
    args = parser.parse_args()

    predictions, metrics = predict_ehr_only_split(
        ehr_dir=args.ehr_dir,
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
