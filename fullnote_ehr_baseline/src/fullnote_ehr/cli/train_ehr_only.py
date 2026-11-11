"""Train and evaluate the structured EHR-only XGBoost ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fullnote_ehr.ehr_only import EHROnlyTrainingConfig, train_ehr_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ehr-dir", type=Path, default=Path("artifacts/ehr"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models/ehr_only_xgboost"),
    )
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-boost-rounds", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7650)
    parser.add_argument("--nthread", type=int, default=0)
    args = parser.parse_args()

    config = EHROnlyTrainingConfig(
        learning_rate=args.learning_rate,
        max_boost_rounds=args.max_boost_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        max_bin=args.max_bin,
        seed=args.seed,
        device=args.device,
        nthread=args.nthread,
    )
    result = train_ehr_only(
        ehr_dir=args.ehr_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
