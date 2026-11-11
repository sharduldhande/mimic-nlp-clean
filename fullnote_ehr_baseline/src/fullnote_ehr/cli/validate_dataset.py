"""Validate the accepted cohort without loading all notes into memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fullnote_ehr.data import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data_cleaned"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/validation/dataset_contract.json"),
    )
    args = parser.parse_args()
    report = validate_dataset(args.data_dir, report_path=args.report)
    print(json.dumps({
        "status": report["status"],
        "rows": report["rows"],
        "patients": report["patients"],
        "positive_labels": report["positive_labels"],
        "report": str(args.report.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
