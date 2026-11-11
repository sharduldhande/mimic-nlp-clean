"""Generate the shared frozen BioClinicalBERT full-note embedding cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fullnote_ehr.data import validate_dataset
from fullnote_ehr.embeddings import (
    DEFAULT_MODEL,
    TorchNoteEncoder,
    generate_embedding_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data_cleaned"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/embeddings/fullnote_bioclinicalbert"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--chunk-batch-size", type=int, default=16)
    parser.add_argument("--note-block-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-contract-validation", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_contract_validation:
        validate_dataset(
            args.data_dir,
            report_path=args.output_dir / "dataset_contract.json",
        )
    encoder = TorchNoteEncoder.from_pretrained(
        model_name_or_path=args.model,
        revision=args.revision,
        device=args.device,
        max_length=args.max_length,
        chunk_batch_size=args.chunk_batch_size,
        l2_normalize=not args.no_l2_normalize,
        use_amp=not args.no_amp,
        local_files_only=args.local_files_only,
    )
    manifest = generate_embedding_artifact(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        encoder=encoder,
        note_block_size=args.note_block_size,
        resume=not args.no_resume,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
