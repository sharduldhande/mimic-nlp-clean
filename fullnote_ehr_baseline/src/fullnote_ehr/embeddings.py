"""Deterministic long-note embedding and failure-safe cache generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import discover_parquet_parts, load_frame
from .schema import IDENTITY_COLUMNS


DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
POOLING_DESCRIPTION = (
    "mean of non-special token hidden states across the full note; "
    "non-overlapping chunks are weighted by content-token count"
)


@dataclass(frozen=True)
class EncodedChunk:
    """One variable-length model input plus its content-token mask."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    content_mask: tuple[int, ...]

    @property
    def content_tokens(self) -> int:
        return int(sum(self.content_mask))


class NoteEncoder(Protocol):
    """Small interface that keeps artifact generation independent of PyTorch."""

    hidden_size: int

    def encode_notes(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return embeddings, content-token counts, and chunk counts."""

    def manifest(self) -> dict:
        """Return model and pooling metadata that uniquely define the vectors."""


def encode_note_chunks(tokenizer, text: str, max_length: int = 512) -> list[EncodedChunk]:
    """Tokenize once and split into non-overlapping chunks with explicit masks."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot embed an empty discharge note")
    if max_length < 4:
        raise ValueError("max_length must leave room for content and special tokens")
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not token_ids:
        raise ValueError("Tokenizer produced no content tokens for a nonempty note")
    empty_with_specials = list(tokenizer.build_inputs_with_special_tokens([]))
    content_limit = max_length - len(empty_with_specials)
    if content_limit < 1:
        raise ValueError(
            f"Tokenizer requires {len(empty_with_specials)} special tokens, exceeding max_length"
        )

    # Locate where a single-sequence template inserts content. This avoids
    # get_special_tokens_mask, whose API differs between fast and slow
    # tokenizers, and it correctly treats a literal special-token ID produced
    # from the note text as content rather than as an added wrapper token.
    special_values = set(empty_with_specials)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    probe_candidates = range(max(vocab_size, len(special_values) + 1))
    probe_id = next(
        (candidate for candidate in probe_candidates if candidate not in special_values),
        max(special_values, default=-1) + 1,
    )
    probe_sequence = list(tokenizer.build_inputs_with_special_tokens([probe_id]))
    insertion_offsets = [
        offset
        for offset, value in enumerate(probe_sequence)
        if value == probe_id
        and probe_sequence[:offset] + probe_sequence[offset + 1:] == empty_with_specials
    ]
    if len(insertion_offsets) != 1:
        raise ValueError("Tokenizer does not expose a stable single-sequence wrapper template")
    content_offset = insertion_offsets[0]

    result: list[EncodedChunk] = []
    for start in range(0, len(token_ids), content_limit):
        content = token_ids[start:start + content_limit]
        sequence = list(tokenizer.build_inputs_with_special_tokens(content))
        if len(sequence) > max_length:
            raise ValueError("Tokenizer added more special tokens than its empty template indicated")
        if (
            sequence[content_offset:content_offset + len(content)] != content
            or sequence[:content_offset] + sequence[content_offset + len(content):]
            != empty_with_specials
        ):
            raise ValueError("Tokenizer changed the single-sequence wrapper template")
        mask = [0] * len(sequence)
        mask[content_offset:content_offset + len(content)] = [1] * len(content)
        chunk = EncodedChunk(
            input_ids=tuple(int(value) for value in sequence),
            attention_mask=tuple(1 for _ in sequence),
            content_mask=tuple(mask),
        )
        if chunk.content_tokens != len(content):
            raise ValueError("Special-token mask does not preserve the expected content-token count")
        result.append(chunk)
    return result


def finalize_token_sums(
    token_sums: np.ndarray,
    token_counts: np.ndarray,
    l2_normalize: bool = True,
) -> np.ndarray:
    """Convert accumulated token sums into full-note mean-pooled vectors."""
    sums = np.asarray(token_sums, dtype=np.float32)
    counts = np.asarray(token_counts, dtype=np.int64)
    if sums.ndim != 2 or counts.shape != (sums.shape[0],):
        raise ValueError("Token sums/counts have incompatible shapes")
    if np.any(counts <= 0):
        raise ValueError("Every note must contribute at least one content token")
    embeddings = sums / counts[:, None].astype(np.float32)
    if l2_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("Cannot L2-normalize zero or non-finite note vectors")
        embeddings = embeddings / norms
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding output contains NaN or infinite values")
    return embeddings.astype(np.float32, copy=False)


class TorchNoteEncoder:
    """Frozen Hugging Face encoder with token-aware full-note pooling."""

    def __init__(
        self,
        tokenizer,
        model,
        device: str,
        max_length: int = 512,
        chunk_batch_size: int = 16,
        l2_normalize: bool = True,
        use_amp: bool = True,
        model_name_or_path: str = DEFAULT_MODEL,
        requested_revision: str | None = None,
    ) -> None:
        if chunk_batch_size < 1:
            raise ValueError("chunk_batch_size must be positive")
        import torch

        self.tokenizer = tokenizer
        self.model = model
        self.device = str(device)
        self.chunk_batch_size = int(chunk_batch_size)
        self.l2_normalize = bool(l2_normalize)
        self.use_amp = bool(use_amp and self.device.startswith("cuda"))
        self.model_name_or_path = model_name_or_path
        self.requested_revision = requested_revision
        model_limit = int(getattr(model.config, "max_position_embeddings", max_length))
        tokenizer_limit = int(getattr(tokenizer, "model_max_length", max_length))
        if tokenizer_limit > 1_000_000:
            tokenizer_limit = max_length
        self.max_length = min(int(max_length), model_limit, tokenizer_limit)
        self.hidden_size = int(getattr(model.config, "hidden_size"))
        self.resolved_revision = getattr(model.config, "_commit_hash", None)
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            raise ValueError("Tokenizer must define pad_token_id")
        self.pad_token_id = int(pad)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_MODEL,
        revision: str | None = None,
        device: str = "auto",
        max_length: int = 512,
        chunk_batch_size: int = 16,
        l2_normalize: bool = True,
        use_amp: bool = True,
        local_files_only: bool = False,
    ) -> "TorchNoteEncoder":
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as error:  # includes native-DLL loading failures
            raise RuntimeError(
                "PyTorch/Transformers could not be loaded. Install requirements-modeling.txt "
                "in a Python 3.10-3.12 environment with a working Torch build."
            ) from error
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        common = {"local_files_only": local_files_only}
        if revision is not None:
            common["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, **common)
        model = AutoModel.from_pretrained(model_name_or_path, **common)
        return cls(
            tokenizer=tokenizer,
            model=model,
            device=resolved_device,
            max_length=max_length,
            chunk_batch_size=chunk_batch_size,
            l2_normalize=l2_normalize,
            use_amp=use_amp,
            model_name_or_path=model_name_or_path,
            requested_revision=revision,
        )

    def _model_hidden(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise TypeError("Model output does not expose last_hidden_state")

    def encode_notes(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        all_chunks: list[EncodedChunk] = []
        owners: list[int] = []
        chunk_counts = np.zeros(len(texts), dtype=np.int32)
        for note_index, text in enumerate(texts):
            chunks = encode_note_chunks(self.tokenizer, text, self.max_length)
            all_chunks.extend(chunks)
            owners.extend([note_index] * len(chunks))
            chunk_counts[note_index] = len(chunks)

        token_sums = np.zeros((len(texts), self.hidden_size), dtype=np.float32)
        token_counts = np.zeros(len(texts), dtype=np.int64)
        for batch_start in range(0, len(all_chunks), self.chunk_batch_size):
            batch = all_chunks[batch_start:batch_start + self.chunk_batch_size]
            batch_owners = owners[batch_start:batch_start + self.chunk_batch_size]
            width = max(len(chunk.input_ids) for chunk in batch)
            input_ids = np.full((len(batch), width), self.pad_token_id, dtype=np.int64)
            attention = np.zeros((len(batch), width), dtype=np.int64)
            content = np.zeros((len(batch), width), dtype=np.float32)
            for index, chunk in enumerate(batch):
                length = len(chunk.input_ids)
                input_ids[index, :length] = chunk.input_ids
                attention[index, :length] = chunk.attention_mask
                content[index, :length] = chunk.content_mask

            ids_tensor = torch.as_tensor(input_ids, device=self.device)
            attention_tensor = torch.as_tensor(attention, device=self.device)
            content_tensor = torch.as_tensor(content, device=self.device)
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        hidden = self._model_hidden(ids_tensor, attention_tensor)
                else:
                    hidden = self._model_hidden(ids_tensor, attention_tensor)
                sums = (
                    hidden.float() * content_tensor.unsqueeze(-1)
                ).sum(dim=1).cpu().numpy()
                counts = content_tensor.sum(dim=1).to(dtype=torch.int64).cpu().numpy()
            for local_index, owner in enumerate(batch_owners):
                token_sums[owner] += sums[local_index]
                token_counts[owner] += int(counts[local_index])

        embeddings = finalize_token_sums(
            token_sums,
            token_counts,
            l2_normalize=self.l2_normalize,
        )
        return embeddings, token_counts, chunk_counts

    def manifest(self) -> dict:
        return {
            "backend": "huggingface-pytorch",
            "model_name_or_path": self.model_name_or_path,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "content_tokens_per_chunk": (
                self.max_length
                - len(self.tokenizer.build_inputs_with_special_tokens([]))
            ),
            "chunk_overlap": 0,
            "chunk_batch_size": self.chunk_batch_size,
            "pooling": POOLING_DESCRIPTION,
            "l2_normalize": self.l2_normalize,
            "device": self.device,
            "automatic_mixed_precision": self.use_amp,
        }


def _note_columns() -> list[str]:
    return list(IDENTITY_COLUMNS) + ["discharge_note"]


def _iter_note_rows(data_dir: str | Path, start_row: int = 0) -> Iterator[dict]:
    current = 0
    for path in discover_parquet_parts(data_dir):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=128, columns=_note_columns()):
            for row in batch.to_pylist():
                if current >= start_row:
                    yield row
                current += 1


def iter_note_blocks(
    data_dir: str | Path,
    start_row: int,
    block_size: int,
) -> Iterator[tuple[int, list[dict]]]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    block: list[dict] = []
    block_start = start_row
    for row in _iter_note_rows(data_dir, start_row=start_row):
        block.append(row)
        if len(block) == block_size:
            yield block_start, block
            block_start += len(block)
            block = []
    if block:
        yield block_start, block


def note_dataset_sha256(data_dir: str | Path) -> str:
    """Hash identity, label, split, and exact note text in deterministic row order."""
    digest = hashlib.sha256()
    for row in _iter_note_rows(data_dir):
        for key in _note_columns():
            value = row[key]
            payload = ("" if value is None else str(value)).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_embedding_artifact(
    data_dir: str | Path,
    output_dir: str | Path,
    encoder: NoteEncoder,
    note_block_size: int = 32,
    resume: bool = True,
) -> dict:
    """Generate a combined embedding cache with resumable block checkpoints."""
    if note_block_size < 1:
        raise ValueError("note_block_size must be positive")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_embeddings = output / "fullnote_embeddings.npy"
    final_index = output / "fullnote_embedding_index.parquet"
    final_manifest = output / "embedding_manifest.json"
    partial_embeddings = output / "fullnote_embeddings.partial.npy"
    progress_path = output / "embedding_progress.json"
    if final_manifest.exists() or final_embeddings.exists() or final_index.exists():
        raise FileExistsError(
            f"Completed or mixed embedding artifacts already exist in {output}. "
            "Choose a new output directory."
        )

    index = load_frame(data_dir, columns=IDENTITY_COLUMNS)
    rows = len(index)
    source_hash = note_dataset_sha256(data_dir)
    encoder_manifest = encoder.manifest()
    run_definition = {
        "artifact_version": "fullnote-embeddings-v1",
        "source_note_sha256": source_hash,
        "rows": rows,
        "hidden_size": int(encoder.hidden_size),
        "note_block_size": int(note_block_size),
        "encoder": encoder_manifest,
    }
    run_fingerprint = _fingerprint(run_definition)

    next_row = total_tokens = total_chunks = 0
    if progress_path.exists():
        if not resume:
            raise FileExistsError(
                f"Partial run exists in {output}; enable resume or choose a new directory."
            )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Partial embedding run does not match the current input/configuration")
        next_row = int(progress["next_row"])
        total_tokens = int(progress["total_content_tokens"])
        total_chunks = int(progress["total_chunks"])
        embeddings = np.lib.format.open_memmap(partial_embeddings, mode="r+")
        if embeddings.shape != (rows, encoder.hidden_size):
            raise ValueError("Partial embedding matrix has an unexpected shape")
    else:
        if partial_embeddings.exists():
            raise FileExistsError("Partial matrix exists without a progress checkpoint")
        embeddings = np.lib.format.open_memmap(
            partial_embeddings,
            mode="w+",
            dtype=np.float32,
            shape=(rows, encoder.hidden_size),
        )
        _atomic_json(progress_path, {
            "run_fingerprint": run_fingerprint,
            "next_row": 0,
            "total_content_tokens": 0,
            "total_chunks": 0,
        })

    identity = list(IDENTITY_COLUMNS)
    for start, block in iter_note_blocks(data_dir, next_row, note_block_size):
        if start != next_row:
            raise RuntimeError(f"Embedding block order changed: expected {next_row}, received {start}")
        end = start + len(block)
        expected = index.iloc[start:end]
        for offset, row in enumerate(block):
            for key in identity:
                if str(row[key]) != str(expected.iloc[offset][key]):
                    raise ValueError(f"Embedding row identity mismatch at row {start + offset}: {key}")
        block_embeddings, token_counts, chunk_counts = encoder.encode_notes(
            [row["discharge_note"] for row in block]
        )
        if block_embeddings.shape != (len(block), encoder.hidden_size):
            raise ValueError("Encoder returned an unexpected embedding shape")
        if not np.isfinite(block_embeddings).all():
            raise ValueError("Encoder returned non-finite embeddings")
        embeddings[start:end] = block_embeddings
        embeddings.flush()
        next_row = end
        total_tokens += int(token_counts.sum())
        total_chunks += int(chunk_counts.sum())
        _atomic_json(progress_path, {
            "run_fingerprint": run_fingerprint,
            "next_row": next_row,
            "total_content_tokens": total_tokens,
            "total_chunks": total_chunks,
        })

    if next_row != rows:
        raise RuntimeError(f"Embedding run stopped at row {next_row} of {rows}")
    embeddings.flush()
    del embeddings

    partial_index = output / "fullnote_embedding_index.partial.parquet"
    index.to_parquet(partial_index, index=False)
    manifest = {
        **run_definition,
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shape": [rows, int(encoder.hidden_size)],
        "dtype": "float32",
        "index_columns": identity,
        "split_rows": {
            str(key): int(value) for key, value in index["split"].value_counts().items()
        },
        "split_positives": {
            split: int(index.loc[index["split"].eq(split), "mortality_30d"].sum())
            for split in sorted(index["split"].unique())
        },
        "total_content_tokens": total_tokens,
        "total_chunks": total_chunks,
        "mean_content_tokens_per_note": total_tokens / rows,
        "mean_chunks_per_note": total_chunks / rows,
        "files": {
            "embeddings": final_embeddings.name,
            "index": final_index.name,
            "manifest": final_manifest.name,
        },
    }
    partial_manifest = output / "embedding_manifest.partial.json"
    partial_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    partial_embeddings.replace(final_embeddings)
    partial_index.replace(final_index)
    partial_manifest.replace(final_manifest)
    progress_path.unlink()
    return manifest


def load_embedding_artifact(
    output_dir: str | Path,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Load and validate the completed embedding matrix/index/manifest trio."""
    root = Path(output_dir).expanduser().resolve()
    manifest = json.loads((root / "embedding_manifest.json").read_text(encoding="utf-8"))
    embeddings = np.load(root / manifest["files"]["embeddings"], mmap_mode="r")
    index = pd.read_parquet(root / manifest["files"]["index"])
    expected_shape = tuple(manifest["shape"])
    if embeddings.shape != expected_shape or len(index) != expected_shape[0]:
        raise ValueError("Embedding artifact matrix/index/manifest shapes disagree")
    if list(index.columns) != manifest["index_columns"]:
        raise ValueError("Embedding index columns differ from the manifest")
    return embeddings, index, manifest
