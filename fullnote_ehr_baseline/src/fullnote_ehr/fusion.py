"""Identity-checked alignment of structured EHR features and note embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .embeddings import load_embedding_artifact
from .schema import IDENTITY_COLUMNS, SPLITS


@dataclass
class AlignedFusionSplit:
    """One split with EHR rows and note vectors in exactly the same order."""

    split: str
    ehr_matrix: sparse.csr_matrix
    note_embeddings: np.ndarray
    labels: np.ndarray
    index: pd.DataFrame

    @property
    def rows(self) -> int:
        return int(self.ehr_matrix.shape[0])

    @property
    def ehr_features(self) -> int:
        return int(self.ehr_matrix.shape[1])

    @property
    def embedding_features(self) -> int:
        return int(self.note_embeddings.shape[1])

    @property
    def direct_features(self) -> int:
        return self.ehr_features + self.embedding_features


def _validate_identity_frame(frame: pd.DataFrame, name: str) -> None:
    expected = list(IDENTITY_COLUMNS)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing identity columns: {missing}")
    if frame["note_id"].isna().any() or frame["note_id"].duplicated().any():
        raise ValueError(f"{name} must contain unique, non-null note_id values")
    if frame["hadm_id"].isna().any() or frame["hadm_id"].duplicated().any():
        raise ValueError(f"{name} must contain unique, non-null hadm_id values")


def _aligned_embedding_rows(
    target_index: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_index: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    _validate_identity_frame(target_index, "EHR index")
    _validate_identity_frame(embedding_index, "embedding index")
    positions_by_note = pd.Series(
        np.arange(len(embedding_index), dtype=np.int64),
        index=embedding_index["note_id"].astype(str),
    )
    positions = positions_by_note.reindex(target_index["note_id"].astype(str))
    if positions.isna().any():
        missing = target_index.loc[positions.isna(), "note_id"].astype(str).head(5).tolist()
        raise ValueError(f"EHR notes are absent from the embedding artifact: {missing}")
    row_positions = positions.to_numpy(dtype=np.int64)
    aligned_index = embedding_index.iloc[row_positions].reset_index(drop=True)
    expected = target_index.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
    for column in IDENTITY_COLUMNS:
        left = expected[column].astype(str).to_numpy()
        right = aligned_index[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            mismatch = int(np.flatnonzero(left != right)[0])
            raise ValueError(
                f"EHR/embedding identity mismatch for {column} at aligned row {mismatch}"
            )
    aligned_embeddings = np.asarray(embeddings[row_positions], dtype=np.float32)
    if aligned_embeddings.ndim != 2 or not np.isfinite(aligned_embeddings).all():
        raise ValueError("Aligned note embeddings must be a finite two-dimensional matrix")
    return aligned_embeddings, aligned_index


def load_aligned_fusion_split(
    ehr_dir: str | Path,
    embedding_dir: str | Path,
    split: str,
) -> AlignedFusionSplit:
    """Load one split and align note vectors by identity, never incidental order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, received {split!r}")
    ehr_root = Path(ehr_dir).expanduser().resolve()
    matrix = sparse.load_npz(ehr_root / f"X_{split}.npz").tocsr()
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    labels = np.load(ehr_root / f"y_{split}.npy", allow_pickle=False)
    labels = np.asarray(labels, dtype=np.int8)
    index = pd.read_parquet(ehr_root / f"index_{split}.parquet")
    _validate_identity_frame(index, f"EHR {split} index")
    rows = len(index)
    if matrix.shape[0] != rows or labels.shape != (rows,):
        raise ValueError(f"EHR matrix, labels, and index disagree for split {split}")
    if not index["split"].eq(split).all():
        raise ValueError(f"EHR index contains rows outside split {split}")
    index_labels = index["mortality_30d"].to_numpy(dtype=np.int8)
    if not np.array_equal(labels, index_labels):
        raise ValueError(f"EHR labels and index labels disagree for split {split}")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"EHR labels are not binary for split {split}")

    embeddings, embedding_index, _ = load_embedding_artifact(embedding_dir)
    aligned_embeddings, _ = _aligned_embedding_rows(
        index,
        embeddings,
        embedding_index,
    )
    if aligned_embeddings.shape[0] != rows:
        raise ValueError(f"Aligned embedding rows disagree for split {split}")
    return AlignedFusionSplit(
        split=split,
        ehr_matrix=matrix,
        note_embeddings=aligned_embeddings,
        labels=labels,
        index=index.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True),
    )


def build_direct_fusion_matrix(data: AlignedFusionSplit) -> sparse.csr_matrix:
    """Concatenate sparse EHR features with all dense note dimensions."""
    if data.ehr_matrix.shape[0] != data.note_embeddings.shape[0]:
        raise ValueError("EHR and embedding row counts differ")
    note_matrix = sparse.csr_matrix(data.note_embeddings, dtype=np.float32)
    combined = sparse.hstack(
        [data.ehr_matrix, note_matrix],
        format="csr",
        dtype=np.float32,
    )
    combined.sort_indices()
    if combined.shape != (data.rows, data.direct_features):
        raise RuntimeError("Direct-fusion matrix has an unexpected shape")
    return combined


def load_direct_feature_names(
    ehr_dir: str | Path,
    embedding_features: int,
) -> list[str]:
    """Return the persisted EHR order followed by named embedding dimensions."""
    if embedding_features < 1:
        raise ValueError("embedding_features must be positive")
    path = Path(ehr_dir).expanduser().resolve() / "feature_names.json"
    ehr_names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ehr_names, list) or not all(isinstance(name, str) for name in ehr_names):
        raise ValueError("EHR feature_names.json must contain a list of strings")
    if len(ehr_names) != len(set(ehr_names)):
        raise ValueError("EHR feature names are not unique")
    note_names = [f"fullnote_embedding::{index:04d}" for index in range(embedding_features)]
    return [*ehr_names, *note_names]


def validate_direct_feature_count(
    feature_names: Sequence[str],
    matrix: sparse.spmatrix,
) -> None:
    if len(feature_names) != matrix.shape[1]:
        raise ValueError(
            f"Feature-name count {len(feature_names)} does not match matrix width "
            f"{matrix.shape[1]}"
        )
