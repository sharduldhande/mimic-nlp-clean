from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fullnote_ehr.fusion import (
    build_direct_fusion_matrix,
    load_aligned_fusion_split,
    load_direct_feature_names,
    load_ehr_split,
)


def _write_fusion_artifacts(tmp_path, embedding_label_override=None):
    ehr = tmp_path / "ehr"
    embeddings = tmp_path / "embeddings"
    ehr.mkdir()
    embeddings.mkdir()
    rows = pd.DataFrame({
        "subject_id": [1, 2],
        "hadm_id": [11, 22],
        "note_id": ["note-a", "note-b"],
        "split": ["train", "train"],
        "mortality_30d": [0, 1],
    })
    sparse.save_npz(
        ehr / "X_train.npz",
        sparse.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32)),
    )
    np.save(ehr / "y_train.npy", np.array([0, 1], dtype=np.int8))
    rows.to_parquet(ehr / "index_train.parquet", index=False)
    (ehr / "feature_names.json").write_text(
        json.dumps(["ehr-a", "ehr-b"]),
        encoding="utf-8",
    )

    shuffled_index = rows.iloc[[1, 0]].reset_index(drop=True)
    if embedding_label_override is not None:
        shuffled_index.loc[1, "mortality_30d"] = embedding_label_override
    matrix = np.array([[20, 21, 22], [10, 11, 12]], dtype=np.float32)
    np.save(embeddings / "fullnote_embeddings.npy", matrix)
    shuffled_index.to_parquet(
        embeddings / "fullnote_embedding_index.parquet",
        index=False,
    )
    manifest = {
        "status": "COMPLETE",
        "shape": [2, 3],
        "index_columns": list(rows.columns),
        "files": {
            "embeddings": "fullnote_embeddings.npy",
            "index": "fullnote_embedding_index.parquet",
            "manifest": "embedding_manifest.json",
        },
    }
    (embeddings / "embedding_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return ehr, embeddings


def test_alignment_reorders_embeddings_and_builds_direct_matrix(tmp_path):
    ehr, embeddings = _write_fusion_artifacts(tmp_path)
    ehr_only = load_ehr_split(ehr, "train")
    assert ehr_only.ehr_matrix.shape == (2, 2)
    np.testing.assert_array_equal(ehr_only.labels, np.array([0, 1], dtype=np.int8))
    aligned = load_aligned_fusion_split(ehr, embeddings, "train")
    np.testing.assert_array_equal(
        aligned.note_embeddings,
        np.array([[10, 11, 12], [20, 21, 22]], dtype=np.float32),
    )
    direct = build_direct_fusion_matrix(aligned).toarray()
    np.testing.assert_array_equal(
        direct,
        np.array([
            [1, 0, 10, 11, 12],
            [0, 2, 20, 21, 22],
        ], dtype=np.float32),
    )
    names = load_direct_feature_names(ehr, 3)
    assert names == [
        "ehr-a",
        "ehr-b",
        "fullnote_embedding::0000",
        "fullnote_embedding::0001",
        "fullnote_embedding::0002",
    ]


def test_alignment_rejects_label_mismatch(tmp_path):
    ehr, embeddings = _write_fusion_artifacts(tmp_path, embedding_label_override=1)
    with pytest.raises(ValueError, match="identity mismatch.*mortality_30d"):
        load_aligned_fusion_split(ehr, embeddings, "train")
