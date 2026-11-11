from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fullnote_ehr.ehr_only import (
    EHROnlyTrainingConfig,
    predict_ehr_only_split,
    train_ehr_only,
)


def _write_ehr_artifact(root) -> None:
    rng = np.random.default_rng(7650)
    feature_names = ["ehr_signal", "ehr_noise", "ehr_missing"]
    (root / "feature_names.json").write_text(
        json.dumps(feature_names),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"artifact_version": "synthetic-ehr-v1"}),
        encoding="utf-8",
    )
    for split, rows, offset in (
        ("train", 60, 0),
        ("validation", 20, 100),
        ("test", 20, 200),
    ):
        labels = np.tile(np.array([0, 1], dtype=np.int8), rows // 2)
        matrix = np.column_stack([
            labels + rng.normal(0, 0.25, size=rows),
            rng.normal(size=rows),
            rng.binomial(1, 0.2, size=rows),
        ]).astype(np.float32)
        index = pd.DataFrame({
            "subject_id": np.arange(rows) + offset,
            "hadm_id": np.arange(rows) + 1000 + offset,
            "note_id": [f"{split}-note-{row}" for row in range(rows)],
            "split": split,
            "mortality_30d": labels,
        })
        sparse.save_npz(root / f"X_{split}.npz", sparse.csr_matrix(matrix))
        np.save(root / f"y_{split}.npy", labels)
        index.to_parquet(root / f"index_{split}.parquet", index=False)


def test_ehr_only_training_and_saved_inference_match(tmp_path):
    pytest.importorskip("xgboost")
    ehr_dir = tmp_path / "ehr"
    model_dir = tmp_path / "model"
    ehr_dir.mkdir()
    _write_ehr_artifact(ehr_dir)
    config = EHROnlyTrainingConfig(
        learning_rate=0.2,
        max_boost_rounds=25,
        early_stopping_rounds=5,
        subsample=1.0,
        colsample_bytree=1.0,
        max_bin=32,
        seed=7650,
        device="cpu",
        nthread=1,
    )
    result = train_ehr_only(
        ehr_dir,
        model_dir,
        config=config,
        candidates=({"max_depth": 2, "min_child_weight": 1.0},),
    )

    assert result["manifest"]["feature_counts"] == {
        "ehr": 3,
        "note": 0,
        "total": 3,
    }
    assert result["manifest"]["test_was_used_for_selection"] is False
    assert result["metrics"]["test"]["selected_threshold"]["rows"] == 20

    predictions, metrics = predict_ehr_only_split(ehr_dir, model_dir, "test")
    saved = pd.read_parquet(model_dir / "predictions_test.parquet")
    np.testing.assert_allclose(
        predictions["probability_mortality_30d"],
        saved["probability_mortality_30d"],
    )
    assert metrics == result["metrics"]["test"]
