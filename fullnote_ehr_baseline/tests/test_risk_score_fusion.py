from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from fullnote_ehr.fusion import AlignedFusionSplit
from fullnote_ehr.risk_score_fusion import (
    NoteRiskConfig,
    build_risk_score_matrix,
    fit_note_risk_scores,
)


def _split(name, subject_ids, labels, embeddings):
    rows = len(labels)
    index = pd.DataFrame({
        "subject_id": subject_ids,
        "hadm_id": np.arange(rows) + {"train": 100, "validation": 200, "test": 300}[name],
        "note_id": [f"{name}-note-{index}" for index in range(rows)],
        "split": name,
        "mortality_30d": labels,
    })
    return AlignedFusionSplit(
        split=name,
        ehr_matrix=sparse.csr_matrix(np.arange(rows * 2).reshape(rows, 2), dtype=np.float32),
        note_embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        index=index,
    )


def test_patient_grouped_oof_scores_and_risk_matrix():
    rng = np.random.default_rng(7650)
    train_subjects = np.repeat(np.arange(12), 2)
    train_labels = np.repeat(np.arange(12) % 2, 2)
    train_embeddings = rng.normal(size=(24, 4))
    train_embeddings[:, 0] += train_labels * 1.5
    validation_labels = np.array([0, 1, 0, 1, 0, 1])
    test_labels = np.array([1, 0, 1, 0, 1, 0])
    train = _split("train", train_subjects, train_labels, train_embeddings)
    validation = _split(
        "validation",
        np.arange(20, 26),
        validation_labels,
        rng.normal(size=(6, 4)),
    )
    test = _split(
        "test",
        np.arange(30, 36),
        test_labels,
        rng.normal(size=(6, 4)),
    )
    result = fit_note_risk_scores(
        train,
        validation,
        test,
        NoteRiskConfig(n_splits=3, max_iter=200),
    )
    assert result.train_oof.shape == (24,)
    assert result.validation.shape == (6,)
    assert result.test.shape == (6,)
    assert np.isfinite(result.train_oof).all()
    assignment = pd.DataFrame({
        "subject": train_subjects,
        "fold": result.fold_assignments,
    })
    assert assignment.groupby("subject")["fold"].nunique().eq(1).all()
    assert set(result.fold_assignments) == {0, 1, 2}
    assert all(fold["patient_overlap"] == 0 for fold in result.manifest["folds"])

    matrix = build_risk_score_matrix(train, result.train_oof)
    assert matrix.shape == (24, 3)
    np.testing.assert_allclose(matrix.toarray()[:, -1], result.train_oof)
