"""Shared data and EHR-feature tooling for FullNote + EHR experiments."""

from .data import DatasetContract, DatasetValidationError, load_frame, validate_dataset
from .ehr_features import EHRFeatureBuilder
from .embeddings import load_embedding_artifact
from .ehr_only import (
    EHROnlyTrainingConfig,
    predict_ehr_only_split,
    train_ehr_only,
)
from .fusion import (
    AlignedFusionSplit,
    EHRModelSplit,
    build_direct_fusion_matrix,
    load_aligned_fusion_split,
    load_ehr_split,
)
from .risk_score_fusion import (
    NoteRiskConfig,
    build_risk_score_matrix,
    fit_note_risk_scores,
)

__all__ = [
    "DatasetContract",
    "DatasetValidationError",
    "EHRFeatureBuilder",
    "EHRModelSplit",
    "AlignedFusionSplit",
    "build_direct_fusion_matrix",
    "load_aligned_fusion_split",
    "load_ehr_split",
    "EHROnlyTrainingConfig",
    "train_ehr_only",
    "predict_ehr_only_split",
    "NoteRiskConfig",
    "build_risk_score_matrix",
    "fit_note_risk_scores",
    "load_embedding_artifact",
    "load_frame",
    "validate_dataset",
]

__version__ = "0.1.0"
