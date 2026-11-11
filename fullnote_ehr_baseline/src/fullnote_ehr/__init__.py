"""Shared data and EHR-feature tooling for FullNote + EHR experiments."""

from .data import DatasetContract, DatasetValidationError, load_frame, validate_dataset
from .ehr_features import EHRFeatureBuilder
from .embeddings import load_embedding_artifact
from .fusion import (
    AlignedFusionSplit,
    build_direct_fusion_matrix,
    load_aligned_fusion_split,
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
    "AlignedFusionSplit",
    "build_direct_fusion_matrix",
    "load_aligned_fusion_split",
    "NoteRiskConfig",
    "build_risk_score_matrix",
    "fit_note_risk_scores",
    "load_embedding_artifact",
    "load_frame",
    "validate_dataset",
]

__version__ = "0.1.0"
