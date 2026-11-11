"""Leakage-safe structured-EHR feature fitting and transformation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from .schema import (
    BASE_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    MEASUREMENT_CONCEPTS,
    MEASUREMENT_NUMERIC_FIELDS,
)


MISSING_CATEGORY = "<MISSING>"


def _record_list(value: object) -> list[dict]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"Expected a list of records, received {type(value).__name__}")
    return value


def _diagnosis_tokens(value: object) -> set[str]:
    tokens: set[str] = set()
    for item in _record_list(value):
        code = item.get("icd_code")
        version = item.get("icd_version")
        if code is None or not str(code).strip():
            continue
        try:
            version_text = str(int(version))
        except (TypeError, ValueError):
            version_text = "UNKNOWN"
        normalized = "".join(str(code).upper().split())
        tokens.add(f"ICD{version_text}_{normalized}")
    return tokens


def _explicit_dense_csr(values: np.ndarray) -> sparse.csr_matrix:
    """Store numeric zeros explicitly so XGBoost does not reinterpret them as absent."""
    values = np.asarray(values, dtype=np.float32, order="C")
    rows, columns = values.shape
    if columns == 0:
        return sparse.csr_matrix((rows, 0), dtype=np.float32)
    data = values.reshape(-1)
    indices = np.tile(np.arange(columns, dtype=np.int32), rows)
    indptr = np.arange(0, (rows + 1) * columns, columns, dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=values.shape)


class EHRFeatureBuilder:
    """Fit train-only EHR vocabularies and transform admissions into a CSR matrix."""

    def __init__(
        self,
        diagnosis_min_frequency: int = 10,
        include_measurement_recency: bool = True,
    ) -> None:
        if diagnosis_min_frequency < 1:
            raise ValueError("diagnosis_min_frequency must be at least 1")
        self.diagnosis_min_frequency = int(diagnosis_min_frequency)
        self.include_measurement_recency = bool(include_measurement_recency)
        self.categorical_encoder_: OneHotEncoder | None = None
        self.diagnosis_vocabulary_: dict[str, int] | None = None
        self.numeric_feature_names_: list[str] | None = None
        self.feature_names_: list[str] | None = None

    @property
    def fitted(self) -> bool:
        return self.feature_names_ is not None

    def _required_columns(self) -> set[str]:
        return set(BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + (
            "dischtime", "diagnoses", "measurements"
        ))

    def _check_frame(self, frame: pd.DataFrame) -> None:
        missing = self._required_columns() - set(frame.columns)
        if missing:
            raise ValueError(f"EHR input frame is missing columns: {sorted(missing)}")

    def _categorical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.loc[:, list(CATEGORICAL_COLUMNS)].copy()
        for column in CATEGORICAL_COLUMNS:
            result[column] = result[column].fillna(MISSING_CATEGORY).astype(str)
        return result

    def _numeric_matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        base = np.column_stack([
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
            for column in BASE_NUMERIC_COLUMNS
        ])
        names = [f"numeric::{column}" for column in BASE_NUMERIC_COLUMNS]
        measurement_fields = list(MEASUREMENT_NUMERIC_FIELDS)
        if self.include_measurement_recency:
            measurement_fields.append("hours_before_discharge")
        width = len(MEASUREMENT_CONCEPTS) * len(measurement_fields)
        measurements = np.full((len(frame), width), np.nan, dtype=np.float32)

        for row_index, (items_value, discharge) in enumerate(
            zip(frame["measurements"], frame["dischtime"])
        ):
            by_name = {
                item.get("feature"): item
                for item in _record_list(items_value)
                if item.get("feature") in MEASUREMENT_CONCEPTS
            }
            for concept_index, concept in enumerate(MEASUREMENT_CONCEPTS):
                item = by_name.get(concept)
                offset = concept_index * len(measurement_fields)
                if item is None:
                    measurements[row_index, offset + measurement_fields.index("is_missing")] = 1.0
                    continue
                for field_index, field in enumerate(measurement_fields):
                    if field == "hours_before_discharge":
                        last_time = item.get("last_time")
                        if last_time is not None and not pd.isna(discharge):
                            delta = pd.Timestamp(discharge) - pd.Timestamp(last_time)
                            measurements[row_index, offset + field_index] = (
                                delta.total_seconds() / 3600.0
                            )
                    else:
                        value = item.get(field)
                        if value is not None:
                            measurements[row_index, offset + field_index] = float(value)

        for concept in MEASUREMENT_CONCEPTS:
            names.extend(
                f"measurement::{concept}::{field}" for field in measurement_fields
            )
        return np.hstack((base, measurements)).astype(np.float32, copy=False), names

    def fit(self, frame: pd.DataFrame) -> "EHRFeatureBuilder":
        self._check_frame(frame)
        numeric, numeric_names = self._numeric_matrix(frame)
        del numeric
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                dtype=np.float32,
            )
        except TypeError:  # pragma: no cover - compatibility with scikit-learn < 1.2
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
                dtype=np.float32,
            )
        encoder.fit(self._categorical_frame(frame))

        diagnosis_counts: Counter[str] = Counter()
        for diagnoses in frame["diagnoses"]:
            diagnosis_counts.update(_diagnosis_tokens(diagnoses))
        kept = sorted(
            token
            for token, count in diagnosis_counts.items()
            if count >= self.diagnosis_min_frequency
        )
        self.categorical_encoder_ = encoder
        self.diagnosis_vocabulary_ = {token: index for index, token in enumerate(kept)}
        self.numeric_feature_names_ = numeric_names
        categorical_names = [
            f"categorical::{name}"
            for name in encoder.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()
        ]
        diagnosis_names = [f"diagnosis::{token}" for token in kept]
        self.feature_names_ = numeric_names + categorical_names + diagnosis_names
        return self

    def _diagnosis_matrix(self, values: Iterable[object], rows: int) -> sparse.csr_matrix:
        assert self.diagnosis_vocabulary_ is not None
        row_indices: list[int] = []
        column_indices: list[int] = []
        for row_index, diagnoses in enumerate(values):
            for token in _diagnosis_tokens(diagnoses):
                column = self.diagnosis_vocabulary_.get(token)
                if column is not None:
                    row_indices.append(row_index)
                    column_indices.append(column)
        data = np.ones(len(row_indices), dtype=np.float32)
        return sparse.csr_matrix(
            (data, (row_indices, column_indices)),
            shape=(rows, len(self.diagnosis_vocabulary_)),
            dtype=np.float32,
        )

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if not self.fitted or self.categorical_encoder_ is None:
            raise RuntimeError("EHRFeatureBuilder must be fitted on training data first")
        self._check_frame(frame)
        numeric, numeric_names = self._numeric_matrix(frame)
        if numeric_names != self.numeric_feature_names_:
            raise RuntimeError("Numeric feature order changed after fitting")
        numeric_matrix = _explicit_dense_csr(numeric)
        categorical_matrix = self.categorical_encoder_.transform(
            self._categorical_frame(frame)
        ).tocsr()
        diagnosis_matrix = self._diagnosis_matrix(frame["diagnoses"], len(frame))
        result = sparse.hstack(
            (numeric_matrix, categorical_matrix, diagnosis_matrix),
            format="csr",
            dtype=np.float32,
        )
        if result.shape[1] != len(self.feature_names_ or []):
            raise RuntimeError("Feature matrix width does not match the fitted feature names")
        return result

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        return self.fit(frame).transform(frame)

    def manifest(self) -> dict:
        if not self.fitted or self.categorical_encoder_ is None:
            raise RuntimeError("EHRFeatureBuilder is not fitted")
        assert self.diagnosis_vocabulary_ is not None
        return {
            "builder_version": "ehr-v1",
            "diagnosis_min_frequency": self.diagnosis_min_frequency,
            "include_measurement_recency": self.include_measurement_recency,
            "numeric_features": len(self.numeric_feature_names_ or []),
            "categorical_features": len(
                self.categorical_encoder_.get_feature_names_out(CATEGORICAL_COLUMNS)
            ),
            "diagnosis_features": len(self.diagnosis_vocabulary_),
            "total_features": len(self.feature_names_ or []),
        }

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted EHRFeatureBuilder")
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "EHRFeatureBuilder":
        result = joblib.load(Path(path).expanduser().resolve())
        if not isinstance(result, cls):
            raise TypeError(f"Artifact does not contain {cls.__name__}")
        return result

    def write_feature_names(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("EHRFeatureBuilder is not fitted")
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.feature_names_, indent=2),
            encoding="utf-8",
        )
