from __future__ import annotations

import numpy as np
import pandas as pd

from fullnote_ehr.ehr_features import EHRFeatureBuilder
from fullnote_ehr.schema import EHR_INPUT_COLUMNS
from helpers import dataset_row


def frame(rows):
    return pd.DataFrame(rows).loc[:, list(EHR_INPUT_COLUMNS)]


def test_train_only_vocab_and_stable_transform(tmp_path):
    train = frame([
        dataset_row(1, 101, "train", 1),
        dataset_row(2, 102, "train", 0),
        dataset_row(3, 103, "train", 0),
    ])
    validation_row = dataset_row(4, 104, "validation", 1)
    validation_row["admission_type"] = "UNSEEN TYPE"
    validation_row["diagnoses"] = [{
        "icd_code": "ZZZ",
        "icd_version": 10,
        "diagnosis": "Unseen",
    }]
    validation = frame([validation_row])

    builder = EHRFeatureBuilder(diagnosis_min_frequency=2)
    train_matrix = builder.fit_transform(train)
    validation_matrix = builder.transform(validation)
    assert train_matrix.shape[0] == 3
    assert validation_matrix.shape == (1, train_matrix.shape[1])
    assert "diagnosis::ICD10_I10" in builder.feature_names_
    assert "diagnosis::ICD10_ZZZ" not in builder.feature_names_
    assert all("UNSEEN TYPE" not in name for name in builder.feature_names_)
    assert np.isfinite(train_matrix.data[~np.isnan(train_matrix.data)]).all()

    path = tmp_path / "builder.joblib"
    builder.save(path)
    restored = EHRFeatureBuilder.load(path)
    np.testing.assert_allclose(
        restored.transform(validation).toarray(),
        validation_matrix.toarray(),
        equal_nan=True,
    )


def test_measurement_recency_and_feature_order():
    train = frame([dataset_row(1, 101, "train", 1)])
    builder = EHRFeatureBuilder(diagnosis_min_frequency=1)
    matrix = builder.fit_transform(train)
    assert matrix.shape[1] == len(builder.feature_names_)
    recency_name = "measurement::creatinine::hours_before_discharge"
    index = builder.feature_names_.index(recency_name)
    assert matrix.toarray()[0, index] == 1.0
    assert builder.feature_names_[:3] == [
        "numeric::age",
        "numeric::length_of_stay",
        "numeric::diagnosis_count",
    ]
