"""Training and evaluation for direct EHR + FullNote XGBoost fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .fusion import (
    build_direct_fusion_matrix,
    load_aligned_fusion_split,
    load_direct_feature_names,
    validate_direct_feature_count,
)


DEFAULT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"max_depth": 3, "min_child_weight": 1.0},
    {"max_depth": 5, "min_child_weight": 1.0},
    {"max_depth": 5, "min_child_weight": 5.0},
    {"max_depth": 7, "min_child_weight": 5.0},
)


@dataclass(frozen=True)
class DirectTrainingConfig:
    learning_rate: float = 0.05
    max_boost_rounds: int = 2000
    early_stopping_rounds: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    max_bin: int = 256
    seed: int = 7650
    device: str = "cuda"
    nthread: int = 0

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_boost_rounds < 1 or self.early_stopping_rounds < 1:
            raise ValueError("boosting and early-stopping rounds must be positive")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("subsample and colsample_bytree must be in (0, 1]")
        if self.max_bin < 2:
            raise ValueError("max_bin must be at least 2")
        if self.device not in {"cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError("device must be cpu, cuda, or cuda:N")


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Select an F1-maximizing threshold without consulting the test set."""
    y = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y.shape != scores.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be same-length vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("labels/probabilities are invalid")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if thresholds.size == 0:
        return {"threshold": 0.5, "validation_f1": 0.0}
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_value = float(np.max(f1))
    tied = np.flatnonzero(np.isclose(f1, best_value, rtol=0, atol=1e-12))
    best_index = int(tied[-1])  # prefer the more conservative threshold on an exact tie
    return {
        "threshold": float(thresholds[best_index]),
        "validation_f1": best_value,
    }


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate discrimination, calibration, and thresholded metrics."""
    y = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y.shape != scores.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be same-length vectors")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    predicted = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _prepare_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_frame(
    index: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    result = index.copy()
    result["mortality_30d"] = np.asarray(labels, dtype=np.int8)
    result["probability_mortality_30d"] = np.asarray(probabilities, dtype=np.float32)
    result["predicted_at_selected_threshold"] = (
        probabilities >= threshold
    ).astype(np.int8)
    result["predicted_at_0_5"] = (probabilities >= 0.5).astype(np.int8)
    return result


def _importance_frame(booster, feature_names: Sequence[str]) -> pd.DataFrame:
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    rows = []
    for index, name in enumerate(feature_names):
        key = f"f{index}"
        rows.append({
            "feature_index": index,
            "feature_name": name,
            "gain": float(gain.get(key, 0.0)),
            "split_count": int(weight.get(key, 0)),
        })
    return pd.DataFrame(rows).sort_values(
        ["gain", "split_count", "feature_index"],
        ascending=[False, False, True],
    )


def train_direct_fusion(
    ehr_dir: str | Path,
    embedding_dir: str | Path,
    output_dir: str | Path,
    config: DirectTrainingConfig | None = None,
    candidates: Sequence[dict[str, Any]] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Tune on validation, select its threshold, and evaluate test exactly once."""
    try:
        import xgboost as xgb
    except Exception as error:
        raise RuntimeError("XGBoost is required; install requirements-modeling.txt") from error

    settings = config or DirectTrainingConfig()
    settings.validate()
    if not candidates:
        raise ValueError("At least one hyperparameter candidate is required")
    output = _prepare_output(output_dir)
    ehr_root = Path(ehr_dir).expanduser().resolve()
    embedding_root = Path(embedding_dir).expanduser().resolve()

    train = load_aligned_fusion_split(ehr_root, embedding_root, "train")
    validation = load_aligned_fusion_split(ehr_root, embedding_root, "validation")
    if train.ehr_features != validation.ehr_features:
        raise ValueError("Train and validation EHR widths differ")
    if train.embedding_features != validation.embedding_features:
        raise ValueError("Train and validation embedding widths differ")
    feature_names = load_direct_feature_names(ehr_root, train.embedding_features)
    X_train = build_direct_fusion_matrix(train)
    X_validation = build_direct_fusion_matrix(validation)
    validate_direct_feature_count(feature_names, X_train)
    validate_direct_feature_count(feature_names, X_validation)
    dtrain = xgb.DMatrix(X_train, label=train.labels)
    dvalidation = xgb.DMatrix(X_validation, label=validation.labels)

    negatives = int((train.labels == 0).sum())
    positives = int((train.labels == 1).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both classes")
    scale_pos_weight = negatives / positives
    common_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc", "aucpr"],
        "tree_method": "hist",
        "device": settings.device,
        "eta": settings.learning_rate,
        "subsample": settings.subsample,
        "colsample_bytree": settings.colsample_bytree,
        "lambda": settings.reg_lambda,
        "alpha": settings.reg_alpha,
        "max_bin": settings.max_bin,
        "scale_pos_weight": scale_pos_weight,
        "seed": settings.seed,
        "nthread": settings.nthread,
    }

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model_path = output / "direct_xgboost_model.ubj"
    best_history: dict[str, Any] = {}
    best_validation_probabilities: np.ndarray | None = None
    for candidate_index, candidate in enumerate(candidates):
        params = {**common_params, **dict(candidate)}
        history: dict[str, Any] = {}
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=settings.max_boost_rounds,
            evals=[(dtrain, "train"), (dvalidation, "validation")],
            early_stopping_rounds=settings.early_stopping_rounds,
            evals_result=history,
            maximize=True,
            verbose_eval=False,
        )
        best_iteration = int(booster.best_iteration)
        probabilities = booster.predict(
            dvalidation,
            iteration_range=(0, best_iteration + 1),
        )
        validation_auprc = float(average_precision_score(validation.labels, probabilities))
        validation_auroc = float(roc_auc_score(validation.labels, probabilities))
        trial = {
            "candidate_index": candidate_index,
            "candidate": dict(candidate),
            "best_iteration_zero_based": best_iteration,
            "boosting_rounds_used": best_iteration + 1,
            "xgboost_best_validation_auprc": float(booster.best_score),
            "validation_auprc": validation_auprc,
            "validation_auroc": validation_auroc,
        }
        trials.append(trial)
        rank = (validation_auprc, validation_auroc, -(best_iteration + 1))
        if best is None or rank > best["rank"]:
            booster.save_model(best_model_path)
            best = {**trial, "rank": rank}
            best_history = history
            best_validation_probabilities = probabilities.astype(np.float32)

    if best is None or best_validation_probabilities is None:
        raise RuntimeError("No XGBoost candidate completed")
    booster = xgb.Booster()
    booster.load_model(best_model_path)
    selected_rounds = int(best["boosting_rounds_used"])
    threshold_selection = select_f1_threshold(
        validation.labels,
        best_validation_probabilities,
    )
    selected_threshold = float(threshold_selection["threshold"])

    validation_metrics = {
        "selected_threshold": binary_metrics(
            validation.labels,
            best_validation_probabilities,
            selected_threshold,
        ),
        "threshold_0_5": binary_metrics(
            validation.labels,
            best_validation_probabilities,
            0.5,
        ),
    }
    validation_predictions = _prediction_frame(
        validation.index,
        validation.labels,
        best_validation_probabilities,
        selected_threshold,
    )
    validation_predictions.to_parquet(output / "predictions_validation.parquet", index=False)

    test = load_aligned_fusion_split(ehr_root, embedding_root, "test")
    if test.direct_features != X_train.shape[1]:
        raise ValueError("Test direct-fusion width differs from training")
    X_test = build_direct_fusion_matrix(test)
    validate_direct_feature_count(feature_names, X_test)
    dtest = xgb.DMatrix(X_test, label=test.labels)
    test_probabilities = booster.predict(
        dtest,
        iteration_range=(0, selected_rounds),
    ).astype(np.float32)
    test_metrics = {
        "selected_threshold": binary_metrics(
            test.labels,
            test_probabilities,
            selected_threshold,
        ),
        "threshold_0_5": binary_metrics(test.labels, test_probabilities, 0.5),
    }
    test_predictions = _prediction_frame(
        test.index,
        test.labels,
        test_probabilities,
        selected_threshold,
    )
    test_predictions.to_parquet(output / "predictions_test.parquet", index=False)

    feature_importance = _importance_frame(booster, feature_names)
    feature_importance.to_parquet(output / "feature_importance.parquet", index=False)
    (output / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2),
        encoding="utf-8",
    )
    (output / "candidate_trials.json").write_text(
        json.dumps(trials, indent=2),
        encoding="utf-8",
    )
    (output / "best_evaluation_history.json").write_text(
        json.dumps(best_history, indent=2),
        encoding="utf-8",
    )
    metrics = {
        "threshold_selection": {
            "method": "maximize F1 on validation; highest threshold wins exact ties",
            **threshold_selection,
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (output / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    ehr_manifest_path = ehr_root / "manifest.json"
    embedding_manifest_path = embedding_root / "embedding_manifest.json"
    manifest = {
        "artifact_version": "direct-fusion-xgboost-v1",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "30-day post-discharge mortality",
        "inputs": {
            "ehr_dir": str(ehr_root),
            "embedding_dir": str(embedding_root),
            "ehr_manifest_sha256": _sha256(ehr_manifest_path),
            "embedding_manifest_sha256": _sha256(embedding_manifest_path),
        },
        "matrix_shapes": {
            "train": list(X_train.shape),
            "validation": list(X_validation.shape),
            "test": list(X_test.shape),
        },
        "feature_counts": {
            "ehr": train.ehr_features,
            "fullnote_embedding": train.embedding_features,
            "total": train.direct_features,
        },
        "split_positives": {
            "train": int(train.labels.sum()),
            "validation": int(validation.labels.sum()),
            "test": int(test.labels.sum()),
        },
        "scale_pos_weight": float(scale_pos_weight),
        "training_config": asdict(settings),
        "selection_metric": "validation AUPRC; AUROC and fewer rounds break ties",
        "selected_candidate": {
            key: value for key, value in best.items() if key != "rank"
        },
        "selected_threshold": selected_threshold,
        "test_was_used_for_selection": False,
        "files": {
            "model": best_model_path.name,
            "feature_names": "feature_names.json",
            "candidate_trials": "candidate_trials.json",
            "evaluation_history": "best_evaluation_history.json",
            "evaluation_metrics": "evaluation_metrics.json",
            "validation_predictions": "predictions_validation.parquet",
            "test_predictions": "predictions_test.parquet",
            "feature_importance": "feature_importance.parquet",
        },
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {"manifest": manifest, "metrics": metrics}


def predict_direct_fusion_split(
    ehr_dir: str | Path,
    embedding_dir: str | Path,
    model_dir: str | Path,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a saved direct model on one prepared, identity-checked split."""
    try:
        import xgboost as xgb
    except Exception as error:
        raise RuntimeError("XGBoost is required; install requirements-modeling.txt") from error

    ehr_root = Path(ehr_dir).expanduser().resolve()
    embedding_root = Path(embedding_dir).expanduser().resolve()
    model_root = Path(model_dir).expanduser().resolve()
    manifest = json.loads(
        (model_root / "training_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Direct-fusion model artifact is not complete")
    expected_hashes = manifest["inputs"]
    actual_ehr_hash = _sha256(ehr_root / "manifest.json")
    actual_embedding_hash = _sha256(embedding_root / "embedding_manifest.json")
    if actual_ehr_hash != expected_hashes["ehr_manifest_sha256"]:
        raise ValueError("EHR artifact manifest differs from the model's training input")
    if actual_embedding_hash != expected_hashes["embedding_manifest_sha256"]:
        raise ValueError("Embedding artifact manifest differs from the model's training input")

    data = load_aligned_fusion_split(ehr_root, embedding_root, split)
    matrix = build_direct_fusion_matrix(data)
    expected_names = json.loads(
        (model_root / "feature_names.json").read_text(encoding="utf-8")
    )
    actual_names = load_direct_feature_names(ehr_root, data.embedding_features)
    if actual_names != expected_names:
        raise ValueError("Current direct feature order differs from the saved model")
    validate_direct_feature_count(expected_names, matrix)

    booster = xgb.Booster()
    booster.load_model(model_root / manifest["files"]["model"])
    rounds = int(manifest["selected_candidate"]["boosting_rounds_used"])
    probabilities = booster.predict(
        xgb.DMatrix(matrix),
        iteration_range=(0, rounds),
    ).astype(np.float32)
    threshold = float(manifest["selected_threshold"])
    predictions = _prediction_frame(
        data.index,
        data.labels,
        probabilities,
        threshold,
    )
    metrics = {
        "selected_threshold": binary_metrics(data.labels, probabilities, threshold),
        "threshold_0_5": binary_metrics(data.labels, probabilities, 0.5),
    }
    return predictions, metrics
