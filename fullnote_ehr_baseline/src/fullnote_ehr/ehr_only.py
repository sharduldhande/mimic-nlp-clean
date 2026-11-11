"""Training, evaluation, and inference for the EHR-only XGBoost ablation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .direct_fusion import (
    DEFAULT_CANDIDATES,
    DirectTrainingConfig,
    binary_metrics,
    select_f1_threshold,
)
from .fusion import load_ehr_feature_names, load_ehr_split


# The alias makes the controlled-comparison intent explicit while keeping every
# optimization default identical to the direct-fusion experiment.
EHROnlyTrainingConfig = DirectTrainingConfig


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


def _validate_feature_count(feature_names: Sequence[str], width: int) -> None:
    if len(feature_names) != width:
        raise ValueError(
            f"Feature-name count {len(feature_names)} does not match matrix width {width}"
        )


def _prediction_frame(
    index: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    result = index.copy()
    result["mortality_30d"] = np.asarray(labels, dtype=np.int8)
    result["probability_mortality_30d"] = np.asarray(
        probabilities,
        dtype=np.float32,
    )
    result["predicted_at_selected_threshold"] = (
        probabilities >= threshold
    ).astype(np.int8)
    result["predicted_at_0_5"] = (probabilities >= 0.5).astype(np.int8)
    return result


def _importance_frame(booster, feature_names: Sequence[str]) -> pd.DataFrame:
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    rows = []
    for feature_index, name in enumerate(feature_names):
        key = f"f{feature_index}"
        rows.append({
            "feature_index": feature_index,
            "feature_name": name,
            "gain": float(gain.get(key, 0.0)),
            "split_count": int(weight.get(key, 0)),
        })
    return pd.DataFrame(rows).sort_values(
        ["gain", "split_count", "feature_index"],
        ascending=[False, False, True],
    )


def train_ehr_only(
    ehr_dir: str | Path,
    output_dir: str | Path,
    config: DirectTrainingConfig | None = None,
    candidates: Sequence[dict[str, Any]] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Tune on validation and evaluate an EHR-only model on test exactly once."""
    try:
        import xgboost as xgb
    except Exception as error:
        raise RuntimeError(
            "XGBoost is required; install requirements-modeling.txt"
        ) from error

    settings = config or EHROnlyTrainingConfig()
    settings.validate()
    if not candidates:
        raise ValueError("At least one hyperparameter candidate is required")

    output = _prepare_output(output_dir)
    ehr_root = Path(ehr_dir).expanduser().resolve()
    train = load_ehr_split(ehr_root, "train")
    validation = load_ehr_split(ehr_root, "validation")
    if train.ehr_features != validation.ehr_features:
        raise ValueError("Train and validation EHR widths differ")
    feature_names = load_ehr_feature_names(ehr_root)
    _validate_feature_count(feature_names, train.ehr_features)

    dtrain = xgb.DMatrix(train.ehr_matrix, label=train.labels)
    dvalidation = xgb.DMatrix(validation.ehr_matrix, label=validation.labels)
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
    best_model_path = output / "ehr_only_xgboost_model.ubj"
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
        validation_auprc = float(
            average_precision_score(validation.labels, probabilities)
        )
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
    _prediction_frame(
        validation.index,
        validation.labels,
        best_validation_probabilities,
        selected_threshold,
    ).to_parquet(output / "predictions_validation.parquet", index=False)

    test = load_ehr_split(ehr_root, "test")
    if test.ehr_features != train.ehr_features:
        raise ValueError("Test EHR width differs from training")
    dtest = xgb.DMatrix(test.ehr_matrix, label=test.labels)
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
    _prediction_frame(
        test.index,
        test.labels,
        test_probabilities,
        selected_threshold,
    ).to_parquet(output / "predictions_test.parquet", index=False)

    _importance_frame(booster, feature_names).to_parquet(
        output / "feature_importance.parquet",
        index=False,
    )
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
    manifest = {
        "artifact_version": "ehr-only-xgboost-v1",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "30-day post-discharge mortality",
        "experiment": "structured EHR only; no note-derived features",
        "inputs": {
            "ehr_dir": str(ehr_root),
            "ehr_manifest_sha256": _sha256(ehr_manifest_path),
        },
        "matrix_shapes": {
            "train": list(train.ehr_matrix.shape),
            "validation": list(validation.ehr_matrix.shape),
            "test": list(test.ehr_matrix.shape),
        },
        "feature_counts": {
            "ehr": train.ehr_features,
            "note": 0,
            "total": train.ehr_features,
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
        "controlled_comparison": {
            "reference_experiment": "direct EHR + FullNote fusion",
            "same_splits": True,
            "same_ehr_features": True,
            "same_candidate_grid": True,
            "same_training_defaults": True,
            "same_selection_and_threshold_rules": True,
        },
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


def predict_ehr_only_split(
    ehr_dir: str | Path,
    model_dir: str | Path,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a saved EHR-only model on one prepared structured-data split."""
    try:
        import xgboost as xgb
    except Exception as error:
        raise RuntimeError(
            "XGBoost is required; install requirements-modeling.txt"
        ) from error

    ehr_root = Path(ehr_dir).expanduser().resolve()
    model_root = Path(model_dir).expanduser().resolve()
    manifest = json.loads(
        (model_root / "training_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "COMPLETE":
        raise ValueError("EHR-only model artifact is not complete")
    actual_ehr_hash = _sha256(ehr_root / "manifest.json")
    if actual_ehr_hash != manifest["inputs"]["ehr_manifest_sha256"]:
        raise ValueError("EHR artifact manifest differs from the model's training input")

    data = load_ehr_split(ehr_root, split)
    expected_names = json.loads(
        (model_root / "feature_names.json").read_text(encoding="utf-8")
    )
    actual_names = load_ehr_feature_names(ehr_root)
    if actual_names != expected_names:
        raise ValueError("Current EHR feature order differs from the saved model")
    _validate_feature_count(expected_names, data.ehr_features)

    booster = xgb.Booster()
    booster.load_model(model_root / manifest["files"]["model"])
    rounds = int(manifest["selected_candidate"]["boosting_rounds_used"])
    probabilities = booster.predict(
        xgb.DMatrix(data.ehr_matrix),
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
