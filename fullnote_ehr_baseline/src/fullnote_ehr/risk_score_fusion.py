"""Leakage-safe FullNote risk scores followed by EHR + score XGBoost fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .direct_fusion import (
    DEFAULT_CANDIDATES,
    DirectTrainingConfig,
    binary_metrics,
    select_f1_threshold,
)
from .fusion import AlignedFusionSplit, load_aligned_fusion_split


@dataclass(frozen=True)
class NoteRiskConfig:
    """Fixed note-only classifier configuration used inside every fold."""

    c: float = 1.0
    max_iter: int = 1000
    tolerance: float = 1e-4
    n_splits: int = 5
    seed: int = 7650
    class_weight: str | None = None

    def validate(self) -> None:
        if self.c <= 0:
            raise ValueError("c must be positive")
        if self.max_iter < 1 or self.tolerance <= 0:
            raise ValueError("max_iter and tolerance must be positive")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'")


@dataclass
class NoteRiskScoreBundle:
    train_oof: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    fold_assignments: np.ndarray
    final_model: LogisticRegression
    manifest: dict[str, Any]


def _new_note_model(config: NoteRiskConfig) -> LogisticRegression:
    return LogisticRegression(
        C=config.c,
        solver="lbfgs",
        max_iter=config.max_iter,
        tol=config.tolerance,
        class_weight=config.class_weight,
        random_state=config.seed,
    )


def _probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "mean_score": float(scores.mean()),
    }


def fit_note_risk_scores(
    train: AlignedFusionSplit,
    validation: AlignedFusionSplit,
    test: AlignedFusionSplit,
    config: NoteRiskConfig | None = None,
) -> NoteRiskScoreBundle:
    """Generate patient-grouped OOF train scores and final-model holdout scores."""
    settings = config or NoteRiskConfig()
    settings.validate()
    if not (
        train.embedding_features
        == validation.embedding_features
        == test.embedding_features
    ):
        raise ValueError("Embedding widths differ across splits")
    groups = train.index["subject_id"].to_numpy()
    if len(np.unique(groups)) < settings.n_splits:
        raise ValueError("Not enough unique patients for the requested grouped folds")
    splitter = StratifiedGroupKFold(
        n_splits=settings.n_splits,
        shuffle=True,
        random_state=settings.seed,
    )
    oof = np.full(train.rows, np.nan, dtype=np.float32)
    fold_assignments = np.full(train.rows, -1, dtype=np.int16)
    folds: list[dict[str, Any]] = []
    for fold, (fit_rows, holdout_rows) in enumerate(
        splitter.split(train.note_embeddings, train.labels, groups=groups)
    ):
        fit_groups = set(groups[fit_rows].tolist())
        holdout_groups = set(groups[holdout_rows].tolist())
        overlap = fit_groups.intersection(holdout_groups)
        if overlap:
            raise RuntimeError(f"Patient leakage detected in note fold {fold}")
        if len(np.unique(train.labels[fit_rows])) != 2:
            raise ValueError(f"Note fold {fold} training rows do not contain both classes")
        model = _new_note_model(settings)
        model.fit(train.note_embeddings[fit_rows], train.labels[fit_rows])
        probabilities = model.predict_proba(
            train.note_embeddings[holdout_rows]
        )[:, 1].astype(np.float32)
        oof[holdout_rows] = probabilities
        fold_assignments[holdout_rows] = fold
        folds.append({
            "fold": fold,
            "fit_rows": int(len(fit_rows)),
            "holdout_rows": int(len(holdout_rows)),
            "fit_patients": int(len(fit_groups)),
            "holdout_patients": int(len(holdout_groups)),
            "patient_overlap": 0,
            "fit_positives": int(train.labels[fit_rows].sum()),
            "holdout_positives": int(train.labels[holdout_rows].sum()),
            "iterations": int(model.n_iter_[0]),
            "holdout_metrics": _probability_metrics(
                train.labels[holdout_rows],
                probabilities,
            ),
        })
    if np.isnan(oof).any() or np.any(fold_assignments < 0):
        raise RuntimeError("Every training admission must receive exactly one OOF score")
    patient_fold_counts = pd.DataFrame({
        "subject_id": groups,
        "fold": fold_assignments,
    }).groupby("subject_id")["fold"].nunique()
    if not patient_fold_counts.eq(1).all():
        raise RuntimeError("A patient was assigned to more than one OOF fold")

    final_model = _new_note_model(settings)
    final_model.fit(train.note_embeddings, train.labels)
    validation_scores = final_model.predict_proba(
        validation.note_embeddings
    )[:, 1].astype(np.float32)
    test_scores = final_model.predict_proba(test.note_embeddings)[:, 1].astype(np.float32)
    for name, values in (
        ("train_oof", oof),
        ("validation", validation_scores),
        ("test", test_scores),
    ):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid {name} note-risk scores")
    manifest = {
        "artifact_version": "note-risk-logistic-v1",
        "method": "L2 logistic regression on frozen 768-dimensional note embeddings",
        "training_score_protocol": "patient-grouped out-of-fold predictions",
        "configuration": asdict(settings),
        "folds": folds,
        "overall_metrics": {
            "train_oof": _probability_metrics(train.labels, oof),
            "validation": _probability_metrics(validation.labels, validation_scores),
            "test": _probability_metrics(test.labels, test_scores),
        },
        "final_model_iterations": int(final_model.n_iter_[0]),
        "embedding_features": train.embedding_features,
    }
    return NoteRiskScoreBundle(
        train_oof=oof,
        validation=validation_scores,
        test=test_scores,
        fold_assignments=fold_assignments,
        final_model=final_model,
        manifest=manifest,
    )


def build_risk_score_matrix(
    data: AlignedFusionSplit,
    risk_scores: np.ndarray,
) -> sparse.csr_matrix:
    """Append one finite note-risk score to the saved EHR matrix."""
    scores = np.asarray(risk_scores, dtype=np.float32)
    if scores.shape != (data.rows,):
        raise ValueError("Risk-score vector does not match the EHR rows")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Risk scores must be finite values in [0, 1]")
    matrix = sparse.hstack(
        [data.ehr_matrix, sparse.csr_matrix(scores[:, None])],
        format="csr",
        dtype=np.float32,
    )
    matrix.sort_indices()
    return matrix


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


def _risk_score_frame(
    data: AlignedFusionSplit,
    scores: np.ndarray,
    folds: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = data.index.copy()
    frame["note_risk_score"] = np.asarray(scores, dtype=np.float32)
    if folds is not None:
        frame["oof_fold"] = np.asarray(folds, dtype=np.int16)
    return frame


def _prediction_frame(
    data: AlignedFusionSplit,
    note_scores: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    frame = data.index.copy()
    frame["note_risk_score"] = np.asarray(note_scores, dtype=np.float32)
    frame["probability_mortality_30d"] = np.asarray(probabilities, dtype=np.float32)
    frame["predicted_at_selected_threshold"] = (
        probabilities >= threshold
    ).astype(np.int8)
    frame["predicted_at_0_5"] = (probabilities >= 0.5).astype(np.int8)
    return frame


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


def _load_ehr_feature_names(ehr_root: Path) -> list[str]:
    names = json.loads((ehr_root / "feature_names.json").read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Invalid EHR feature_names.json")
    if len(names) != len(set(names)):
        raise ValueError("EHR feature names are not unique")
    return names


def train_risk_score_fusion(
    ehr_dir: str | Path,
    embedding_dir: str | Path,
    output_dir: str | Path,
    note_config: NoteRiskConfig | None = None,
    xgboost_config: DirectTrainingConfig | None = None,
    candidates: Sequence[dict[str, Any]] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    """Train the leakage-safe note-risk score and EHR + score XGBoost model."""
    try:
        import xgboost as xgb
    except Exception as error:
        raise RuntimeError("XGBoost is required; install requirements-modeling.txt") from error

    note_settings = note_config or NoteRiskConfig()
    xgb_settings = xgboost_config or DirectTrainingConfig()
    note_settings.validate()
    xgb_settings.validate()
    if not candidates:
        raise ValueError("At least one XGBoost candidate is required")
    output = _prepare_output(output_dir)
    ehr_root = Path(ehr_dir).expanduser().resolve()
    embedding_root = Path(embedding_dir).expanduser().resolve()

    train = load_aligned_fusion_split(ehr_root, embedding_root, "train")
    validation = load_aligned_fusion_split(ehr_root, embedding_root, "validation")
    test = load_aligned_fusion_split(ehr_root, embedding_root, "test")
    if not (
        train.ehr_features == validation.ehr_features == test.ehr_features
    ):
        raise ValueError("EHR feature widths differ across splits")

    risk = fit_note_risk_scores(
        train,
        validation,
        test,
        config=note_settings,
    )
    joblib.dump(risk.final_model, output / "note_risk_model.joblib")
    (output / "note_risk_manifest.json").write_text(
        json.dumps(risk.manifest, indent=2),
        encoding="utf-8",
    )
    _risk_score_frame(train, risk.train_oof, risk.fold_assignments).to_parquet(
        output / "risk_scores_train_oof.parquet",
        index=False,
    )
    _risk_score_frame(validation, risk.validation).to_parquet(
        output / "risk_scores_validation.parquet",
        index=False,
    )
    _risk_score_frame(test, risk.test).to_parquet(
        output / "risk_scores_test.parquet",
        index=False,
    )

    X_train = build_risk_score_matrix(train, risk.train_oof)
    X_validation = build_risk_score_matrix(validation, risk.validation)
    feature_names = [*_load_ehr_feature_names(ehr_root), "fullnote_risk_score"]
    if len(feature_names) != X_train.shape[1] or X_validation.shape[1] != X_train.shape[1]:
        raise ValueError("Risk-fusion feature order and matrices disagree")
    dtrain = xgb.DMatrix(X_train, label=train.labels)
    dvalidation = xgb.DMatrix(X_validation, label=validation.labels)

    negatives = int((train.labels == 0).sum())
    positives = int((train.labels == 1).sum())
    scale_pos_weight = negatives / positives
    common_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc", "aucpr"],
        "tree_method": "hist",
        "device": xgb_settings.device,
        "eta": xgb_settings.learning_rate,
        "subsample": xgb_settings.subsample,
        "colsample_bytree": xgb_settings.colsample_bytree,
        "lambda": xgb_settings.reg_lambda,
        "alpha": xgb_settings.reg_alpha,
        "max_bin": xgb_settings.max_bin,
        "scale_pos_weight": scale_pos_weight,
        "seed": xgb_settings.seed,
        "nthread": xgb_settings.nthread,
    }
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_history: dict[str, Any] = {}
    best_validation_probabilities: np.ndarray | None = None
    model_path = output / "risk_fusion_xgboost_model.ubj"
    for candidate_index, candidate in enumerate(candidates):
        params = {**common_params, **dict(candidate)}
        history: dict[str, Any] = {}
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=xgb_settings.max_boost_rounds,
            evals=[(dtrain, "train"), (dvalidation, "validation")],
            early_stopping_rounds=xgb_settings.early_stopping_rounds,
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
            booster.save_model(model_path)
            best = {**trial, "rank": rank}
            best_history = history
            best_validation_probabilities = probabilities.astype(np.float32)

    if best is None or best_validation_probabilities is None:
        raise RuntimeError("No risk-fusion XGBoost candidate completed")
    booster = xgb.Booster()
    booster.load_model(model_path)
    selected_rounds = int(best["boosting_rounds_used"])
    threshold_selection = select_f1_threshold(
        validation.labels,
        best_validation_probabilities,
    )
    threshold = float(threshold_selection["threshold"])
    validation_metrics = {
        "selected_threshold": binary_metrics(
            validation.labels,
            best_validation_probabilities,
            threshold,
        ),
        "threshold_0_5": binary_metrics(
            validation.labels,
            best_validation_probabilities,
            0.5,
        ),
    }
    _prediction_frame(
        validation,
        risk.validation,
        best_validation_probabilities,
        threshold,
    ).to_parquet(output / "predictions_validation.parquet", index=False)

    X_test = build_risk_score_matrix(test, risk.test)
    dtest = xgb.DMatrix(X_test, label=test.labels)
    test_probabilities = booster.predict(
        dtest,
        iteration_range=(0, selected_rounds),
    ).astype(np.float32)
    test_metrics = {
        "selected_threshold": binary_metrics(test.labels, test_probabilities, threshold),
        "threshold_0_5": binary_metrics(test.labels, test_probabilities, 0.5),
    }
    _prediction_frame(
        test,
        risk.test,
        test_probabilities,
        threshold,
    ).to_parquet(output / "predictions_test.parquet", index=False)

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
    _importance_frame(booster, feature_names).to_parquet(
        output / "feature_importance.parquet",
        index=False,
    )
    metrics = {
        "threshold_selection": {
            "method": "maximize F1 on validation; highest threshold wins exact ties",
            **threshold_selection,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "note_risk_classifier": risk.manifest["overall_metrics"],
    }
    (output / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    ehr_manifest = ehr_root / "manifest.json"
    embedding_manifest = embedding_root / "embedding_manifest.json"
    manifest = {
        "artifact_version": "risk-score-fusion-xgboost-v1",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "30-day post-discharge mortality",
        "inputs": {
            "ehr_dir": str(ehr_root),
            "embedding_dir": str(embedding_root),
            "ehr_manifest_sha256": _sha256(ehr_manifest),
            "embedding_manifest_sha256": _sha256(embedding_manifest),
        },
        "matrix_shapes": {
            "train": list(X_train.shape),
            "validation": list(X_validation.shape),
            "test": list(X_test.shape),
        },
        "feature_counts": {
            "ehr": train.ehr_features,
            "fullnote_risk_score": 1,
            "total": int(X_train.shape[1]),
        },
        "split_positives": {
            "train": int(train.labels.sum()),
            "validation": int(validation.labels.sum()),
            "test": int(test.labels.sum()),
        },
        "note_risk_configuration": asdict(note_settings),
        "xgboost_training_config": asdict(xgb_settings),
        "scale_pos_weight": float(scale_pos_weight),
        "selection_metric": "validation AUPRC; AUROC and fewer rounds break ties",
        "selected_candidate": {
            key: value for key, value in best.items() if key != "rank"
        },
        "selected_threshold": threshold,
        "test_was_used_for_selection": False,
        "files": {
            "note_risk_model": "note_risk_model.joblib",
            "note_risk_manifest": "note_risk_manifest.json",
            "risk_fusion_model": model_path.name,
            "feature_names": "feature_names.json",
            "candidate_trials": "candidate_trials.json",
            "evaluation_history": "best_evaluation_history.json",
            "evaluation_metrics": "evaluation_metrics.json",
            "train_oof_scores": "risk_scores_train_oof.parquet",
            "validation_scores": "risk_scores_validation.parquet",
            "test_scores": "risk_scores_test.parquet",
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


def predict_risk_score_fusion_split(
    ehr_dir: str | Path,
    embedding_dir: str | Path,
    model_dir: str | Path,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the saved note-risk and XGBoost models on one prepared split."""
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
        raise ValueError("Risk-score model artifact is not complete")
    if _sha256(ehr_root / "manifest.json") != manifest["inputs"]["ehr_manifest_sha256"]:
        raise ValueError("EHR artifact differs from the risk model's training input")
    if (
        _sha256(embedding_root / "embedding_manifest.json")
        != manifest["inputs"]["embedding_manifest_sha256"]
    ):
        raise ValueError("Embedding artifact differs from the risk model's training input")

    data = load_aligned_fusion_split(ehr_root, embedding_root, split)
    note_model = joblib.load(model_root / manifest["files"]["note_risk_model"])
    note_scores = note_model.predict_proba(data.note_embeddings)[:, 1].astype(np.float32)
    matrix = build_risk_score_matrix(data, note_scores)
    expected_names = json.loads(
        (model_root / manifest["files"]["feature_names"]).read_text(encoding="utf-8")
    )
    current_names = [*_load_ehr_feature_names(ehr_root), "fullnote_risk_score"]
    if current_names != expected_names or len(expected_names) != matrix.shape[1]:
        raise ValueError("Current risk-fusion feature order differs from the saved model")
    booster = xgb.Booster()
    booster.load_model(model_root / manifest["files"]["risk_fusion_model"])
    rounds = int(manifest["selected_candidate"]["boosting_rounds_used"])
    probabilities = booster.predict(
        xgb.DMatrix(matrix),
        iteration_range=(0, rounds),
    ).astype(np.float32)
    threshold = float(manifest["selected_threshold"])
    predictions = _prediction_frame(data, note_scores, probabilities, threshold)
    metrics = {
        "selected_threshold": binary_metrics(data.labels, probabilities, threshold),
        "threshold_0_5": binary_metrics(data.labels, probabilities, 0.5),
    }
    return predictions, metrics
