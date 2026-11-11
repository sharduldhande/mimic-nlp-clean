"""Create dependency-light SVG plots for the saved direct-fusion experiment."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#E69F00"
INK = "#222222"
MUTED = "#666666"
GRID = "#D9D9D9"
PAPER = "#FFFFFF"


def _svg_document(
    width: int,
    height: int,
    title: str,
    description: str,
    content: Iterable[str],
) -> str:
    body = "\n".join(content)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(description)}</desc>
  <rect width="100%" height="100%" fill="{PAPER}"/>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: {INK}; }}
    .title {{ font-size: 24px; font-weight: 600; }}
    .subtitle {{ font-size: 14px; fill: {MUTED}; }}
    .axis {{ font-size: 13px; fill: {INK}; }}
    .tick {{ font-size: 12px; fill: {MUTED}; }}
    .value {{ font-size: 12px; font-weight: 600; }}
    .legend {{ font-size: 13px; }}
  </style>
{body}
</svg>
"""


def _write_svg(
    path: Path,
    width: int,
    height: int,
    title: str,
    description: str,
    content: Iterable[str],
) -> None:
    path.write_text(
        _svg_document(width, height, title, description, content),
        encoding="utf-8",
    )


def _line_path(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x0: float,
    y0: float,
    plot_width: float,
    plot_height: float,
) -> str:
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = x0 + float(x_value) * plot_width
        y = y0 + (1 - float(y_value)) * plot_height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _curve_axes(
    width: int,
    height: int,
    x_label: str,
    y_label: str,
) -> tuple[list[str], tuple[float, float, float, float]]:
    left, top, right, bottom = 90, 100, 40, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    items = [
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="{INK}" stroke-width="1"/>',
    ]
    for value in np.linspace(0, 1, 6):
        x = left + value * plot_width
        y = top + (1 - value) * plot_height
        items.extend([
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="{GRID}" stroke-width="1"/>',
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>',
            f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle">{value:.1f}</text>',
            f'<text class="tick" x="{left - 14}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>',
        ])
    items.extend([
        f'<text class="axis" x="{left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle">{escape(x_label)}</text>',
        f'<text class="axis" transform="translate(24 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>',
    ])
    return items, (left, top, plot_width, plot_height)


def _add_title(items: list[str], title: str, subtitle: str) -> None:
    items.extend([
        f'<text class="title" x="50" y="38">{escape(title)}</text>',
        f'<text class="subtitle" x="50" y="64">{escape(subtitle)}</text>',
    ])


def _plot_metric_comparison(metrics: dict, output: Path) -> None:
    names = [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall_sensitivity"),
        ("Specificity", "specificity"),
        ("F1", "f1"),
    ]
    validation = metrics["validation"]["selected_threshold"]
    test = metrics["test"]["selected_threshold"]
    width, height = 1200, 680
    left, top, right, bottom = 90, 110, 40, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    items: list[str] = []
    _add_title(
        items,
        "Direct-fusion XGBoost metrics",
        "Thresholded metrics use the validation-selected threshold 0.2614; higher is better.",
    )
    for value in np.linspace(0, 1, 6):
        y = top + (1 - value) * plot_height
        items.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="{GRID}"/>',
            f'<text class="tick" x="{left - 14}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>',
        ])
    group_width = plot_width / len(names)
    bar_width = min(44, group_width * 0.3)
    for index, (label, key) in enumerate(names):
        center = left + (index + 0.5) * group_width
        for offset, value, color in (
            (-bar_width / 2, validation[key], BLUE),
            (bar_width / 2, test[key], ORANGE),
        ):
            x = center + offset - bar_width / 2
            bar_height = value * plot_height
            y = top + plot_height - bar_height
            items.extend([
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>',
                f'<text class="value" x="{x + bar_width / 2:.2f}" y="{max(top + 12, y - 8):.2f}" text-anchor="middle">{value:.3f}</text>',
            ])
        items.append(
            f'<text class="axis" x="{center:.2f}" y="{top + plot_height + 32}" text-anchor="middle">{escape(label)}</text>'
        )
    items.extend([
        f'<rect x="{width - 260}" y="32" width="16" height="16" fill="{BLUE}"/><text class="legend" x="{width - 236}" y="45">Validation</text>',
        f'<rect x="{width - 150}" y="32" width="16" height="16" fill="{ORANGE}"/><text class="legend" x="{width - 126}" y="45">Test</text>',
        f'<text class="axis" transform="translate(24 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">Score (0–1)</text>',
    ])
    _write_svg(
        output / "metric-comparison.svg",
        width,
        height,
        "Direct-fusion metric comparison",
        "Validation and test AUROC, AUPRC, accuracy, precision, recall, specificity, and F1.",
        items,
    )


def _plot_losses(metrics: dict, output: Path) -> None:
    names = [("Brier score", "brier_score"), ("Log loss", "log_loss")]
    validation = metrics["validation"]["selected_threshold"]
    test = metrics["test"]["selected_threshold"]
    width, height = 760, 560
    left, top, right, bottom = 90, 105, 40, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    ceiling = 0.20
    items: list[str] = []
    _add_title(items, "Probability error metrics", "Lower is better; both metrics use predicted probabilities, not the classification threshold.")
    for value in np.linspace(0, ceiling, 5):
        y = top + (1 - value / ceiling) * plot_height
        items.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="{GRID}"/>',
            f'<text class="tick" x="{left - 14}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>',
        ])
    group_width = plot_width / len(names)
    bar_width = 70
    for index, (label, key) in enumerate(names):
        center = left + (index + 0.5) * group_width
        for offset, value, color in (
            (-bar_width / 2, validation[key], BLUE),
            (bar_width / 2, test[key], ORANGE),
        ):
            x = center + offset - bar_width / 2
            bar_height = value / ceiling * plot_height
            y = top + plot_height - bar_height
            items.extend([
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{color}"/>',
                f'<text class="value" x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle">{value:.3f}</text>',
            ])
        items.append(f'<text class="axis" x="{center:.2f}" y="{top + plot_height + 32}" text-anchor="middle">{label}</text>')
    items.extend([
        f'<rect x="{width - 260}" y="32" width="16" height="16" fill="{BLUE}"/><text class="legend" x="{width - 236}" y="45">Validation</text>',
        f'<rect x="{width - 150}" y="32" width="16" height="16" fill="{ORANGE}"/><text class="legend" x="{width - 126}" y="45">Test</text>',
    ])
    _write_svg(output / "probability-losses.svg", width, height, "Probability loss metrics", "Validation and test Brier score and log loss.", items)


def _downsample(x_values: np.ndarray, y_values: np.ndarray, limit: int = 1200):
    if len(x_values) <= limit:
        return x_values, y_values
    positions = np.linspace(0, len(x_values) - 1, limit).round().astype(int)
    return x_values[positions], y_values[positions]


def _plot_roc(predictions: dict[str, pd.DataFrame], output: Path) -> None:
    width, height = 760, 680
    items, (left, top, plot_width, plot_height) = _curve_axes(width, height, "False-positive rate", "True-positive rate (sensitivity)")
    _add_title(items, "ROC curve", "Measures ranking across every possible threshold; the diagonal is random ranking.")
    items.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top}" stroke="{MUTED}" stroke-dasharray="7 6"/>')
    legend_y = 88
    for index, (split, color) in enumerate((("validation", BLUE), ("test", ORANGE))):
        frame = predictions[split]
        labels = frame["mortality_30d"].to_numpy()
        scores = frame["probability_mortality_30d"].to_numpy()
        fpr, tpr, _ = roc_curve(labels, scores)
        fpr, tpr = _downsample(fpr, tpr)
        path = _line_path(fpr, tpr, left, top, plot_width, plot_height)
        auc = roc_auc_score(labels, scores)
        items.extend([
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>',
            f'<line x1="{width - 260}" y1="{legend_y + index * 24}" x2="{width - 230}" y2="{legend_y + index * 24}" stroke="{color}" stroke-width="3"/>',
            f'<text class="legend" x="{width - 220}" y="{legend_y + 5 + index * 24}">{split.title()} AUROC {auc:.3f}</text>',
        ])
    _write_svg(output / "roc-curve.svg", width, height, "ROC curve", "Validation and test receiver operating characteristic curves.", items)


def _plot_precision_recall(predictions: dict[str, pd.DataFrame], output: Path) -> None:
    width, height = 760, 680
    items, (left, top, plot_width, plot_height) = _curve_axes(width, height, "Recall (sensitivity)", "Precision (positive predictive value)")
    _add_title(items, "Precision–recall curve", "Preferred for this imbalanced cohort; no-skill precision is approximately the 4.7% prevalence.")
    prevalence = np.mean([
        predictions[split]["mortality_30d"].mean() for split in ("validation", "test")
    ])
    baseline_y = top + (1 - prevalence) * plot_height
    items.extend([
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{left + plot_width}" y2="{baseline_y:.2f}" stroke="{MUTED}" stroke-dasharray="7 6"/>',
        f'<text class="tick" x="{left + 8}" y="{baseline_y - 7:.2f}">No-skill ≈ {prevalence:.3f}</text>',
    ])
    legend_y = 88
    for index, (split, color) in enumerate((("validation", BLUE), ("test", ORANGE))):
        frame = predictions[split]
        labels = frame["mortality_30d"].to_numpy()
        scores = frame["probability_mortality_30d"].to_numpy()
        precision, recall, _ = precision_recall_curve(labels, scores)
        recall, precision = _downsample(recall, precision)
        path = _line_path(recall, precision, left, top, plot_width, plot_height)
        auprc = average_precision_score(labels, scores)
        items.extend([
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>',
            f'<line x1="{width - 260}" y1="{legend_y + index * 24}" x2="{width - 230}" y2="{legend_y + index * 24}" stroke="{color}" stroke-width="3"/>',
            f'<text class="legend" x="{width - 220}" y="{legend_y + 5 + index * 24}">{split.title()} AUPRC {auprc:.3f}</text>',
        ])
    _write_svg(output / "precision-recall-curve.svg", width, height, "Precision–recall curve", "Validation and test precision-recall curves with prevalence baseline.", items)


def _plot_confusion_matrices(metrics: dict, output: Path) -> None:
    width, height = 980, 590
    items: list[str] = []
    threshold = metrics["threshold_selection"]["threshold"]
    _add_title(items, "Confusion matrices", f"Counts at the validation-selected threshold {threshold:.4f}; percentages are normalized within the actual class.")
    cell = 150
    top = 150
    panel_origins = {"validation": 155, "test": 590}
    for split, origin_x in panel_origins.items():
        values = metrics[split]["selected_threshold"]["confusion_matrix"]
        matrix = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
        row_totals = matrix.sum(axis=1, keepdims=True)
        rates = matrix / row_totals
        items.append(f'<text class="axis" x="{origin_x + cell}" y="110" text-anchor="middle">{split.title()}</text>')
        items.append(f'<text class="axis" x="{origin_x + cell}" y="{top - 22}" text-anchor="middle">Predicted class</text>')
        items.append(f'<text class="axis" transform="translate({origin_x - 72} {top + cell}) rotate(-90)" text-anchor="middle">Actual class</text>')
        for column, label in enumerate(("Survived", "Died")):
            items.append(f'<text class="tick" x="{origin_x + (column + 0.5) * cell}" y="{top - 3}" text-anchor="middle">{label}</text>')
        for row, label in enumerate(("Survived", "Died")):
            items.append(f'<text class="tick" x="{origin_x - 14}" y="{top + (row + 0.5) * cell + 4}" text-anchor="end">{label}</text>')
            for column in range(2):
                rate = float(rates[row, column])
                opacity = 0.12 + 0.78 * rate
                x = origin_x + column * cell
                y = top + row * cell
                text_color = PAPER if rate > 0.65 else INK
                items.extend([
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{BLUE}" fill-opacity="{opacity:.3f}" stroke="{PAPER}" stroke-width="3"/>',
                    f'<text class="value" x="{x + cell / 2}" y="{y + cell / 2 - 8}" text-anchor="middle" style="fill:{text_color};font-size:22px">{int(matrix[row, column]):,}</text>',
                    f'<text class="tick" x="{x + cell / 2}" y="{y + cell / 2 + 19}" text-anchor="middle" style="fill:{text_color}">{rate:.1%} of actual class</text>',
                ])
    _write_svg(output / "confusion-matrices.svg", width, height, "Confusion matrices", "Validation and test confusion matrices at the selected threshold.", items)


def _plot_threshold_tradeoffs(validation: pd.DataFrame, threshold: float, output: Path) -> None:
    labels = validation["mortality_30d"].to_numpy(dtype=np.int8)
    scores = validation["probability_mortality_30d"].to_numpy(dtype=np.float64)
    thresholds = np.linspace(0.01, 0.99, 199)
    series = {"Precision": [], "Recall": [], "F1": [], "Specificity": [], "Accuracy": []}
    for candidate in thresholds:
        predicted = (scores >= candidate).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        series["Precision"].append(precision_score(labels, predicted, zero_division=0))
        series["Recall"].append(recall_score(labels, predicted, zero_division=0))
        series["F1"].append(f1_score(labels, predicted, zero_division=0))
        series["Specificity"].append(tn / (tn + fp) if tn + fp else 0)
        series["Accuracy"].append(accuracy_score(labels, predicted))
    width, height = 900, 680
    items, (left, top, plot_width, plot_height) = _curve_axes(width, height, "Classification threshold", "Validation score")
    _add_title(items, "Validation threshold tradeoffs", "The threshold 0.2614 was selected by maximum validation F1; the test set did not choose it.")
    colors = {"Precision": BLUE, "Recall": ORANGE, "F1": GREEN, "Specificity": PURPLE, "Accuracy": SKY}
    for index, (name, values) in enumerate(series.items()):
        path = _line_path(thresholds, values, left, top, plot_width, plot_height)
        legend_x = 545 + (index % 3) * 115
        legend_y = 82 + (index // 3) * 24
        items.extend([
            f'<polyline points="{path}" fill="none" stroke="{colors[name]}" stroke-width="2.5"/>',
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 25}" y2="{legend_y}" stroke="{colors[name]}" stroke-width="3"/>',
            f'<text class="legend" x="{legend_x + 32}" y="{legend_y + 5}">{name}</text>',
        ])
    selected_x = left + threshold * plot_width
    items.extend([
        f'<line x1="{selected_x:.2f}" y1="{top}" x2="{selected_x:.2f}" y2="{top + plot_height}" stroke="{INK}" stroke-width="2" stroke-dasharray="7 6"/>',
        f'<text class="value" x="{selected_x + 7:.2f}" y="{top + 18}">selected {threshold:.3f}</text>',
    ])
    _write_svg(output / "threshold-tradeoffs.svg", width, height, "Threshold tradeoffs", "Validation precision, recall, F1, specificity, and accuracy across thresholds.", items)


def _plot_calibration(predictions: dict[str, pd.DataFrame], metrics: dict, output: Path) -> None:
    width, height = 760, 680
    items, (left, top, plot_width, plot_height) = _curve_axes(width, height, "Mean predicted mortality probability", "Observed mortality rate")
    _add_title(items, "Calibration curve", "Quantile bins compare predicted risk with observed mortality; the diagonal is perfect calibration.")
    items.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top}" stroke="{MUTED}" stroke-dasharray="7 6"/>')
    legend_y = 88
    for index, (split, color) in enumerate((("validation", BLUE), ("test", ORANGE))):
        frame = predictions[split]
        observed, predicted = calibration_curve(
            frame["mortality_30d"].to_numpy(),
            frame["probability_mortality_30d"].to_numpy(),
            n_bins=10,
            strategy="quantile",
        )
        path = _line_path(predicted, observed, left, top, plot_width, plot_height)
        brier = metrics[split]["selected_threshold"]["brier_score"]
        items.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x_value, y_value in zip(predicted, observed):
            x = left + x_value * plot_width
            y = top + (1 - y_value) * plot_height
            items.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" stroke="{PAPER}" stroke-width="1.5"/>')
        items.extend([
            f'<line x1="{width - 290}" y1="{legend_y + index * 24}" x2="{width - 260}" y2="{legend_y + index * 24}" stroke="{color}" stroke-width="3"/>',
            f'<text class="legend" x="{width - 250}" y="{legend_y + 5 + index * 24}">{split.title()} Brier {brier:.3f}</text>',
        ])
    _write_svg(output / "calibration-curve.svg", width, height, "Calibration curve", "Validation and test calibration curves with Brier scores.", items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models/direct_fusion_xgboost"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir / "plots"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((model_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    predictions = {
        split: pd.read_parquet(model_dir / f"predictions_{split}.parquet")
        for split in ("validation", "test")
    }
    threshold = float(metrics["threshold_selection"]["threshold"])

    _plot_metric_comparison(metrics, output)
    _plot_losses(metrics, output)
    _plot_roc(predictions, output)
    _plot_precision_recall(predictions, output)
    _plot_confusion_matrices(metrics, output)
    _plot_threshold_tradeoffs(predictions["validation"], threshold, output)
    _plot_calibration(predictions, metrics, output)

    files = [
        "metric-comparison.svg",
        "probability-losses.svg",
        "roc-curve.svg",
        "precision-recall-curve.svg",
        "confusion-matrices.svg",
        "threshold-tradeoffs.svg",
        "calibration-curve.svg",
    ]
    manifest = {
        "artifact_version": "direct-fusion-plots-v1",
        "source_model_dir": str(model_dir),
        "selected_threshold": threshold,
        "files": files,
    }
    (output / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
