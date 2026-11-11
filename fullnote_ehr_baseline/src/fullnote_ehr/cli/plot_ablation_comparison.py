"""Plot EHR-only, direct-note, and note-risk XGBoost test comparisons."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


COLORS = {
    "EHR only": "#6B7280",
    "EHR + FullNote": "#2563EB",
    "EHR + note risk": "#EA580C",
}
INK = "#172033"
GRID = "#D7DDE8"
PAPER = "#FFFFFF"


def _metrics(model_dir: Path) -> dict:
    payload = json.loads(
        (model_dir / "evaluation_metrics.json").read_text(encoding="utf-8")
    )
    return payload["test"]["selected_threshold"]


def _document(width: int, height: int, title: str, content: list[str]) -> str:
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{escape(title)}</title>",
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#172033}",
        ".title{font-size:25px;font-weight:700}",
        ".subtitle{font-size:14px;fill:#5B6475}",
        ".axis{font-size:14px;font-weight:600}",
        ".tick{font-size:12px;fill:#5B6475}",
        ".value{font-size:11px;font-weight:700}",
        ".legend{font-size:13px}",
        "</style>",
        *content,
        "</svg>",
    ])


def _write(path: Path, width: int, height: int, title: str, content: list[str]) -> None:
    path.write_text(_document(width, height, title, content), encoding="utf-8")


def _metric_comparison(models: dict[str, dict], output: Path) -> None:
    metrics = [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall_sensitivity"),
        ("Specificity", "specificity"),
        ("F1", "f1"),
    ]
    width, height = 1320, 720
    left, top, right, bottom = 90, 125, 40, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    items = [
        '<text class="title" x="50" y="38">Held-out test performance</text>',
        '<text class="subtitle" x="50" y="65">Each model uses its own validation-F1-selected threshold; AUROC and AUPRC are threshold-independent.</text>',
    ]
    for value in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = top + (1 - value) * plot_height
        items.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="{GRID}"/>',
            f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>',
        ])
    group_width = plot_width / len(metrics)
    bar_width = 36
    model_names = list(models)
    for metric_index, (label, key) in enumerate(metrics):
        center = left + (metric_index + 0.5) * group_width
        for model_index, model_name in enumerate(model_names):
            value = float(models[model_name][key])
            offset = (model_index - (len(model_names) - 1) / 2) * bar_width
            x = center + offset - bar_width / 2
            bar_height = value * plot_height
            y = top + plot_height - bar_height
            items.extend([
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{COLORS[model_name]}"/>',
                f'<text class="value" x="{x + bar_width / 2:.2f}" y="{max(top + 12, y - 7):.2f}" text-anchor="middle">{value:.3f}</text>',
            ])
        items.append(
            f'<text class="axis" x="{center:.2f}" y="{top + plot_height + 32}" text-anchor="middle">{escape(label)}</text>'
        )
    legend_x = 730
    for index, model_name in enumerate(model_names):
        x = legend_x + index * 180
        items.extend([
            f'<rect x="{x}" y="34" width="16" height="16" fill="{COLORS[model_name]}"/>',
            f'<text class="legend" x="{x + 24}" y="47">{escape(model_name)}</text>',
        ])
    _write(output / "test-metric-comparison.svg", width, height, "Test metric comparison", items)


def _note_deltas(ehr: dict, direct: dict, output: Path) -> None:
    metrics = [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall_sensitivity"),
        ("Specificity", "specificity"),
        ("F1", "f1"),
    ]
    deltas = [(label, float(direct[key]) - float(ehr[key])) for label, key in metrics]
    bound = max(0.12, max(abs(value) for _, value in deltas) * 1.15)
    width, height = 1100, 680
    left, top, right, bottom = 100, 120, 50, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    zero_y = top + plot_height / 2
    items = [
        '<text class="title" x="50" y="38">Effect of adding the full note embedding</text>',
        '<text class="subtitle" x="50" y="65">Direct FullNote fusion minus EHR-only on the held-out test set. Positive bars favor FullNote.</text>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" stroke="{INK}" stroke-width="1.5"/>',
    ]
    for value in [-bound, -bound / 2, 0, bound / 2, bound]:
        y = top + (bound - value) / (2 * bound) * plot_height
        if value != 0:
            items.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="{GRID}"/>'
            )
        items.append(
            f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:+.3f}</text>'
        )
    group_width = plot_width / len(deltas)
    bar_width = min(75, group_width * 0.55)
    for index, (label, delta) in enumerate(deltas):
        center = left + (index + 0.5) * group_width
        scaled = abs(delta) / (2 * bound) * plot_height
        y = zero_y - scaled if delta >= 0 else zero_y
        color = "#2563EB" if delta >= 0 else "#DC2626"
        items.extend([
            f'<rect x="{center - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{scaled:.2f}" fill="{color}"/>',
            f'<text class="value" x="{center:.2f}" y="{y - 8 if delta >= 0 else y + scaled + 18:.2f}" text-anchor="middle">{delta:+.3f}</text>',
            f'<text class="axis" x="{center:.2f}" y="{top + plot_height + 32}" text-anchor="middle">{escape(label)}</text>',
        ])
    _write(output / "note-contribution-deltas.svg", width, height, "FullNote contribution deltas", items)


def _loss_comparison(models: dict[str, dict], output: Path) -> None:
    metrics = [("Brier score", "brier_score"), ("Log loss", "log_loss")]
    width, height = 850, 590
    left, top, right, bottom = 90, 115, 40, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    ceiling = max(models[name][key] for name in models for _, key in metrics) * 1.15
    items = [
        '<text class="title" x="50" y="38">Held-out probability error</text>',
        '<text class="subtitle" x="50" y="65">Lower is better. These values use continuous scores and do not depend on the classification threshold.</text>',
    ]
    for fraction in [0, 0.25, 0.5, 0.75, 1.0]:
        value = ceiling * fraction
        y = top + (1 - fraction) * plot_height
        items.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="{GRID}"/>',
            f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>',
        ])
    group_width = plot_width / len(metrics)
    bar_width = 55
    model_names = list(models)
    for metric_index, (label, key) in enumerate(metrics):
        center = left + (metric_index + 0.5) * group_width
        for model_index, model_name in enumerate(model_names):
            value = float(models[model_name][key])
            offset = (model_index - (len(model_names) - 1) / 2) * bar_width
            x = center + offset - bar_width / 2
            bar_height = value / ceiling * plot_height
            y = top + plot_height - bar_height
            items.extend([
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{COLORS[model_name]}"/>',
                f'<text class="value" x="{x + bar_width / 2:.2f}" y="{y - 7:.2f}" text-anchor="middle">{value:.3f}</text>',
            ])
        items.append(
            f'<text class="axis" x="{center:.2f}" y="{top + plot_height + 32}" text-anchor="middle">{escape(label)}</text>'
        )
    for index, model_name in enumerate(model_names):
        x = 320 + index * 170
        items.extend([
            f'<rect x="{x}" y="82" width="16" height="16" fill="{COLORS[model_name]}"/>',
            f'<text class="legend" x="{x + 24}" y="95">{escape(model_name)}</text>',
        ])
    _write(output / "probability-loss-comparison.svg", width, height, "Probability loss comparison", items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ehr-only-dir",
        type=Path,
        default=Path("artifacts/models/ehr_only_xgboost"),
    )
    parser.add_argument(
        "--direct-dir",
        type=Path,
        default=Path("artifacts/models/direct_fusion_xgboost"),
    )
    parser.add_argument(
        "--risk-score-dir",
        type=Path,
        default=Path("artifacts/models/risk_score_fusion_xgboost"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models/model_comparison"),
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = {
        "EHR only": _metrics(args.ehr_only_dir.expanduser().resolve()),
        "EHR + FullNote": _metrics(args.direct_dir.expanduser().resolve()),
        "EHR + note risk": _metrics(args.risk_score_dir.expanduser().resolve()),
    }
    _metric_comparison(models, output)
    _note_deltas(models["EHR only"], models["EHR + FullNote"], output)
    _loss_comparison(models, output)

    metric_keys = [
        "auroc",
        "auprc",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "brier_score",
        "log_loss",
    ]
    summary = {
        "split": "test",
        "threshold_policy": "each model uses its validation-F1-selected threshold",
        "models": models,
        "direct_fullnote_minus_ehr_only": {
            key: models["EHR + FullNote"][key] - models["EHR only"][key]
            for key in metric_keys
        },
        "files": [
            "test-metric-comparison.svg",
            "note-contribution-deltas.svg",
            "probability-loss-comparison.svg",
        ],
    }
    (output / "comparison_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
