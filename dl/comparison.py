"""Model comparison between CNN-BiLSTM baseline and Additive Temporal Attention.

Generates structured comparison files for cross-validation and final independent test:
- artifacts/dl/comparison/cv_comparison.json
- artifacts/dl/comparison/final_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def compute_comparison(
    baseline_metrics: Mapping[str, Any],
    attention_metrics: Mapping[str, Any],
    *,
    comparison_name: str = "CV",
) -> dict[str, Any]:
    """Compare baseline CNN-BiLSTM and Attention evaluation results side-by-side."""
    window_keys = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    event_keys = [
        "actual_seizure_events",
        "detected_events",
        "predicted_alarm_episodes",
        "false_alarm_episodes",
        "event_sensitivity",
        "event_precision",
        "event_f1",
        "false_alarms_per_hour",
        "mean_detection_latency_seconds",
        "median_detection_latency_seconds",
    ]

    base_agg = baseline_metrics.get("aggregate", baseline_metrics)
    attn_agg = attention_metrics.get("aggregate", attention_metrics)

    base_window = base_agg.get("window_metrics", {})
    attn_window = attn_agg.get("window_metrics", {})

    window_comparison: dict[str, Any] = {}
    for key in window_keys:
        b_val = base_window.get(key)
        a_val = attn_window.get(key)
        diff = (
            float(a_val - b_val)
            if b_val is not None and a_val is not None
            else None
        )
        window_comparison[key] = {
            "baseline": b_val,
            "attention": a_val,
            "delta_attention_minus_baseline": diff,
        }

    event_comparison: dict[str, Any] = {}
    for key in event_keys:
        b_val = base_agg.get(key)
        a_val = attn_agg.get(key)
        diff = (
            float(a_val - b_val)
            if isinstance(b_val, (int, float)) and isinstance(a_val, (int, float))
            else None
        )
        event_comparison[key] = {
            "baseline": b_val,
            "attention": a_val,
            "delta_attention_minus_baseline": diff,
        }

    return {
        "comparison_type": comparison_name,
        "window_level": window_comparison,
        "event_level": event_comparison,
        "baseline_summary": base_agg,
        "attention_summary": attn_agg,
    }


def save_comparison_json(
    data: Mapping[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=False, default=float),
        encoding="utf-8",
    )
