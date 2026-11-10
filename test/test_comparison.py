"""Tests for comparison module."""

from pathlib import Path
import json
import pytest

from backend.dl.comparison import compute_comparison, save_comparison_json


def test_compute_comparison_and_deltas() -> None:
    baseline = {
        "aggregate": {
            "window_metrics": {
                "accuracy": 0.90,
                "precision": 0.50,
                "recall": 0.60,
                "specificity": 0.95,
                "f1": 0.545,
                "roc_auc": 0.85,
                "average_precision": 0.40,
            },
            "actual_seizure_events": 10,
            "detected_events": 6,
            "predicted_alarm_episodes": 12,
            "false_alarm_episodes": 6,
            "event_sensitivity": 0.60,
            "event_precision": 0.50,
            "event_f1": 0.545,
            "false_alarms_per_hour": 1.2,
            "mean_detection_latency_seconds": 22.0,
            "median_detection_latency_seconds": 20.0,
        }
    }
    attention = {
        "aggregate": {
            "window_metrics": {
                "accuracy": 0.92,
                "precision": 0.55,
                "recall": 0.65,
                "specificity": 0.96,
                "f1": 0.596,
                "roc_auc": 0.89,
                "average_precision": 0.48,
            },
            "actual_seizure_events": 10,
            "detected_events": 7,
            "predicted_alarm_episodes": 10,
            "false_alarm_episodes": 3,
            "event_sensitivity": 0.70,
            "event_precision": 0.70,
            "event_f1": 0.70,
            "false_alarms_per_hour": 0.6,
            "mean_detection_latency_seconds": 18.0,
            "median_detection_latency_seconds": 18.0,
        }
    }

    comp = compute_comparison(baseline, attention, comparison_name="CV Test")
    assert comp["comparison_type"] == "CV Test"

    # Check window deltas
    w_comp = comp["window_level"]
    assert w_comp["accuracy"]["delta_attention_minus_baseline"] == pytest.approx(0.02)
    assert w_comp["f1"]["delta_attention_minus_baseline"] == pytest.approx(0.051)

    # Check event deltas
    e_comp = comp["event_level"]
    assert e_comp["event_sensitivity"]["delta_attention_minus_baseline"] == pytest.approx(0.10)
    assert e_comp["false_alarms_per_hour"]["delta_attention_minus_baseline"] == pytest.approx(-0.6)
    assert e_comp["mean_detection_latency_seconds"]["delta_attention_minus_baseline"] == pytest.approx(-4.0)


def test_save_comparison_json(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "comp.json"
    data = {"test": 123, "val": 0.456}
    save_comparison_json(data, out)
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["test"] == 123
