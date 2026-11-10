"""Unit and integration tests for comprehensive event-level evaluation."""

import numpy as np
import pytest

from backend.dl.event_evaluation import (
    GroundTruthEvent,
    PredictedAlarmEpisode,
    evaluate_events,
    evaluate_window_metrics,
    group_positive_windows,
    per_patient_and_aggregate_event_evaluation,
)


def test_window_metrics_calculation() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.4, 0.05, 0.3])
    metrics = evaluate_window_metrics(labels, probs, threshold=0.5)

    assert metrics["threshold"] == 0.5
    assert metrics["total_windows"] == 8
    assert metrics["positive_windows"] == 3
    assert metrics["negative_windows"] == 5

    # Predictions at 0.5: [0, 0, 1, 1, 1, 0, 0, 0]
    # TP: indices 2, 3 -> 2
    # FP: index 4 -> 1
    # FN: index 5 -> 1
    # TN: indices 0, 1, 6, 7 -> 4
    cm = metrics["confusion_matrix"]
    assert cm["tp"] == 2
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tn"] == 4

    assert metrics["accuracy"] == 6 / 8
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert metrics["specificity"] == 4 / 5
    assert metrics["f1"] == pytest.approx(2 * (2/3) * (2/3) / ((2/3) + (2/3)))
    assert metrics["roc_auc"] is not None
    assert metrics["average_precision"] is not None


def test_window_metrics_single_class() -> None:
    labels = np.array([0, 0, 0, 0])
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = evaluate_window_metrics(labels, probs, threshold=0.5)
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["recall"] == 0.0
    assert metrics["specificity"] == 1.0


def test_group_positive_windows_merging_and_alert_time() -> None:
    # 20-second windows with max_gap_windows = 1 (20s gap allowed)
    metadata = [
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 20.0, "end_seconds": 40.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 40.0, "end_seconds": 60.0, "label": 0},  # gap 20s
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 60.0, "end_seconds": 80.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 120.0, "end_seconds": 140.0, "label": 1}, # gap 40s > 20s
    ]
    # probabilities: first two positive, third negative, fourth positive (merged!), fifth positive (separate episode)
    probs = [0.8, 0.9, 0.1, 0.85, 0.95]

    episodes = group_positive_windows(
        metadata,
        probs,
        threshold=0.5,
        min_consecutive_windows=1,
        max_gap_windows=1,
        window_seconds=20.0,
    )

    assert len(episodes) == 2
    ep1, ep2 = episodes[0], episodes[1]

    # ep1 should span 0.0 to 80.0 (merged across the 20s gap)
    assert ep1.patient_id == "P1"
    assert ep1.recording_id == "R1"
    assert ep1.start_seconds == 0.0
    assert ep1.end_seconds == 80.0
    # alert time is the end of the first positive window that triggers the alarm
    assert ep1.alert_time_seconds == 20.0
    assert ep1.window_count == 3
    assert ep1.duration_seconds == 80.0
    assert ep1.max_probability == 0.9

    # ep2 is from 120.0 to 140.0
    assert ep2.start_seconds == 120.0
    assert ep2.end_seconds == 140.0
    assert ep2.alert_time_seconds == 140.0
    assert ep2.window_count == 1


def test_group_positive_windows_min_consecutive_filter() -> None:
    metadata = [
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 100.0, "end_seconds": 120.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 120.0, "end_seconds": 140.0, "label": 1},
    ]
    probs = [0.8, 0.9, 0.95]
    # Filter for min_consecutive_windows = 2
    episodes = group_positive_windows(
        metadata,
        probs,
        threshold=0.5,
        min_consecutive_windows=2,
        max_gap_windows=0,
    )
    assert len(episodes) == 1
    assert episodes[0].start_seconds == 100.0
    assert episodes[0].end_seconds == 140.0
    assert episodes[0].alert_time_seconds == 140.0


def test_group_positive_windows_mismatched_length_raises() -> None:
    with pytest.raises(ValueError, match="metadata length"):
        group_positive_windows([{"patient_id": "P1", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0}], [0.8, 0.2])


def test_evaluate_events_synthetic_case() -> None:
    """Test full event matching, sensitivity, precision, false alarms/hour, latency."""
    # Recording R1 has 2 seizures: [15.0, 45.0], [100.0, 130.0]
    # Total observation time: 3600s (1 hour)
    gt_events = {
        ("P1", "R1"): [(15.0, 45.0), (100.0, 130.0)]
    }
    durations = {("P1", "R1"): 3600.0}

    # Windows (20-second steps from 0 to 200s, plus a dummy far window to total 3600s if durations not used)
    # Predicted episodes:
    # Ep 1: window [20.0, 40.0] (prob 0.8) -> matches seizure 1 ([15, 45]), alert_time=40.0, latency = 40.0 - 15.0 = 25.0s
    # Ep 2: window [100.0, 120.0] (prob 0.9) -> matches seizure 2 ([100, 130]), alert_time=120.0, latency = 120.0 - 100.0 = 20.0s
    # Ep 3: window [500.0, 520.0] (prob 0.85) -> false alarm episode!
    metadata = [
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 20.0, "end_seconds": 40.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 100.0, "end_seconds": 120.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 500.0, "end_seconds": 520.0, "label": 0},
    ]
    probabilities = [0.1, 0.8, 0.9, 0.85]

    results = evaluate_events(
        metadata,
        probabilities,
        gt_events,
        threshold=0.5,
        min_overlap_seconds=1.0,
        recording_durations=durations,
    )

    assert results["actual_seizure_events"] == 2
    assert results["detected_events"] == 2
    assert results["predicted_alarm_episodes"] == 3
    assert results["matched_episodes"] == 2
    assert results["false_alarm_episodes"] == 1

    assert results["event_sensitivity"] == 1.0  # 2 / 2
    assert results["event_precision"] == pytest.approx(2 / 3)
    assert results["event_f1"] == pytest.approx(2 * (1.0 * 2/3) / (1.0 + 2/3))

    # Latencies: [25.0, 20.0]
    assert results["latencies"] == [25.0, 20.0]
    assert results["mean_detection_latency_seconds"] == 22.5
    assert results["median_detection_latency_seconds"] == 22.5

    # Observation: 3600s total, seizure seconds = (45-15) + (130-100) = 60s
    # Non-seizure seconds = 3540s = 3540 / 3600 = 0.98333 hours
    # False alarms / hour = 1 / 0.98333 = 1.0169
    expected_fa_per_hour = 1.0 / (3540.0 / 3600.0)
    assert results["false_alarms_per_hour"] == pytest.approx(expected_fa_per_hour, rel=1e-3)


def test_evaluate_events_no_seizures_recording() -> None:
    """Handle edge case with zero ground truth seizures."""
    gt_events = {("P2", "R1"): []}
    durations = {("P2", "R1"): 3600.0}

    # Case A: 0 predictions
    meta = [
        {"patient_id": "P2", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0, "label": 0},
    ]
    res_a = evaluate_events(meta, [0.1], gt_events, threshold=0.5, recording_durations=durations)
    assert res_a["actual_seizure_events"] == 0
    assert res_a["detected_events"] == 0
    assert res_a["predicted_alarm_episodes"] == 0
    assert res_a["false_alarm_episodes"] == 0
    assert res_a["event_sensitivity"] == 1.0
    assert res_a["event_precision"] == 1.0
    assert res_a["false_alarms_per_hour"] == 0.0
    assert res_a["mean_detection_latency_seconds"] is None

    # Case B: 1 false alarm prediction
    res_b = evaluate_events(meta, [0.9], gt_events, threshold=0.5, recording_durations=durations)
    assert res_b["actual_seizure_events"] == 0
    assert res_b["detected_events"] == 0
    assert res_b["predicted_alarm_episodes"] == 1
    assert res_b["false_alarm_episodes"] == 1
    assert res_b["event_sensitivity"] == 0.0
    assert res_b["event_precision"] == 0.0
    assert res_b["false_alarms_per_hour"] == pytest.approx(1.0)


def test_evaluate_events_missed_seizure() -> None:
    gt_events = {("P1", "R1"): [(10.0, 50.0)]}
    durations = {("P1", "R1"): 3600.0}
    meta = [
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0, "label": 1},
    ]
    # Model predicted 0.2 (no alarm)
    res = evaluate_events(meta, [0.2], gt_events, threshold=0.5, recording_durations=durations)
    assert res["actual_seizure_events"] == 1
    assert res["detected_events"] == 0
    assert res["predicted_alarm_episodes"] == 0
    assert res["event_sensitivity"] == 0.0
    assert res["event_precision"] == 0.0
    assert res["false_alarms_per_hour"] == 0.0


def test_per_patient_and_aggregate_event_evaluation() -> None:
    gt_events = {
        ("P1", "R1"): [(10.0, 30.0)],
        ("P2", "R1"): [],
    }
    meta = [
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0, "label": 1},
        {"patient_id": "P2", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0, "label": 0},
    ]
    probs = [0.8, 0.1]
    res = per_patient_and_aggregate_event_evaluation(meta, probs, gt_events, threshold=0.5)

    assert "aggregate" in res
    assert "per_patient" in res
    assert "P1" in res["per_patient"]
    assert "P2" in res["per_patient"]

    assert res["per_patient"]["P1"]["detected_events"] == 1
    assert res["per_patient"]["P2"]["detected_events"] == 0
