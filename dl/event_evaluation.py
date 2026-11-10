# """Comprehensive event-level evaluation and comparison for DL pipelines.

# Computes window-level and event-level metrics consistently across CNN-BiLSTM
# and Attention architectures, preserving patient independence and clinical event
# matching rules.
# """

# from __future__ import annotations

# from dataclasses import asdict, dataclass
# import json
# from pathlib import Path
# from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# import numpy as np
# from sklearn.metrics import (
#     accuracy_score,
#     average_precision_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )

# from .config import (
#     DEFAULT_SEIZURE_THRESHOLD,
#     EVENT_MATCH_MIN_OVERLAP_SECONDS,
#     EVENT_MAX_GAP_WINDOWS,
#     EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS,
#     WINDOW_SECONDS,
# )


# @dataclass(frozen=True)
# class GroundTruthEvent:
#     patient_id: str
#     recording_id: str
#     start_seconds: float
#     end_seconds: float


# @dataclass(frozen=True)
# class PredictedAlarmEpisode:
#     patient_id: str
#     recording_id: str
#     start_seconds: float
#     end_seconds: float
#     alert_time_seconds: float
#     duration_seconds: float
#     window_count: int
#     mean_probability: float
#     max_probability: float


# def group_positive_windows(
#     metadata: Sequence[Mapping[str, Any]],
#     probabilities: Sequence[float] | np.ndarray,
#     *,
#     threshold: float = DEFAULT_SEIZURE_THRESHOLD,
#     min_consecutive_windows: int = EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS,
#     max_gap_windows: int = EVENT_MAX_GAP_WINDOWS,
#     window_seconds: float = float(WINDOW_SECONDS),
# ) -> list[PredictedAlarmEpisode]:
#     """Group positive windows into alarm episodes.

#     Uses configurable merging rules:
#     - positive window if probability >= threshold
#     - merge gap if current_start - previous_end <= max_gap_seconds
#     - filters out episodes shorter than min_consecutive_windows
#     - alert time is the end_seconds of the first positive window that triggers the alarm
#     """
#     if len(metadata) != len(probabilities):
#         raise ValueError(
#             f"metadata length ({len(metadata)}) does not match probabilities ({len(probabilities)})"
#         )

#     grouped_by_recording: dict[tuple[str, str], list[tuple[Mapping[str, Any], float]]] = {}
#     for meta, prob in zip(metadata, probabilities):
#         key = (str(meta["patient_id"]), str(meta["recording_id"]))
#         grouped_by_recording.setdefault(key, []).append((meta, float(prob)))

#     max_gap_seconds = window_seconds * max_gap_windows
#     episodes: list[PredictedAlarmEpisode] = []

#     for (patient_id, recording_id), items in sorted(grouped_by_recording.items()):
#         # Sort chronologically by start_seconds
#         sorted_items = sorted(items, key=lambda x: float(x[0]["start_seconds"]))
#         positive_items = [(meta, prob) for meta, prob in sorted_items if prob >= threshold]
#         candidate_groups: list[list[tuple[Mapping[str, Any], float]]] = []
#         active: list[tuple[Mapping[str, Any], float]] = []

#         for meta, prob in positive_items:
#             if not active:
#                 active = [(meta, prob)]
#                 continue
#             prev_end = float(active[-1][0]["end_seconds"])
#             curr_start = float(meta["start_seconds"])
#             gap = curr_start - prev_end
#             if gap <= max_gap_seconds + 1e-5:
#                 active.append((meta, prob))
#             else:
#                 candidate_groups.append(active)
#                 active = [(meta, prob)]

#         if active:
#             candidate_groups.append(active)


#         for group in candidate_groups:
#             if len(group) < min_consecutive_windows:
#                 continue
#             start_sec = float(group[0][0]["start_seconds"])
#             end_sec = float(group[-1][0]["end_seconds"])
#             # Alert time is the end of the detection window that satisfies min_consecutive_windows
#             alert_window = group[min_consecutive_windows - 1][0]
#             alert_time = float(alert_window["end_seconds"])
#             probs = [p for _, p in group]
#             episodes.append(
#                 PredictedAlarmEpisode(
#                     patient_id=patient_id,
#                     recording_id=recording_id,
#                     start_seconds=start_sec,
#                     end_seconds=end_sec,
#                     alert_time_seconds=alert_time,
#                     duration_seconds=end_sec - start_sec,
#                     window_count=len(group),
#                     mean_probability=float(np.mean(probs)),
#                     max_probability=float(np.max(probs)),
#                 )
#             )

#     return episodes


# def evaluate_window_metrics(
#     labels: np.ndarray,
#     probabilities: np.ndarray,
#     threshold: float = DEFAULT_SEIZURE_THRESHOLD,
# ) -> dict[str, Any]:
#     """Compute standard window-level classification metrics."""
#     labels_arr = np.asarray(labels, dtype=np.int64)
#     probs_arr = np.asarray(probabilities, dtype=np.float64)
#     preds_arr = (probs_arr >= threshold).astype(np.int64)

#     total_windows = int(labels_arr.size)
#     positive_windows = int(labels_arr.sum())
#     negative_windows = total_windows - positive_windows

#     tn = int(((labels_arr == 0) & (preds_arr == 0)).sum())
#     fp = int(((labels_arr == 0) & (preds_arr == 1)).sum())
#     fn = int(((labels_arr == 1) & (preds_arr == 0)).sum())
#     tp = int(((labels_arr == 1) & (preds_arr == 1)).sum())

#     specificity = float(tn / negative_windows) if negative_windows > 0 else 0.0

#     result: dict[str, Any] = {
#         "threshold": float(threshold),
#         "total_windows": total_windows,
#         "positive_windows": positive_windows,
#         "negative_windows": negative_windows,
#         "accuracy": float(accuracy_score(labels_arr, preds_arr)) if total_windows > 0 else 0.0,
#         "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
#         "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
#         "specificity": specificity,
#         "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
#         "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
#     }

#     if np.unique(labels_arr).size == 2:
#         try:
#             result["roc_auc"] = float(roc_auc_score(labels_arr, probs_arr))
#         except Exception:
#             result["roc_auc"] = None
#         try:
#             result["average_precision"] = float(average_precision_score(labels_arr, probs_arr))
#         except Exception:
#             result["average_precision"] = None
#     else:
#         result["roc_auc"] = None
#         result["average_precision"] = None

#     return result


# def evaluate_events(
#     metadata: Sequence[Mapping[str, Any]],
#     probabilities: Sequence[float] | np.ndarray,
#     ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
#     *,
#     threshold: float = DEFAULT_SEIZURE_THRESHOLD,
#     min_consecutive_windows: int = EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS,
#     max_gap_windows: int = EVENT_MAX_GAP_WINDOWS,
#     min_overlap_seconds: float = EVENT_MATCH_MIN_OVERLAP_SECONDS,
#     recording_durations: Mapping[tuple[str, str], float] | None = None,
# ) -> dict[str, Any]:
#     """Calculate event-level detection performance metrics.

#     Matching rules:
#     - Earliest-overlap matching between predicted episodes and ground-truth events.
#     - An episode matches a true event if max(0, min(pred_end, true_end) - max(pred_start, true_start)) >= min_overlap_seconds.
#     - One predicted episode may match at most one true event, and vice versa.
#     - Detection latency = alert_time_seconds - true_start_seconds.
#     - False alarms / hour = false_alarm_episodes / non_seizure_hours.
#     """
#     labels = np.asarray([int(m["label"]) for m in metadata], dtype=np.int64)
#     probs = np.asarray(probabilities, dtype=np.float64)

#     window_metrics = evaluate_window_metrics(labels, probs, threshold=threshold)
#     predicted_episodes = group_positive_windows(
#         metadata,
#         probs,
#         threshold=threshold,
#         min_consecutive_windows=min_consecutive_windows,
#         max_gap_windows=max_gap_windows,
#     )

#     # Filter ground_truth_events to only include recordings present in metadata
#     represented_keys = {(str(m["patient_id"]), str(m["recording_id"])) for m in metadata}
#     filtered_gt: dict[tuple[str, str], list[tuple[float, float]]] = {
#         key: list(events)
#         for key, events in ground_truth_events.items()
#         if key in represented_keys
#     }

#     total_actual_events = sum(len(evs) for evs in filtered_gt.values())
#     total_predicted_episodes = len(predicted_episodes)

#     matched_true = 0
#     matched_predicted = 0
#     latencies: list[float] = []
#     used_predicted_indices: set[int] = set()

#     for rec_key, true_events in sorted(filtered_gt.items()):
#         rec_pred_indices = [
#             i
#             for i, ep in enumerate(predicted_episodes)
#             if (ep.patient_id, ep.recording_id) == rec_key
#         ]
#         # Sort true events chronologically
#         sorted_true_events = sorted(true_events, key=lambda x: x[0])

#         for true_start, true_end in sorted_true_events:
#             best_idx: Optional[int] = None
#             earliest_alert = float("inf")

#             for idx in rec_pred_indices:
#                 if idx in used_predicted_indices:
#                     continue
#                 pred = predicted_episodes[idx]
#                 overlap = max(
#                     0.0,
#                     min(pred.end_seconds, true_end) - max(pred.start_seconds, true_start),
#                 )
#                 if overlap >= min_overlap_seconds:
#                     if pred.alert_time_seconds < earliest_alert:
#                         earliest_alert = pred.alert_time_seconds
#                         best_idx = idx

#             if best_idx is not None:
#                 used_predicted_indices.add(best_idx)
#                 matched_true += 1
#                 matched_predicted += 1
#                 pred_matched = predicted_episodes[best_idx]
#                 latency = max(0.0, pred_matched.alert_time_seconds - true_start)
#                 latencies.append(latency)

#     false_alarm_episodes = total_predicted_episodes - matched_predicted

#     # Compute observation time (duration)
#     total_observed_seconds = 0.0
#     if recording_durations:
#         for rec_key in represented_keys:
#             total_observed_seconds += float(recording_durations.get(rec_key, 0.0))
#     else:
#         # Sum duration from distinct non-overlapping windows per recording
#         seen_windows: set[tuple[str, str, float, float]] = set()
#         for meta in metadata:
#             w_key = (
#                 str(meta["patient_id"]),
#                 str(meta["recording_id"]),
#                 float(meta["start_seconds"]),
#                 float(meta["end_seconds"]),
#             )
#             if w_key not in seen_windows:
#                 seen_windows.add(w_key)
#                 total_observed_seconds += max(0.0, w_key[3] - w_key[2])

#     seizure_duration_seconds = 0.0
#     for true_events in filtered_gt.values():
#         for start, end in true_events:
#             seizure_duration_seconds += max(0.0, float(end) - float(start))

#     non_seizure_seconds = max(0.0, total_observed_seconds - seizure_duration_seconds)
#     non_seizure_hours = max(non_seizure_seconds / 3600.0, 1e-6)

#     false_alarms_per_hour = float(false_alarm_episodes / non_seizure_hours)
#     event_sensitivity = float(matched_true / total_actual_events) if total_actual_events > 0 else 1.0 if total_predicted_episodes == 0 else 0.0
#     event_precision = float(matched_predicted / total_predicted_episodes) if total_predicted_episodes > 0 else 1.0 if total_actual_events == 0 else 0.0
#     event_f1 = (
#         float(2 * event_precision * event_sensitivity / (event_precision + event_sensitivity))
#         if (event_precision + event_sensitivity) > 0
#         else 0.0
#     )

#     return {
#         "threshold": float(threshold),
#         "actual_seizure_events": total_actual_events,
#         "detected_events": matched_true,
#         "predicted_alarm_episodes": total_predicted_episodes,
#         "matched_episodes": matched_predicted,
#         "false_alarm_episodes": false_alarm_episodes,
#         "event_sensitivity": event_sensitivity,
#         "event_precision": event_precision,
#         "event_f1": event_f1,
#         "false_alarms_per_hour": false_alarms_per_hour,
#         "mean_detection_latency_seconds": float(np.mean(latencies)) if latencies else None,
#         "median_detection_latency_seconds": float(np.median(latencies)) if latencies else None,
#         "latencies": [float(lat) for lat in latencies],
#         "observation_hours": float(total_observed_seconds / 3600.0),
#         "non_seizure_hours": float(non_seizure_hours),
#         "window_metrics": window_metrics,
#         "episodes": [asdict(ep) for ep in predicted_episodes],
#     }


# def per_patient_and_aggregate_event_evaluation(
#     metadata: Sequence[Mapping[str, Any]],
#     probabilities: Sequence[float] | np.ndarray,
#     ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
#     *,
#     threshold: float = DEFAULT_SEIZURE_THRESHOLD,
#     recording_durations: Mapping[tuple[str, str], float] | None = None,
# ) -> dict[str, Any]:
#     """Run event evaluation overall and broken down per patient."""
#     overall = evaluate_events(
#         metadata,
#         probabilities,
#         ground_truth_events,
#         threshold=threshold,
#         recording_durations=recording_durations,
#     )

#     by_patient_meta: dict[str, list[Mapping[str, Any]]] = {}
#     by_patient_probs: dict[str, list[float]] = {}

#     for meta, prob in zip(metadata, probabilities):
#         pid = str(meta["patient_id"])
#         by_patient_meta.setdefault(pid, []).append(meta)
#         by_patient_probs.setdefault(pid, []).append(float(prob))

#     per_patient: dict[str, Any] = {}
#     for pid in sorted(by_patient_meta.keys()):
#         p_gt = {
#             k: v for k, v in ground_truth_events.items() if k[0] == pid
#         }
#         per_patient[pid] = evaluate_events(
#             by_patient_meta[pid],
#             by_patient_probs[pid],
#             p_gt,
#             threshold=threshold,
#             recording_durations=recording_durations,
#         )

#     return {
#         "aggregate": overall,
#         "per_patient": per_patient,
#     }
# #

"""Comprehensive event-level evaluation and comparison for DL pipelines.

Computes window-level and event-level metrics consistently across CNN-BiLSTM
and Attention architectures, preserving patient independence and clinical event
matching rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    DEFAULT_SEIZURE_THRESHOLD,
    EVENT_MATCH_MIN_OVERLAP_SECONDS,
    EVENT_MAX_GAP_WINDOWS,
    EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    WINDOW_SECONDS,
)


@dataclass(frozen=True)
class GroundTruthEvent:
    patient_id: str
    recording_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class PredictedAlarmEpisode:
    patient_id: str
    recording_id: str
    start_seconds: float
    end_seconds: float
    alert_time_seconds: float
    duration_seconds: float
    window_count: int
    mean_probability: float
    max_probability: float


def group_positive_windows(
    metadata: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_SEIZURE_THRESHOLD,
    min_consecutive_windows: int = EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    max_gap_windows: int = EVENT_MAX_GAP_WINDOWS,
    window_seconds: float = float(WINDOW_SECONDS),
) -> list[PredictedAlarmEpisode]:
    """Create alarm episodes using explicit consecutive-window and merge rules.

    1. A window is positive when ``probability >= threshold``.
    2. A trigger run must contain ``min_consecutive_windows`` truly adjacent
       positive windows (no intervening negative window).
    3. Trigger runs separated by at most ``max_gap_windows`` missing windows
       are merged into one alarm episode. Merging does not make a short run
       qualify as a trigger.
    4. Alert time is the end of the Nth positive window in the first qualifying
       run, where N is ``min_consecutive_windows``.
    """
    if len(metadata) != len(probabilities):
        raise ValueError(
            f"metadata length ({len(metadata)}) does not match probabilities ({len(probabilities)})"
        )
    if min_consecutive_windows < 1:
        raise ValueError("min_consecutive_windows must be >= 1")
    if max_gap_windows < 0:
        raise ValueError("max_gap_windows must be >= 0")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    grouped_by_recording: dict[tuple[str, str], list[tuple[Mapping[str, Any], float]]] = {}
    for meta, probability in zip(metadata, probabilities):
        key = (str(meta["patient_id"]), str(meta["recording_id"]))
        grouped_by_recording.setdefault(key, []).append((meta, float(probability)))

    max_gap_seconds = window_seconds * max_gap_windows
    episodes: list[PredictedAlarmEpisode] = []

    for (patient_id, recording_id), items in sorted(grouped_by_recording.items()):
        sorted_items = sorted(items, key=lambda item: float(item[0]["start_seconds"]))
        positive_runs: list[list[tuple[Mapping[str, Any], float]]] = []
        active_run: list[tuple[Mapping[str, Any], float]] = []

        for meta, probability in sorted_items:
            if probability < threshold:
                if active_run:
                    if len(active_run) >= min_consecutive_windows:
                        positive_runs.append(active_run)
                    active_run = []
                continue

            if not active_run:
                active_run = [(meta, probability)]
                continue

            previous_end = float(active_run[-1][0]["end_seconds"])
            current_start = float(meta["start_seconds"])
            # Adjacent windows touch at their boundary. A missing window creates
            # a positive temporal gap and therefore starts a new trigger run.
            if current_start - previous_end <= 1e-5:
                active_run.append((meta, probability))
            else:
                if len(active_run) >= min_consecutive_windows:
                    positive_runs.append(active_run)
                active_run = [(meta, probability)]

        if active_run and len(active_run) >= min_consecutive_windows:
            positive_runs.append(active_run)

        if not positive_runs:
            continue

        # Merge only already-qualified trigger runs. The gap is measured between
        # the end of one positive run and the start of the next positive run.
        merged_runs: list[list[list[tuple[Mapping[str, Any], float]]]] = []
        current_episode = [positive_runs[0]]
        for run in positive_runs[1:]:
            previous_run = current_episode[-1]
            gap = float(run[0][0]["start_seconds"]) - float(previous_run[-1][0]["end_seconds"])
            if gap <= max_gap_seconds + 1e-5:
                current_episode.append(run)
            else:
                merged_runs.append(current_episode)
                current_episode = [run]
        merged_runs.append(current_episode)

        for run_group in merged_runs:
            positive_items = [item for run in run_group for item in run]
            first_run = run_group[0]
            start_seconds = float(first_run[0][0]["start_seconds"])
            end_seconds = float(run_group[-1][-1][0]["end_seconds"])
            alert_time = float(first_run[min_consecutive_windows - 1][0]["end_seconds"])
            run_probabilities = [probability for _, probability in positive_items]
            episodes.append(
                PredictedAlarmEpisode(
                    patient_id=patient_id,
                    recording_id=recording_id,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    alert_time_seconds=alert_time,
                    duration_seconds=end_seconds - start_seconds,
                    window_count=len(positive_items),
                    mean_probability=float(np.mean(run_probabilities)),
                    max_probability=float(np.max(run_probabilities)),
                )
            )

    return episodes


def evaluate_window_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DEFAULT_SEIZURE_THRESHOLD,
) -> dict[str, Any]:
    """Compute standard window-level classification metrics."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    probs_arr = np.asarray(probabilities, dtype=np.float64)
    preds_arr = (probs_arr >= threshold).astype(np.int64)

    total_windows = int(labels_arr.size)
    positive_windows = int(labels_arr.sum())
    negative_windows = total_windows - positive_windows

    tn = int(((labels_arr == 0) & (preds_arr == 0)).sum())
    fp = int(((labels_arr == 0) & (preds_arr == 1)).sum())
    fn = int(((labels_arr == 1) & (preds_arr == 0)).sum())
    tp = int(((labels_arr == 1) & (preds_arr == 1)).sum())

    specificity = float(tn / negative_windows) if negative_windows > 0 else 0.0

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "total_windows": total_windows,
        "positive_windows": positive_windows,
        "negative_windows": negative_windows,
        "accuracy": float(accuracy_score(labels_arr, preds_arr)) if total_windows > 0 else 0.0,
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    if np.unique(labels_arr).size == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(labels_arr, probs_arr))
        except Exception:
            result["roc_auc"] = None
        try:
            result["average_precision"] = float(average_precision_score(labels_arr, probs_arr))
        except Exception:
            result["average_precision"] = None
    else:
        result["roc_auc"] = None
        result["average_precision"] = None

    return result


def evaluate_events(
    metadata: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float] | np.ndarray,
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
    *,
    threshold: float = DEFAULT_SEIZURE_THRESHOLD,
    min_consecutive_windows: int = EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    max_gap_windows: int = EVENT_MAX_GAP_WINDOWS,
    min_overlap_seconds: float = EVENT_MATCH_MIN_OVERLAP_SECONDS,
    recording_durations: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    """Calculate event-level detection performance metrics.

    Matching rules:
    - Earliest-overlap matching between predicted episodes and ground-truth events.
    - An episode matches a true event if max(0, min(pred_end, true_end) - max(pred_start, true_start)) >= min_overlap_seconds.
    - One predicted episode may match at most one true event, and vice versa.
    - Detection latency = alert_time_seconds - true_start_seconds.
    - False alarms / hour = false_alarm_episodes / non_seizure_hours.
    """
    labels = np.asarray([int(m["label"]) for m in metadata], dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)

    window_metrics = evaluate_window_metrics(labels, probs, threshold=threshold)
    predicted_episodes = group_positive_windows(
        metadata,
        probs,
        threshold=threshold,
        min_consecutive_windows=min_consecutive_windows,
        max_gap_windows=max_gap_windows,
    )

    # Filter ground_truth_events to only include recordings present in metadata
    represented_keys = {(str(m["patient_id"]), str(m["recording_id"])) for m in metadata}
    filtered_gt: dict[tuple[str, str], list[tuple[float, float]]] = {
        key: list(events)
        for key, events in ground_truth_events.items()
        if key in represented_keys
    }

    total_actual_events = sum(len(evs) for evs in filtered_gt.values())
    total_predicted_episodes = len(predicted_episodes)

    matched_true = 0
    matched_predicted = 0
    latencies: list[float] = []
    used_predicted_indices: set[int] = set()

    for rec_key, true_events in sorted(filtered_gt.items()):
        rec_pred_indices = [
            i
            for i, ep in enumerate(predicted_episodes)
            if (ep.patient_id, ep.recording_id) == rec_key
        ]
        # Sort true events chronologically
        sorted_true_events = sorted(true_events, key=lambda x: x[0])

        for true_start, true_end in sorted_true_events:
            best_idx: Optional[int] = None
            earliest_alert = float("inf")

            for idx in rec_pred_indices:
                if idx in used_predicted_indices:
                    continue
                pred = predicted_episodes[idx]
                overlap = max(
                    0.0,
                    min(pred.end_seconds, true_end) - max(pred.start_seconds, true_start),
                )
                if overlap >= min_overlap_seconds:
                    if pred.alert_time_seconds < earliest_alert:
                        earliest_alert = pred.alert_time_seconds
                        best_idx = idx

            if best_idx is not None:
                used_predicted_indices.add(best_idx)
                matched_true += 1
                matched_predicted += 1
                pred_matched = predicted_episodes[best_idx]
                latency = max(0.0, pred_matched.alert_time_seconds - true_start)
                latencies.append(latency)

    false_alarm_episodes = total_predicted_episodes - matched_predicted

    # Compute observation time (duration)
    total_observed_seconds = 0.0
    if recording_durations:
        for rec_key in represented_keys:
            total_observed_seconds += float(recording_durations.get(rec_key, 0.0))
    else:
        # Sum duration from distinct non-overlapping windows per recording
        seen_windows: set[tuple[str, str, float, float]] = set()
        for meta in metadata:
            w_key = (
                str(meta["patient_id"]),
                str(meta["recording_id"]),
                float(meta["start_seconds"]),
                float(meta["end_seconds"]),
            )
            if w_key not in seen_windows:
                seen_windows.add(w_key)
                total_observed_seconds += max(0.0, w_key[3] - w_key[2])

    seizure_duration_seconds = 0.0
    for true_events in filtered_gt.values():
        for start, end in true_events:
            seizure_duration_seconds += max(0.0, float(end) - float(start))

    non_seizure_seconds = max(0.0, total_observed_seconds - seizure_duration_seconds)
    non_seizure_hours = max(non_seizure_seconds / 3600.0, 1e-6)

    false_alarms_per_hour = float(false_alarm_episodes / non_seizure_hours)
    event_sensitivity = float(matched_true / total_actual_events) if total_actual_events > 0 else 1.0 if total_predicted_episodes == 0 else 0.0
    event_precision = float(matched_predicted / total_predicted_episodes) if total_predicted_episodes > 0 else 1.0 if total_actual_events == 0 else 0.0
    event_f1 = (
        float(2 * event_precision * event_sensitivity / (event_precision + event_sensitivity))
        if (event_precision + event_sensitivity) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "actual_seizure_events": total_actual_events,
        "detected_events": matched_true,
        "predicted_alarm_episodes": total_predicted_episodes,
        "matched_episodes": matched_predicted,
        "false_alarm_episodes": false_alarm_episodes,
        "event_sensitivity": event_sensitivity,
        "event_precision": event_precision,
        "event_f1": event_f1,
        "false_alarms_per_hour": false_alarms_per_hour,
        "mean_detection_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "median_detection_latency_seconds": float(np.median(latencies)) if latencies else None,
        "latencies": [float(lat) for lat in latencies],
        "observation_hours": float(total_observed_seconds / 3600.0),
        "non_seizure_hours": float(non_seizure_hours),
        "window_metrics": window_metrics,
        "episodes": [asdict(ep) for ep in predicted_episodes],
    }


def per_patient_and_aggregate_event_evaluation(
    metadata: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float] | np.ndarray,
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
    *,
    threshold: float = DEFAULT_SEIZURE_THRESHOLD,
    recording_durations: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    """Run event evaluation overall and broken down per patient."""
    overall = evaluate_events(
        metadata,
        probabilities,
        ground_truth_events,
        threshold=threshold,
        recording_durations=recording_durations,
    )

    by_patient_meta: dict[str, list[Mapping[str, Any]]] = {}
    by_patient_probs: dict[str, list[float]] = {}

    for meta, prob in zip(metadata, probabilities):
        pid = str(meta["patient_id"])
        by_patient_meta.setdefault(pid, []).append(meta)
        by_patient_probs.setdefault(pid, []).append(float(prob))

    per_patient: dict[str, Any] = {}
    for pid in sorted(by_patient_meta.keys()):
        p_gt = {
            k: v for k, v in ground_truth_events.items() if k[0] == pid
        }
        per_patient[pid] = evaluate_events(
            by_patient_meta[pid],
            by_patient_probs[pid],
            p_gt,
            threshold=threshold,
            recording_durations=recording_durations,
        )

    return {
        "aggregate": overall,
        "per_patient": per_patient,
    }
