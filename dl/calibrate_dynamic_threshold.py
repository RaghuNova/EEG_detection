# # FOR ATTENTION!!!!!
# """Calibrate an adaptive per-recording EEG seizure threshold from OOF scores.

# Run from the repository root after attention CV has produced:
#   artifacts/dl/attention/oof_predictions.npz
#   artifacts/dl/attention/oof_metadata.json

# The script uses only development OOF predictions for parameter selection. It
# loads event annotations from the project's manifest and never evaluates or
# optimizes on the independent test set.
# """
# from __future__ import annotations

# import argparse
# import csv
# import json
# import sys
# from collections import defaultdict
# from pathlib import Path
# from typing import Any, Mapping, Sequence

# import numpy as np
# from sklearn.metrics import (
#     accuracy_score,
#     average_precision_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )

# # When invoked as ``uv run python /path/to/script.py`` from the project root,
# # make that root importable. This is also harmless when run as a module.
# _CWD = Path.cwd()
# if (_CWD / "backend").is_dir() and str(_CWD) not in sys.path:
#     sys.path.insert(0, str(_CWD))

# try:  # package/module execution: python -m backend.dl.calibrate_dynamic_threshold
#     from .dynamic_threshold import (
#         build_alarm_episodes,
#         calculate_recording_thresholds,
#     )
#     from .config import (
#         ATTENTION_ARTIFACTS_DIR,
#         DL_ARTIFACTS_DIR,
#         EVENT_MATCH_MIN_OVERLAP_SECONDS,
#         EVENT_MAX_GAP_WINDOWS,
#         PREPROCESSED_MANIFEST_PATH,
#         WINDOW_SECONDS,
#     )
#     from .raw_data import load_ground_truth_events, load_recording_durations
#     from .train import _load_split
# except ImportError:  # direct script invocation from repository root
#     from backend.dl.dynamic_threshold import (
#         build_alarm_episodes,
#         calculate_recording_thresholds,
#     )
#     from backend.dl.config import (
#         ATTENTION_ARTIFACTS_DIR,
#         DL_ARTIFACTS_DIR,
#         EVENT_MATCH_MIN_OVERLAP_SECONDS,
#         EVENT_MAX_GAP_WINDOWS,
#         PREPROCESSED_MANIFEST_PATH,
#         WINDOW_SECONDS,
#     )
#     from backend.dl.raw_data import load_ground_truth_events, load_recording_durations
#     from backend.dl.train import _load_split


# def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
#     if np.unique(labels).size < 2:
#         return None
#     return float(roc_auc_score(labels, scores))


# def _macro_auc_by_group(
#     labels: np.ndarray,
#     probabilities: np.ndarray,
#     metadata: Sequence[Mapping[str, Any]],
#     group_key: str,
# ) -> tuple[float | None, list[dict[str, Any]]]:
#     grouped: dict[str, list[int]] = defaultdict(list)
#     for i, row in enumerate(metadata):
#         grouped[str(row[group_key])].append(i)
#     values: list[float] = []
#     rows: list[dict[str, Any]] = []
#     for group, indices in sorted(grouped.items()):
#         ix = np.asarray(indices, dtype=np.int64)
#         auc = _safe_roc_auc(labels[ix], probabilities[ix])
#         if auc is not None:
#             values.append(auc)
#         rows.append(
#             {
#                 group_key: group,
#                 "windows": int(len(ix)),
#                 "positive_windows": int(labels[ix].sum()),
#                 "roc_auc": auc,
#             }
#         )
#     return (float(np.mean(values)) if values else None), rows


# def _dynamic_event_metrics(
#     probabilities: np.ndarray,
#     labels: np.ndarray,
#     metadata: Sequence[Mapping[str, Any]],
#     thresholds: np.ndarray,
#     ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
#     recording_durations: Mapping[tuple[str, str], float],
#     *,
#     min_consecutive_windows: int,
#     max_gap_windows: int,
#     min_overlap_seconds: float,
#     window_seconds: float,
# ) -> dict[str, Any]:
#     """Event evaluation matching backend.dl.event_evaluation semantics."""
#     policy = {
#         "selected": {
#             "method": "recording_quantile",
#             "quantile": 0.99,
#             "floor": 0.95,
#             "quantile_method": "higher",
#             "min_consecutive_windows": min_consecutive_windows,
#             "max_gap_windows": max_gap_windows,
#             "window_seconds": window_seconds,
#         }
#     }
#     # The supplied thresholds are already candidate-specific. Generate event
#     # groups using the shared runtime helper and its persistence policy.
#     episodes, _ = build_alarm_episodes(
#         probabilities,
#         thresholds,
#         metadata,
#         min_consecutive_windows=min_consecutive_windows,
#         max_gap_windows=max_gap_windows,
#         window_seconds=window_seconds,
#     )

#     represented_keys = {
#         (str(row["patient_id"]), str(row["recording_id"])) for row in metadata
#     }
#     filtered_gt: dict[tuple[str, str], list[tuple[float, float]]] = {
#         (str(k[0]), str(k[1])): list(v)
#         for k, v in ground_truth_events.items()
#         if (str(k[0]), str(k[1])) in represented_keys
#     }

#     matched_true = 0
#     matched_predicted = 0
#     latencies: list[float] = []
#     used_predicted_indices: set[int] = set()
#     matched_by_patient: dict[str, int] = defaultdict(int)
#     actual_by_patient: dict[str, int] = defaultdict(int)
#     alarms_by_patient: dict[str, int] = defaultdict(int)

#     for ep in episodes:
#         alarms_by_patient[str(ep["patient_id"])] += 1
#     for rec_key, true_events in sorted(filtered_gt.items()):
#         pid = rec_key[0]
#         actual_by_patient[pid] += len(true_events)
#         rec_pred_indices = [
#             i
#             for i, ep in enumerate(episodes)
#             if (str(ep["patient_id"]), str(ep["recording_id"])) == rec_key
#         ]
#         for true_start, true_end in sorted(true_events, key=lambda x: x[0]):
#             best_idx: int | None = None
#             earliest_alert = float("inf")
#             for idx in rec_pred_indices:
#                 if idx in used_predicted_indices:
#                     continue
#                 pred = episodes[idx]
#                 overlap = max(
#                     0.0,
#                     min(float(pred["end_seconds"]), float(true_end))
#                     - max(float(pred["start_seconds"]), float(true_start)),
#                 )
#                 if overlap >= min_overlap_seconds:
#                     if float(pred["alert_time_seconds"]) < earliest_alert:
#                         earliest_alert = float(pred["alert_time_seconds"])
#                         best_idx = idx
#             if best_idx is not None:
#                 used_predicted_indices.add(best_idx)
#                 matched_true += 1
#                 matched_predicted += 1
#                 matched_by_patient[pid] += 1
#                 latency = max(
#                     0.0,
#                     float(episodes[best_idx]["alert_time_seconds"])
#                     - float(true_start),
#                 )
#                 latencies.append(latency)

#     actual_events = sum(len(v) for v in filtered_gt.values())
#     predicted_alarms = len(episodes)
#     false_alarms = predicted_alarms - matched_predicted

#     if recording_durations:
#         observed_seconds = sum(
#             float(recording_durations.get(key, 0.0)) for key in represented_keys
#         )
#     else:
#         seen: set[tuple[str, str, float, float]] = set()
#         observed_seconds = 0.0
#         for row in metadata:
#             key = (
#                 str(row["patient_id"]),
#                 str(row["recording_id"]),
#                 float(row["start_seconds"]),
#                 float(row["end_seconds"]),
#             )
#             if key not in seen:
#                 seen.add(key)
#                 observed_seconds += max(0.0, key[3] - key[2])

#     seizure_seconds = sum(
#         max(0.0, float(end) - float(start))
#         for events in filtered_gt.values()
#         for start, end in events
#     )
#     non_seizure_hours = max((observed_seconds - seizure_seconds) / 3600.0, 1e-6)
#     event_sensitivity = (
#         matched_true / actual_events
#         if actual_events
#         else (1.0 if predicted_alarms == 0 else 0.0)
#     )
#     event_precision = (
#         matched_predicted / predicted_alarms
#         if predicted_alarms
#         else (1.0 if actual_events == 0 else 0.0)
#     )
#     event_f1 = (
#         2.0 * event_precision * event_sensitivity / (event_precision + event_sensitivity)
#         if event_precision + event_sensitivity > 0
#         else 0.0
#     )

#     raw_predictions = (probabilities >= thresholds).astype(np.int8)
#     window_metrics = _window_metrics(labels, probabilities, raw_predictions)

#     patient_metrics: list[dict[str, Any]] = []
#     all_patients = sorted({str(row["patient_id"]) for row in metadata})
#     # Per-patient duration and seizure interval totals are used only for the
#     # false-alarm rate display; patient-level alarm/event counts are exact.
#     for pid in all_patients:
#         patient_keys = [key for key in represented_keys if key[0] == pid]
#         patient_observed = sum(float(recording_durations.get(key, 0.0)) for key in patient_keys)
#         patient_seizure = sum(
#             max(0.0, float(end) - float(start))
#             for key, evs in filtered_gt.items()
#             if key[0] == pid
#             for start, end in evs
#         )
#         patient_non_seizure_h = max((patient_observed - patient_seizure) / 3600.0, 1e-6)
#         p_actual = int(actual_by_patient.get(pid, 0))
#         p_detected = int(matched_by_patient.get(pid, 0))
#         p_alarms = int(alarms_by_patient.get(pid, 0))
#         patient_metrics.append(
#             {
#                 "patient_id": pid,
#                 "actual_events": p_actual,
#                 "detected_events": p_detected,
#                 "alarm_episodes": p_alarms,
#                 "false_alarm_episodes": p_alarms - p_detected,
#                 "event_sensitivity": p_detected / p_actual if p_actual else None,
#                 "event_precision": p_detected / p_alarms if p_alarms else None,
#                 "false_alarms_per_hour": (p_alarms - p_detected) / patient_non_seizure_h,
#             }
#         )

#     return {
#         "actual_seizure_events": int(actual_events),
#         "detected_events": int(matched_true),
#         "predicted_alarm_episodes": int(predicted_alarms),
#         "matched_episodes": int(matched_predicted),
#         "false_alarm_episodes": int(false_alarms),
#         "event_sensitivity": float(event_sensitivity),
#         "event_precision": float(event_precision),
#         "event_f1": float(event_f1),
#         "false_alarms_per_hour": float(false_alarms / non_seizure_hours),
#         "mean_detection_latency_seconds": float(np.mean(latencies)) if latencies else None,
#         "median_detection_latency_seconds": float(np.median(latencies)) if latencies else None,
#         "observation_hours": float(observed_seconds / 3600.0),
#         "non_seizure_hours": float(non_seizure_hours),
#         "window_metrics": window_metrics,
#         "patient_metrics": patient_metrics,
#         "alarm_episodes": episodes,
#     }


# def _window_metrics(
#     labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray
# ) -> dict[str, Any]:
#     tn = int(((labels == 0) & (predictions == 0)).sum())
#     fp = int(((labels == 0) & (predictions == 1)).sum())
#     fn = int(((labels == 1) & (predictions == 0)).sum())
#     tp = int(((labels == 1) & (predictions == 1)).sum())
#     negatives = int((labels == 0).sum())
#     specificity = tn / negatives if negatives else 0.0
#     return {
#         "accuracy": float(accuracy_score(labels, predictions)),
#         "precision": float(precision_score(labels, predictions, zero_division=0)),
#         "recall": float(recall_score(labels, predictions, zero_division=0)),
#         "specificity": float(specificity),
#         "f1": float(f1_score(labels, predictions, zero_division=0)),
#         "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
#         "roc_auc": _safe_roc_auc(labels, probabilities),
#         "average_precision": float(average_precision_score(labels, probabilities)),
#         "positive_windows": int(predictions.sum()),
#         "total_windows": int(len(labels)),
#     }


# def _candidate_policy(
#     probabilities: np.ndarray,
#     metadata: Sequence[Mapping[str, Any]],
#     ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
#     recording_durations: Mapping[tuple[str, str], float],
#     *,
#     quantile: float,
#     floor: float,
#     min_consecutive_windows: int,
#     max_gap_windows: int,
#     min_overlap_seconds: float,
#     window_seconds: float,
# ) -> dict[str, Any]:
#     policy = {
#         "method": "recording_quantile",
#         "quantile": float(quantile),
#         "floor": float(floor),
#         "quantile_method": "higher",
#         "min_consecutive_windows": int(min_consecutive_windows),
#         "max_gap_windows": int(max_gap_windows),
#         "window_seconds": float(window_seconds),
#     }
#     thresholds, _ = calculate_recording_thresholds(probabilities, metadata, {"selected": policy})
#     metrics = _dynamic_event_metrics(
#         probabilities,
#         np.asarray([int(row["label"]) for row in metadata], dtype=np.int64),
#         metadata,
#         thresholds,
#         ground_truth_events,
#         recording_durations,
#         min_consecutive_windows=min_consecutive_windows,
#         max_gap_windows=max_gap_windows,
#         min_overlap_seconds=min_overlap_seconds,
#         window_seconds=window_seconds,
#     )
#     wm = metrics["window_metrics"]
#     return {
#         **policy,
#         "event_sensitivity": metrics["event_sensitivity"],
#         "event_precision": metrics["event_precision"],
#         "event_f1": metrics["event_f1"],
#         "false_alarms_per_hour": metrics["false_alarms_per_hour"],
#         "detected_events": metrics["detected_events"],
#         "actual_seizure_events": metrics["actual_seizure_events"],
#         "predicted_alarm_episodes": metrics["predicted_alarm_episodes"],
#         "false_alarm_episodes": metrics["false_alarm_episodes"],
#         "mean_detection_latency_seconds": metrics["mean_detection_latency_seconds"],
#         "window_precision": wm["precision"],
#         "window_recall": wm["recall"],
#         "window_f1": wm["f1"],
#         "window_roc_auc": wm["roc_auc"],
#         "window_average_precision": wm["average_precision"],
#         "positive_windows": wm["positive_windows"],
#     }


# def _select_candidate(
#     candidates: list[dict[str, Any]],
#     *,
#     target_precision: float,
#     max_faph: float,
#     min_alarms: int,
# ) -> tuple[dict[str, Any], str]:
#     feasible = [
#         row
#         for row in candidates
#         if row["event_precision"] >= target_precision
#         and row["false_alarms_per_hour"] <= max_faph
#         and row["predicted_alarm_episodes"] >= min_alarms
#         and row["detected_events"] > 0
#     ]
#     if feasible:
#         selected = max(
#             feasible,
#             key=lambda row: (
#                 row["event_sensitivity"],
#                 row["event_precision"],
#                 -row["false_alarms_per_hour"],
#                 row["window_precision"],
#                 -(float(row["mean_detection_latency_seconds"]) if row["mean_detection_latency_seconds"] is not None else float("inf")),
#             ),
#         )
#         return selected, "maximize event sensitivity subject to precision, false-alarm, and minimum-alarm constraints"

#     eligible = [row for row in candidates if row["predicted_alarm_episodes"] >= min_alarms]
#     if not eligible:
#         eligible = candidates
#     selected = max(
#         eligible,
#         key=lambda row: (
#             row["event_precision"],
#             -row["false_alarms_per_hour"],
#             row["event_sensitivity"],
#             row["window_precision"],
#         ),
#     )
#     return selected, "no candidate met all requested constraints; prioritized event precision, then fewer false alarms, then sensitivity"


# def _parse_floats(values: Sequence[str]) -> list[float]:
#     return [float(value) for value in values]


# def main(argv: Sequence[str] | None = None) -> int:
#     parser = argparse.ArgumentParser(
#         description="Calibrate a per-recording dynamic threshold from attention OOF predictions"
#     )
#     parser.add_argument("--oof", type=Path, default=ATTENTION_ARTIFACTS_DIR / "oof_predictions.npz")
#     parser.add_argument("--metadata", type=Path, default=ATTENTION_ARTIFACTS_DIR / "oof_metadata.json")
#     parser.add_argument("--manifest", type=Path, default=PREPROCESSED_MANIFEST_PATH)
#     parser.add_argument("--output-dir", type=Path, default=ATTENTION_ARTIFACTS_DIR)
#     parser.add_argument("--target-event-precision", type=float, default=0.80)
#     parser.add_argument("--max-false-alarms-per-hour", type=float, default=0.02)
#     parser.add_argument("--min-alarm-episodes", type=int, default=5)
#     parser.add_argument(
#         "--quantiles", nargs="+", default=["0.985", "0.987", "0.989", "0.990", "0.992", "0.995", "0.997"]
#     )
#     parser.add_argument("--floors", nargs="+", default=["0.90", "0.92", "0.95", "0.98"])
#     parser.add_argument("--min-consecutive", nargs="+", type=int, default=[2, 3])
#     parser.add_argument("--max-gap-windows", type=int, default=EVENT_MAX_GAP_WINDOWS)
#     parser.add_argument("--min-overlap-seconds", type=float, default=EVENT_MATCH_MIN_OVERLAP_SECONDS)
#     parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
#     args = parser.parse_args(argv)

#     if not args.oof.is_file():
#         raise FileNotFoundError(f"OOF prediction file not found: {args.oof}")
#     if not args.metadata.is_file():
#         raise FileNotFoundError(f"OOF metadata file not found: {args.metadata}")
#     if not args.manifest.is_file():
#         raise FileNotFoundError(f"Manifest file not found: {args.manifest}")
#     if not 0.0 < args.target_event_precision <= 1.0:
#         raise ValueError("--target-event-precision must be in (0, 1]")
#     if args.max_false_alarms_per_hour < 0:
#         raise ValueError("--max-false-alarms-per-hour must be >= 0")
#     if args.min_alarm_episodes < 1:
#         raise ValueError("--min-alarm-episodes must be >= 1")

#     with np.load(args.oof, allow_pickle=False) as data:
#         probabilities = np.asarray(data["probabilities"], dtype=np.float64).reshape(-1)
#         labels_from_npz = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
#     metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
#     if not isinstance(metadata, list):
#         raise TypeError("OOF metadata must be a JSON list")
#     labels = np.asarray([int(row["label"]) for row in metadata], dtype=np.int64)
#     if len(probabilities) != len(metadata) or len(labels_from_npz) != len(metadata):
#         raise ValueError("OOF predictions, labels, and metadata have different lengths")
#     if not np.array_equal(labels_from_npz, labels):
#         raise ValueError("OOF NPZ labels do not match labels in oof_metadata.json")
#     if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
#         raise ValueError("OOF probabilities must be finite and in [0, 1]")
#     if np.unique(labels).size != 2:
#         raise ValueError("OOF labels must contain both seizure and non-seizure windows")

#     # Safety check: calibration predictions must belong to development patients,
#     # not the independent test partition.
#     split = _load_split()
#     expected_dev = {str(x) for x in split["dev_patients"]}
#     present_patients = {str(row["patient_id"]) for row in metadata}
#     unexpected = sorted(present_patients - expected_dev)
#     missing = sorted(expected_dev - present_patients)
#     if unexpected:
#         raise ValueError(
#             "OOF metadata contains patients outside the development split: "
#             + ", ".join(unexpected)
#         )
#     if missing:
#         print(
#             "WARNING: some development patients are absent from OOF metadata: "
#             + ", ".join(missing),
#             file=sys.stderr,
#         )

#     quarantine_path = DL_ARTIFACTS_DIR / "audit_report.json"
#     ground_truth_events = load_ground_truth_events(
#         args.manifest, quarantine_report_path=quarantine_path
#     )
#     recording_durations = load_recording_durations(
#         args.manifest, quarantine_report_path=quarantine_path
#     )

#     quantiles = _parse_floats(args.quantiles)
#     floors = _parse_floats(args.floors)
#     candidates: list[dict[str, Any]] = []
#     for quantile in quantiles:
#         for floor in floors:
#             for min_consecutive in args.min_consecutive:
#                 if min_consecutive < 1:
#                     raise ValueError("--min-consecutive values must be >= 1")
#                 candidates.append(
#                     _candidate_policy(
#                         probabilities,
#                         metadata,
#                         ground_truth_events,
#                         recording_durations,
#                         quantile=quantile,
#                         floor=floor,
#                         min_consecutive_windows=min_consecutive,
#                         max_gap_windows=args.max_gap_windows,
#                         min_overlap_seconds=args.min_overlap_seconds,
#                         window_seconds=args.window_seconds,
#                     )
#                 )

#     selected, selection_policy = _select_candidate(
#         candidates,
#         target_precision=args.target_event_precision,
#         max_faph=args.max_false_alarms_per_hour,
#         min_alarms=args.min_alarm_episodes,
#     )
#     selected_policy = {
#         "method": selected["method"],
#         "quantile": selected["quantile"],
#         "floor": selected["floor"],
#         "quantile_method": selected["quantile_method"],
#         "min_consecutive_windows": selected["min_consecutive_windows"],
#         "max_gap_windows": selected["max_gap_windows"],
#         "window_seconds": selected["window_seconds"],
#     }

#     patient_macro_auc, patient_auc_rows = _macro_auc_by_group(
#         labels, probabilities, metadata, "patient_id"
#     )
#     recording_macro_auc, recording_auc_rows = _macro_auc_by_group(
#         labels, probabilities, metadata, "recording_id"
#     )
#     pooled_auc = _safe_roc_auc(labels, probabilities)
#     pooled_ap = float(average_precision_score(labels, probabilities))

#     thresholds, recording_summary = calculate_recording_thresholds(
#         probabilities, metadata, {"selected": selected_policy}
#     )
#     selected_metrics = _dynamic_event_metrics(
#         probabilities,
#         labels,
#         metadata,
#         thresholds,
#         ground_truth_events,
#         recording_durations,
#         min_consecutive_windows=selected["min_consecutive_windows"],
#         max_gap_windows=args.max_gap_windows,
#         min_overlap_seconds=args.min_overlap_seconds,
#         window_seconds=args.window_seconds,
#     )

#     # Static-cutoff references are evaluated with the same persistence rule as
#     # the selected dynamic policy, so event-level comparisons are fair.
#     static_references: list[dict[str, Any]] = []
#     for threshold in (0.50, 0.90, 0.95, 0.99):
#         static_thresholds = np.full(len(probabilities), threshold, dtype=np.float64)
#         sm = _dynamic_event_metrics(
#             probabilities,
#             labels,
#             metadata,
#             static_thresholds,
#             ground_truth_events,
#             recording_durations,
#             min_consecutive_windows=selected["min_consecutive_windows"],
#             max_gap_windows=args.max_gap_windows,
#             min_overlap_seconds=args.min_overlap_seconds,
#             window_seconds=args.window_seconds,
#         )
#         static_references.append(
#             {
#                 "threshold": threshold,
#                 "event_sensitivity": sm["event_sensitivity"],
#                 "event_precision": sm["event_precision"],
#                 "false_alarms_per_hour": sm["false_alarms_per_hour"],
#                 "detected_events": sm["detected_events"],
#                 "predicted_alarm_episodes": sm["predicted_alarm_episodes"],
#                 "false_alarm_episodes": sm["false_alarm_episodes"],
#                 "window_precision": sm["window_metrics"]["precision"],
#                 "window_recall": sm["window_metrics"]["recall"],
#             }
#         )

#     threshold_values = np.asarray([row["threshold"] for row in recording_summary], dtype=float)
#     report = {
#         "method": "per_recording_quantile_threshold_plus_persistence",
#         "selection_policy": selection_policy,
#         "optimization_constraints": {
#             "target_event_precision": args.target_event_precision,
#             "maximum_false_alarms_per_hour": args.max_false_alarms_per_hour,
#             "minimum_alarm_episodes": args.min_alarm_episodes,
#         },
#         "selected": {**selected_policy, **{k: selected[k] for k in (
#             "event_sensitivity", "event_precision", "event_f1", "false_alarms_per_hour",
#             "detected_events", "actual_seizure_events", "predicted_alarm_episodes",
#             "false_alarm_episodes", "mean_detection_latency_seconds", "window_precision",
#             "window_recall", "window_f1", "window_roc_auc", "window_average_precision",
#             "positive_windows"
#         )}},
#         "oof_data": {
#             "windows": int(len(labels)),
#             "positive_windows": int(labels.sum()),
#             "negative_windows": int((labels == 0).sum()),
#             "patients": int(len(present_patients)),
#             "recordings": int(len(recording_summary)),
#             "pooled_roc_auc_raw_scores": pooled_auc,
#             "patient_macro_roc_auc": patient_macro_auc,
#             "recording_macro_roc_auc": recording_macro_auc,
#             "pooled_average_precision": pooled_ap,
#             "patient_auc": patient_auc_rows,
#             "recording_auc": recording_auc_rows,
#         },
#         "recording_threshold_distribution": {
#             "min": float(np.min(threshold_values)),
#             "p25": float(np.quantile(threshold_values, 0.25)),
#             "median": float(np.median(threshold_values)),
#             "p75": float(np.quantile(threshold_values, 0.75)),
#             "p95": float(np.quantile(threshold_values, 0.95)),
#             "max": float(np.max(threshold_values)),
#             "floor_applied_recordings": int(sum(
#                 row["threshold"] == selected_policy["floor"] for row in recording_summary
#             )),
#         },
#         "selected_policy_oof_event_evaluation": selected_metrics,
#         "static_threshold_references": static_references,
#         "warning": (
#             "Development OOF calibration only. The independent test set was not used. "
#             "The selected operating point is not a clinical guarantee. Re-evaluate once "
#             "on the untouched independent test set after applying the policy to final.pt."
#         ),
#     }

#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     calibration_path = args.output_dir / "dynamic_threshold_calibration.json"
#     candidates_path = args.output_dir / "dynamic_threshold_candidates.csv"
#     report_path = args.output_dir / "dynamic_threshold_oof_report.json"
#     calibration_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
#     with candidates_path.open("w", newline="", encoding="utf-8") as handle:
#         writer = csv.DictWriter(handle, fieldnames=list(candidates[0].keys()))
#         writer.writeheader()
#         writer.writerows(candidates)
#     report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

#     print("\nDYNAMIC THRESHOLD CALIBRATION")
#     print(f"OOF windows: {len(labels):,} | positive: {int(labels.sum()):,} | patients: {len(present_patients)}")
#     print(f"Pooled OOF ROC-AUC: {pooled_auc:.4f}")
#     print(f"Patient-macro ROC-AUC: {patient_macro_auc:.4f}" if patient_macro_auc is not None else "Patient-macro ROC-AUC: n/a")
#     print(f"Recording-macro ROC-AUC: {recording_macro_auc:.4f}" if recording_macro_auc is not None else "Recording-macro ROC-AUC: n/a")
#     print("\nSELECTED POLICY")
#     print(json.dumps(report["selected"], indent=2))
#     print("\nSaved:")
#     print(f"  {calibration_path}")
#     print(f"  {candidates_path}")
#     print(f"  {report_path}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
"""Calibrate hybrid global + recording-adaptive Attention thresholds from OOF data.

Run from repository root:
    uv run python -m backend.dl.calibrate_dynamic_threshold

Only development OOF predictions are used for selection. The independent test
partition is explicitly rejected if it appears in the OOF metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

_CWD = Path.cwd()
if (_CWD / "backend").is_dir() and str(_CWD) not in sys.path:
    sys.path.insert(0, str(_CWD))

try:
    from .dynamic_threshold import calculate_recording_thresholds, build_alarm_episodes
    from .config import (
        ATTENTION_ARTIFACTS_DIR, DL_ARTIFACTS_DIR, EVENT_MATCH_MIN_OVERLAP_SECONDS,
        EVENT_MAX_GAP_WINDOWS, PREPROCESSED_MANIFEST_PATH, WINDOW_SECONDS,
    )
    from .raw_data import load_ground_truth_events, load_recording_durations
    from .train import _load_split
except ImportError:
    from backend.dl.dynamic_threshold import calculate_recording_thresholds, build_alarm_episodes
    from backend.dl.config import (
        ATTENTION_ARTIFACTS_DIR, DL_ARTIFACTS_DIR, EVENT_MATCH_MIN_OVERLAP_SECONDS,
        EVENT_MAX_GAP_WINDOWS, PREPROCESSED_MANIFEST_PATH, WINDOW_SECONDS,
    )
    from backend.dl.raw_data import load_ground_truth_events, load_recording_durations
    from backend.dl.train import _load_split


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None


def _window_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc": _safe_auc(y, p),
        "average_precision": float(average_precision_score(y, p)),
        "positive_windows": int(pred.sum()),
        "total_windows": int(len(y)),
    }


def _validate_annotations(
    metadata: Sequence[Mapping[str, Any]],
    gt: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
    durations: Mapping[tuple[str, str], float],
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], list[tuple[float, float]]]]:
    represented = {(str(r["patient_id"]), str(r["recording_id"])) for r in metadata}
    normalized_gt = {(str(k[0]), str(k[1])): list(v) for k, v in gt.items()}
    matched = represented.intersection(normalized_gt)
    if not matched:
        raise ValueError(
            "No OOF recording IDs matched ground-truth annotations. "
            "Check raw_data.py deserialization and recording_id naming."
        )
    actual = sum(len(normalized_gt[k]) for k in matched)
    if actual <= 0:
        raise ValueError("Matched OOF recordings contain zero ground-truth seizure events.")
    # Ground-truth loaders commonly omit recordings with no annotated seizure.
    # That is valid: their alarms still count as false alarms. Require complete
    # duration coverage for every recording, and at least one matched seizure.
    missing_durations = represented - set(durations)
    if missing_durations:
        sample = sorted(missing_durations)[:5]
        raise ValueError(f"{len(missing_durations)} OOF recordings lack durations; examples={sample}")
    invalid = [k for k in represented if not np.isfinite(float(durations[k])) or float(durations[k]) <= 0]
    if invalid:
        raise ValueError(f"{len(invalid)} OOF recordings have invalid/zero durations; examples={invalid[:5]}")
    return represented, normalized_gt


def _evaluate(
    p: np.ndarray, y: np.ndarray, metadata: Sequence[Mapping[str, Any]],
    thresholds: np.ndarray, gt: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
    durations: Mapping[tuple[str, str], float], *, min_consecutive: int,
    max_gap: int, min_overlap: float, window_seconds: float,
) -> dict[str, Any]:
    episodes, persistent = build_alarm_episodes(
        p, thresholds, metadata, min_consecutive_windows=min_consecutive,
        max_gap_windows=max_gap, window_seconds=window_seconds,
    )
    represented = {(str(r["patient_id"]), str(r["recording_id"])) for r in metadata}
    actual = sum(len(gt[k]) for k in represented)
    used: set[int] = set()
    detected = 0
    latencies: list[float] = []
    actual_by_patient: dict[str, int] = defaultdict(int)
    detected_by_patient: dict[str, int] = defaultdict(int)
    alarms_by_patient: dict[str, int] = defaultdict(int)
    for ep in episodes:
        alarms_by_patient[str(ep["patient_id"])] += 1

    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, ep in enumerate(episodes):
        by_key[(str(ep["patient_id"]), str(ep["recording_id"]))].append(i)
    for key in sorted(represented):
        pid = key[0]
        for start, end in sorted(gt[key], key=lambda x: x[0]):
            actual_by_patient[pid] += 1
            matches = []
            for idx in by_key.get(key, []):
                if idx in used:
                    continue
                ep = episodes[idx]
                overlap = max(0.0, min(float(ep["end_seconds"]), float(end)) -
                              max(float(ep["start_seconds"]), float(start)))
                if overlap >= min_overlap:
                    matches.append(idx)
            if matches:
                best = min(matches, key=lambda i: float(episodes[i]["alert_time_seconds"]))
                used.add(best)
                detected += 1
                detected_by_patient[pid] += 1
                latencies.append(max(0.0, float(episodes[best]["alert_time_seconds"]) - float(start)))

    predicted = len(episodes)
    false_alarms = predicted - detected
    observed_seconds = sum(float(durations[k]) for k in represented)
    seizure_seconds = sum(max(0.0, float(e) - float(s)) for k in represented for s, e in gt[k])
    non_seizure_hours = max((observed_seconds - seizure_seconds) / 3600.0, 1e-9)
    sensitivity = detected / actual if actual else 0.0
    precision = detected / predicted if predicted else 0.0
    f1 = 2 * sensitivity * precision / (sensitivity + precision) if sensitivity + precision else 0.0

    raw = (p >= thresholds).astype(np.int8)
    wm = _window_metrics(y, p, raw)
    patients = sorted({str(r["patient_id"]) for r in metadata})
    patient_rows = []
    for pid in patients:
        keys = [k for k in represented if k[0] == pid]
        obs = sum(float(durations[k]) for k in keys)
        seiz = sum(max(0.0, float(e) - float(s)) for k in keys for s, e in gt[k])
        non_seiz_h = max((obs - seiz) / 3600.0, 1e-9)
        a, d, n = actual_by_patient[pid], detected_by_patient[pid], alarms_by_patient[pid]
        patient_rows.append({
            "patient_id": pid, "actual_events": a, "detected_events": d,
            "alarm_episodes": n, "false_alarm_episodes": n - d,
            "event_sensitivity": d / a if a else None,
            "event_precision": d / n if n else 0.0,
            "false_alarms_per_hour": (n - d) / non_seiz_h,
        })
    return {
        "actual_seizure_events": int(actual), "detected_events": int(detected),
        "predicted_alarm_episodes": int(predicted), "false_alarm_episodes": int(false_alarms),
        "event_sensitivity": float(sensitivity), "event_precision": float(precision),
        "event_f1": float(f1), "false_alarms_per_hour": float(false_alarms / non_seizure_hours),
        "mean_detection_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "median_detection_latency_seconds": float(np.median(latencies)) if latencies else None,
        "observation_hours": float(observed_seconds / 3600.0),
        "non_seizure_hours": float(non_seizure_hours),
        "window_metrics": wm, "patient_metrics": patient_rows,
        "alarm_episodes": episodes,
        "persistent_positive_windows": int(persistent.sum()),
    }


def _candidate(
    p: np.ndarray, y: np.ndarray, metadata: Sequence[Mapping[str, Any]],
    gt: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
    durations: Mapping[tuple[str, str], float], *, global_threshold: float,
    quantile: float, floor: float, blend: float, min_consecutive: int,
    max_gap: int, min_overlap: float, window_seconds: float,
) -> dict[str, Any]:
    policy = {
        "method": "hybrid_global_recording_quantile",
        "global_threshold": round(global_threshold, 5),
        "quantile": round(quantile, 5), "floor": round(floor, 5),
        "adaptive_blend": round(blend, 5), "quantile_method": "higher",
        "min_consecutive_windows": int(min_consecutive),
        "max_gap_windows": int(max_gap), "window_seconds": round(window_seconds, 5),
    }
    thresholds, summary = calculate_recording_thresholds(p, metadata, {"selected": policy})
    metrics = _evaluate(
        p, y, metadata, thresholds, gt, durations,
        min_consecutive=min_consecutive, max_gap=max_gap,
        min_overlap=min_overlap, window_seconds=window_seconds,
    )
    wm = metrics["window_metrics"]
    return {
        **policy,
        "event_sensitivity": metrics["event_sensitivity"],
        "event_precision": metrics["event_precision"],
        "event_f1": metrics["event_f1"],
        "false_alarms_per_hour": metrics["false_alarms_per_hour"],
        "detected_events": metrics["detected_events"],
        "actual_seizure_events": metrics["actual_seizure_events"],
        "predicted_alarm_episodes": metrics["predicted_alarm_episodes"],
        "false_alarm_episodes": metrics["false_alarm_episodes"],
        "mean_detection_latency_seconds": metrics["mean_detection_latency_seconds"],
        "window_precision": wm["precision"], "window_recall": wm["recall"],
        "window_f1": wm["f1"], "window_specificity": wm["specificity"],
        "positive_windows": wm["positive_windows"],
        "threshold_min": float(min(x["threshold"] for x in summary)),
        "threshold_median": float(np.median([x["threshold"] for x in summary])),
        "threshold_max": float(max(x["threshold"] for x in summary)),
    }


def _rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    # Sensitivity is first; then F1, precision, lower FA/hr, and shorter latency.
    latency = row["mean_detection_latency_seconds"]
    return (
        float(row["event_sensitivity"]),
        float(row["event_f1"]),
        float(row["event_precision"]),
        -float(row["false_alarms_per_hour"]),
        -(float(latency) if latency is not None else 1e12),
        float(row["window_specificity"]),
    )


def _load_global_threshold(path: Path | None, override: float | None) -> float:
    if override is not None:
        value = float(override)
    elif path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"Global calibration file not found: {path}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        selected = obj.get("selected", obj)
        value = float(selected["threshold"])
    else:
        value = 0.03
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Global threshold must be finite and in [0, 1]")
    return value


def _parse_floats(values: Sequence[str]) -> list[float]:
    return [float(x) for x in values]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate hybrid global + adaptive Attention threshold")
    parser.add_argument("--oof", type=Path, default=ATTENTION_ARTIFACTS_DIR / "oof_predictions.npz")
    parser.add_argument("--metadata", type=Path, default=ATTENTION_ARTIFACTS_DIR / "oof_metadata.json")
    parser.add_argument("--manifest", type=Path, default=PREPROCESSED_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=ATTENTION_ARTIFACTS_DIR)
    parser.add_argument("--global-calibration", type=Path,
                        default=ATTENTION_ARTIFACTS_DIR / "threshold_calibration.json")
    parser.add_argument("--global-threshold", type=float, default=None)
    parser.add_argument("--quantiles", nargs="+",
                        default=["0.90000", "0.92500", "0.95000", "0.96500", "0.97500",
                                 "0.98500", "0.99000", "0.99500"])
    parser.add_argument("--floors", nargs="+",
                        default=["0.03000", "0.10000", "0.25000", "0.50000",
                                 "0.75000", "0.90000", "0.95000", "0.98000"])
    parser.add_argument("--blends", nargs="+", default=["0.50000", "0.75000", "1.00000"])
    parser.add_argument("--min-consecutive", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-gaps", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-false-alarms-per-hour", type=float, default=None,
                        help="Optional constraint; omitted means optimize sensitivity/F1 without a hard cap.")
    parser.add_argument("--min-event-precision", type=float, default=None)
    parser.add_argument("--min-detected-events", type=int, default=1)
    parser.add_argument("--min-overlap-seconds", type=float, default=EVENT_MATCH_MIN_OVERLAP_SECONDS)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    args = parser.parse_args(argv)

    for path in (args.oof, args.metadata, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.window_seconds <= 0 or args.min_detected_events < 1:
        raise ValueError("window-seconds must be positive and min-detected-events >= 1")
    if args.max_false_alarms_per_hour is not None and args.max_false_alarms_per_hour < 0:
        raise ValueError("max-false-alarms-per-hour must be >= 0")
    if args.min_event_precision is not None and not 0 <= args.min_event_precision <= 1:
        raise ValueError("min-event-precision must be in [0, 1]")

    with np.load(args.oof, allow_pickle=False) as data:
        p = np.asarray(data["probabilities"], dtype=np.float64).reshape(-1)
        y_npz = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("OOF metadata must be a non-empty JSON list")
    y = np.asarray([int(r["label"]) for r in metadata], dtype=np.int64)
    if len(p) != len(metadata) or len(y_npz) != len(metadata) or not np.array_equal(y_npz, y):
        raise ValueError("OOF predictions, NPZ labels and metadata labels are inconsistent")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("OOF probabilities must be finite and in [0, 1]")
    if np.unique(y).size != 2:
        raise ValueError("OOF labels must contain both classes")

    split = _load_split()
    expected_dev = {str(x) for x in split["dev_patients"]}
    present = {str(r["patient_id"]) for r in metadata}
    unexpected = present - expected_dev
    if unexpected:
        raise ValueError(f"OOF contains non-development patients: {sorted(unexpected)}")
    if not present:
        raise ValueError("No patients in OOF metadata")

    quarantine = DL_ARTIFACTS_DIR / "audit_report.json"
    gt = load_ground_truth_events(args.manifest, quarantine_report_path=quarantine)
    durations = load_recording_durations(args.manifest, quarantine_report_path=quarantine)
    represented, gt = _validate_annotations(metadata, gt, durations)
    actual_events = sum(len(gt[k]) for k in represented)
    if actual_events != 78:
        print(f"WARNING: matched OOF ground-truth event count is {actual_events}, expected historical count 78.",
              file=sys.stderr)

    global_threshold = _load_global_threshold(args.global_calibration, args.global_threshold)
    quantiles, floors, blends = map(_parse_floats, (args.quantiles, args.floors, args.blends))
    if any(not 0 < q < 1 for q in quantiles):
        raise ValueError("All quantiles must be in (0, 1)")
    if any(not 0 <= f <= 1 for f in floors) or any(not 0 <= b <= 1 for b in blends):
        raise ValueError("Floors and blends must be in [0, 1]")
    if any(n < 1 for n in args.min_consecutive) or any(g < 0 for g in args.max_gaps):
        raise ValueError("Invalid persistence/gap candidates")

    candidates: list[dict[str, Any]] = []
    total = len(quantiles) * len(floors) * len(blends) * len(args.min_consecutive) * len(args.max_gaps)
    print(f"Evaluating {total:,} hybrid candidates on {len(p):,} development OOF windows...")
    for q in quantiles:
        for floor in floors:
            for blend in blends:
                for consecutive in args.min_consecutive:
                    for gap in args.max_gaps:
                        candidates.append(_candidate(
                            p, y, metadata, gt, durations,
                            global_threshold=global_threshold, quantile=q, floor=floor,
                            blend=blend, min_consecutive=consecutive, max_gap=gap,
                            min_overlap=args.min_overlap_seconds, window_seconds=args.window_seconds,
                        ))

    eligible = [r for r in candidates if r["detected_events"] >= args.min_detected_events]
    if args.min_event_precision is not None:
        constrained = [r for r in eligible if r["event_precision"] >= args.min_event_precision]
        if constrained:
            eligible = constrained
    if args.max_false_alarms_per_hour is not None:
        constrained = [r for r in eligible if r["false_alarms_per_hour"] <= args.max_false_alarms_per_hour]
        if constrained:
            eligible = constrained
    if not eligible:
        raise RuntimeError("No candidate detected any seizure event; inspect annotations, IDs and score distributions.")
    selected = max(eligible, key=_rank)
    selected_policy = {
        k: selected[k] for k in (
            "method", "global_threshold", "quantile", "floor", "adaptive_blend",
            "quantile_method", "min_consecutive_windows", "max_gap_windows", "window_seconds"
        )
    }
    thresholds, rec_summary = calculate_recording_thresholds(p, metadata, {"selected": selected_policy})
    selected_metrics = _evaluate(
        p, y, metadata, thresholds, gt, durations,
        min_consecutive=selected["min_consecutive_windows"],
        max_gap=selected["max_gap_windows"],
        min_overlap=args.min_overlap_seconds, window_seconds=args.window_seconds,
    )

    # Baselines use the same persistence settings to make comparisons fair.
    baselines = []
    for name, floor, q, blend in (
        ("global_only", global_threshold, 0.5, 0.0),
        ("dynamic_only", 0.98, 0.985, 1.0),
        ("hybrid_selected", selected["floor"], selected["quantile"], selected["adaptive_blend"]),
    ):
        policy = {
            "global_threshold": global_threshold, "floor": floor, "quantile": q,
            "adaptive_blend": blend, "quantile_method": "higher",
        }
        th, _ = calculate_recording_thresholds(p, metadata, {"selected": policy})
        m = _evaluate(
            p, y, metadata, th, gt, durations,
            min_consecutive=selected["min_consecutive_windows"],
            max_gap=selected["max_gap_windows"],
            min_overlap=args.min_overlap_seconds, window_seconds=args.window_seconds,
        )
        baselines.append({
            "name": name, "event_sensitivity": m["event_sensitivity"],
            "event_precision": m["event_precision"], "event_f1": m["event_f1"],
            "false_alarms_per_hour": m["false_alarms_per_hour"],
            "detected_events": m["detected_events"],
            "actual_seizure_events": m["actual_seizure_events"],
            "predicted_alarm_episodes": m["predicted_alarm_episodes"],
            "false_alarm_episodes": m["false_alarm_episodes"],
        })

    def round_floats(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: round_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [round_floats(v) for v in obj]
        if isinstance(obj, (float, np.floating)):
            return round(float(obj), 5)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        return obj

    report = round_floats({
        "schema_version": 2,
        "method": "hybrid_global_floor_plus_recording_quantile",
        "selection_policy": (
            "maximize event sensitivity, then event F1, event precision, "
            "minimize false alarms/hour, then latency"
        ),
        "global_threshold_source": str(args.global_calibration) if args.global_threshold is None else "CLI override",
        "global_threshold": global_threshold,
        "optimization_constraints": {
            "max_false_alarms_per_hour": args.max_false_alarms_per_hour,
            "min_event_precision": args.min_event_precision,
            "min_detected_events": args.min_detected_events,
        },
        "selected": {
            **selected_policy,
            **{k: selected[k] for k in (
                "event_sensitivity", "event_precision", "event_f1", "false_alarms_per_hour",
                "detected_events", "actual_seizure_events", "predicted_alarm_episodes",
                "false_alarm_episodes", "mean_detection_latency_seconds",
                "window_precision", "window_recall", "window_f1", "window_specificity",
                "positive_windows", "threshold_min", "threshold_median", "threshold_max",
            )},
        },
        "oof_data": {
            "windows": len(y), "positive_windows": int(y.sum()),
            "negative_windows": int((y == 0).sum()), "patients": len(present),
            "recordings": len(represented), "actual_seizure_events": actual_events,
            "pooled_roc_auc": _safe_auc(y, p),
            "pooled_average_precision": float(average_precision_score(y, p)),
        },
        "recording_threshold_distribution": {
            "min": min(x["threshold"] for x in rec_summary),
            "median": float(np.median([x["threshold"] for x in rec_summary])),
            "p95": float(np.quantile([x["threshold"] for x in rec_summary], 0.95)),
            "max": max(x["threshold"] for x in rec_summary),
            "recordings_at_global_floor": sum(abs(x["threshold"] - global_threshold) < 1e-10 for x in rec_summary),
        },
        "baselines_same_persistence": baselines,
        "selected_policy_oof_event_evaluation": selected_metrics,
        "candidate_count": len(candidates),
        "warning": (
            "Development OOF calibration only. No independent test data was used. "
            "The final checkpoint must be evaluated once with this frozen policy. "
            "Offline full-recording quantiles use the complete recording and are not causal real-time thresholds."
        ),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "hybrid_attention_threshold_calibration.json"
    candidates_path = args.output_dir / "hybrid_attention_threshold_candidates.csv"
    report_path = args.output_dir / "hybrid_attention_threshold_oof_report.json"
    calibration_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with candidates_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nHYBRID ATTENTION CALIBRATION")
    print(f"OOF windows: {len(y):,} | positive: {int(y.sum()):,} | patients: {len(present)}")
    print(f"Matched seizure events: {actual_events} | recordings: {len(represented)}")
    print(f"Global threshold: {global_threshold:.5f}")
    print("\nSELECTED POLICY")
    print(json.dumps(report["selected"], indent=2))
    print("\nBASELINES (same persistence)")
    print(json.dumps(report["baselines_same_persistence"], indent=2))
    print("\nSaved:")
    print(calibration_path)
    print(candidates_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
