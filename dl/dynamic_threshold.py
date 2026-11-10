# #for ATTENTION!!!!
# """Patient-folder adaptive thresholding for EEG seizure probabilities.

# The threshold is estimated separately for every EDF/recording from that
# recording's complete probability distribution:

#     threshold = max(minimum_floor, recording_probability_quantile)

# A prediction becomes an alarm only after N adjacent 20-second windows exceed
# that threshold. This module does not load a model and does not change any
# checkpoint. It is intended for offline EDF-folder processing, where the model
# can score a complete recording before alarms are assembled.
# """
# from __future__ import annotations

# from collections import defaultdict
# from typing import Any, Mapping, Sequence

# import numpy as np


# def _quantile(values: np.ndarray, q: float, method: str = "higher") -> float:
#     """NumPy-version-compatible quantile helper."""
#     try:
#         return float(np.quantile(values, q, method=method))
#     except TypeError:  # NumPy < 1.22
#         return float(np.quantile(values, q, interpolation=method))


# def _resolve_policy(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
#     """Accept either the full calibration JSON or its selected policy."""
#     selected = calibration.get("selected")
#     if isinstance(selected, Mapping):
#         return selected
#     selected_policy = calibration.get("selected_policy")
#     if isinstance(selected_policy, Mapping):
#         return selected_policy
#     return calibration


# def calculate_recording_thresholds(
#     probabilities: Sequence[float] | np.ndarray,
#     metadata: Sequence[Mapping[str, Any]],
#     calibration: Mapping[str, Any],
# ) -> tuple[np.ndarray, list[dict[str, Any]]]:
#     """Return one adaptive threshold per window and a per-recording summary.

#     Required metadata keys: patient_id, recording_id, start_seconds,
#     end_seconds. Labels are not used by this function.
#     """
#     probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
#     if len(probs) != len(metadata):
#         raise ValueError(
#             f"probability count ({len(probs)}) does not match metadata ({len(metadata)})"
#         )
#     if len(probs) == 0:
#         raise ValueError("No probabilities supplied")
#     if not np.isfinite(probs).all():
#         raise ValueError("Probabilities contain NaN or infinite values")
#     if np.any((probs < 0.0) | (probs > 1.0)):
#         raise ValueError("Probabilities must be in [0, 1]")

#     policy = _resolve_policy(calibration)
#     quantile = float(policy.get("quantile", 0.989))
#     floor = float(policy.get("floor", 0.92))
#     quantile_method = str(policy.get("quantile_method", "higher"))
#     if not 0.0 < quantile < 1.0:
#         raise ValueError("quantile must be strictly between 0 and 1")
#     if not 0.0 <= floor <= 1.0:
#         raise ValueError("floor must be in [0, 1]")

#     grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
#     for i, row in enumerate(metadata):
#         try:
#             key = (str(row["patient_id"]), str(row["recording_id"]))
#         except KeyError as exc:
#             raise ValueError(f"Metadata row {i} is missing {exc.args[0]!r}") from exc
#         grouped[key].append(i)

#     thresholds = np.empty(len(probs), dtype=np.float64)
#     recording_summary: list[dict[str, Any]] = []
#     for (patient_id, recording_id), indices in sorted(grouped.items()):
#         values = probs[np.asarray(indices, dtype=np.int64)]
#         quantile_value = _quantile(values, quantile, quantile_method)
#         threshold = max(floor, quantile_value)
#         thresholds[np.asarray(indices, dtype=np.int64)] = threshold
#         recording_summary.append(
#             {
#                 "patient_id": patient_id,
#                 "recording_id": recording_id,
#                 "n_windows": int(len(indices)),
#                 "quantile_value": float(quantile_value),
#                 "threshold": float(threshold),
#             }
#         )
#     return thresholds, recording_summary


# def build_alarm_episodes(
#     probabilities: Sequence[float] | np.ndarray,
#     thresholds: Sequence[float] | np.ndarray,
#     metadata: Sequence[Mapping[str, Any]],
#     *,
#     min_consecutive_windows: int = 2,
#     max_gap_windows: int = 0,
#     window_seconds: float = 20.0,
# ) -> tuple[list[dict[str, Any]], np.ndarray]:
#     """Group threshold-positive windows into alarm episodes.

#     A qualifying run requires ``min_consecutive_windows`` truly adjacent
#     positive windows. Qualified runs can then be merged if their time gap is
#     no larger than ``max_gap_windows`` windows. The returned boolean array
#     marks all windows belonging to qualifying runs (before episode merging).
#     """
#     probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
#     th = np.asarray(thresholds, dtype=np.float64).reshape(-1)
#     if len(probs) != len(metadata) or len(th) != len(probs):
#         raise ValueError("probabilities, thresholds, and metadata lengths differ")
#     if min_consecutive_windows < 1:
#         raise ValueError("min_consecutive_windows must be >= 1")
#     if max_gap_windows < 0:
#         raise ValueError("max_gap_windows must be >= 0")
#     if window_seconds <= 0:
#         raise ValueError("window_seconds must be positive")

#     raw_positive = probs >= th
#     persistent_positive = np.zeros(len(probs), dtype=bool)
#     grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
#     for i, row in enumerate(metadata):
#         grouped[(str(row["patient_id"]), str(row["recording_id"]))].append(i)

#     episodes: list[dict[str, Any]] = []
#     max_gap_seconds = window_seconds * max_gap_windows
#     for (patient_id, recording_id), indices in sorted(grouped.items()):
#         indices.sort(key=lambda i: float(metadata[i]["start_seconds"]))
#         qualifying_runs: list[list[int]] = []
#         active: list[int] = []
#         previous_end: float | None = None

#         for idx in indices:
#             row = metadata[idx]
#             start = float(row["start_seconds"])
#             end = float(row["end_seconds"])
#             is_positive = bool(raw_positive[idx])
#             adjacent = (
#                 previous_end is not None
#                 and abs(start - previous_end) <= 1e-5
#             )
#             if is_positive:
#                 if active and adjacent:
#                     active.append(idx)
#                 else:
#                     if len(active) >= min_consecutive_windows:
#                         qualifying_runs.append(active)
#                     active = [idx]
#                 previous_end = end
#             else:
#                 if len(active) >= min_consecutive_windows:
#                     qualifying_runs.append(active)
#                 active = []
#                 previous_end = None

#         if len(active) >= min_consecutive_windows:
#             qualifying_runs.append(active)
#         if not qualifying_runs:
#             continue

#         for run in qualifying_runs:
#             persistent_positive[np.asarray(run, dtype=np.int64)] = True

#         # Merge only runs that already passed the persistence requirement.
#         merged: list[list[list[int]]] = [[qualifying_runs[0]]]
#         for run in qualifying_runs[1:]:
#             previous_run = merged[-1][-1]
#             gap = (
#                 float(metadata[run[0]]["start_seconds"])
#                 - float(metadata[previous_run[-1]]["end_seconds"])
#             )
#             if gap <= max_gap_seconds + 1e-5:
#                 merged[-1].append(run)
#             else:
#                 merged.append([run])

#         for run_group in merged:
#             first_run = run_group[0]
#             last_run = run_group[-1]
#             positive_indices = [i for run in run_group for i in run]
#             alert_index = first_run[min_consecutive_windows - 1]
#             episodes.append(
#                 {
#                     "patient_id": patient_id,
#                     "recording_id": recording_id,
#                     "start_seconds": float(metadata[first_run[0]]["start_seconds"]),
#                     "end_seconds": float(metadata[last_run[-1]]["end_seconds"]),
#                     "alert_time_seconds": float(metadata[alert_index]["end_seconds"]),
#                     "duration_seconds": float(
#                         metadata[last_run[-1]]["end_seconds"]
#                         - metadata[first_run[0]]["start_seconds"]
#                     ),
#                     "window_count": int(len(positive_indices)),
#                     "mean_probability": float(np.mean(probs[positive_indices])),
#                     "max_probability": float(np.max(probs[positive_indices])),
#                     "threshold_used": float(th[positive_indices[0]]),
#                 }
#             )

#     return episodes, persistent_positive


# def apply_dynamic_threshold(
#     probabilities: Sequence[float] | np.ndarray,
#     metadata: Sequence[Mapping[str, Any]],
#     calibration: Mapping[str, Any],
#     *,
#     window_seconds: float | None = None,
# ) -> dict[str, Any]:
#     """Apply the selected policy to a complete set of recording predictions.

#     Returns per-window thresholds, raw threshold predictions, persistence-
#     filtered window flags, per-recording thresholds, and alarm episodes.
#     """
#     policy = _resolve_policy(calibration)
#     thresholds, recording_summary = calculate_recording_thresholds(
#         probabilities, metadata, calibration
#     )
#     configured_window_seconds = float(
#         window_seconds
#         if window_seconds is not None
#         else policy.get("window_seconds", 20.0)
#     )
#     min_consecutive = int(policy.get("min_consecutive_windows", 2))
#     max_gap = int(policy.get("max_gap_windows", 0))
#     episodes, persistent_flags = build_alarm_episodes(
#         probabilities,
#         thresholds,
#         metadata,
#         min_consecutive_windows=min_consecutive,
#         max_gap_windows=max_gap,
#         window_seconds=configured_window_seconds,
#     )
#     probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
#     raw_flags = probs >= thresholds
#     return {
#         "thresholds": thresholds,
#         "window_predictions": raw_flags.astype(np.int8),
#         "persistent_window_predictions": persistent_flags.astype(np.int8),
#         "recording_thresholds": recording_summary,
#         "alarm_episodes": episodes,
#     }
"""Hybrid global + recording-adaptive thresholding for Attention EEG inference.

Offline recording-level policy:
    T_recording = clip(
        global_threshold + blend * (max(floor, Q_q(recording_scores)) - global_threshold),
        global_threshold,
        1.0,
    )

The global threshold is learned from development OOF predictions. The adaptive
quantile adjusts it to each recording's score distribution. This module does
not load or modify a model checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _quantile(values: np.ndarray, q: float, method: str = "higher") -> float:
    try:
        return float(np.quantile(values, q, method=method))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, q, interpolation=method))


def _resolve_policy(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = calibration.get("selected")
    if isinstance(selected, Mapping):
        return selected
    selected = calibration.get("selected_policy")
    if isinstance(selected, Mapping):
        return selected
    return calibration


def calculate_recording_thresholds(
    probabilities: Sequence[float] | np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return one hybrid threshold per window and recording-level diagnostics."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(probs) != len(metadata):
        raise ValueError("probability count does not match metadata count")
    if not len(probs):
        raise ValueError("No probabilities supplied")
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")

    policy = _resolve_policy(calibration)
    q = float(policy.get("quantile", 0.985))
    floor = float(policy.get("floor", 0.03))
    global_threshold = float(policy.get("global_threshold", 0.03))
    blend = float(policy.get("adaptive_blend", 1.0))
    method = str(policy.get("quantile_method", "higher"))

    if not 0 < q < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    if not 0 <= floor <= 1 or not 0 <= global_threshold <= 1:
        raise ValueError("floor and global_threshold must be in [0, 1]")
    if not 0 <= blend <= 1:
        raise ValueError("adaptive_blend must be in [0, 1]")
    if method not in {"inverted_cdf", "averaged_inverted_cdf", "closest_observation",
                      "interpolated_inverted_cdf", "hazen", "weibull", "linear",
                      "median_unbiased", "normal_unbiased", "lower", "higher",
                      "midpoint", "nearest"}:
        raise ValueError(f"Unsupported quantile method: {method}")

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(metadata):
        try:
            key = (str(row["patient_id"]), str(row["recording_id"]))
        except KeyError as exc:
            raise ValueError(f"Metadata row {i} missing {exc.args[0]!r}") from exc
        if "start_seconds" not in row or "end_seconds" not in row:
            raise ValueError(f"Metadata row {i} lacks start_seconds/end_seconds")
        grouped[key].append(i)

    thresholds = np.empty(len(probs), dtype=np.float64)
    summary: list[dict[str, Any]] = []
    for (pid, rid), indices in sorted(grouped.items()):
        ix = np.asarray(indices, dtype=np.int64)
        values = probs[ix]
        q_value = _quantile(values, q, method)
        adaptive_target = max(floor, q_value)
        # Blend from the global operating point toward the recording-adaptive one.
        threshold = float(np.clip(
            global_threshold + blend * (adaptive_target - global_threshold),
            global_threshold, 1.0
        ))
        thresholds[ix] = threshold
        summary.append({
            "patient_id": pid,
            "recording_id": rid,
            "n_windows": int(len(ix)),
            "global_threshold": round(global_threshold, 5),
            "quantile": round(q, 5),
            "quantile_value": round(q_value, 5),
            "floor": round(floor, 5),
            "adaptive_blend": round(blend, 5),
            "threshold": round(threshold, 5),
        })
    return thresholds, summary


def build_alarm_episodes(
    probabilities: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    min_consecutive_windows: int = 2,
    max_gap_windows: int = 0,
    window_seconds: float = 20.0,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Create persistent alarm episodes; gaps only merge already-qualified runs."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    th = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if len(probs) != len(metadata) or len(th) != len(probs):
        raise ValueError("probabilities, thresholds, and metadata lengths differ")
    if not np.isfinite(probs).all() or not np.isfinite(th).all():
        raise ValueError("probabilities and thresholds must be finite")
    if min_consecutive_windows < 1 or max_gap_windows < 0 or window_seconds <= 0:
        raise ValueError("Invalid persistence/gap/window configuration")

    raw_positive = probs >= th
    persistent = np.zeros(len(probs), dtype=bool)
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(metadata):
        grouped[(str(row["patient_id"]), str(row["recording_id"]))].append(i)

    episodes: list[dict[str, Any]] = []
    max_gap_seconds = window_seconds * max_gap_windows
    for (pid, rid), indices in sorted(grouped.items()):
        indices.sort(key=lambda i: float(metadata[i]["start_seconds"]))
        runs: list[list[int]] = []
        active: list[int] = []
        previous_end: float | None = None
        for idx in indices:
            row = metadata[idx]
            start, end = float(row["start_seconds"]), float(row["end_seconds"])
            adjacent = previous_end is not None and abs(start - previous_end) <= 1e-5
            if raw_positive[idx]:
                if active and adjacent:
                    active.append(idx)
                else:
                    if len(active) >= min_consecutive_windows:
                        runs.append(active)
                    active = [idx]
                previous_end = end
            else:
                if len(active) >= min_consecutive_windows:
                    runs.append(active)
                active = []
                previous_end = None
        if len(active) >= min_consecutive_windows:
            runs.append(active)
        if not runs:
            continue

        for run in runs:
            persistent[np.asarray(run, dtype=np.int64)] = True

        merged: list[list[list[int]]] = [[runs[0]]]
        for run in runs[1:]:
            prev = merged[-1][-1]
            gap = float(metadata[run[0]]["start_seconds"]) - float(metadata[prev[-1]]["end_seconds"])
            if gap <= max_gap_seconds + 1e-5:
                merged[-1].append(run)
            else:
                merged.append([run])

        for group in merged:
            first, last = group[0], group[-1]
            positive_indices = [i for run in group for i in run]
            alert_idx = first[min_consecutive_windows - 1]
            episodes.append({
                "patient_id": pid,
                "recording_id": rid,
                "start_seconds": float(metadata[first[0]]["start_seconds"]),
                "end_seconds": float(metadata[last[-1]]["end_seconds"]),
                "alert_time_seconds": float(metadata[alert_idx]["end_seconds"]),
                "duration_seconds": float(metadata[last[-1]]["end_seconds"] -
                                           metadata[first[0]]["start_seconds"]),
                "window_count": int(len(positive_indices)),
                "mean_probability": float(np.mean(probs[positive_indices])),
                "max_probability": float(np.max(probs[positive_indices])),
                "threshold_used": float(th[positive_indices[0]]),
            })
    episodes.sort(key=lambda x: (x["patient_id"], x["recording_id"], x["start_seconds"]))
    return episodes, persistent


def apply_dynamic_threshold(
    probabilities: Sequence[float] | np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    *,
    window_seconds: float | None = None,
) -> dict[str, Any]:
    """Apply a saved hybrid policy to complete recording predictions."""
    policy = _resolve_policy(calibration)
    thresholds, recording_summary = calculate_recording_thresholds(
        probabilities, metadata, calibration
    )
    win = float(window_seconds if window_seconds is not None
                else policy.get("window_seconds", 20.0))
    min_consecutive = int(policy.get("min_consecutive_windows", 2))
    max_gap = int(policy.get("max_gap_windows", 0))
    episodes, persistent = build_alarm_episodes(
        probabilities, thresholds, metadata,
        min_consecutive_windows=min_consecutive,
        max_gap_windows=max_gap,
        window_seconds=win,
    )
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    return {
        "thresholds": thresholds,
        "window_predictions": (probs >= thresholds).astype(np.int8),
        "persistent_window_predictions": persistent.astype(np.int8),
        "recording_thresholds": recording_summary,
        "alarm_episodes": episodes,
    }
