"""Manifest-backed raw EEG windows for the independent DL branch.

Window decoding remains CPU-side; training transfers batches to the NVIDIA
CUDA device for CNN/Bi-LSTM computation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from .cache import _load_recording
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from .audit import canonicalize_channel_names
from .config import (
    EXPECTED_PATIENT_COUNT,
    POSITIVE_OVERLAP_SECONDS,
    SAMPLING_RATE_HZ,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
)


@dataclass(frozen=True)
class RawWindow:
    patient_id: str
    recording_id: str
    path: str
    window_id: int
    start_seconds: float
    end_seconds: float
    label: int
    channel_indices: tuple[int, ...]


NormalizationCache = dict[
    tuple[str, int, tuple[int, ...]],
    tuple[np.ndarray, np.ndarray, int],
]


def _overlap_label(start: float, end: float, intervals: Sequence[Any]) -> int:
    maximum = max(
        (
            max(0.0, min(end, float(interval.end_seconds)) - max(start, float(interval.start_seconds)))
            for interval in intervals
        ),
        default=0.0,
    )
    return int(maximum >= POSITIVE_OVERLAP_SECONDS)


def load_manifest_windows(
    manifest_path: Path,
    *,
    canonical_channels: Sequence[str],
    patients: set[str] | None = None,
    quarantine_report_path: Path | None = None,
) -> list[RawWindow]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"preprocessed manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = [item.get("patient_id") for item in manifest.get("patients", [])]
    if len(declared) != EXPECTED_PATIENT_COUNT or len(set(declared)) != len(declared):
        raise ValueError("manifest must contain exactly 20 unique patients")
    report_path = quarantine_report_path or manifest_path.parent / "dl" / "audit_report.json"
    quarantined_paths: set[str] = set()
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        quarantined_paths = {
            str(item["path"])
            for item in report.get("quarantined_recordings", [])
            if item.get("path")
        }
    windows: list[RawWindow] = []
    for patient in manifest["patients"]:
        patient_id = patient["patient_id"]
        if patients is not None and patient_id not in patients:
            continue
        for entry in patient.get("recordings", []):
            path = Path(entry["path"])
            if not path.is_file():
                raise FileNotFoundError(f"manifest shard does not exist: {path}")
            if str(path) in quarantined_paths:
                continue
            recording = joblib.load(path)
            data = np.asarray(recording.data_uv)
            if data.ndim != 2 or not np.isfinite(data).all():
                raise ValueError(f"{path}: data_uv must be finite and 2D")
            if float(recording.sampling_rate_hz) != SAMPLING_RATE_HZ:
                raise ValueError(f"{path}: expected {SAMPLING_RATE_HZ} Hz")
            canonical, indices, _ = canonicalize_channel_names(recording.channel_names)
            del canonical
            missing = [channel for channel in canonical_channels if channel not in indices]
            if missing:
                raise ValueError(f"{path}: missing canonical channels {missing}")
            selected = tuple(indices[channel] for channel in canonical_channels)
            for window_id, start_sample in enumerate(
                range(0, data.shape[1] - WINDOW_SAMPLES + 1, WINDOW_SAMPLES)
            ):
                end_sample = start_sample + WINDOW_SAMPLES
                start = start_sample / SAMPLING_RATE_HZ
                end = end_sample / SAMPLING_RATE_HZ
                windows.append(
                    RawWindow(
                        patient_id=patient_id,
                        recording_id=str(recording.recording_id),
                        path=str(path),
                        window_id=window_id,
                        start_seconds=start,
                        end_seconds=end,
                        label=_overlap_label(start, end, recording.seizure_intervals),
                        channel_indices=selected,
#                     )
#                 )
#     return windows
                    )
                )
    return windows


def load_ground_truth_events(
    manifest_path: Path,
    quarantine_report_path: Path | None = None,
) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Extract ground-truth seizure events [start_seconds, end_seconds] per recording."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"preprocessed manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = quarantine_report_path or manifest_path.parent / "dl" / "audit_report.json"
    quarantined_paths: set[str] = set()
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        quarantined_paths = {
            str(item["path"])
            for item in report.get("quarantined_recordings", [])
            if item.get("path")
        }
    events_by_recording: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for patient in manifest.get("patients", []):
        patient_id = patient.get("patient_id")
        for entry in patient.get("recordings", []):
            path = Path(entry["path"])
            if str(path) in quarantined_paths or not path.is_file():
                continue
            rec_id = str(entry.get("recording_id"))
            key = (str(patient_id), rec_id)
            try:
                # recording = joblib.load(path)
                recording = _load_recording(path)
                events = [
                    (float(interval.start_seconds), float(interval.end_seconds))
                    for interval in getattr(recording, "seizure_intervals", [])
                ]
                events_by_recording[key] = events
            except Exception:
                pass
    return events_by_recording


def load_recording_durations(
    manifest_path: Path,
    quarantine_report_path: Path | None = None,
) -> dict[tuple[str, str], float]:
    """Extract recording duration in seconds per recording."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"preprocessed manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = quarantine_report_path or manifest_path.parent / "dl" / "audit_report.json"
    quarantined_paths: set[str] = set()
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        quarantined_paths = {
            str(item["path"])
            for item in report.get("quarantined_recordings", [])
            if item.get("path")
        }
    durations: dict[tuple[str, str], float] = {}
    for patient in manifest.get("patients", []):
        patient_id = patient.get("patient_id")
        for entry in patient.get("recordings", []):
            path = Path(entry["path"])
            if str(path) in quarantined_paths or not path.is_file():
                continue
            rec_id = str(entry.get("recording_id"))
            key = (str(patient_id), rec_id)
            try:
                # recording = joblib.load(path)
                recording = _load_recording(path)
                durations[key] = float(getattr(recording, "duration_seconds", 0.0))
            except Exception:
                pass
    return durations



class RawEEGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy shard-backed dataset; normalization is supplied by the fold."""

    def __init__(self, windows: Sequence[RawWindow], mean: np.ndarray, std: np.ndarray) -> None:
        self.windows = list(windows)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
        self._cache_path: str | None = None
        self._cache_data: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        if self._cache_path != window.path:
            if window.path.endswith(".npy"):
                self._cache_data = np.asarray(np.load(window.path, mmap_mode="r"))
            else:
                self._cache_data = np.asarray(joblib.load(window.path).data_uv)
            self._cache_path = window.path
        assert self._cache_data is not None
        start = window.window_id * WINDOW_SAMPLES
        signal = self._cache_data[list(window.channel_indices), start : start + WINDOW_SAMPLES]
        signal = (signal.astype(np.float32) - self.mean[:, None]) / self.std[:, None]
        return torch.from_numpy(np.ascontiguousarray(signal)), torch.tensor(window.label, dtype=torch.float32)


def order_windows_for_shard_io(
    windows: Sequence[RawWindow],
    *,
    shuffle_shards: bool,
    seed: int,
) -> list[RawWindow]:
    """Keep each shard contiguous so lazy loading performs one disk read/shard."""

    grouped: dict[str, list[RawWindow]] = {}
    for window in windows:
        grouped.setdefault(window.path, []).append(window)
    rng = np.random.default_rng(seed)
    paths = list(grouped)
    if shuffle_shards:
        rng.shuffle(paths)
    ordered: list[RawWindow] = []
    for path in paths:
        items = grouped[path]
        if shuffle_shards:
            items = [items[index] for index in rng.permutation(len(items))]
        ordered.extend(items)
    return ordered


def fit_channel_normalization(
    windows: Sequence[RawWindow],
    *,
    cache: NormalizationCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit channel-wise mean/std using only the supplied training windows."""

    if not windows:
        raise ValueError("cannot fit normalization on no windows")
    grouped: dict[str, list[RawWindow]] = {}
    for window in windows:
        grouped.setdefault(window.path, []).append(window)
    stats_cache = cache if cache is not None else {}
    total = None
    total_sq = None
    count = 0
    for path, shard_windows in grouped.items():
        data: np.ndarray | None = None
        for window in shard_windows:
            key = (path, window.window_id, window.channel_indices)
            stats = stats_cache.get(key)
            if stats is None:
                if data is None:
                    data = (
                        np.asarray(np.load(path, mmap_mode="r"))
                        if path.endswith(".npy")
                        else np.asarray(joblib.load(path).data_uv)
                    )
                start = window.window_id * WINDOW_SAMPLES
                values = data[list(window.channel_indices), start : start + WINDOW_SAMPLES].astype(np.float64)
                stats = (values.sum(axis=1), (values * values).sum(axis=1), values.shape[1])
                stats_cache[key] = stats
            window_sum, window_sum_sq, window_count = stats
            total = window_sum if total is None else total + window_sum
            total_sq = window_sum_sq if total_sq is None else total_sq + window_sum_sq
            count += window_count
    assert total is not None and total_sq is not None
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
