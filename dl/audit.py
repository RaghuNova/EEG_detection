"""Audit raw preprocessing shards before CUDA DL cross-validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np

from .config import (
    AUDIT_REPORT_PATH,
    EXPECTED_PATIENT_COUNT,
    POSITIVE_OVERLAP_SECONDS,
    SAMPLING_RATE_HZ,
    SEED,
    SPLIT_MANIFEST_PATH,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
)

_SEPARATOR_RE = re.compile(r"[\s_\-–—]+")


@dataclass(frozen=True)
class Interval:
    start_seconds: float
    end_seconds: float


@dataclass
class RecordingAudit:
    patient_id: str
    recording_id: str
    path: str
    channel_names: list[str]
    canonical_indices: dict[str, int]
    sample_count: int
    duration_seconds: float
    seizure_intervals: list[Any]
    original_to_canonical: list[dict[str, Any]] | None = None
    duplicate_channels: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class WindowMetadata:
    patient_id: str
    recording_id: str
    window_id: int
    start_seconds: float
    end_seconds: float
    label: int


def normalize_channel_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"channel name must be a string, received {type(name).__name__}")
    normalized = _SEPARATOR_RE.sub("-", name.strip().upper()).strip("-")
    if normalized in {"T8-P8-0", "T8-P8-1"}:
        normalized = "T8-P8"
    if not normalized:
        raise ValueError("channel name is empty after normalization")
    return normalized


def canonicalize_channel_names(
    channel_names: Sequence[str],
) -> tuple[list[str], dict[str, int], list[dict[str, Any]]]:
    canonical: list[str] = []
    index_by_channel: dict[str, int] = {}
    duplicates: list[dict[str, Any]] = []
    for index, raw_name in enumerate(channel_names):
        name = normalize_channel_name(raw_name)
        if name not in index_by_channel:
            index_by_channel[name] = index
            canonical.append(name)
            continue
        if name != "T8-P8":
            raise ValueError(
                f"ambiguous channel identity: {name!r} occurs at indices "
                f"{index_by_channel[name]} and {index}; only duplicated T8-P8 is permitted"
            )
        duplicates.append(
            {
                "canonical_channel": name,
                "kept_index": index_by_channel[name],
                "dropped_index": index,
            }
        )
    return canonical, index_by_channel, duplicates


def _get_attr(recording: Any, attr: str) -> Any:
    if not hasattr(recording, attr):
        raise ValueError(f"CleanEEGRecording is missing required attribute {attr!r}")
    return getattr(recording, attr)


def load_recording(path: str | Path, declared_patient_id: str, declared_recording_id: str) -> RecordingAudit:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"preprocessed recording file does not exist: {path}")
    try:
        recording = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"failed to load preprocessed recording {path}: {exc}") from exc

    patient_id = _get_attr(recording, "patient_id")
    recording_id = _get_attr(recording, "recording_id")
    if patient_id != declared_patient_id:
        raise ValueError(f"{path}: patient_id {patient_id!r} disagrees with manifest {declared_patient_id!r}")
    if recording_id != declared_recording_id:
        raise ValueError(f"{path}: recording_id {recording_id!r} disagrees with manifest {declared_recording_id!r}")

    data_uv = np.asarray(_get_attr(recording, "data_uv"))
    channel_names = list(_get_attr(recording, "channel_names"))
    sampling_rate = _get_attr(recording, "sampling_rate_hz")
    duration_seconds = float(_get_attr(recording, "duration_seconds"))
    seizure_intervals = list(_get_attr(recording, "seizure_intervals"))

    if data_uv.ndim != 2:
        raise ValueError(f"{path}: data_uv must be 2D [channels, samples], got {data_uv.shape}")
    if not np.isfinite(data_uv).all():
        raise ValueError(f"{path}: data_uv contains non-finite values")
    if float(sampling_rate) != SAMPLING_RATE_HZ:
        raise ValueError(f"{path}: sampling rate must be {SAMPLING_RATE_HZ} Hz, got {sampling_rate}")
    if data_uv.shape[0] != len(channel_names):
        raise ValueError(f"{path}: channel count {data_uv.shape[0]} does not match {len(channel_names)} names")
    if data_uv.shape[1] == 0:
        raise ValueError(f"{path}: data_uv has zero samples")
    expected_duration = data_uv.shape[1] / SAMPLING_RATE_HZ
    if duration_seconds <= 0 or not np.isclose(duration_seconds, expected_duration, rtol=0, atol=1 / SAMPLING_RATE_HZ):
        raise ValueError(
            f"{path}: duration {duration_seconds} disagrees with sample count / rate {expected_duration}"
        )
    for interval in seizure_intervals:
        start = float(_get_attr(interval, "start_seconds"))
        end = float(_get_attr(interval, "end_seconds"))
        if start < 0 or end <= start or end > duration_seconds:
            raise ValueError(f"{path}: invalid seizure interval [{start}, {end}] for duration {duration_seconds}")

    canonical, indices, duplicates = canonicalize_channel_names(channel_names)
    mappings = [
        {"index": index, "original": raw, "canonical": normalize_channel_name(raw)}
        for index, raw in enumerate(channel_names)
    ]
    return RecordingAudit(
        patient_id=patient_id,
        recording_id=recording_id,
        path=str(path),
        channel_names=canonical,
        canonical_indices=indices,
        sample_count=int(data_uv.shape[1]),
        duration_seconds=duration_seconds,
        seizure_intervals=seizure_intervals,
        original_to_canonical=mappings,
        duplicate_channels=duplicates,
    )


def make_windows(
    record: RecordingAudit,
    *,
    sample_rate_hz: int = SAMPLING_RATE_HZ,
    window_seconds: int = WINDOW_SECONDS,
    stride_seconds: int = WINDOW_STRIDE_SECONDS,
) -> list[WindowMetadata]:
    window_samples = sample_rate_hz * window_seconds
    stride_samples = sample_rate_hz * stride_seconds
    windows: list[WindowMetadata] = []
    for window_id, start_sample in enumerate(range(0, record.sample_count - window_samples + 1, stride_samples)):
        end_sample = start_sample + window_samples
        start_seconds = start_sample / sample_rate_hz
        end_seconds = end_sample / sample_rate_hz
        maximum_overlap = max(
            (
                max(0.0, min(end_seconds, float(interval.end_seconds)) - max(start_seconds, float(interval.start_seconds)))
                for interval in record.seizure_intervals
            ),
            default=0.0,
        )
        windows.append(
            WindowMetadata(
                patient_id=record.patient_id,
                recording_id=record.recording_id,
                window_id=window_id,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                label=int(maximum_overlap >= POSITIVE_OVERLAP_SECONDS),
            )
        )
    return windows


def build_patient_split(patient_counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    if len(patient_counts) != EXPECTED_PATIENT_COUNT:
        raise ValueError(f"expected {EXPECTED_PATIENT_COUNT} patients, received {len(patient_counts)}")
    seizure_bearing = sorted(
        (patient for patient, counts in patient_counts.items() if counts["seizure_events"] > 0),
        key=lambda patient: (-patient_counts[patient]["seizure_events"], patient),
    )
    dev: list[str] = []
    test: list[str] = []
    pattern = ("dev", "test", "test", "dev")
    for index, patient in enumerate(seizure_bearing):
        (dev if pattern[index % len(pattern)] == "dev" else test).append(patient)

    non_seizure = sorted(patient for patient in patient_counts if patient not in seizure_bearing)
    # When seizure-bearing assignment leaves dev and test tied, start non-seizure
    # assignment with "test" (deterministic tie-breaker; either side produces a
    # valid 10/10 split because 20 − 2k non-seizure patients is always even).
    next_group = "dev" if len(dev) < len(test) else "test"
    for patient in non_seizure:
        if next_group == "dev":
            dev.append(patient)
            next_group = "test"
        else:
            test.append(patient)
            next_group = "dev"
    if len(dev) != 10 or len(test) != 10:
        raise ValueError(f"deterministic split must be 10/10, got dev={len(dev)} test={len(test)}")

    ordered_dev = sorted(dev, key=lambda patient: (-patient_counts[patient]["seizure_events"], patient))
    validation_pairs = [(ordered_dev[i], ordered_dev[-(i + 1)]) for i in range(5)]
    folds: list[dict[str, Any]] = []
    for fold_id, pair in enumerate(validation_pairs, start=1):
        validation = list(pair)
        training = [patient for patient in ordered_dev if patient not in validation]
        folds.append({"fold": fold_id, "training_patients": training, "validation_patients": validation})
    validation_patients = [patient for fold in folds for patient in fold["validation_patients"]]
    if len(set(validation_patients)) != 10 or set(validation_patients) != set(dev):
        raise AssertionError("each development patient must validate exactly once")
    if set(dev) & set(test):
        raise AssertionError("development and test patients overlap")
    return {"dev_patients": ordered_dev, "test_patients": sorted(test), "folds": folds}


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"preprocessed manifest does not exist: {manifest_path}")
    try:
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"preprocessed manifest is not valid JSON: {exc}") from exc
    if not isinstance(result.get("patients"), list):
        raise ValueError("preprocessed manifest must contain a patients list")
    return result


def audit_and_freeze(manifest_path: Path, *, output_dir: Path) -> dict[str, Any]:
    split_path = output_dir / SPLIT_MANIFEST_PATH.name
    if split_path.exists():
        raise FileExistsError(f"split manifest already exists and is immutable: {split_path}")
    manifest = _load_manifest(manifest_path)
    declared_patients = [item.get("patient_id") for item in manifest["patients"]]
    duplicate_ids = sorted({patient for patient in declared_patients if declared_patients.count(patient) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate patient IDs in preprocessed manifest: {duplicate_ids}")
    if len(declared_patients) != EXPECTED_PATIENT_COUNT:
        raise ValueError(f"expected {EXPECTED_PATIENT_COUNT} manifest patients, got {len(declared_patients)}")
    if any(not isinstance(patient, str) or not patient for patient in declared_patients):
        raise ValueError("manifest contains missing or invalid patient IDs")

    recordings: list[RecordingAudit] = []
    quarantined: list[dict[str, Any]] = []
    for patient in manifest["patients"]:
        patient_id = patient["patient_id"]
        entries = patient.get("recordings")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"patient {patient_id}: manifest has no recording list")
        for entry in entries:
            try:
                recordings.append(load_recording(entry["path"], patient_id, entry["recording_id"]))
            except ValueError as exc:
                if "ambiguous channel identity" in str(exc):
                    quarantined.append(
                        {
                            "patient_id": patient_id,
                            "recording_id": entry.get("recording_id"),
                            "path": entry.get("path"),
                            "reason": str(exc),
                            "seizure_seconds_lost": None,
                        }
                    )
                else:
                    raise

    if not recordings:
        raise ValueError("all recordings were quarantined; no canonical channels can be determined")
    common_channels = set(recordings[0].channel_names)
    for record in recordings[1:]:
        common_channels.intersection_update(record.channel_names)
    ordering_counts: dict[tuple[str, ...], int] = {}
    for record in recordings:
        ordering = tuple(channel for channel in record.channel_names if channel in common_channels)
        ordering_counts[ordering] = ordering_counts.get(ordering, 0) + 1
    canonical_strategy = "intersection"
    if not common_channels:
        # CHB-MIT contains recordings with different valid montages. Use the
        # most frequent complete montage and quarantine incompatible shards.
        ordering_counts = {
            tuple(record.channel_names): sum(
                tuple(candidate.channel_names) == tuple(record.channel_names)
                for candidate in recordings
            )
            for record in recordings
        }
        canonical_strategy = "most_common_complete_montage"
    highest_ordering_count = max(ordering_counts.values())
    most_common_orderings = [
        ordering for ordering, count in ordering_counts.items() if count == highest_ordering_count
    ]
    if len(most_common_orderings) != 1:
        raise ValueError(
            "canonical channel ordering is ambiguous because multiple orderings are tied for most common: "
            f"{[list(ordering) for ordering in most_common_orderings]}"
        )
    canonical_channels = list(most_common_orderings[0])
    if not canonical_channels:
        raise ValueError("canonical channel audit produced an empty montage")

    accepted: list[RecordingAudit] = []
    for record in recordings:
        missing = [channel for channel in canonical_channels if channel not in record.canonical_indices]
        if missing:
            quarantined.append(
                {
                    "patient_id": record.patient_id,
                    "recording_id": record.recording_id,
                    "path": record.path,
                    "reason": f"missing canonical channels: {missing}",
                    "seizure_seconds_lost": sum(float(item.end_seconds) - float(item.start_seconds) for item in record.seizure_intervals),
                }
            )
        else:
            accepted.append(record)

    patient_stats: dict[str, dict[str, Any]] = {
        patient_id: {"recordings": 0, "hours": 0.0, "seizure_events": 0, "positive_windows": 0}
        for patient_id in declared_patients
    }
    for record in accepted:
        stat = patient_stats[record.patient_id]
        stat["recordings"] += 1
        stat["hours"] += record.duration_seconds / 3600.0
        stat["seizure_events"] += len(record.seizure_intervals)
        stat["positive_windows"] += sum(window.label for window in make_windows(record))

    split = build_patient_split(patient_stats)
    def counts_for(patients: Iterable[str]) -> dict[str, int]:
        patients = list(patients)
        return {
            "patients": len(patients),
            "seizure_events": sum(patient_stats[p]["seizure_events"] for p in patients),
            "positive_windows": sum(patient_stats[p]["positive_windows"] for p in patients),
        }

    for fold in split["folds"]:
        fold["training_counts"] = counts_for(fold["training_patients"])
        fold["validation_counts"] = counts_for(fold["validation_patients"])
        fold["validation_has_zero_positive_windows"] = fold["validation_counts"]["positive_windows"] == 0

    report = {
        "manifest_path": str(manifest_path),
        "manifest_hash": _manifest_hash(manifest),
        "expected_patient_count": EXPECTED_PATIENT_COUNT,
        "patients": patient_stats,
        "canonical_channels": canonical_channels,
        "canonical_channel_strategy": canonical_strategy,
        "channel_count": len(canonical_channels),
        "channel_maps": [
            {
                "patient_id": record.patient_id,
                "recording_id": record.recording_id,
                "original_to_canonical": record.original_to_canonical,
                "duplicate_channels_dropped": record.duplicate_channels,
            }
            for record in accepted
        ],
        "quarantined_recordings": quarantined,
        "quarantined_recording_count": len(quarantined),
        "quarantined_seizure_seconds_lost": sum(
            item["seizure_seconds_lost"] or 0.0 for item in quarantined
        ),
        "split_counts": {"dev": counts_for(split["dev_patients"]), "test": counts_for(split["test_patients"])},
        "folds": split["folds"],
    }
    split_manifest = {
        "seed": SEED,
        "manifest_hash": report["manifest_hash"],
        "canonical_channels": canonical_channels,
        "canonical_channel_strategy": canonical_strategy,
        "window_config": {
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": WINDOW_STRIDE_SECONDS,
            "samples_per_channel": WINDOW_SAMPLES,
            "positive_overlap_seconds": POSITIVE_OVERLAP_SECONDS,
        },
        **split,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / AUDIT_REPORT_PATH.name).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    split_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"report": report, "split_manifest": split_manifest}
