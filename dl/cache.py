"""Persistent read-only NumPy cache for DL recording shards."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .audit import canonicalize_channel_names
from .config import SAMPLING_RATE_HZ

CACHE_VERSION = "dl-npy-cache-v1"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_recording(path: Path) -> Any:
    import sys
    from backend.preprocessing import CleanEEGRecording, SeizureInterval

    setattr(sys.modules["__main__"], "CleanEEGRecording", CleanEEGRecording)
    setattr(sys.modules["__main__"], "SeizureInterval", SeizureInterval)
    return joblib.load(path)


def build_cache(
    manifest_path: Path,
    cache_dir: Path,
    *,
    canonical_channels: list[str],
    rebuild: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recordings_dir = cache_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for patient in manifest["patients"]:
        for item in patient["recordings"]:
            source = Path(item["path"])
            source_hash = _sha256(source)
            target = recordings_dir / patient["patient_id"] / f"{item['recording_id']}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            cache_is_current = False
            if target.is_file() and not rebuild:
                try:
                    cache_is_current = (
                        np.load(target, mmap_mode="r").dtype == np.float32
                    )
                except (OSError, ValueError):
                    cache_is_current = False
            if rebuild or not cache_is_current:
                recording = _load_recording(source)
                canonical, indices, _ = canonicalize_channel_names(recording.channel_names)
                missing = [channel for channel in canonical_channels if channel not in indices]
                if missing:
                    continue
                values = np.asarray(recording.data_uv)[
                    [indices[channel] for channel in canonical_channels]
                ].astype(np.float32, copy=False)
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, suffix=".tmp", delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                try:
                    with temporary_path.open("wb") as handle:
                        np.save(handle, values)
                    os.replace(temporary_path, target)
                finally:
                    temporary_path.unlink(missing_ok=True)
            entries.append(
                {
                    "patient_id": patient["patient_id"],
                    "recording_id": item["recording_id"],
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "cache_path": str(target),
                    "shape": list(np.load(target, mmap_mode="r").shape),
                    "dtype": "float32",
                    "sampling_rate_hz": SAMPLING_RATE_HZ,
                    "canonical_channels": canonical_channels,
                    "cache_version": CACHE_VERSION,
                }
            )
    result = {
        "cache_version": CACHE_VERSION,
        "manifest": str(manifest_path),
        "entries": entries,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def verify_cache(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid: list[str] = []
    for entry in manifest.get("entries", []):
        source = Path(entry["source_path"])
        target = Path(entry["cache_path"])
        if not source.is_file() or not target.is_file():
            invalid.append(entry["recording_id"])
            continue
        if _sha256(source) != entry["source_sha256"]:
            invalid.append(entry["recording_id"])
            continue
        array = np.load(target, mmap_mode="r")
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
            invalid.append(entry["recording_id"])
    if invalid:
        raise ValueError(f"invalid or stale DL cache entries: {invalid[:10]}")
    return {"cache_version": manifest["cache_version"], "entries": len(manifest["entries"])}
