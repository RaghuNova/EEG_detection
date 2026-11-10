from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from backend.dl.raw_data import RawWindow, load_manifest_windows, order_windows_for_shard_io
from backend.preprocessing import CleanEEGRecording, SeizureInterval


def _recording(path: str) -> CleanEEGRecording:
    return CleanEEGRecording(
        patient_id="CHB01",
        recording_id="r1",
        file_path=path,
        channel_names=["A", "B"],
        sampling_rate_hz=256.0,
        data_uv=np.zeros((2, 5120), dtype=np.float32),
        duration_seconds=20.0,
        seizure_intervals=[SeizureInterval(1.0, 3.0)],
    )


def _manifest(tmp_path, shard_path: str) -> str:
    patients = [
        {
            "patient_id": f"CHB{i:02d}",
            "recordings": (
                [{"recording_id": "r1", "path": shard_path}]
                if i == 1
                else [{"recording_id": f"r{i}", "path": shard_path}]
            ),
        }
        for i in range(1, 21)
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"patients": patients}), encoding="utf-8")
    return str(path)


def test_manifest_loader_reads_existing_joblib_and_labels_window(tmp_path) -> None:
    shard = tmp_path / "recording.joblib"
    joblib.dump(_recording(str(shard)), shard)

    windows = load_manifest_windows(
        Path(_manifest(tmp_path, str(shard))),
        canonical_channels=["A", "B"],
        patients={"CHB01"},
    )

    assert len(windows) == 1
    assert windows[0].label == 1
    assert windows[0].channel_indices == (0, 1)


def test_manifest_loader_reports_missing_shard(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_manifest_windows(
            Path(_manifest(tmp_path, str(tmp_path / "missing.joblib"))),
            canonical_channels=["A", "B"],
            patients={"CHB01"},
        )


def test_manifest_loader_skips_audited_quarantine(tmp_path) -> None:
    shard = tmp_path / "recording.joblib"
    joblib.dump(_recording(str(shard)), shard)
    report = tmp_path / "audit.json"
    report.write_text(
        json.dumps({"quarantined_recordings": [{"path": str(shard)}]}),
        encoding="utf-8",
    )

    windows = load_manifest_windows(
        Path(_manifest(tmp_path, str(shard))),
        canonical_channels=["A", "B"],
        patients={"CHB01"},
        quarantine_report_path=report,
    )

    assert windows == []


def test_window_order_groups_shards_for_single_read() -> None:
    windows = [
        RawWindow("p", "r2", "b", 0, 0, 20, 0, (0,)),
        RawWindow("p", "r1", "a", 1, 20, 40, 0, (0,)),
        RawWindow("p", "r1", "a", 0, 0, 20, 0, (0,)),
    ]
    ordered = order_windows_for_shard_io(windows, shuffle_shards=True, seed=42)
    paths = [window.path for window in ordered]
    assert paths in (["a", "a", "b"], ["b", "a", "a"])
