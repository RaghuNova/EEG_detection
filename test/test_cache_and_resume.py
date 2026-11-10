import json
from pathlib import Path

import joblib
import numpy as np
import torch

from backend.dl.cache import build_cache, verify_cache
from backend.preprocessing import CleanEEGRecording


def test_cache_build_and_verify(tmp_path: Path) -> None:
    source = tmp_path / "source.joblib"
    joblib.dump(
        CleanEEGRecording(
            patient_id="CHB01",
            recording_id="r1",
            file_path=str(source),
            channel_names=["A"],
            sampling_rate_hz=256.0,
            data_uv=np.zeros((1, 5120), dtype=np.float32),
            duration_seconds=20.0,
            seizure_intervals=[],
        ),
        source,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "patients": [
                    {"patient_id": f"CHB{i:02d}", "recordings": []}
                    for i in range(1, 21)
                ]
            }
        ),
        encoding="utf-8",
    )
    data = json.loads(manifest.read_text())
    data["patients"][0]["recordings"] = [{"recording_id": "r1", "path": str(source)}]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    build_cache(manifest, cache_dir, canonical_channels=["A"])
    assert verify_cache(cache_dir)["entries"] == 1
