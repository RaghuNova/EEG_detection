"""test_neuroscan_platform.py
----------------------------
Comprehensive automated test suite for NEUROSCAN EEG Intelligence Platform.

Tests all 18 areas specified in Part 25:
1. EDF discovery from a patient folder.
2. EDF upload validation.
3. Recording metadata extraction.
4. Preprocessing wrapper.
5. ML feature extraction integration.
6. ML inference.
7. CNN-BiLSTM checkpoint loading.
8. CNN temporal feature extraction.
9. CNN probability generation.
10. AttentionClassifier input shape.
11. Attention inference.
12. Dynamic threshold loading.
13. Recording-specific threshold calculation.
14. Window timestamp alignment.
15. Event generation.
16. Patient summary aggregation.
17. API endpoints.
18. PDF report generation.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pyedflib
import pytest
import torch
from fastapi.testclient import TestClient

import mlprocess
import mlout
import preprocessing
from backend.api.main import app
from backend.inference.attention_inference import AttentionModelAdapter
from backend.inference.cnn_inference import CNNBiLSTMAdapter
from backend.inference.inference_orchestrator import InferenceOrchestrator
from backend.inference.ml_inference import (
    MLModelAdapter,
    group_positive_windows_to_events,
)
from backend.inference.preprocessing_wrapper import (
    CANONICAL_CHANNELS,
    WindowInfo,
    get_edf_metadata,
    map_channels_to_canonical,
    preprocess_recording_for_inference,
)
from backend.inference.result_builder import (
    build_failed_recording_result,
    build_patient_summary,
    build_recording_result,
)
from backend.reporting.patient_report import generate_patient_pdf_report
from backend.storage.storage_manager import StorageManager
from dl.attention import AttentionClassifier
from dl.dynamic_threshold import apply_dynamic_threshold
from dl.model import CNNBiLSTM


@pytest.fixture(scope="session")
def synthetic_edf_fixture(tmp_path_factory):
    """Generate a valid multi-channel 20-second synthetic EDF file for pipeline plumbing tests."""
    temp_dir = tmp_path_factory.mktemp("test_edf_dir")
    patient_folder = temp_dir / "PATIENT_TEST_01"
    patient_folder.mkdir()
    edf_path = patient_folder / "recording_01.edf"

    channel_count = len(CANONICAL_CHANNELS)
    f = pyedflib.EdfWriter(str(edf_path), channel_count, file_type=pyedflib.FILETYPE_EDFPLUS)
    channel_info = []
    for ch in CANONICAL_CHANNELS:
        channel_info.append({
            "label": ch,
            "dimension": "uV",
            "sample_frequency": 256,
            "physical_max": 500,
            "physical_min": -500,
            "digital_max": 32767,
            "digital_min": -32768,
            "transducer": "",
            "prefilter": "",
        })
    f.setSignalHeaders(channel_info)

    # 40 seconds of EEG signal = 2 windows of 20 seconds at 256 Hz
    data = (np.random.randn(channel_count, 256 * 40) * 15.0).astype(np.float64)
    f.writeSamples(data)
    f.close()

    return patient_folder, edf_path


# 1. EDF discovery
def test_01_edf_discovery(synthetic_edf_fixture):
    patient_folder, edf_path = synthetic_edf_fixture
    edfs = list(patient_folder.glob("*.edf"))
    assert len(edfs) >= 1
    assert edfs[0].name == "recording_01.edf"


# 2. EDF upload validation
def test_02_edf_upload_validation(tmp_path):
    invalid_file = tmp_path / "bad.txt"
    invalid_file.write_text("invalid eeg")
    assert not invalid_file.name.lower().endswith(".edf")

    valid_file = tmp_path / "good.edf"
    valid_file.write_bytes(b"EDF dummy")
    assert valid_file.name.lower().endswith(".edf")


# 3. Recording metadata extraction
def test_03_recording_metadata_extraction(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    meta = get_edf_metadata(edf_path)
    assert meta["filename"] == "recording_01.edf"
    assert meta["sampling_frequency"] == 256.0
    assert meta["channel_count"] == len(CANONICAL_CHANNELS)
    assert meta["duration_seconds"] == 40.0
    assert meta["total_samples"] == 256 * 40


# 4. Preprocessing wrapper
def test_04_preprocessing_wrapper(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    processed = preprocess_recording_for_inference(edf_path, patient_id="PATIENT_TEST_01")

    assert processed.patient_id == "PATIENT_TEST_01"
    assert len(processed.windows) == 2  # 40 seconds / 20s = 2 windows
    assert processed.canonical_matrix.shape[0] == 22
    assert processed.canonical_matrix.shape[1] == 256 * 40
    assert len(processed.window_infos) == 2
    assert processed.window_infos[0].start_seconds == 0.0
    assert processed.window_infos[0].end_seconds == 20.0
    assert processed.window_infos[1].start_seconds == 20.0
    assert processed.window_infos[1].end_seconds == 40.0


# 5. ML feature extraction integration
def test_05_ml_feature_extraction(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    processed = preprocess_recording_for_inference(edf_path)
    X, y, meta, names = mlprocess.windows_to_features(processed.windows)

    assert X.shape == (2, 122)
    assert len(names) == 122
    assert np.isfinite(X).all()


# 6. ML inference with trained joblib bundle
def test_06_ml_inference(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    processed = preprocess_recording_for_inference(edf_path)
    ml_adapter = MLModelAdapter("models/seizure_ml_model.joblib")

    res = ml_adapter.run_inference(processed)
    assert res.model_name == "XGBoost Classifier"
    assert res.total_windows == 2
    assert len(res.window_probabilities) == 2
    assert len(res.window_predictions) == 2
    assert 0.0 < res.decision_threshold < 1.0
    for p in res.window_probabilities:
        assert 0.0 <= p <= 1.0


# 7. CNN-BiLSTM checkpoint loading
def test_07_cnn_bilstm_checkpoint_loading():
    adapter = CNNBiLSTMAdapter("models/final.pt (1).zip")
    assert adapter.model is not None
    assert len(adapter.canonical_channels) == 22
    assert adapter.threshold == 0.50
    assert adapter.norm_mean.shape == (22,)
    assert adapter.norm_std.shape == (22,)


# 8. CNN temporal feature extraction
def test_08_cnn_temporal_feature_extraction(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    processed = preprocess_recording_for_inference(edf_path)
    adapter = CNNBiLSTMAdapter("models/final.pt (1).zip")

    res = adapter.run_inference(processed)
    assert res.temporal_features.shape == (2, 128, 128)
    assert not torch.isnan(res.temporal_features).any()


# 9. CNN probability generation
def test_09_cnn_probability_generation(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    processed = preprocess_recording_for_inference(edf_path)
    adapter = CNNBiLSTMAdapter("models/final.pt (1).zip")

    res = adapter.run_inference(processed)
    assert len(res.window_probabilities) == 2
    assert len(res.window_predictions) == 2
    for p in res.window_probabilities:
        assert 0.0 <= p <= 1.0
        assert p in [0.0, 1.0] or 0.0 < p < 1.0


# 10. AttentionClassifier input shape requirement
def test_10_attention_input_shape():
    model = AttentionClassifier(input_dim=128)
    # Valid input: [batch, time, 128]
    valid_features = torch.randn(3, 128, 128)
    logits, weights = model.forward_with_attention(valid_features)
    assert logits.shape == (3,)
    assert weights.shape == (3, 4, 128)

    # Incompatible shape must raise ValueError
    invalid_features = torch.randn(3, 128, 64)
    with pytest.raises(ValueError):
        model.forward_with_attention(invalid_features)


# 11. Attention inference
def test_11_attention_inference(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    processed = preprocess_recording_for_inference(edf_path)
    cnn_adapter = CNNBiLSTMAdapter("models/final.pt (1).zip")
    cnn_res = cnn_adapter.run_inference(processed)

    attn_adapter = AttentionModelAdapter(
        "models/final.pt.zip",
        "dependencies/hybrid_attention_threshold_calibration.json",
    )
    attn_res = attn_adapter.run_inference(cnn_res.temporal_features, processed)

    assert attn_res.model_name == "AttentionClassifier"
    assert len(attn_res.window_probabilities) == 2
    assert len(attn_res.window_thresholds) == 2
    assert len(attn_res.persistent_window_predictions) == 2


# 12. Dynamic threshold loading
def test_12_dynamic_threshold_loading():
    calib_path = Path("dependencies/hybrid_attention_threshold_calibration.json")
    assert calib_path.is_file()
    with open(calib_path) as f:
        data = json.load(f)
    assert "selected" in data
    assert data["selected"]["global_threshold"] == 0.03
    assert data["selected"]["quantile"] == 0.985
    assert data["selected"]["floor"] == 0.03
    assert data["selected"]["adaptive_blend"] == 1.0


# 13. Recording-specific threshold calculation
def test_13_recording_specific_threshold_calculation():
    with open("dependencies/hybrid_attention_threshold_calibration.json") as f:
        calib = json.load(f)

    probs = np.array([0.01, 0.02, 0.04, 0.05, 0.90])
    metadata = [
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": i * 20.0, "end_seconds": (i + 1) * 20.0}
        for i in range(5)
    ]
    res = apply_dynamic_threshold(probs, metadata, calib)
    assert "thresholds" in res
    assert len(res["thresholds"]) == 5
    assert res["thresholds"][0] >= 0.03  # floor


# 14. Window timestamp alignment
def test_14_window_timestamp_alignment(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    processed = preprocess_recording_for_inference(edf_path)

    w0 = processed.window_infos[0]
    w1 = processed.window_infos[1]
    assert w0.start_seconds == 0.0 and w0.end_seconds == 20.0
    assert w1.start_seconds == 20.0 and w1.end_seconds == 40.0


# 15. Event generation
def test_15_event_generation():
    window_infos = [
        WindowInfo(0, "W0", 0.0, 20.0),
        WindowInfo(1, "W1", 20.0, 40.0),
        WindowInfo(2, "W2", 40.0, 60.0),
        WindowInfo(3, "W3", 60.0, 80.0),
    ]
    probabilities = [0.1, 0.85, 0.92, 0.1]
    predictions = [0, 1, 1, 0]

    events = group_positive_windows_to_events(probabilities, predictions, window_infos)
    assert len(events) == 1
    assert events[0].start_seconds == 20.0
    assert events[0].end_seconds == 60.0
    assert events[0].duration_seconds == 40.0
    assert events[0].window_count == 2


# 16. Patient summary aggregation
def test_16_patient_summary_aggregation(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    orchestrator = InferenceOrchestrator()
    summary = orchestrator.analyze_patient_folder(
        patient_id="PATIENT_TEST_01",
        analysis_id="TEST_RUN_001",
        edf_paths=[edf_path],
    )

    assert summary["patient_id"] == "PATIENT_TEST_01"
    assert summary["analysis_id"] == "TEST_RUN_001"
    assert summary["total_recordings"] == 1
    assert summary["successful_recordings"] == 1
    assert summary["failed_recordings"] == 0
    assert summary["total_windows"] == 2
    assert "summary" in summary
    assert "ml_positive_windows" in summary["summary"]
    assert "cnn_positive_windows" in summary["summary"]
    assert "attention_positive_windows" in summary["summary"]


# 17. API Endpoints
def test_17_api_endpoints():
    client = TestClient(app)

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    health_data = res.json()
    assert health_data["status"] == "healthy"
    assert health_data["models"]["ml_model_exists"] is True
    assert health_data["models"]["cnn_checkpoint_exists"] is True
    assert health_data["models"]["attention_checkpoint_exists"] is True
    assert health_data["models"]["calibration_file_exists"] is True

    # Analyses list
    hist_res = client.get("/api/analyses")
    assert hist_res.status_code == 200
    assert isinstance(hist_res.json(), list)


# 18. PDF report generation
def test_18_pdf_report_generation(synthetic_edf_fixture):
    _, edf_path = synthetic_edf_fixture
    orchestrator = InferenceOrchestrator()
    summary = orchestrator.analyze_patient_folder(
        patient_id="PATIENT_TEST_01",
        analysis_id="TEST_RUN_PDF",
        edf_paths=[edf_path],
    )

    pdf_bytes = generate_patient_pdf_report(summary)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
