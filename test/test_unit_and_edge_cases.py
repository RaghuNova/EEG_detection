"""Comprehensive unit tests covering edge cases, invalid inputs, and boundaries."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from backend.dl.config import (
    DEFAULT_DEVICE,
    DEFAULT_SEIZURE_THRESHOLD,
    EVENT_MATCH_MIN_OVERLAP_SECONDS,
    EVENT_MAX_GAP_WINDOWS,
    EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    EXPECTED_PATIENT_COUNT,
    POSITIVE_OVERLAP_SECONDS,
    SAMPLING_RATE_HZ,
    SEED,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
)
from backend.dl.model import CNNBiLSTM
from backend.dl.attention import AdditiveTemporalAttention, AttentionClassifier
from backend.dl.sequence_dataset import SequenceRecord, build_sequence_records
from backend.dl.audit import (
    Interval,
    RecordingAudit,
    build_patient_split,
    canonicalize_channel_names,
    load_recording,
    make_windows,
    normalize_channel_name,
)
from backend.dl.raw_data import (
    RawWindow,
    fit_channel_normalization,
    load_ground_truth_events,
    load_manifest_windows,
    load_recording_durations,
    order_windows_for_shard_io,
)
from backend.dl.train import select_device, seed_everything


# =====================================================================
# Config & Device Selection tests
# =====================================================================
def test_select_device_options() -> None:
    cpu_dev = select_device("cpu")
    assert cpu_dev.type == "cpu"

    auto_dev = select_device("auto")
    assert auto_dev.type in ("cuda", "cpu")

    with pytest.raises(ValueError, match="unsupported device"):
        select_device("tpu")


# =====================================================================
# Model & Attention Dimension Boundary tests
# =====================================================================
def test_cnn_bilstm_boundary_conditions() -> None:
    with pytest.raises(ValueError, match="positive"):
        CNNBiLSTM(input_channels=0)
    with pytest.raises(ValueError, match="positive"):
        CNNBiLSTM(input_channels=4, lstm_hidden=-1)
    with pytest.raises(ValueError, match="dropout"):
        CNNBiLSTM(input_channels=4, dropout=1.5)
    with pytest.raises(ValueError, match="dropout"):
        CNNBiLSTM(input_channels=4, dropout=-0.1)


def test_attention_boundary_conditions() -> None:
    with pytest.raises(ValueError, match="positive"):
        AdditiveTemporalAttention(input_dim=0)
    with pytest.raises(ValueError, match="positive"):
        AdditiveTemporalAttention(input_dim=128, attention_dim=-5)

    attn = AdditiveTemporalAttention(input_dim=32, attention_dim=16)
    with pytest.raises(ValueError, match="shape"):
        attn(torch.randn(2, 32))  # 2D instead of 3D
    with pytest.raises(ValueError, match="shape"):
        attn(torch.randn(2, 10, 64))  # Dim 64 != 32


# =====================================================================
# Sequence dataset helpers tests
# =====================================================================
def test_sequence_dataset_builder_valid_and_invalid() -> None:
    valid_meta = [
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 0.0, "end_seconds": 20.0, "label": 0},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 20.0, "end_seconds": 40.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 40.0, "end_seconds": 60.0, "label": 1},
        {"patient_id": "P1", "recording_id": "R1", "start_seconds": 60.0, "end_seconds": 80.0, "label": 0},
    ]
    records = build_sequence_records(valid_meta, sequence_length=2, drop_incomplete=True)
    assert len(records) == 2
    assert records[0].patient_id == "P1"
    assert records[0].recording_id == "R1"
    assert records[0].indices == (0, 1)
    assert records[1].indices == (2, 3)

    # Test EEGSequenceDataset wrapper
    from backend.dl.sequence_dataset import EEGSequenceDataset
    feat = np.ones((4, 16), dtype=np.float32)
    seq_ds = EEGSequenceDataset(feat, valid_meta, sequence_length=2)
    assert len(seq_ds) == 2
    x, y = seq_ds[0]
    assert x.shape == (2, 16)

    with pytest.raises(ValueError, match="sequence_length must be positive"):
        build_sequence_records(valid_meta, sequence_length=0)

    invalid_meta = [{"patient_id": "P1"}]  # Missing keys
    with pytest.raises(ValueError, match="invalid sequence metadata"):
        build_sequence_records(invalid_meta, sequence_length=2)



# =====================================================================
# Normalization and Raw Data Edge Cases
# =====================================================================
def test_fit_channel_normalization_empty_raises() -> None:
    with pytest.raises(ValueError, match="cannot fit normalization on no windows"):
        fit_channel_normalization([])


def test_order_windows_for_shard_io_preserves_shards() -> None:
    w1 = RawWindow("P1", "R1", "path1.npy", 0, 0.0, 20.0, 0, (0,))
    w2 = RawWindow("P1", "R1", "path1.npy", 1, 20.0, 40.0, 0, (0,))
    w3 = RawWindow("P1", "R2", "path2.npy", 0, 0.0, 20.0, 0, (0,))
    ordered = order_windows_for_shard_io([w1, w2, w3], shuffle_shards=False, seed=42)
    assert len(ordered) == 3
    # Shards are contiguous
    assert ordered[0].path == ordered[1].path


def test_load_ground_truth_events_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_ground_truth_events(Path("non_existent_manifest.json"))


def test_load_recording_durations_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_recording_durations(Path("non_existent_manifest.json"))


# =====================================================================
# Audit Normalization Edge Cases
# =====================================================================
def test_normalize_channel_name_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        normalize_channel_name(123)  # type: ignore

    with pytest.raises(ValueError, match="empty after normalization"):
        normalize_channel_name("   --- ___ ")


def test_build_patient_split_wrong_count_raises() -> None:
    with pytest.raises(ValueError, match="expected 20 patients"):
        build_patient_split({"P1": {"seizure_events": 1}})
