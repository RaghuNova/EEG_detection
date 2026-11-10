"""Data integrity, patient leakage, and split verification tests."""

import json
from pathlib import Path
import numpy as np
import pytest

from backend.dl.audit import (
    Interval,
    RecordingAudit,
    build_patient_split,
    canonicalize_channel_names,
    make_windows,
    normalize_channel_name,
)
from backend.dl.raw_data import fit_channel_normalization, RawWindow
from backend.dl.config import EXPECTED_PATIENT_COUNT


def test_split_has_zero_patient_leakage() -> None:
    # 20 patients: 12 with seizures, 8 non-seizure
    patient_counts = {
        f"CHB{i:02d}": {"seizure_events": 10 if i <= 12 else 0, "positive_windows": 100 if i <= 12 else 0}
        for i in range(1, 21)
    }
    split = build_patient_split(patient_counts)
    dev_patients = set(split["dev_patients"])
    test_patients = set(split["test_patients"])

    # 1. No overlap between dev and test
    assert len(dev_patients & test_patients) == 0
    assert len(dev_patients) == 10
    assert len(test_patients) == 10

    # 2. Every dev patient validates exactly once across the 5 folds
    validated = []
    for fold in split["folds"]:
        assert len(fold["training_patients"]) == 8
        assert len(fold["validation_patients"]) == 2
        # Training and validation within fold are disjoint
        assert set(fold["training_patients"]).isdisjoint(set(fold["validation_patients"]))
        # No test patient in training or validation
        assert set(fold["training_patients"]).isdisjoint(test_patients)
        assert set(fold["validation_patients"]).isdisjoint(test_patients)
        validated.extend(fold["validation_patients"])

    assert len(validated) == 10
    assert set(validated) == dev_patients


def test_normalization_fitted_exclusively_on_training_patients(tmp_path: Path) -> None:
    # Create two mock .npy files for patient A (training) and patient B (validation)
    data_a = np.ones((2, 5120 * 2), dtype=np.float32) * 5.0
    data_b = np.ones((2, 5120 * 2), dtype=np.float32) * 100.0

    path_a = tmp_path / "A.npy"
    path_b = tmp_path / "B.npy"
    np.save(path_a, data_a)
    np.save(path_b, data_b)

    windows_train = [
        RawWindow("P_train", "R1", str(path_a), 0, 0.0, 20.0, 0, (0, 1)),
        RawWindow("P_train", "R1", str(path_a), 1, 20.0, 40.0, 0, (0, 1)),
    ]
    windows_val = [
        RawWindow("P_val", "R2", str(path_b), 0, 0.0, 20.0, 0, (0, 1)),
    ]

    mean, std = fit_channel_normalization(windows_train)
    # Mean of training is exactly 5.0, unaffected by patient B's 100.0
    assert np.allclose(mean, [5.0, 5.0])
    assert np.allclose(std, [0.0, 0.0], atol=1e-5)


def test_no_synthetic_samples_or_smote_in_raw_dataset(tmp_path: Path) -> None:
    # Verify that raw dataset reads raw continuous samples from disk without SMOTE interpolation
    data = np.arange(5120, dtype=np.float32).reshape(1, -1)
    path = tmp_path / "data.npy"
    np.save(path, data)

    windows = [RawWindow("P1", "R1", str(path), 0, 0.0, 20.0, 0, (0,))]
    from backend.dl.raw_data import RawEEGDataset
    ds = RawEEGDataset(windows, mean=np.zeros(1), std=np.ones(1))
    sample, target = ds[0]
    assert sample.shape == (1, 5120)
    assert np.allclose(sample[0].numpy(), np.arange(5120))


def test_recording_boundaries_preserved_in_windowing() -> None:
    # Recording of 45 seconds at 256 Hz = 45 * 256 = 11520 samples
    # 20-second windows (stride 20s):
    # Window 0: 0-20s (0-5120 samples)
    # Window 1: 20-40s (5120-10240 samples)
    # Remainder 5s is not a full window and should never be emitted as partial window!
    record = RecordingAudit(
        patient_id="P1",
        recording_id="R1",
        path="dummy.joblib",
        channel_names=["A"],
        canonical_indices={"A": 0},
        sample_count=45 * 256,
        duration_seconds=45.0,
        seizure_intervals=[Interval(10.0, 25.0)],
    )
    windows = make_windows(record, sample_rate_hz=256, window_seconds=20, stride_seconds=20)
    assert len(windows) == 2
    assert windows[0].start_seconds == 0.0
    assert windows[0].end_seconds == 20.0
    # Overlap with [10, 25] is from 10 to 20 = 10s >= 1s -> label 1
    assert windows[0].label == 1

    assert windows[1].start_seconds == 20.0
    assert windows[1].end_seconds == 40.0
    # Overlap with [10, 25] is from 20 to 25 = 5s >= 1s -> label 1
    assert windows[1].label == 1
