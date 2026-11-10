from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from backend.dl import train
from backend.dl.audit import (
    RecordingAudit,
    build_patient_split,
    canonicalize_channel_names,
    make_windows,
)


@dataclass
class Interval:
    start_seconds: float
    end_seconds: float


def test_train_entrypoint_exposes_legacy_recording_class_for_joblib() -> None:
    assert train.CleanEEGRecording.__name__ == "CleanEEGRecording"
    assert train.SeizureInterval.__name__ == "SeizureInterval"


def test_channel_mapping_normalizes_and_drops_only_duplicate_t8_p8() -> None:
    canonical, index_by_channel, duplicates = canonicalize_channel_names(
        [" fp1--f7 ", "T8-P8-0", "T8-P8-1", "O1-O2"]
    )

    assert canonical == ["FP1-F7", "T8-P8", "O1-O2"]
    assert index_by_channel == {"FP1-F7": 0, "T8-P8": 1, "O1-O2": 3}
    assert duplicates == [
        {"canonical_channel": "T8-P8", "kept_index": 1, "dropped_index": 2}
    ]


def test_channel_mapping_rejects_ambiguous_non_t8_p8_identity() -> None:
    with pytest.raises(ValueError, match="ambiguous channel identity"):
        canonicalize_channel_names(["FP1-F7", "fp1 f7"])


def test_windowing_uses_boundary_overlap_rule_and_never_emits_partial_window() -> None:
    record = RecordingAudit(
        patient_id="CHB01",
        recording_id="r1",
        path="unused",
        channel_names=["FP1-F7"],
        canonical_indices={"FP1-F7": 0},
        sample_count=10_300,
        duration_seconds=40.234375,
        seizure_intervals=[Interval(19.0, 21.0), Interval(39.2, 40.2)],
    )

    windows = make_windows(record, sample_rate_hz=256, window_seconds=20, stride_seconds=20)

    assert [(w.start_seconds, w.end_seconds, w.label) for w in windows] == [
        (0.0, 20.0, 1),
        (20.0, 40.0, 1),
    ]
    assert [w.window_id for w in windows] == [0, 1]


def test_split_is_disjoint_and_each_development_patient_validates_once() -> None:
    patient_counts = {
        "CHB01": {"seizure_events": 5, "positive_windows": 6},
        "CHB02": {"seizure_events": 4, "positive_windows": 5},
        "CHB03": {"seizure_events": 3, "positive_windows": 4},
        "CHB04": {"seizure_events": 2, "positive_windows": 3},
        "CHB05": {"seizure_events": 1, "positive_windows": 2},
        "CHB06": {"seizure_events": 0, "positive_windows": 0},
        "CHB07": {"seizure_events": 0, "positive_windows": 0},
        "CHB08": {"seizure_events": 0, "positive_windows": 0},
        "CHB09": {"seizure_events": 0, "positive_windows": 0},
        "CHB10": {"seizure_events": 0, "positive_windows": 0},
        "CHB11": {"seizure_events": 0, "positive_windows": 0},
        "CHB12": {"seizure_events": 0, "positive_windows": 0},
        "CHB13": {"seizure_events": 0, "positive_windows": 0},
        "CHB14": {"seizure_events": 0, "positive_windows": 0},
        "CHB15": {"seizure_events": 0, "positive_windows": 0},
        "CHB16": {"seizure_events": 0, "positive_windows": 0},
        "CHB17": {"seizure_events": 0, "positive_windows": 0},
        "CHB18": {"seizure_events": 0, "positive_windows": 0},
        "CHB19": {"seizure_events": 0, "positive_windows": 0},
        "CHB20": {"seizure_events": 0, "positive_windows": 0},
    }

    split = build_patient_split(patient_counts)

    dev = set(split["dev_patients"])
    test = set(split["test_patients"])
    assert len(dev) == len(test) == 10
    assert not dev & test
    assert dev | test == set(patient_counts)
    assert set().union(*(set(fold["validation_patients"]) for fold in split["folds"])) == dev
    assert sum(len(fold["validation_patients"]) for fold in split["folds"]) == 10
    for fold in split["folds"]:
        assert len(fold["training_patients"]) == 8
        assert len(fold["validation_patients"]) == 2
        assert not set(fold["training_patients"]) & set(fold["validation_patients"])
        assert not set(fold["training_patients"]) & test
        assert not set(fold["validation_patients"]) & test


def test_split_handles_tied_seizure_bearing_counts() -> None:
    # 4 seizure-bearing patients: dev gets 2, test gets 2 (tied)
    # 16 non-seizure patients: should cleanly divide into 8 dev, 8 test
    patient_counts = {
        f"CHB{i:02d}": {
            "seizure_events": 2 if i <= 4 else 0,
            "positive_windows": 5 if i <= 4 else 0,
        }
        for i in range(1, 21)
    }
    split = build_patient_split(patient_counts)
    assert len(split["dev_patients"]) == 10
    assert len(split["test_patients"]) == 10
    assert len(set(split["dev_patients"]) & set(split["test_patients"])) == 0
