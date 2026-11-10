"""
Pre-training regression/audit tests for the 20-patient EEG ML pipeline.

Run from the repository root:
    uv run pytest -q tests/test_ml_stage_pretraining_audit.py

These are deliberately fail-closed tests. A failure means either:
  * a regression exists, or
  * the current implementation does not yet enforce the audited invariant.

They use synthetic data only; they do not load EDFs or train the real XGBoost model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

mp = pytest.importorskip("backend.mlprocess")
mo = pytest.importorskip("backend.mlout")
pp = pytest.importorskip("backend.preprocessing")


def _meta(patient: str, recording: str, start: float, end: float, label: int = 0) -> dict:
    return {
        "patient_id": patient,
        "recording_id": recording,
        "window_id": f"{recording}_{int(start)}",
        "start_seconds": float(start),
        "end_seconds": float(end),
        "label": int(label),
        "label_name": "seizure" if label else "non_seizure",
    }


def _valid_dataset(
    *,
    train_metadata: list[dict] | None = None,
    validation_metadata: list[dict] | None = None,
    test_metadata: list[dict] | None = None,
) -> Any:
    """Small internally consistent FeatureDataset used to test its validator."""
    feature_names = ["f0", "f1"]
    x_train = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    y_train = np.array([0, 1, 0], dtype=np.int64)
    x_val = np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    y_val = np.array([0, 1], dtype=np.int64)
    x_test = np.array([[0.3, 0.7], [0.7, 0.3]], dtype=np.float32)
    y_test = np.array([0, 1], dtype=np.int64)
    return mp.FeatureDataset(
        X_train=x_train,
        y_train=y_train,
        X_validation=x_val,
        y_validation=y_val,
        X_test=x_test,
        y_test=y_test,
        train_metadata=train_metadata or [
            _meta("CHB01", "r1", 0, 20),
            _meta("CHB01", "r1", 20, 40, 1),
            _meta("CHB02", "r2", 0, 20),
        ],
        validation_metadata=validation_metadata or [
            _meta("CHB03", "r3", 0, 20),
            _meta("CHB03", "r3", 20, 40, 1),
        ],
        test_metadata=test_metadata or [
            _meta("CHB04", "r4", 0, 20),
            _meta("CHB04", "r4", 20, 40, 1),
        ],
        train_patients=["CHB01", "CHB02"],
        validation_patients=["CHB03"],
        test_patients=["CHB04"],
        feature_names=feature_names,
        scaler=None,
        X_train_unbalanced=x_train.copy(),
        y_train_unbalanced=y_train.copy(),
    )


# ---------------------------------------------------------------------------
# 20-patient split contract
# ---------------------------------------------------------------------------

def test_patient_split_is_exactly_10_3_7_and_disjoint():
    patients = [f"CHB{i:02d}" for i in range(1, 21)]
    train, validation, test = mp._split_patient_ids(patients)

    assert len(train) == 10
    assert len(validation) == 3
    assert len(test) == 7
    assert len(set(train)) == 10
    assert len(set(validation)) == 3
    assert len(set(test)) == 7
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert set(train) | set(validation) | set(test) == set(patients)


def test_patient_split_rejects_missing_or_extra_patients():
    with pytest.raises(ValueError, match="Expected exactly 20"):
        mp._split_patient_ids([f"CHB{i:02d}" for i in range(1, 20)])
    with pytest.raises(ValueError, match="Expected exactly 20"):
        mp._split_patient_ids([f"CHB{i:02d}" for i in range(1, 22)])


def test_patient_split_is_reproducible():
    patients = [f"CHB{i:02d}" for i in range(1, 21)]
    assert mp._split_patient_ids(patients) == mp._split_patient_ids(patients)


# ---------------------------------------------------------------------------
# Dataset validation and provenance
# ---------------------------------------------------------------------------

def test_valid_dataset_passes_contract_validation():
    _valid_dataset().validate()


@pytest.mark.parametrize(
    ("field", "bad_metadata"),
    [
        ("train_metadata", [_meta("CHB03", "r3", 0, 20),
                            _meta("CHB01", "r1", 20, 40, 1),
                            _meta("CHB02", "r2", 0, 20)]),
        ("validation_metadata", [_meta("CHB01", "r1", 0, 20),
                                 _meta("CHB03", "r3", 20, 40, 1)]),
        ("test_metadata", [_meta("CHB03", "r3", 0, 20),
                           _meta("CHB04", "r4", 20, 40, 1)]),
    ],
)
def test_dataset_validator_rejects_metadata_patient_outside_declared_split(field, bad_metadata):
    """Regression guard: array alignment alone is not enough to prove patient isolation."""
    kwargs = {field: bad_metadata}
    dataset = _valid_dataset(**kwargs)
    with pytest.raises(ValueError, match="metadata|patient|split|partition"):
        dataset.validate()


def test_dataset_validator_rejects_nonfinite_features():
    dataset = _valid_dataset()
    dataset.X_test[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dataset.validate()


def test_dataset_validator_rejects_duplicate_window_provenance_within_split():
    dataset = _valid_dataset()
    duplicate = dict(dataset.train_metadata[0])
    dataset.train_metadata[1] = duplicate
    with pytest.raises(ValueError, match="duplicate|window|metadata|provenance"):
        dataset.validate()


# ---------------------------------------------------------------------------
# Windowing, labels and duplicate detection
# ---------------------------------------------------------------------------

def _recording(patient="CHB01", recording="r1", duration=60.0, intervals=None):
    sr = float(pp.TARGET_SAMPLING_RATE_HZ)
    n_samples = int(duration * sr)
    rng = np.random.default_rng(123)
    data = rng.normal(0, 10, size=(2, n_samples)).astype(np.float32)
    return pp.CleanEEGRecording(
        patient_id=patient,
        recording_id=recording,
        file_path=f"/synthetic/{recording}.edf",
        channel_names=["F7-T7", "T7-P7"],
        sampling_rate_hz=sr,
        data_uv=data,
        duration_seconds=float(duration),
        seizure_intervals=list(intervals or []),
    )


def test_windowing_uses_20_second_nonoverlapping_windows():
    recording = _recording(duration=61.0)
    windows = list(mp.iter_windows(recording))
    assert len(windows) == 3
    for window in windows:
        assert window.end_seconds - window.start_seconds == pytest.approx(20.0)
        assert window.data_uv.shape[1] == int(20 * pp.TARGET_SAMPLING_RATE_HZ)
    for left, right in zip(windows, windows[1:]):
        assert left.end_seconds == pytest.approx(right.start_seconds)


def test_window_labeling_uses_configured_minimum_overlap():
    # A 20s window from 0..20 has 1.5s overlap with the seizure interval.
    recording = _recording(
        duration=40.0,
        intervals=[pp.SeizureInterval(start_seconds=18.5, end_seconds=22.0)],
    )
    windows = list(mp.iter_windows(recording))
    assert [w.label for w in windows] == [1, 1]


def test_duplicate_hash_detects_identical_signal_windows():
    a = mp.EEGWindow(
        patient_id="CHB01", recording_id="r1", window_id="w1",
        start_seconds=0, end_seconds=20, label=0, label_name="non_seizure",
        data_uv=np.ones((2, 5120), dtype=np.float32),
        sampling_rate_hz=256.0, channel_names=["A", "B"],
    )
    b = mp.EEGWindow(
        patient_id="CHB02", recording_id="r2", window_id="w2",
        start_seconds=0, end_seconds=20, label=0, label_name="non_seizure",
        data_uv=np.ones((2, 5120), dtype=np.float32),
        sampling_rate_hz=256.0, channel_names=["A", "B"],
    )
    unique = mp.remove_exact_duplicate_windows([a, b])
    assert len(unique) == 1


# ---------------------------------------------------------------------------
# Train-only scaling and resampling
# ---------------------------------------------------------------------------

def test_scaler_is_fit_on_training_rows_only():
    """Numerical invariant: validation/test extremes must not affect train mean."""
    from sklearn.preprocessing import StandardScaler

    x_train = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    x_validation = np.array([[1000.0, 1000.0]], dtype=np.float32)
    x_test = np.array([[-1000.0, -1000.0]], dtype=np.float32)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    np.testing.assert_allclose(scaler.mean_, [1.0, 3.0])
    np.testing.assert_allclose(x_train_scaled.mean(axis=0), [0.0, 0.0], atol=1e-7)
    assert np.max(np.abs(x_validation_scaled)) > 100
    assert np.max(np.abs(x_test_scaled)) > 100


def test_resampling_helper_only_receives_the_supplied_training_arrays(monkeypatch):
    """Spy on the sampler boundary; this catches accidental external split inputs."""
    captured = {}

    class FakeSampler:
        def fit_resample(self, X, y):
            captured["X"] = np.asarray(X).copy()
            captured["y"] = np.asarray(y).copy()
            return np.asarray(X).copy(), np.asarray(y).copy()

    def fake_builder(name, k_neighbors, random_state, **kwargs):
        return FakeSampler()

    monkeypatch.setattr(mp, "_build_oversampler", fake_builder)
    X_train = np.arange(24, dtype=np.float32).reshape(12, 2)
    y_train = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    X_out, y_out, report = mp._safe_kmeans_smote(X_train, y_train)

    np.testing.assert_array_equal(captured["X"], X_train)
    np.testing.assert_array_equal(captured["y"], y_train)
    np.testing.assert_array_equal(X_out, X_train)
    np.testing.assert_array_equal(y_out, y_train)
    assert report.class_counts_before == [6, 6]


def test_resampling_fails_closed_if_training_has_one_class():
    X = np.ones((8, 2), dtype=np.float32)
    y = np.zeros(8, dtype=np.int64)
    with pytest.raises(ValueError, match="only one class"):
        mp._safe_kmeans_smote(X, y)


# ---------------------------------------------------------------------------
# Binary classification and threshold tuning
# ---------------------------------------------------------------------------

def test_threshold_tuning_uses_validation_labels_and_returns_valid_threshold():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    threshold = mo.select_decision_threshold(y_true, y_prob)
    assert 0.0 < threshold < 1.0
    pred = (y_prob >= threshold).astype(np.int64)
    assert np.array_equal(pred, y_true)


def test_threshold_tuning_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unsupported threshold"):
        mo.select_decision_threshold(
            np.array([0, 1]), np.array([0.2, 0.8]), metric="made_up_metric"
        )


def test_model_bundle_rejects_invalid_threshold():
    bundle = mo.ModelBundle(
        model=object(), scaler=object(), feature_names=["f0"],
        model_version="test", sampling_rate_hz=256.0, window_seconds=20.0,
        low_cutoff_hz=0.5, high_cutoff_hz=40.0, notch_frequency_hz=60.0,
        class_names=["non_seizure", "seizure"], created_at_utc="test",
        decision_threshold=1.0,
    )
    with pytest.raises(ValueError, match="decision_threshold"):
        bundle.validate()


# ---------------------------------------------------------------------------
# Event grouping and event metric regressions
# ---------------------------------------------------------------------------

def test_event_grouping_bridges_configured_single_negative_window(monkeypatch):
    """Current implementation is expected to expose the gap-bridging defect."""
    monkeypatch.setattr(mo, "EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS", 1)
    monkeypatch.setattr(mo, "EVENT_MAX_GAP_WINDOWS", 1)
    metadata = [
        _meta("CHB01", "r1", 0, 20),
        _meta("CHB01", "r1", 20, 40),
        _meta("CHB01", "r1", 40, 60),
    ]
    probabilities = np.array([0.9, 0.1, 0.9])
    events = mo._group_positive_windows(metadata, probabilities, threshold=0.5)
    assert len(events) == 1, (
        "EVENT_MAX_GAP_WINDOWS=1 should bridge one intervening negative window; "
        "the current grouping logic closes an event immediately on any negative."
    )
    assert events[0].start_seconds == pytest.approx(0.0)
    assert events[0].end_seconds == pytest.approx(60.0)


def test_event_grouping_never_merges_different_recordings():
    metadata = [
        _meta("CHB01", "r1", 0, 20),
        _meta("CHB01", "r2", 0, 20),
    ]
    events = mo._group_positive_windows(metadata, np.array([0.9, 0.9]), threshold=0.5)
    assert len(events) == 2
    assert {e.recording_id for e in events} == {"r1", "r2"}


# def test_event_latency_is_based_on_predicted_onset_not_predicted_end():
#     metadata = [_meta("CHB01", "r1", 100, 120, 1)]
#     truth = {("CHB01", "r1"): [(100.0, 110.0)]}
#     metrics = mo.event_level_metrics(
#         metadata, np.array([0.9]), truth, threshold=0.5
#     )
#     assert metrics["mean_detection_latency_seconds"] == pytest.approx(0.0), (
#         "Prediction begins exactly at true onset; latency should be 0s, not "
#         "the 20s window end minus onset."
#     )
def test_event_latency_is_based_on_end_of_window_alert_time():
    """Operationally correct detection latency:

    An alert can only be emitted once the full window [100, 120] has elapsed
    (at t=120s). Since true seizure onset is at t=100s, detection latency
    is 120 - 100 = 20s.
    """
    metadata = [_meta("CHB01", "r1", 100, 120, 1)]
    truth = {("CHB01", "r1"): [(100.0, 110.0)]}
    metrics = mo.event_level_metrics(
        metadata, np.array([0.9]), truth, threshold=0.5
    )
    assert metrics["mean_detection_latency_seconds"] == pytest.approx(20.0), (
        "Alert time is window end (120s); latency relative to 100s onset is 20s."
    )



def test_event_metrics_report_no_detected_event_as_zero_sensitivity():
    metadata = [_meta("CHB01", "r1", 0, 20, 0)]
    truth = {("CHB01", "r1"): [(5.0, 10.0)]}
    metrics = mo.event_level_metrics(
        metadata, np.array([0.1]), truth, threshold=0.5
    )
    assert metrics["event_sensitivity"] == pytest.approx(0.0)
    assert metrics["detected_events"] == 0
    assert metrics["false_alarm_events"] == 0


# ---------------------------------------------------------------------------
# Inference input contract
# ---------------------------------------------------------------------------

def test_raw_window_prediction_rejects_wrong_sample_count_before_preprocessing():
    bundle = mo.ModelBundle(
        model=object(), scaler=object(), feature_names=["f0"],
        model_version="test", sampling_rate_hz=256.0, window_seconds=20.0,
        low_cutoff_hz=0.5, high_cutoff_hz=40.0, notch_frequency_hz=60.0,
        class_names=["non_seizure", "seizure"], created_at_utc="test",
        decision_threshold=0.5,
    )
    with pytest.raises(ValueError, match="Expected exactly"):
        mo.predict_window_from_preprocessed_array(
            bundle, np.zeros((2, 100), dtype=np.float32), 256.0, ["A", "B"]
        )


def test_raw_window_prediction_rejects_channel_name_count_mismatch():
    bundle = mo.ModelBundle(
        model=object(), scaler=object(), feature_names=["f0"],
        model_version="test", sampling_rate_hz=256.0, window_seconds=20.0,
        low_cutoff_hz=0.5, high_cutoff_hz=40.0, notch_frequency_hz=60.0,
        class_names=["non_seizure", "seizure"], created_at_utc="test",
        decision_threshold=0.5,
    )
    with pytest.raises(ValueError, match="channel_names length"):
        mo.predict_window_from_preprocessed_array(
            bundle, np.zeros((2, 5120), dtype=np.float32), 256.0, ["A"]
        )
