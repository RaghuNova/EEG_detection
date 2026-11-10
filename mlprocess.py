# """
# mlprocess.py
# ------------
# Leakage-safe EEG dataset engineering for the minor project.

# Pipeline
# --------
# 1. Load the cleaned recordings from preprocessing.py.
# 2. Generate non-overlapping 20-second EEG windows.
# 3. Keep patient/recording/window/time metadata for every window.
# 4. Label each window using CHB-MIT seizure annotations.
# 5. Remove exact duplicate windows BEFORE splitting.
# 6. Split by PATIENT, not by individual windows.
# 7. Extract fixed-size, channel-order-invariant ML features.
# 8. Fit the feature scaler on TRAIN only.
# 9. Apply K-Means SMOTE to TRAIN only.
# 10. Save the resulting leakage-safe train/validation/test datasets.

# Important:
#     No synthetic sample is ever created from validation/test data.
#     No test/validation statistics are used to fit the scaler or SMOTE model.
# """

# from __future__ import annotations

# import gc
# import hashlib
# import logging
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from typing import Dict, List, Optional, Sequence, Tuple

# import joblib
# import numpy as np
# from scipy import signal, stats
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import KMeansSMOTE, SMOTE, RandomOverSampler


# # Random patient-level split: exactly 10 / 3 / 7 out of 20.
# TRAIN_PATIENT_COUNT = 10
# VALIDATION_PATIENT_COUNT = 3
# TEST_PATIENT_COUNT = 7

# RANDOM_STATE = 42  # Reproducible random selection

# try:
#     from . import runtime  # noqa: F401
#     from .preprocessing import (
#         CleanEEGRecording,
#         SeizureInterval,
#         WINDOW_DURATION_SECONDS,
#         load_dataset,
#         load_preprocessed_recordings,
#         iter_preprocessed_recordings,
#         load_preprocessed_manifest,
#         preprocess_dataset_to_disk,
#         PREPROCESSED_CACHE_PATH,
#         PREPROCESSED_MANIFEST_PATH,
#         TARGET_SAMPLING_RATE_HZ,
#     )
# except ImportError:  # supports `python backend/mlprocess.py`
#     import runtime  # type: ignore  # noqa: F401
#     from preprocessing import (  # type: ignore
#     CleanEEGRecording,
#     SeizureInterval,
#     WINDOW_DURATION_SECONDS,
#     load_dataset,
#     load_preprocessed_recordings,
#     iter_preprocessed_recordings,
#     load_preprocessed_manifest,
#     preprocess_dataset_to_disk,
#     PREPROCESSED_CACHE_PATH,
#     PREPROCESSED_MANIFEST_PATH,
#     TARGET_SAMPLING_RATE_HZ,
#     )

# # -----------------------------------------------------------------------------
# # USER CONFIGURATION
# # -----------------------------------------------------------------------------
# WINDOW_SECONDS: float = WINDOW_DURATION_SECONDS
# WINDOW_STRIDE_SECONDS: float = WINDOW_SECONDS  # non-overlapping windows by design
# SEIZURE_MIN_OVERLAP_SECONDS: float = 1.0

# # Patient-independent split. The split is performed once at patient level.
# TEST_SIZE: float = 0.20
# VALIDATION_SIZE_WITHIN_REMAINDER: float = 0.20
# RANDOM_STATE: int = 42

# # Duplicate policy.
# HASH_DECIMAL_PLACES: int = 4

# # K-Means SMOTE configuration. Fit only on training features after scaling.
# KMEANS_SMOTE_K_NEIGHBORS: int = 2
# KMEANS_SMOTE_RANDOM_STATE: int = RANDOM_STATE
# KMEANS_SMOTE_CLUSTER_BALANCE_THRESHOLD: float = 0.0

# # Artifact paths.
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
# ML_DATASET_ARTIFACT_PATH: str = str(ARTIFACT_DIR / "ml_dataset.joblib")
# FEATURE_SCHEMA_PATH: str = str(ARTIFACT_DIR / "feature_schema.joblib")

# # -----------------------------------------------------------------------------
# # LOGGING
# # -----------------------------------------------------------------------------

# LOGGER = logging.getLogger("minorproj.mlprocess")


# # -----------------------------------------------------------------------------
# # DATA CONTRACTS
# # -----------------------------------------------------------------------------


# @dataclass
# class EEGWindow:
#     patient_id: str
#     recording_id: str
#     window_id: str
#     start_seconds: float
#     end_seconds: float
#     label: int
#     label_name: str
#     data_uv: np.ndarray
#     sampling_rate_hz: float
#     channel_names: List[str]


# @dataclass
# class ResamplingReport:
#     """
#     Transparency record of how class imbalance was handled on TRAIN only.

#     With ~10 patients and 35-50 EDF files, seizure windows are a small
#     minority of all 20-second windows, so this report exists specifically so
#     that bias from imbalance (and how it was corrected) is visible in every
#     training run rather than buried in logs.
#     """

#     method_used: str
#     method_params: Dict[str, object]
#     attempts_tried: List[str]
#     class_counts_before: List[int]
#     class_counts_after: List[int]
#     imbalance_ratio_before: float  # majority : minority, before oversampling
#     imbalance_ratio_after: float
#     n_synthetic_samples_added: int


# @dataclass
# class FeatureDataset:
#     """Serializable ML dataset and all metadata needed by mlout.py."""

#     X_train: np.ndarray
#     y_train: np.ndarray
#     X_validation: np.ndarray
#     y_validation: np.ndarray
#     X_test: np.ndarray
#     y_test: np.ndarray

#     train_metadata: List[dict]
#     validation_metadata: List[dict]
#     test_metadata: List[dict]

#     train_patients: List[str]
#     validation_patients: List[str]
#     test_patients: List[str]

#     feature_names: List[str]
#     feature_schema_version: str = "v1-channel-invariant"
#     scaler: Optional[StandardScaler] = None
#     resampling_report: Optional[ResamplingReport] = None
#     # Real, unresampled training rows are required to build chronological DL sequences.
#     X_train_unbalanced: Optional[np.ndarray] = None
#     y_train_unbalanced: Optional[np.ndarray] = None

#     def validate(self) -> None:
#         arrays = [
#             ("X_train", self.X_train, self.y_train),
#             ("X_validation", self.X_validation, self.y_validation),
#             ("X_test", self.X_test, self.y_test),
#         ]
#         for name, X, y in arrays:
#             if X.ndim != 2:
#                 raise ValueError(f"{name} must be a 2D feature matrix.")
#             if y.ndim != 1:
#                 raise ValueError(f"{name.replace('X_', 'y_')} must be 1D.")
#             if X.shape[0] != y.shape[0]:
#                 raise ValueError(f"{name} and labels have inconsistent row counts.")
#             if X.shape[1] != len(self.feature_names):
#                 raise ValueError(f"{name} has an unexpected number of features.")
#             if not np.isfinite(X).all():
#                 raise ValueError(f"{name} contains non-finite values.")
#         train_alignment_X = self.X_train_unbalanced if self.X_train_unbalanced is not None else self.X_train
#         for metadata_name, metadata, X in (
#             ("train_metadata", self.train_metadata, train_alignment_X),
#             ("validation_metadata", self.validation_metadata, self.X_validation),
#             ("test_metadata", self.test_metadata, self.X_test),
#         ):
#             if len(metadata) != X.shape[0]:
#                 raise ValueError(f"{metadata_name} must align with its feature matrix.")
#         split_sets = [set(self.train_patients), set(self.validation_patients), set(self.test_patients)]
#         if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
#             raise ValueError("Patient IDs overlap between train, validation, and test splits.")
#         if self.X_train_unbalanced is not None:
#             if self.X_train_unbalanced.ndim != 2 or self.X_train_unbalanced.shape[1] != len(self.feature_names):
#                 raise ValueError("X_train_unbalanced has an invalid feature shape.")
#             if self.y_train_unbalanced is None or len(self.y_train_unbalanced) != len(self.X_train_unbalanced):
#                 raise ValueError("Unbalanced training features and labels are misaligned.")


# def compute_scale_pos_weight(y_train: np.ndarray) -> float:
#     """
#     Ratio of negative to positive TRAIN samples, computed BEFORE oversampling.

#     This is exposed so mlout.py can optionally use XGBoost's own
#     `scale_pos_weight` as an alternative (or ablation baseline) to K-Means
#     SMOTE. Combining strong oversampling with a large scale_pos_weight at the
#     same time tends to over-correct for class imbalance and can hurt
#     precision (more false alarms) far more than it helps recall, so treat
#     these as two competing strategies to compare, not something to stack by
#     default.
#     """

#     counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
#     if counts[1] == 0:
#         raise ValueError("Cannot compute scale_pos_weight with zero positive samples.")
#     return float(counts[0] / counts[1])


# def summarize_dataset(dataset: FeatureDataset) -> Dict[str, object]:
#     summary: Dict[str, object] = {
#         "n_train": int(len(dataset.y_train)),
#         "n_validation": int(len(dataset.y_validation)),
#         "n_test": int(len(dataset.y_test)),
#         "train_class_counts": np.bincount(dataset.y_train, minlength=2).tolist(),
#         "validation_class_counts": np.bincount(dataset.y_validation, minlength=2).tolist(),
#         "test_class_counts": np.bincount(dataset.y_test, minlength=2).tolist(),
#         "n_train_patients": len(dataset.train_patients),
#         "n_validation_patients": len(dataset.validation_patients),
#         "n_test_patients": len(dataset.test_patients),
#         "feature_count": len(dataset.feature_names),
#         "feature_schema_version": dataset.feature_schema_version,
#     }
#     if dataset.resampling_report is not None:
#         summary["resampling"] = asdict(dataset.resampling_report)
#     return summary


# def print_resampling_report(report: Optional[ResamplingReport]) -> None:
#     """Human-readable class-balancing report for CLI output (see mlout.py train_command)."""

#     if report is None:
#         print("No resampling report available (dataset was built before this feature existed).")
#         return

#     non_seiz_before, seiz_before = report.class_counts_before
#     non_seiz_after, seiz_after = report.class_counts_after

#     print("\n=== TRAIN-SET CLASS BALANCING (SMOTE) REPORT ===")
#     print(f"Method used            : {report.method_used}  params={report.method_params}")
#     print(f"Fallback ladder tried   : {report.attempts_tried}")
#     print(f"Before  -> non_seizure={non_seiz_before:>6d}  seizure={seiz_before:>6d}  "
#           f"(imbalance {report.imbalance_ratio_before:.1f} : 1)")
#     print(f"After   -> non_seizure={non_seiz_after:>6d}  seizure={seiz_after:>6d}  "
#           f"(imbalance {report.imbalance_ratio_after:.1f} : 1)")
#     print(f"Synthetic samples added: {report.n_synthetic_samples_added}")
#     print("(Validation and test sets are never resampled or synthetically balanced.)")


# # if __name__ == "__main__":
# #     dataset = build_feature_dataset()
# #     path = save_feature_dataset(dataset)
# #     print(summarize_dataset(dataset))
# #     print_resampling_report(dataset.resampling_report)
# #     print(f"Saved: {path.resolve()}")


# if __name__ == "__main__":
#     # Step 1: Build the dataset once. Do not rerun preprocessing if a
#     # compatible preprocessed manifest/cache is already available.
#     dataset = build_feature_dataset()

#     # Step 2: Save the main ML dataset.
#     ml_path = save_feature_dataset(dataset)

#     # Step 3: Save the feature schema separately.
#     schema_path = Path(FEATURE_SCHEMA_PATH)
#     schema_path.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(
#         {
#             "feature_schema_version": dataset.feature_schema_version,
#             "feature_names": dataset.feature_names,
#             "feature_count": len(dataset.feature_names),
#         },
#         schema_path,
#         compress=3,
#     )

#     # Step 5: Report the outputs.
#     print("\n=== DATASET SUMMARY ===")
#     print(summarize_dataset(dataset))
#     print_resampling_report(dataset.resampling_report)

#     print("\n=== SAVED ARTIFACTS ===")
#     for path in (ml_path, schema_path):
#         print(f"{path.resolve()} | {path.stat().st_size:,} bytes")


"""
mlprocess.py
------------
Leakage-safe EEG dataset engineering for the minor project.

Pipeline
--------
1. Load the cleaned recordings from preprocessing.py.
2. Generate non-overlapping 20-second EEG windows.
3. Keep patient/recording/window/time metadata for every window.
4. Label each window using CHB-MIT seizure annotations.
5. Remove exact duplicate windows BEFORE splitting.
6. Split by PATIENT, not by individual windows.
7. Extract fixed-size, channel-order-invariant ML features.
8. Fit the feature scaler on TRAIN only.
9. Apply K-Means SMOTE to TRAIN only.
10. Save the resulting leakage-safe train/validation/test datasets.

Important:
    No synthetic sample is ever created from validation/test data.
    No test/validation statistics are used to fit the scaler or SMOTE model.
"""

from __future__ import annotations

import gc
import hashlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy import signal, stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import KMeansSMOTE, SMOTE, RandomOverSampler


# Random patient-level split: exactly 10 / 3 / 7 out of 20.
TRAIN_PATIENT_COUNT = 10
VALIDATION_PATIENT_COUNT = 3
TEST_PATIENT_COUNT = 7

RANDOM_STATE = 42  # Reproducible random selection

try:
    from . import runtime  # noqa: F401
    from .preprocessing import (
        CleanEEGRecording,
        SeizureInterval,
        WINDOW_DURATION_SECONDS,
        load_dataset,
        load_preprocessed_recordings,
        iter_preprocessed_recordings,
        load_preprocessed_manifest,
        preprocess_dataset_to_disk,
        PREPROCESSED_CACHE_PATH,
        PREPROCESSED_MANIFEST_PATH,
        TARGET_SAMPLING_RATE_HZ,
    )
except ImportError:  # supports `python backend/mlprocess.py`
    import runtime  # type: ignore  # noqa: F401
    from preprocessing import (  # type: ignore
    CleanEEGRecording,
    SeizureInterval,
    WINDOW_DURATION_SECONDS,
    load_dataset,
    load_preprocessed_recordings,
    iter_preprocessed_recordings,
    load_preprocessed_manifest,
    preprocess_dataset_to_disk,
    PREPROCESSED_CACHE_PATH,
    PREPROCESSED_MANIFEST_PATH,
    TARGET_SAMPLING_RATE_HZ,
    )

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------
WINDOW_SECONDS: float = WINDOW_DURATION_SECONDS
WINDOW_STRIDE_SECONDS: float = WINDOW_SECONDS  # non-overlapping windows by design
SEIZURE_MIN_OVERLAP_SECONDS: float = 1.0

# Patient-independent split. The split is performed once at patient level.
TEST_SIZE: float = 0.20
VALIDATION_SIZE_WITHIN_REMAINDER: float = 0.20
RANDOM_STATE: int = 42

# Duplicate policy.
HASH_DECIMAL_PLACES: int = 4

# K-Means SMOTE configuration. Fit only on training features after scaling.
KMEANS_SMOTE_K_NEIGHBORS: int = 2
KMEANS_SMOTE_RANDOM_STATE: int = RANDOM_STATE
KMEANS_SMOTE_CLUSTER_BALANCE_THRESHOLD: float = 0.0

# Artifact paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ML_DATASET_ARTIFACT_PATH: str = str(ARTIFACT_DIR / "ml_dataset.joblib")
FEATURE_SCHEMA_PATH: str = str(ARTIFACT_DIR / "feature_schema.joblib")

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

LOGGER = logging.getLogger("minorproj.mlprocess")


# -----------------------------------------------------------------------------
# DATA CONTRACTS
# -----------------------------------------------------------------------------


@dataclass
class EEGWindow:
    patient_id: str
    recording_id: str
    window_id: str
    start_seconds: float
    end_seconds: float
    label: int
    label_name: str
    data_uv: np.ndarray
    sampling_rate_hz: float
    channel_names: List[str]


@dataclass
class ResamplingReport:
    """
    Transparency record of how class imbalance was handled on TRAIN only.

    With ~10 patients and 35-50 EDF files, seizure windows are a small
    minority of all 20-second windows, so this report exists specifically so
    that bias from imbalance (and how it was corrected) is visible in every
    training run rather than buried in logs.
    """

    method_used: str
    method_params: Dict[str, object]
    attempts_tried: List[str]
    class_counts_before: List[int]
    class_counts_after: List[int]
    imbalance_ratio_before: float  # majority : minority, before oversampling
    imbalance_ratio_after: float
    n_synthetic_samples_added: int


@dataclass
class FeatureDataset:
    """Serializable ML dataset and all metadata needed by mlout.py."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    train_metadata: List[dict]
    validation_metadata: List[dict]
    test_metadata: List[dict]

    train_patients: List[str]
    validation_patients: List[str]
    test_patients: List[str]

    feature_names: List[str]
    feature_schema_version: str = "v1-channel-invariant"
    scaler: Optional[StandardScaler] = None
    resampling_report: Optional[ResamplingReport] = None
    # Real, unresampled training rows are required to build chronological DL sequences.
    X_train_unbalanced: Optional[np.ndarray] = None
    y_train_unbalanced: Optional[np.ndarray] = None

    def validate(self) -> None:
        arrays = [
            ("X_train", self.X_train, self.y_train),
            ("X_validation", self.X_validation, self.y_validation),
            ("X_test", self.X_test, self.y_test),
        ]
        for name, X, y in arrays:
            if X.ndim != 2:
                raise ValueError(f"{name} must be a 2D feature matrix.")
            if y.ndim != 1:
                raise ValueError(f"{name.replace('X_', 'y_')} must be 1D.")
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"{name} and labels have inconsistent row counts.")
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"{name} has an unexpected number of features.")
            if not np.isfinite(X).all():
                raise ValueError(f"{name} contains non-finite values.")
        train_alignment_X = self.X_train_unbalanced if self.X_train_unbalanced is not None else self.X_train
        for metadata_name, metadata, X in (
            ("train_metadata", self.train_metadata, train_alignment_X),
            ("validation_metadata", self.validation_metadata, self.X_validation),
            ("test_metadata", self.test_metadata, self.X_test),
        ):
            if len(metadata) != X.shape[0]:
                raise ValueError(f"{metadata_name} must align with its feature matrix.")
        split_sets = [set(self.train_patients), set(self.validation_patients), set(self.test_patients)]
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Patient IDs overlap between train, validation, and test splits.")
        # Validate metadata patient IDs against declared split patients
        partition_map = [
            ("train_metadata", self.train_metadata, set(self.train_patients)),
            ("validation_metadata", self.validation_metadata, set(self.validation_patients)),
            ("test_metadata", self.test_metadata, set(self.test_patients)),
        ]
        for meta_name, meta_list, allowed_patients in partition_map:
            seen_provenance: set[tuple[str, str, str]] = set()
            for meta in meta_list:
                pid = meta.get("patient_id")
                if pid not in allowed_patients:
                    raise ValueError(
                        f"{meta_name} contains patient {pid!r} outside declared partition patients {allowed_patients}."
                    )
                prov_key = (str(pid), str(meta.get("recording_id")), str(meta.get("window_id")))
                if prov_key in seen_provenance:
                    raise ValueError(
                        f"duplicate window provenance detected in {meta_name}: {prov_key}"
                    )
                seen_provenance.add(prov_key)

        if self.X_train_unbalanced is not None:
            if self.X_train_unbalanced.ndim != 2 or self.X_train_unbalanced.shape[1] != len(self.feature_names):
                raise ValueError("X_train_unbalanced has an invalid feature shape.")
            if self.y_train_unbalanced is None or len(self.y_train_unbalanced) != len(self.X_train_unbalanced):
                raise ValueError("Unbalanced training features and labels are misaligned.")


@dataclass
class DLFeatureArtifact:
    """ML feature output reshaped for the next CNN -> Bi-LSTM stage.

    Each row is one ML feature vector. The CNN receives [batch, channels, length]
    with one input channel and ``feature_count`` positions; the later sequence
    builder can group these rows by recording and chronological window metadata.
    """

    X_train_cnn: np.ndarray
    y_train: np.ndarray
    X_validation_cnn: np.ndarray
    y_validation: np.ndarray
    X_test_cnn: np.ndarray
    y_test: np.ndarray
    train_metadata: List[dict]
    validation_metadata: List[dict]
    test_metadata: List[dict]
    feature_names: List[str]
    source_ml_dataset: str = ML_DATASET_ARTIFACT_PATH

    def validate(self) -> None:
        for name, X, y in (
            ("X_train_cnn", self.X_train_cnn, self.y_train),
            ("X_validation_cnn", self.X_validation_cnn, self.y_validation),
            ("X_test_cnn", self.X_test_cnn, self.y_test),
        ):
            if X.ndim != 3 or X.shape[1] != 1:
                raise ValueError(f"{name} must have shape [rows, 1, features].")
            if X.shape[0] != y.shape[0] or X.shape[2] != len(self.feature_names):
                raise ValueError(f"{name} and labels/features are inconsistent.")
            if not np.isfinite(X).all():
                raise ValueError(f"{name} contains non-finite values.")


# -----------------------------------------------------------------------------
# WINDOW CREATION + LABELING
# -----------------------------------------------------------------------------


def _window_is_seizure(
    start_seconds: float,
    end_seconds: float,
    seizure_intervals: Sequence[SeizureInterval],
) -> bool:
    """Label a window positive if it overlaps a seizure for >= configured seconds."""

    for interval in seizure_intervals:
        overlap = max(
            0.0,
            min(end_seconds, interval.end_seconds)
            - max(start_seconds, interval.start_seconds),
        )
        if overlap >= SEIZURE_MIN_OVERLAP_SECONDS:
            return True
    return False


def iter_windows(recording: CleanEEGRecording):
    """Yield windows one at a time so cached recordings do not get duplicated."""

    sampling_rate = float(recording.sampling_rate_hz)
    expected_sr = float(TARGET_SAMPLING_RATE_HZ)
    if not np.isclose(sampling_rate, expected_sr):
        raise ValueError(
            f"Recording {recording.recording_id} has {sampling_rate} Hz, expected {expected_sr} Hz."
        )

    window_samples = int(round(WINDOW_SECONDS * sampling_rate))
    stride_samples = int(round(WINDOW_STRIDE_SECONDS * sampling_rate))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must produce at least one sample.")

    n_samples = recording.data_uv.shape[1]
    produced = False
    for window_number, start_sample in enumerate(
        range(0, n_samples - window_samples + 1, stride_samples)
    ):
        produced = True
        stop_sample = start_sample + window_samples
        start_seconds = start_sample / sampling_rate
        end_seconds = stop_sample / sampling_rate
        data = np.ascontiguousarray(
            recording.data_uv[:, start_sample:stop_sample], dtype=np.float32
        )
        label = int(_window_is_seizure(start_seconds, end_seconds, recording.seizure_intervals))
        yield EEGWindow(
            patient_id=recording.patient_id,
            recording_id=recording.recording_id,
            window_id=f"{recording.recording_id}_W{window_number:06d}",
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            label=label,
            label_name="seizure" if label else "non_seizure",
            data_uv=data,
            sampling_rate_hz=sampling_rate,
            channel_names=list(recording.channel_names),
        )
    if not produced:
        LOGGER.warning("Recording %s is shorter than one 20-second window.", recording.recording_id)


def make_windows(recording: CleanEEGRecording) -> List[EEGWindow]:
    """Convert one cleaned recording into non-overlapping 20-second windows."""

    return list(iter_windows(recording))


def build_all_windows(recordings: Sequence[CleanEEGRecording]) -> List[EEGWindow]:
    all_windows: List[EEGWindow] = []
    for recording in recordings:
        all_windows.extend(make_windows(recording))
    if not all_windows:
        raise RuntimeError("No 20-second EEG windows were created.")
    return all_windows


# -----------------------------------------------------------------------------
# DUPLICATE HANDLING
# -----------------------------------------------------------------------------


def window_signal_hash(window: EEGWindow) -> str:
    """Stable exact-content hash after controlled float quantization."""

    quantized = np.round(window.data_uv.astype(np.float32), HASH_DECIMAL_PLACES)
    digest = hashlib.sha256(np.ascontiguousarray(quantized).tobytes()).hexdigest()
    return digest


def remove_exact_duplicate_windows(windows: Sequence[EEGWindow]) -> List[EEGWindow]:
    """Remove exact signal duplicates while preserving the first metadata occurrence."""

    seen: set[str] = set()
    unique: List[EEGWindow] = []
    duplicates = 0

    for window in windows:
        digest = window_signal_hash(window)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        unique.append(window)

    LOGGER.info(
        "Duplicate handling removed %d exact duplicate windows; %d remain.",
        duplicates,
        len(unique),
    )
    return unique


# -----------------------------------------------------------------------------
# PATIENT-LEVEL SPLITTING
# -----------------------------------------------------------------------------
### 10 paitent split-- 6/2/2

# def _split_patient_ids(patient_ids: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
#     unique_patients = sorted(set(patient_ids))
#     if len(unique_patients) < 3:
#         raise ValueError(
#             "Patient-independent train/validation/test splitting requires at least 3 "
#             "distinct patients. The current path appears to contain fewer patients. "
#             "Point PATIENT_DATASET_PATH in preprocessing.py to the dataset root "
#             "containing multiple ChbXX folders."
#         )

#     dummy = np.zeros((len(unique_patients), 1), dtype=np.float32)
#     splitter_test = GroupShuffleSplit(
#         n_splits=1,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#     )
#     train_val_idx, test_idx = next(
#         splitter_test.split(dummy, groups=np.asarray(unique_patients))
#     )

#     train_val_patients = [unique_patients[i] for i in train_val_idx]
#     test_patients = [unique_patients[i] for i in test_idx]

#     dummy_tv = np.zeros((len(train_val_patients), 1), dtype=np.float32)
#     splitter_val = GroupShuffleSplit(
#         n_splits=1,
#         test_size=VALIDATION_SIZE_WITHIN_REMAINDER,
#         random_state=RANDOM_STATE,
#     )
#     train_idx, val_idx = next(
#         splitter_val.split(dummy_tv, groups=np.asarray(train_val_patients))
#     )

#     train_patients = [train_val_patients[i] for i in train_idx]
#     validation_patients = [train_val_patients[i] for i in val_idx]

#     return (
#         sorted(train_patients),
#         sorted(validation_patients),
#         sorted(test_patients),
#     )

def _split_patient_ids(
    patient_ids: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Randomly split patients into 10 train / 3 validation / 7 test."""

    unique_patients = sorted(set(patient_ids))

    expected_total = (
        TRAIN_PATIENT_COUNT
        + VALIDATION_PATIENT_COUNT
        + TEST_PATIENT_COUNT
    )

    if len(unique_patients) != expected_total:
        raise ValueError(
            f"Expected exactly {expected_total} distinct patients, "
            f"but found {len(unique_patients)}: {unique_patients}"
        )

    # Randomize PATIENT IDs, never individual EEG windows.
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = rng.permutation(unique_patients).tolist()

    train_end = TRAIN_PATIENT_COUNT
    val_end = train_end + VALIDATION_PATIENT_COUNT

    train_patients = sorted(shuffled[:train_end])
    validation_patients = sorted(shuffled[train_end:val_end])
    test_patients = sorted(shuffled[val_end:])

    # Leakage-safety assertions.
    train_set = set(train_patients)
    val_set = set(validation_patients)
    test_set = set(test_patients)

    assert not (train_set & val_set)
    assert not (train_set & test_set)
    assert not (val_set & test_set)

    assert len(train_patients) == 10
    assert len(validation_patients) == 3
    assert len(test_patients) == 7

    LOGGER.info(
        "Random patient split | train=%s | validation=%s | test=%s",
        train_patients,
        validation_patients,
        test_patients,
    )

    print("\n=== RANDOM PATIENT-LEVEL SPLIT ===")
    print(f"Train      ({len(train_patients)}): {train_patients}")
    print(f"Validation ({len(validation_patients)}): {validation_patients}")
    print(f"Test       ({len(test_patients)}): {test_patients}")

    return train_patients, validation_patients, test_patients


def split_windows_by_patient(
    windows: Sequence[EEGWindow],
) -> Tuple[List[EEGWindow], List[EEGWindow], List[EEGWindow], Tuple[List[str], List[str], List[str]]]:
    patient_ids = [window.patient_id for window in windows]
    train_patients, validation_patients, test_patients = _split_patient_ids(patient_ids)

    train_set = set(train_patients)
    validation_set = set(validation_patients)
    test_set = set(test_patients)

    train_windows = [w for w in windows if w.patient_id in train_set]
    validation_windows = [w for w in windows if w.patient_id in validation_set]
    test_windows = [w for w in windows if w.patient_id in test_set]

    # Strong invariant: patients may occur in exactly one split.
    intersections = [
        train_set & validation_set,
        train_set & test_set,
        validation_set & test_set,
    ]
    if any(intersections):
        raise AssertionError("Patient leakage detected between dataset splits.")

    LOGGER.info(
        "Patient split: train=%s | validation=%s | test=%s",
        train_patients,
        validation_patients,
        test_patients,
    )

    # With only a handful of patients (this project currently uses ~10), a random
    # group split can plausibly place a patient with zero seizures alone into
    # validation or test, silently making ROC-AUC/PR-AUC undefined for that split.
    # This is not fatal (downstream code already guards against it), but it should
    # never pass unnoticed, since it usually means the split should be re-run with
    # a different RANDOM_STATE or re-checked against the seizure-per-patient counts.
    for split_name, split_windows in (
        ("validation", validation_windows),
        ("test", test_windows),
    ):
        labels = np.asarray([w.label for w in split_windows], dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        if counts[1] == 0:
            LOGGER.warning(
                "%s split contains ZERO seizure windows (patients=%s). "
                "Event-level and ROC/PR metrics for this split will be meaningless.",
                split_name,
                sorted(validation_set if split_name == "validation" else test_set),
            )

    return (
        train_windows,
        validation_windows,
        test_windows,
        (train_patients, validation_patients, test_patients),
    )


# -----------------------------------------------------------------------------
# FEATURE EXTRACTION
# -----------------------------------------------------------------------------

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

BASE_FEATURES = [
    "mean",
    "std",
    "rms",
    "ptp",
    "line_length",
    "zero_crossing_rate",
    "skewness",
    "kurtosis",
    "hjorth_mobility",
    "hjorth_complexity",
]

AGGREGATIONS = ["mean", "std", "median", "p10", "p90", "max"]


def _safe_stat(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _hjorth_parameters(x: np.ndarray) -> Tuple[float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = float(np.var(x))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))

    if var_x <= 1e-12:
        return 0.0, 0.0
    mobility = np.sqrt(max(var_dx, 0.0) / var_x)
    if mobility <= 1e-12:
        return float(mobility), 0.0
    complexity = np.sqrt(max(var_ddx, 0.0) / max(var_dx, 1e-12)) / mobility
    return _safe_stat(mobility), _safe_stat(complexity)


def _channel_base_features(x: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x_centered = x - np.mean(x)

    mean = _safe_stat(np.mean(x))
    std = _safe_stat(np.std(x))
    rms = _safe_stat(np.sqrt(np.mean(np.square(x))))
    ptp = _safe_stat(np.ptp(x))
    line_length = _safe_stat(np.mean(np.abs(np.diff(x))))
    zero_crossing_rate = _safe_stat(
        np.mean(np.signbit(x_centered[1:]) != np.signbit(x_centered[:-1]))
    )
    skewness = _safe_stat(stats.skew(x, bias=False, nan_policy="omit"))
    kurtosis = _safe_stat(stats.kurtosis(x, fisher=True, bias=False, nan_policy="omit"))
    hjorth_mobility, hjorth_complexity = _hjorth_parameters(x)

    features: Dict[str, float] = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "ptp": ptp,
        "line_length": line_length,
        "zero_crossing_rate": zero_crossing_rate,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
    }

    freqs, psd = signal.welch(
        x,
        fs=sampling_rate,
        window="hann",
        nperseg=min(len(x), int(4 * sampling_rate)),
        noverlap=min(int(2 * sampling_rate), max(0, len(x) // 2 - 1)),
        scaling="density",
    )
    total_mask = (freqs >= 0.5) & (freqs <= 40.0)
    total_power = float(np.trapezoid(psd[total_mask], freqs[total_mask])) if total_mask.any() else 0.0

    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() < 2:
            absolute_power = 0.0
        else:
            absolute_power = float(np.trapezoid(psd[mask], freqs[mask]))
        relative_power = absolute_power / max(total_power, 1e-12)
        features[f"bandpower_{band_name}"] = _safe_stat(absolute_power)
        features[f"relative_bandpower_{band_name}"] = _safe_stat(relative_power)

    return features


def _aggregate_channel_features(
    channel_feature_rows: Sequence[Dict[str, float]],
) -> Tuple[np.ndarray, List[str]]:
    if not channel_feature_rows:
        raise ValueError("No channel features were produced.")

    feature_keys = sorted(channel_feature_rows[0].keys())
    vector: List[float] = []
    names: List[str] = []

    for feature_key in feature_keys:
        values = np.asarray(
            [_safe_stat(row.get(feature_key, 0.0)) for row in channel_feature_rows],
            dtype=np.float64,
        )
        quantiles = np.percentile(values, [10, 50, 90])
        aggregation_values = {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": quantiles[1],
            "p10": quantiles[0],
            "p90": quantiles[2],
            "max": np.max(values),
        }
        for aggregation in AGGREGATIONS:
            names.append(f"channel_{aggregation}_{feature_key}")
            vector.append(_safe_stat(aggregation_values[aggregation]))

    return np.asarray(vector, dtype=np.float32), names


def extract_window_features(window: EEGWindow) -> Tuple[np.ndarray, List[str]]:
    """Extract channel-order-invariant statistical + spectral features."""

    rows = [
        _channel_base_features(channel, window.sampling_rate_hz)
        for channel in window.data_uv
    ]
    vector, names = _aggregate_channel_features(rows)

    # Add two global inter-channel synchrony features. These use correlation across
    # channels but do not depend on channel ordering.
    if window.data_uv.shape[0] >= 2:
        corr = np.corrcoef(window.data_uv)
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[np.isfinite(upper)]
        mean_abs_corr = float(np.mean(np.abs(upper))) if upper.size else 0.0
        max_abs_corr = float(np.max(np.abs(upper))) if upper.size else 0.0
    else:
        mean_abs_corr = 0.0
        max_abs_corr = 0.0

    vector = np.concatenate(
        [vector, np.asarray([mean_abs_corr, max_abs_corr], dtype=np.float32)]
    )
    names.extend(["global_mean_abs_channel_correlation", "global_max_abs_channel_correlation"])

    if not np.isfinite(vector).all():
        raise ValueError(f"Feature extraction produced non-finite values for {window.window_id}.")
    return vector, names


def windows_to_features(
    windows: Sequence[EEGWindow],
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str]]:
    if not windows:
        raise ValueError("Cannot extract features from an empty window list.")

    feature_rows: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[dict] = []
    reference_names: Optional[List[str]] = None

    for window in windows:
        features, names = extract_window_features(window)
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise AssertionError("Feature schema changed between windows.")
        feature_rows.append(features)
        labels.append(window.label)
        metadata.append(
            {
                "patient_id": window.patient_id,
                "recording_id": window.recording_id,
                "window_id": window.window_id,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "label": int(window.label),
                "label_name": window.label_name,
            }
        )

    X = np.vstack(feature_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    assert reference_names is not None
    return X, y, metadata, reference_names


# -----------------------------------------------------------------------------
# TRAINING-ONLY SMOTE + COMPLETE BUILD
# -----------------------------------------------------------------------------


def _build_oversampler(name: str, k_neighbors: int, random_state: int, **kwargs):
    if name == "kmeans_smote":
        return KMeansSMOTE(
            sampling_strategy="auto",
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=-1,
            **kwargs,
        )
    if name == "smote":
        return SMOTE(
            sampling_strategy="auto",
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
    if name == "random_oversample":
        return RandomOverSampler(sampling_strategy="auto", random_state=random_state)
    raise ValueError(f"Unknown oversampler: {name}")


def _safe_kmeans_smote(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, "ResamplingReport"]:
    """
    Oversample the TRAIN split only, with a fallback ladder, and return a
    ResamplingReport documenting exactly what happened (method used, class
    counts before/after, imbalance ratio before/after). This report is
    carried through FeatureDataset so mlout.py can print/display it —
    important given the small (~10 patient, 35-50 EDF) dataset where
    class-imbalance bias is a real risk and should be visible, not buried
    in logs.

    K-Means SMOTE clusters the training feature space and only oversamples
    clusters that pass a minority-density threshold. With a small number of
    patients, that clustering step can legitimately find no usable cluster
    and raise a RuntimeError even though both classes are present. Rather
    than letting the whole pipeline crash on a specific patient split, we
    fall back to progressively simpler (but still TRAIN-ONLY) oversampling
    strategies:

        1. KMeansSMOTE with an 'auto' density threshold (best quality, but
           the most likely to fail on sparse data).
        2. KMeansSMOTE with the project's original fixed threshold (accepts
           every cluster regardless of balance -> more permissive).
        3. Plain SMOTE (no clustering, just k-NN interpolation).
        4. RandomOverSampler (pure duplication) as a last resort so the
           pipeline never silently drops the minority class.

    Every attempt still fits and resamples on X_train/y_train only, so the
    leakage-safety guarantee (validation/test are never touched) is
    unchanged regardless of which strategy ultimately succeeds.
    """

    class_counts_before = np.bincount(y_train, minlength=2)
    if class_counts_before.min() == 0:
        raise ValueError(
            "Training data contains only one class. Patient-level splitting produced "
            "a training set without both seizure and non-seizure windows. Adjust the "
            "patient selection or dataset size."
        )
    if class_counts_before[1] < 3 or class_counts_before[0] < 3:
        raise ValueError(
            f"Too few samples for reliable oversampling: class counts={class_counts_before.tolist()}."
        )

    k_neighbors = min(KMEANS_SMOTE_K_NEIGHBORS, int(class_counts_before.min() - 1))
    attempts: List[Tuple[str, Dict[str, object]]] = [
        ("kmeans_smote", {"cluster_balance_threshold": "auto"}),
        ("kmeans_smote", {"cluster_balance_threshold": KMEANS_SMOTE_CLUSTER_BALANCE_THRESHOLD}),
        ("smote", {}),
        ("random_oversample", {}),
    ]

    attempts_tried: List[str] = []
    last_error: Optional[Exception] = None
    for name, kwargs in attempts:
        attempts_tried.append(f"{name}({kwargs})")
        try:
            sampler = _build_oversampler(name, k_neighbors, KMEANS_SMOTE_RANDOM_STATE, **kwargs)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            class_counts_after = np.bincount(y_resampled, minlength=2)

            ratio_before = float(class_counts_before.max() / max(class_counts_before.min(), 1))
            ratio_after = float(class_counts_after.max() / max(class_counts_after.min(), 1))

            report = ResamplingReport(
                method_used=name,
                method_params={**kwargs, "k_neighbors": k_neighbors},
                attempts_tried=list(attempts_tried),
                class_counts_before=class_counts_before.tolist(),
                class_counts_after=class_counts_after.tolist(),
                imbalance_ratio_before=ratio_before,
                imbalance_ratio_after=ratio_after,
                n_synthetic_samples_added=int(class_counts_after.sum() - class_counts_before.sum()),
            )

            LOGGER.info(
                "Oversampling succeeded using %s(%s). Class counts: before=%s (ratio %.1f:1), "
                "after=%s (ratio %.1f:1).",
                name,
                kwargs,
                class_counts_before.tolist(),
                ratio_before,
                class_counts_after.tolist(),
                ratio_after,
            )
            return X_resampled.astype(np.float32), y_resampled.astype(np.int64), report
        except Exception as exc:  # noqa: BLE001 - deliberately broad fallback ladder
            LOGGER.warning("Oversampler %s(%s) failed (%s); trying next fallback.", name, kwargs, exc)
            last_error = exc

    raise RuntimeError(f"All oversampling strategies failed on the training split: {last_error}")


def _split_feature_rows_by_manifest(
    manifest_path: str | Path,
) -> FeatureDataset:
    """Build features from disk shards while keeping only tabular matrices in RAM."""

    manifest = load_preprocessed_manifest(manifest_path)
    patient_ids = [str(item["patient_id"]) for item in manifest.get("patients", [])]
    train_patients, validation_patients, test_patients = _split_patient_ids(patient_ids)
    split_sets = {
        "train": set(train_patients),
        "validation": set(validation_patients),
        "test": set(test_patients),
    }
    rows: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    labels: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
    metadata: Dict[str, List[dict]] = {"train": [], "validation": [], "test": []}
    seen_hashes: set[str] = set()
    feature_names: Optional[List[str]] = None

    for recording in iter_preprocessed_recordings(manifest_path):
        split_name = next(
            name for name, patients in split_sets.items() if recording.patient_id in patients
        )
        try:
            for window in iter_windows(recording):
                digest = window_signal_hash(window)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                vector, names = extract_window_features(window)
                if feature_names is None:
                    feature_names = names
                elif feature_names != names:
                    raise AssertionError("Feature schema changed between cached recordings.")
                rows[split_name].append(vector)
                labels[split_name].append(window.label)
                metadata[split_name].append(
                    {
                        "patient_id": window.patient_id,
                        "recording_id": window.recording_id,
                        "window_id": window.window_id,
                        "start_seconds": window.start_seconds,
                        "end_seconds": window.end_seconds,
                        "label": int(window.label),
                        "label_name": window.label_name,
                    }
                )
        finally:
            del recording
            gc.collect()

    if feature_names is None or any(not rows[name] for name in rows):
        raise RuntimeError("Disk-backed feature extraction produced an empty split.")

    X_train_raw = np.vstack(rows["train"]).astype(np.float32)
    X_val_raw = np.vstack(rows["validation"]).astype(np.float32)
    X_test_raw = np.vstack(rows["test"]).astype(np.float32)
    y_train = np.asarray(labels["train"], dtype=np.int64)
    y_val = np.asarray(labels["validation"], dtype=np.int64)
    y_test = np.asarray(labels["test"], dtype=np.int64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    X_train_balanced, y_train_balanced, resampling_report = _safe_kmeans_smote(
        X_train_scaled, y_train
    )
    dataset = FeatureDataset(
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        X_validation=X_val_scaled,
        y_validation=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        train_metadata=metadata["train"],
        validation_metadata=metadata["validation"],
        test_metadata=metadata["test"],
        train_patients=train_patients,
        validation_patients=validation_patients,
        test_patients=test_patients,
        feature_names=feature_names,
        scaler=scaler,
        resampling_report=resampling_report,
        X_train_unbalanced=X_train_scaled,
        y_train_unbalanced=y_train,
    )
    dataset.validate()
    return dataset


def build_feature_dataset(
    recordings: Optional[Sequence[CleanEEGRecording]] = None,
) -> FeatureDataset:
    """Run the leakage-safe pipeline, preferring the disk-backed path."""

    if recordings is None:
        if Path(PREPROCESSED_MANIFEST_PATH).exists():
            return _split_feature_rows_by_manifest(PREPROCESSED_MANIFEST_PATH)
        if Path(PREPROCESSED_CACHE_PATH).exists():
            recordings = load_preprocessed_recordings(PREPROCESSED_CACHE_PATH)
        else:
            manifest = preprocess_dataset_to_disk()
            return _split_feature_rows_by_manifest(manifest)

    recordings = list(recordings)
    windows = remove_exact_duplicate_windows(build_all_windows(recordings))
    train_windows, validation_windows, test_windows, patient_splits = split_windows_by_patient(windows)
    X_train_raw, y_train, train_meta, feature_names = windows_to_features(train_windows)
    X_val_raw, y_val, val_meta, val_feature_names = windows_to_features(validation_windows)
    X_test_raw, y_test, test_meta, test_feature_names = windows_to_features(test_windows)
    if feature_names != val_feature_names or feature_names != test_feature_names:
        raise AssertionError("Feature schema is inconsistent across splits.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    X_train_balanced, y_train_balanced, resampling_report = _safe_kmeans_smote(
        X_train_scaled, y_train
    )
    dataset = FeatureDataset(
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        X_validation=X_val_scaled,
        y_validation=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        train_metadata=train_meta,
        validation_metadata=val_meta,
        test_metadata=test_meta,
        train_patients=patient_splits[0],
        validation_patients=patient_splits[1],
        test_patients=patient_splits[2],
        feature_names=feature_names,
        scaler=scaler,
        resampling_report=resampling_report,
        X_train_unbalanced=X_train_scaled,
        y_train_unbalanced=y_train,
    )
    dataset.validate()
    return dataset


# def _prepare_pickle_classes() -> None:
#     """Ensure artifacts created via `python -m backend.mlprocess` are portable."""
#     for cls in (FeatureDataset, ResamplingReport, DLFeatureArtifact, EEGWindow):
#         cls.__module__ = "backend.mlprocess"


def _prepare_pickle_classes() -> None:
    """Register the active classes under the canonical importable module name."""
    import sys

    module_name = "backend.mlprocess"

    # Use the active module when this file is executed as a script.
    active_module = sys.modules[__name__]
    registered_module = sys.modules.get(module_name)

    if registered_module is None:
        sys.modules[module_name] = active_module
        registered_module = active_module

    for cls in (FeatureDataset, ResamplingReport, DLFeatureArtifact, EEGWindow):
        cls.__module__ = module_name
        setattr(registered_module, cls.__name__, cls)

def save_feature_dataset(    dataset: FeatureDataset,
    output_path: str | Path = ML_DATASET_ARTIFACT_PATH,
) -> Path:
    _prepare_pickle_classes()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset, output, compress=3)
    return output


def load_feature_dataset(
    input_path: str | Path = ML_DATASET_ARTIFACT_PATH,
) -> FeatureDataset:
    import sys
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for cls in (FeatureDataset, ResamplingReport, DLFeatureArtifact, EEGWindow):
            setattr(main_module, cls.__name__, cls)
    dataset: FeatureDataset = joblib.load(input_path)
    if not isinstance(dataset, FeatureDataset):
        raise TypeError("Unexpected object in ML dataset artifact.")
    dataset.validate()
    return dataset


def build_dl_feature_artifact(dataset: FeatureDataset) -> DLFeatureArtifact:
    """Expose ML feature output in the tensor layout expected by a 1D CNN."""

    dataset.validate()
    X_train = dataset.X_train_unbalanced if dataset.X_train_unbalanced is not None else dataset.X_train
    y_train = dataset.y_train_unbalanced if dataset.y_train_unbalanced is not None else dataset.y_train
    train_metadata = list(dataset.train_metadata)
    if len(X_train) != len(train_metadata):
        raise ValueError("DL train artifact requires real training rows with matching metadata.")
    artifact = DLFeatureArtifact(
        X_train_cnn=np.asarray(X_train, dtype=np.float32)[:, np.newaxis, :],
        y_train=np.asarray(y_train, dtype=np.int64),
        X_validation_cnn=np.asarray(dataset.X_validation, dtype=np.float32)[:, np.newaxis, :],
        y_validation=np.asarray(dataset.y_validation, dtype=np.int64),
        X_test_cnn=np.asarray(dataset.X_test, dtype=np.float32)[:, np.newaxis, :],
        y_test=np.asarray(dataset.y_test, dtype=np.int64),
        train_metadata=list(dataset.train_metadata),
        validation_metadata=list(dataset.validation_metadata),
        test_metadata=list(dataset.test_metadata),
        feature_names=list(dataset.feature_names),
    )
    artifact.validate()
    return artifact


def save_dl_feature_artifact(
    artifact: DLFeatureArtifact,
    output_path: str | Path = ARTIFACT_DIR / "dl_feature_input.joblib",
) -> Path:
    """Persist the ML -> DL hand-off without rerunning EEG preprocessing."""

    artifact.validate()
    _prepare_pickle_classes()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    return output


def load_dl_feature_artifact(
    input_path: str | Path = ARTIFACT_DIR / "dl_feature_input.joblib",
) -> DLFeatureArtifact:
    artifact = joblib.load(input_path)
    if not isinstance(artifact, DLFeatureArtifact):
        raise TypeError("Unexpected object in DL feature artifact.")
    artifact.validate()
    return artifact


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Ratio of negative to positive TRAIN samples, computed BEFORE oversampling.

    This is exposed so mlout.py can optionally use XGBoost's own
    `scale_pos_weight` as an alternative (or ablation baseline) to K-Means
    SMOTE. Combining strong oversampling with a large scale_pos_weight at the
    same time tends to over-correct for class imbalance and can hurt
    precision (more false alarms) far more than it helps recall, so treat
    these as two competing strategies to compare, not something to stack by
    default.
    """

    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
    if counts[1] == 0:
        raise ValueError("Cannot compute scale_pos_weight with zero positive samples.")
    return float(counts[0] / counts[1])


def summarize_dataset(dataset: FeatureDataset) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "n_train": int(len(dataset.y_train)),
        "n_validation": int(len(dataset.y_validation)),
        "n_test": int(len(dataset.y_test)),
        "train_class_counts": np.bincount(dataset.y_train, minlength=2).tolist(),
        "validation_class_counts": np.bincount(dataset.y_validation, minlength=2).tolist(),
        "test_class_counts": np.bincount(dataset.y_test, minlength=2).tolist(),
        "n_train_patients": len(dataset.train_patients),
        "n_validation_patients": len(dataset.validation_patients),
        "n_test_patients": len(dataset.test_patients),
        "feature_count": len(dataset.feature_names),
        "feature_schema_version": dataset.feature_schema_version,
    }
    if dataset.resampling_report is not None:
        summary["resampling"] = asdict(dataset.resampling_report)
    return summary


def print_resampling_report(report: Optional[ResamplingReport]) -> None:
    """Human-readable class-balancing report for CLI output (see mlout.py train_command)."""

    if report is None:
        print("No resampling report available (dataset was built before this feature existed).")
        return

    non_seiz_before, seiz_before = report.class_counts_before
    non_seiz_after, seiz_after = report.class_counts_after

    print("\n=== TRAIN-SET CLASS BALANCING (SMOTE) REPORT ===")
    print(f"Method used            : {report.method_used}  params={report.method_params}")
    print(f"Fallback ladder tried   : {report.attempts_tried}")
    print(f"Before  -> non_seizure={non_seiz_before:>6d}  seizure={seiz_before:>6d}  "
          f"(imbalance {report.imbalance_ratio_before:.1f} : 1)")
    print(f"After   -> non_seizure={non_seiz_after:>6d}  seizure={seiz_after:>6d}  "
          f"(imbalance {report.imbalance_ratio_after:.1f} : 1)")
    print(f"Synthetic samples added: {report.n_synthetic_samples_added}")
    print("(Validation and test sets are never resampled or synthetically balanced.)")


# if __name__ == "__main__":
#     dataset = build_feature_dataset()
#     path = save_feature_dataset(dataset)
#     print(summarize_dataset(dataset))
#     print_resampling_report(dataset.resampling_report)
#     print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    # Step 1: Build the dataset once. Do not rerun preprocessing if a
    # compatible preprocessed manifest/cache is already available.
    dataset = build_feature_dataset()

    # Step 2: Save the main ML dataset.
    ml_path = save_feature_dataset(dataset)

    # Step 3: Save the DL handoff using real, unresampled training rows.
    dl_artifact = build_dl_feature_artifact(dataset)
    dl_path = save_dl_feature_artifact(dl_artifact)

    # Step 4: Save the feature schema separately.
    schema_path = Path(FEATURE_SCHEMA_PATH)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_schema_version": dataset.feature_schema_version,
            "feature_names": dataset.feature_names,
            "feature_count": len(dataset.feature_names),
        },
        schema_path,
        compress=3,
    )

    # Step 5: Report the outputs.
    print("\n=== DATASET SUMMARY ===")
    print(summarize_dataset(dataset))
    print_resampling_report(dataset.resampling_report)

    print("\n=== SAVED ARTIFACTS ===")
    for path in (ml_path, dl_path, schema_path):
        print(f"{path.resolve()} | {path.stat().st_size:,} bytes")
