# """
# mlout.py
# --------
# Train, evaluate, save, load, and serve the classical ML seizure detector.
# Stage 1 (this file) of the project. Stage 2 replaces the XGBoost block below
# with a 1D CNN -> Bi-LSTM -> Attention deep model; preprocessing.py and the
# window/feature contracts in mlprocess.py are designed to carry over unchanged.

# Model flow
# ----------
# 20-second EEG window
#     -> same preprocessing used by preprocessing.py
#     -> channel-invariant feature extraction from mlprocess.py
#     -> training-fitted StandardScaler
#     -> XGBoost classifier (validation-tuned decision threshold)
#     -> seizure probability

# The API exposes:
#     GET  /health
#     GET  /model-info
#     GET  /training-report        (SMOTE / class-balancing report + metrics)
#     POST /predict-window
#     POST /predict-event-sequence

# The last POST endpoint performs temporal event post-processing over
# consecutive 20-second window predictions. This is the project-level
# improvement intended to turn independent window predictions into clinically
# interpretable seizure events.

# Project abstract
# -----------------
# This project builds a patient-independent, event-level epileptic seizure
# detector on the CHB-MIT scalp EEG dataset (~10 patients, 35-50 EDF
# recordings for this prototype stage). Raw EDF signals are cleaned with MNE
# (band-pass 0.5-40 Hz, 60 Hz notch, resampling to 256 Hz) in preprocessing.py,
# then segmented into non-overlapping 20-second windows and converted into a
# channel-order-invariant statistical/spectral feature vector (time-domain
# statistics, Hjorth parameters, Welch band power per canonical EEG band, and
# inter-channel correlation) in mlprocess.py. Because the CHB-MIT class
# distribution is heavily skewed toward non-seizure windows, class imbalance
# is corrected on the TRAINING split only, after a strict patient-level
# train/validation/test split, using a K-Means SMOTE -> SMOTE ->
# RandomOverSampler fallback ladder (KMeansSMOTE is the preferred, highest-
# quality method, but can fail to find valid clusters on this small a cohort).
# XGBoost, chosen for its strength on structured/tabular features, small-data
# robustness, and interpretability, is trained on the balanced features with
# early stopping and a validation-tuned decision threshold, then evaluated
# with both window-level metrics (accuracy, sensitivity, specificity, F1,
# ROC/PR-AUC) and clinically-motivated event-level metrics (event sensitivity,
# event precision/F1, false alarms per hour, detection latency) produced by
# grouping consecutive positive-probability windows into seizure episodes.
# This XGBoost pipeline is Stage 1: a scalable, leakage-safe classical ML
# baseline. Stage 2 will replace the hand-crafted feature step with an
# end-to-end deep model — a 1D CNN over raw EEG windows for automatic local
# feature extraction, feeding a Bi-LSTM to capture bidirectional temporal
# dependencies across the window sequence, with an attention layer to weight
# the most seizure-relevant time steps before the final classification head —
# reusing the same preprocessing, patient-level split, and event-level
# evaluation built here.
# """

# from __future__ import annotations

# import argparse
# import logging
# import os
# from dataclasses import asdict, dataclass
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Dict, List, Optional, Sequence, Tuple

# import joblib
# import numpy as np
# from sklearn.metrics import (
#     accuracy_score,
#     average_precision_score,
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )

# try:
#     from . import runtime  # noqa: F401
#     from .mlprocess import (
#         EEGWindow,
#         FeatureDataset,
#         ResamplingReport,
#         extract_window_features,
#         load_feature_dataset,
#         print_resampling_report,
#         ML_DATASET_ARTIFACT_PATH,
#         WINDOW_SECONDS,
#     )
#     from .preprocessing import (
#         CleanEEGRecording,
#         SeizureInterval,
#         LOW_CUTOFF_HZ,
#         HIGH_CUTOFF_HZ,
#         NOTCH_FREQUENCY_HZ,
#         TARGET_SAMPLING_RATE_HZ,
#         WINDOW_DURATION_SECONDS,
#         preprocess_raw,
#         PREPROCESSED_CACHE_PATH,
#         PREPROCESSED_MANIFEST_PATH,
#     )
#     from .report import generate_model_report
# except ImportError:  # supports `python backend/mlout.py`
#     import runtime  # type: ignore  # noqa: F401
#     from mlprocess import (  # type: ignore
#     EEGWindow,
#     FeatureDataset,
#     ResamplingReport,
#     extract_window_features,
#     load_feature_dataset,
#     print_resampling_report,
#     ML_DATASET_ARTIFACT_PATH,
#     WINDOW_SECONDS,
#     )
#     from preprocessing import (  # type: ignore
#     CleanEEGRecording,
#     SeizureInterval,
#     LOW_CUTOFF_HZ,
#     HIGH_CUTOFF_HZ,
#     NOTCH_FREQUENCY_HZ,
#     TARGET_SAMPLING_RATE_HZ,
#     WINDOW_DURATION_SECONDS,
#     preprocess_raw,
#     PREPROCESSED_CACHE_PATH,
#     PREPROCESSED_MANIFEST_PATH,
#     )
#     from report import generate_model_report  # type: ignore

#     # XGBoost is a strong classical baseline for nonlinear EEG feature spaces.
# try:
#     from xgboost import XGBClassifier
# except ImportError as exc:  # pragma: no cover - environment-specific dependency guard
#     raise ImportError(
#         "xgboost is required for mlout.py. Install it with `pip install xgboost`."
#     ) from exc

# # FastAPI is imported lazily by build_app() so model training can run without
# # needing the web server stack in memory.

# # -----------------------------------------------------------------------------
# # USER CONFIGURATION
# # -----------------------------------------------------------------------------
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
# MODEL_ARTIFACT_PATH: str = str(ARTIFACT_DIR / "seizure_ml_model.joblib")
# MODEL_METRICS_PATH: str = str(ARTIFACT_DIR / "ml_metrics.joblib")
# MODEL_REPORT_PATH: str = str(ARTIFACT_DIR / "ml_model_report.html")
# MODEL_VERSION: str = "ml-baseline-v2"

# RANDOM_STATE: int = 42

# # Fallback threshold, only used if validation-based threshold tuning below is
# # unavailable (e.g. the validation split has zero seizure windows). Prefer
# # bundle.decision_threshold (tuned per training run) over this constant.
# DEFAULT_SEIZURE_THRESHOLD: float = 0.50

# # Strong, but not excessively large, XGBoost baseline for tabular EEG features.
# # early_stopping_rounds is set here (not just eval_set) so training actually
# # stops once validation logloss stops improving, instead of always fitting
# # the full 500 trees and risking overfitting on a small patient sample.
# XGB_PARAMS: Dict[str, object] = {
#     "n_estimators": 500,
#     "max_depth": 5,
#     "learning_rate": 0.05,
#     "subsample": 0.90,
#     "colsample_bytree": 0.85,
#     "min_child_weight": 2,
#     "gamma": 0.0,
#     "reg_alpha": 0.05,
#     "reg_lambda": 1.0,
#     "objective": "binary:logistic",
#     "eval_metric": "logloss",
#     "tree_method": "hist",
#     "device": "cpu",
#     "random_state": RANDOM_STATE,
#     "n_jobs": -1,
#     "early_stopping_rounds": 30,
# }

# # Threshold selection strategy applied to the validation set after training.
# # "f1" maximizes window-level F1. If you care more about false alarms than
# # missed windows (usually true for continuous EEG monitoring), consider
# # scanning for the smallest threshold that keeps validation recall above a
# # clinically-chosen floor instead.
# THRESHOLD_SELECTION_METRIC: str = "f1"

# # Temporal event-level post-processing.
# EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS: int = 1
# EVENT_MAX_GAP_WINDOWS: int = 1

# # -----------------------------------------------------------------------------
# # LOGGING
# # -----------------------------------------------------------------------------

# LOGGER = logging.getLogger("minorproj.mlout")


# # -----------------------------------------------------------------------------
# # MODEL BUNDLE
# # -----------------------------------------------------------------------------


# @dataclass
# class ModelBundle:
#     model: XGBClassifier
#     scaler: object
#     feature_names: List[str]
#     model_version: str
#     sampling_rate_hz: float
#     window_seconds: float
#     low_cutoff_hz: float
#     high_cutoff_hz: float
#     notch_frequency_hz: float
#     class_names: List[str]
#     created_at_utc: str
#     decision_threshold: float = DEFAULT_SEIZURE_THRESHOLD

#     def validate(self) -> None:
#         if len(self.class_names) != 2:
#             raise ValueError("This project expects binary seizure classification.")
#         if not self.feature_names:
#             raise ValueError("Feature schema cannot be empty.")
#         if self.window_seconds <= 0:
#             raise ValueError("Window duration must be positive.")
#         if not (0.0 < self.decision_threshold < 1.0):
#             raise ValueError("decision_threshold must be strictly between 0 and 1.")


# # -----------------------------------------------------------------------------
# # MODEL TRAINING
# # -----------------------------------------------------------------------------


# def make_model() -> XGBClassifier:
#     return XGBClassifier(**XGB_PARAMS)


# def select_decision_threshold(
#     y_true: np.ndarray,
#     y_prob: np.ndarray,
#     metric: str = THRESHOLD_SELECTION_METRIC,
#     fallback: float = DEFAULT_SEIZURE_THRESHOLD,
# ) -> float:
#     """
#     Scan candidate thresholds on a held-out split and pick the one that
#     maximizes `metric` (currently only 'f1' is implemented).

#     A fixed 0.50 cutoff has no particular reason to be optimal for an
#     imbalanced, oversampled-at-train-time model; the operating point should
#     be chosen using data the model did not train on. Falls back to
#     `fallback` if the split has only one class present (metric undefined).
#     """

#     y_true = np.asarray(y_true, dtype=np.int64)
#     y_prob = np.asarray(y_prob, dtype=np.float64)
#     if np.unique(y_true).size < 2:
#         LOGGER.warning(
#             "Cannot tune decision threshold: validation split has only one class. "
#             "Falling back to %.2f.",
#             fallback,
#         )
#         return fallback

#     candidates = np.unique(np.clip(y_prob, 1e-6, 1 - 1e-6))
#     if candidates.size == 0:
#         return fallback

#     best_threshold = fallback
#     best_score = -1.0
#     for threshold in candidates:
#         y_pred = (y_prob >= threshold).astype(np.int64)
#         if metric == "f1":
#             score = f1_score(y_true, y_pred, zero_division=0)
#         else:
#             raise ValueError(f"Unsupported threshold selection metric: {metric}")
#         if score > best_score:
#             best_score = score
#             best_threshold = float(threshold)

#     LOGGER.info(
#         "Selected decision threshold=%.4f on validation set (%s=%.4f).",
#         best_threshold,
#         metric,
#         best_score,
#     )
#     return best_threshold


# def train_model(dataset: FeatureDataset) -> ModelBundle:
#     dataset.validate()
#     model = make_model()

#     # Validation is never used as training data. It is used for two things
#     # only, both of which stop short of fitting on it: (1) early stopping,
#     # so training halts once validation logloss stops improving, and
#     # (2) picking the decision threshold below. The tree ensemble itself is
#     # fit on X_train/y_train only.
#     model.fit(
#         dataset.X_train,
#         dataset.y_train,
#         eval_set=[(dataset.X_validation, dataset.y_validation)],
#         verbose=False,
#     )

#     val_probabilities = model.predict_proba(dataset.X_validation)[:, 1]
#     decision_threshold = select_decision_threshold(dataset.y_validation, val_probabilities)

#     bundle = ModelBundle(
#         model=model,
#         scaler=dataset.scaler,
#         feature_names=list(dataset.feature_names),
#         model_version=MODEL_VERSION,
#         sampling_rate_hz=float(TARGET_SAMPLING_RATE_HZ),
#         window_seconds=float(WINDOW_DURATION_SECONDS),
#         low_cutoff_hz=float(LOW_CUTOFF_HZ),
#         high_cutoff_hz=float(HIGH_CUTOFF_HZ),
#         notch_frequency_hz=float(NOTCH_FREQUENCY_HZ),
#         class_names=["non_seizure", "seizure"],
#         created_at_utc=datetime.now(timezone.utc).isoformat(),
#         decision_threshold=decision_threshold,
#     )
#     bundle.validate()
#     return bundle


# def save_model_bundle(
#     bundle: ModelBundle,
#     output_path: str | Path = MODEL_ARTIFACT_PATH,
# ) -> Path:
#     bundle.validate()
#     output = Path(output_path)
#     output.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(bundle, output, compress=3)
#     LOGGER.info("Saved trained model bundle to %s", output.resolve())
#     return output


# def load_model_bundle(
#     input_path: str | Path = MODEL_ARTIFACT_PATH,
# ) -> ModelBundle:
#     import sys
#     main_module = sys.modules.get("__main__")
#     if main_module is not None:
#         setattr(main_module, "ModelBundle", ModelBundle)
#     bundle: ModelBundle = joblib.load(input_path)
#     if not isinstance(bundle, ModelBundle):
#         raise TypeError("Unexpected object in model artifact.")
#     bundle.validate()
#     return bundle


# # -----------------------------------------------------------------------------
# # WINDOW-LEVEL INFERENCE
# # -----------------------------------------------------------------------------


# def predict_feature_matrix(
#     bundle: ModelBundle,
#     X: np.ndarray,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     X = np.asarray(X, dtype=np.float32)
#     if X.ndim != 2:
#         raise ValueError("Feature input must be 2D.")
#     if X.shape[1] != len(bundle.feature_names):
#         raise ValueError(
#             f"Expected {len(bundle.feature_names)} features, got {X.shape[1]}."
#         )
#     if not np.isfinite(X).all():
#         raise ValueError("Feature input contains non-finite values.")

#     probabilities = bundle.model.predict_proba(X)[:, 1].astype(np.float64)
#     predictions = (probabilities >= bundle.decision_threshold).astype(np.int64)
#     return predictions, probabilities


# def predict_window_from_preprocessed_array(
#     bundle: ModelBundle,
#     eeg_window_uv: np.ndarray,
#     sampling_rate_hz: float,
#     channel_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, object]:
#     """
#     Predict one raw 20-second EEG window.

#     The array may be supplied as [channels, samples]. If a caller provides a
#     single-channel vector [samples], it is promoted to [1, samples].
#     """

#     eeg = np.asarray(eeg_window_uv, dtype=np.float32)
#     if eeg.ndim == 1:
#         eeg = eeg[np.newaxis, :]
#     if eeg.ndim != 2:
#         raise ValueError("EEG input must have shape [channels, samples].")

#     if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
#         raise ValueError("sampling_rate_hz must be a positive finite number.")

#     expected_input_samples = int(round(float(sampling_rate_hz) * bundle.window_seconds))
#     if eeg.shape[1] != expected_input_samples:
#         raise ValueError(
#             f"Expected exactly {expected_input_samples} samples ({bundle.window_seconds:.0f}s at "
#             f"{sampling_rate_hz:.0f}Hz), got {eeg.shape[1]}."
#         )

#     # Recreate the exact preprocessing used for training from the raw array.
#     # Using MNE here keeps filtering behavior aligned with preprocessing.py.
#     import mne

#     names = list(channel_names) if channel_names is not None else [f"EEG{i+1:02d}" for i in range(eeg.shape[0])]
#     if len(names) != eeg.shape[0]:
#         raise ValueError("channel_names length does not match EEG channel count.")

#     info = mne.create_info(
#         ch_names=names,
#         sfreq=float(sampling_rate_hz),
#         ch_types=["eeg"] * len(names),
#     )
#     raw = mne.io.RawArray(eeg.astype(np.float64), info, verbose=False)
#     cleaned, cleaned_names, cleaned_sr = preprocess_raw(raw)

#     if not np.isclose(cleaned_sr, bundle.sampling_rate_hz):
#         raise ValueError("Preprocessing did not produce the model's target sampling rate.")

#     window = EEGWindow(
#         patient_id="api_patient",
#         recording_id="api_recording",
#         window_id="api_window",
#         start_seconds=0.0,
#         end_seconds=bundle.window_seconds,
#         label=0,
#         label_name="unknown",
#         data_uv=cleaned,
#         sampling_rate_hz=cleaned_sr,
#         channel_names=cleaned_names,
#     )

#     X_raw, feature_names = extract_window_features(window)
#     if feature_names != bundle.feature_names:
#         raise ValueError(
#             "Feature schema mismatch between API preprocessing and trained model. "
#             "Retrain after changing preprocessing or feature extraction."
#         )

#     X_scaled = bundle.scaler.transform(X_raw[np.newaxis, :]).astype(np.float32)
#     predictions, probabilities = predict_feature_matrix(bundle, X_scaled)
#     probability = float(probabilities[0])
#     prediction = int(predictions[0])

#     return {
#         "prediction": bundle.class_names[prediction],
#         "class_id": prediction,
#         "seizure_probability": probability,
#         "confidence": probability if prediction == 1 else 1.0 - probability,
#         "model_version": bundle.model_version,
#         "window_seconds": bundle.window_seconds,
#         "sampling_rate_hz": bundle.sampling_rate_hz,
#     }


# # -----------------------------------------------------------------------------
# # CLASSIFICATION METRICS
# # -----------------------------------------------------------------------------


# def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
#     y_true = np.asarray(y_true, dtype=np.int64)
#     y_pred = np.asarray(y_pred, dtype=np.int64)
#     y_prob = np.asarray(y_prob, dtype=np.float64)

#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#     tn, fp, fn, tp = cm.ravel()
#     specificity = tn / max(tn + fp, 1)

#     metrics: Dict[str, object] = {
#         "accuracy": float(accuracy_score(y_true, y_pred)),
#         "precision": float(precision_score(y_true, y_pred, zero_division=0)),
#         "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
#         "specificity": float(specificity),
#         "f1": float(f1_score(y_true, y_pred, zero_division=0)),
#         "confusion_matrix": cm.tolist(),
#         "classification_report": classification_report(
#             y_true,
#             y_pred,
#             target_names=["non_seizure", "seizure"],
#             output_dict=True,
#             zero_division=0,
#         ),
#     }

#     if np.unique(y_true).size == 2:
#         metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
#         metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
#     else:
#         metrics["roc_auc"] = None
#         metrics["pr_auc"] = None
#     return metrics


# # -----------------------------------------------------------------------------
# # TEMPORAL EVENT-LEVEL DETECTION
# # -----------------------------------------------------------------------------


# @dataclass
# class PredictedEvent:
#     patient_id: str
#     recording_id: str
#     start_seconds: float
#     end_seconds: float
#     duration_seconds: float
#     max_probability: float
#     mean_probability: float


# def _group_positive_windows(
#     metadata: Sequence[dict],
#     probabilities: np.ndarray,
#     threshold: float = DEFAULT_SEIZURE_THRESHOLD,
# ) -> List[PredictedEvent]:
#     grouped: Dict[Tuple[str, str], List[Tuple[dict, float]]] = {}
#     for meta, probability in zip(metadata, probabilities):
#         key = (meta["patient_id"], meta["recording_id"])
#         grouped.setdefault(key, []).append((meta, float(probability)))

#     events: List[PredictedEvent] = []
#     max_gap_seconds = WINDOW_SECONDS * EVENT_MAX_GAP_WINDOWS

#     for (patient_id, recording_id), rows in grouped.items():
#         rows = sorted(rows, key=lambda item: float(item[0]["start_seconds"]))
#         active: List[Tuple[dict, float]] = []
#         candidate_groups: List[List[Tuple[dict, float]]] = []

#         for meta, probability in rows:
#             positive = probability >= threshold
#             if not positive:
#                 if active:
#                     candidate_groups.append(active)
#                     active = []
#                 continue

#             if not active:
#                 active = [(meta, probability)]
#                 continue

#             previous_end = float(active[-1][0]["end_seconds"])
#             current_start = float(meta["start_seconds"])
#             gap = current_start - previous_end
#             if gap <= max_gap_seconds + 1e-6:
#                 active.append((meta, probability))
#             else:
#                 candidate_groups.append(active)
#                 active = [(meta, probability)]

#         if active:
#             candidate_groups.append(active)

#         for group in candidate_groups:
#             if len(group) < EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS:
#                 continue
#             start = float(group[0][0]["start_seconds"])
#             end = float(group[-1][0]["end_seconds"])
#             probs = np.asarray([prob for _, prob in group], dtype=np.float64)
#             events.append(
#                 PredictedEvent(
#                     patient_id=patient_id,
#                     recording_id=recording_id,
#                     start_seconds=start,
#                     end_seconds=end,
#                     duration_seconds=end - start,
#                     max_probability=float(np.max(probs)),
#                     mean_probability=float(np.mean(probs)),
#                 )
#             )

#     return events


# def _ground_truth_events_from_metadata(
#     metadata: Sequence[dict],
#     recordings_by_key: Dict[Tuple[str, str], Sequence[Tuple[float, float]]],
# ) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
#     result: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
#     keys = {(m["patient_id"], m["recording_id"]) for m in metadata}
#     for key in keys:
#         result[key] = list(recordings_by_key.get(key, []))
#     return result


# def event_level_metrics(
#     metadata: Sequence[dict],
#     probabilities: np.ndarray,
#     ground_truth_events: Dict[Tuple[str, str], Sequence[Tuple[float, float]]],
#     threshold: float = DEFAULT_SEIZURE_THRESHOLD,
# ) -> Dict[str, object]:
#     """Calculate event sensitivity, event precision/F1, false alarms/hour and latency."""

#     predicted_events = _group_positive_windows(metadata, probabilities, threshold=threshold)

#     matched_true = 0
#     matched_predicted = 0
#     latencies: List[float] = []

#     # One prediction may match one true event only; greedy earliest-overlap matching
#     # is deterministic and avoids double counting.
#     used_predictions: set[int] = set()
#     total_true_events = sum(len(events) for events in ground_truth_events.values())

#     for true_key, true_events in ground_truth_events.items():
#         key_pred_indices = [
#             i
#             for i, event in enumerate(predicted_events)
#             if (event.patient_id, event.recording_id) == true_key
#         ]
#         for true_start, true_end in true_events:
#             candidate_index: Optional[int] = None
#             candidate_onset = float("inf")

#             for index in key_pred_indices:
#                 if index in used_predictions:
#                     continue
#                 pred = predicted_events[index]
#                 overlap = max(
#                     0.0,
#                     min(pred.end_seconds, true_end)
#                     - max(pred.start_seconds, true_start),
#                 )
#                 if overlap <= 0:
#                     continue
#                 if pred.start_seconds < candidate_onset:
#                     candidate_onset = pred.start_seconds
#                     candidate_index = index

#             if candidate_index is not None:
#                 used_predictions.add(candidate_index)
#                 matched_true += 1
#                 matched_predicted += 1
#                 pred = predicted_events[candidate_index]
#                 latencies.append(max(0.0, pred.end_seconds - true_start))

#     false_positive_events = len(predicted_events) - matched_predicted

#     # Estimate non-seizure observation hours from the metadata covered by TEST.
#     # Each metadata row is a 20-second window, so this remains valid even when
#     # multiple recordings/patients have different durations.
#     total_test_seconds = 0.0
#     seizure_seconds = 0.0
#     seen_windows: set[Tuple[str, str, float]] = set()
#     for meta in metadata:
#         key = (
#             meta["patient_id"],
#             meta["recording_id"],
#             float(meta["start_seconds"]),
#         )
#         if key in seen_windows:
#             continue
#         seen_windows.add(key)
#         total_test_seconds += max(
#             0.0,
#             float(meta["end_seconds"]) - float(meta["start_seconds"]),
#         )

#     for true_events in ground_truth_events.values():
#         for start, end in true_events:
#             seizure_seconds += max(0.0, end - start)

#     non_seizure_hours = max((total_test_seconds - seizure_seconds) / 3600.0, 1e-9)
#     false_alarms_per_hour = false_positive_events / non_seizure_hours

#     event_sensitivity = matched_true / max(total_true_events, 1)
#     event_precision = matched_predicted / max(len(predicted_events), 1)
#     event_f1 = (
#         2 * event_precision * event_sensitivity / max(event_precision + event_sensitivity, 1e-12)
#     )

#     return {
#         "event_sensitivity": float(event_sensitivity),
#         "event_precision": float(event_precision),
#         "event_f1": float(event_f1),
#         "false_alarm_events": int(false_positive_events),
#         "false_alarms_per_hour": float(false_alarms_per_hour),
#         "mean_detection_latency_seconds": float(np.mean(latencies)) if latencies else None,
#         "median_detection_latency_seconds": float(np.median(latencies)) if latencies else None,
#         "detected_events": int(len(predicted_events)),
#         "ground_truth_events": int(total_true_events),
#         "matched_events": int(matched_true),
#         "predicted_event_details": [event.__dict__ for event in predicted_events],
#     }


# # -----------------------------------------------------------------------------
# # FULL EVALUATION
# # -----------------------------------------------------------------------------


# def evaluate_bundle(
#     bundle: ModelBundle,
#     dataset: FeatureDataset,
# ) -> Dict[str, object]:
#     y_val_pred, y_val_prob = predict_feature_matrix(bundle, dataset.X_validation)
#     y_test_pred, y_test_prob = predict_feature_matrix(bundle, dataset.X_test)

#     results: Dict[str, object] = {
#         "model_version": bundle.model_version,
#         "validation": classification_metrics(
#             dataset.y_validation, y_val_pred, y_val_prob
#         ),
#         "test": classification_metrics(dataset.y_test, y_test_pred, y_test_prob),
#         "patient_splits": {
#             "train": dataset.train_patients,
#             "validation": dataset.validation_patients,
#             "test": dataset.test_patients,
#         },
#     }

#     # Event metrics require ground-truth event intervals. Those are supplied by
#     # reloading the original recordings rather than duplicating them per window.
#     # The artifact itself keeps window-level metadata; the recording intervals are
#     # reconstructed from the original preprocessing output.
#     try:
#         if Path(PREPROCESSED_MANIFEST_PATH).exists():
#             try:
#                 from .preprocessing import iter_preprocessed_recordings
#             except ImportError:
#                 from preprocessing import iter_preprocessed_recordings  # type: ignore
#             recordings = iter_preprocessed_recordings(PREPROCESSED_MANIFEST_PATH)
#         else:
#             try:
#                 from .preprocessing import load_preprocessed_recordings
#             except ImportError:
#                 from preprocessing import load_preprocessed_recordings  # type: ignore
#             recordings = load_preprocessed_recordings(PREPROCESSED_CACHE_PATH)
#         event_truth: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
#         for recording in recordings:
#             event_truth[(recording.patient_id, recording.recording_id)] = [
#                 (interval.start_seconds, interval.end_seconds)
#                 for interval in recording.seizure_intervals
#             ]
#             del recording

#         results["test_event_level"] = event_level_metrics(
#             dataset.test_metadata,
#             y_test_prob,
#             _ground_truth_events_from_metadata(dataset.test_metadata, event_truth),
#             threshold=bundle.decision_threshold,
#         )
#     except FileNotFoundError:
#         LOGGER.warning(
#             "Preprocessing cache is unavailable; skipping event-level ground-truth metrics."
#         )
#         results["test_event_level"] = None

#     return results


# # -----------------------------------------------------------------------------
# # API
# # -----------------------------------------------------------------------------


# def build_app(bundle_path: str | Path = MODEL_ARTIFACT_PATH):
#     from fastapi import FastAPI, HTTPException
#     from pydantic import BaseModel, Field

#     bundle = load_model_bundle(bundle_path)
#     app = FastAPI(
#         title="EEG Seizure Detection API",
#         version=bundle.model_version,
#         description="Patient-independent 20-second EEG seizure detector with temporal event post-processing.",
#     )

#     class WindowRequest(BaseModel):
#         eeg: List[List[float]] = Field(..., description="EEG matrix [channels][samples].")
#         sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ
#         channel_names: Optional[List[str]] = None

#     class WindowResponse(BaseModel):
#         prediction: str
#         class_id: int
#         seizure_probability: float
#         confidence: float
#         model_version: str
#         window_seconds: float
#         sampling_rate_hz: float

#     class FeatureRequest(BaseModel):
#         features: List[float] = Field(..., description="ML feature vector in model schema order.")

#     class FeatureResponse(BaseModel):
#         prediction: str
#         class_id: int
#         seizure_probability: float
#         confidence: float
#         model_version: str
#         feature_count: int

#     class SequenceWindow(BaseModel):
#         start_seconds: float = Field(..., ge=0.0)
#         end_seconds: float = Field(..., gt=0.0)
#         eeg: List[List[float]] = Field(...)
#         sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ
#         channel_names: Optional[List[str]] = None

#     class EventSequenceRequest(BaseModel):
#         patient_id: str = "api_patient"
#         recording_id: str = "api_recording"
#         windows: List[SequenceWindow] = Field(..., min_length=1)

#     @app.get("/health")
#     def health() -> Dict[str, object]:
#         return {
#             "status": "healthy",
#             "model_loaded": True,
#             "model_version": bundle.model_version,
#         }

#     @app.get("/model-info")
#     def model_info() -> Dict[str, object]:
#         return {
#             "model_version": bundle.model_version,
#             "created_at_utc": bundle.created_at_utc,
#             "sampling_rate_hz": bundle.sampling_rate_hz,
#             "window_seconds": bundle.window_seconds,
#             "expected_samples": int(bundle.sampling_rate_hz * bundle.window_seconds),
#             "classes": bundle.class_names,
#             "feature_count": len(bundle.feature_names),
#             "decision_threshold": bundle.decision_threshold,
#         }

#     @app.get("/training-report")
#     def training_report() -> Dict[str, object]:
#         """
#         Class-balancing (SMOTE) and evaluation report saved by `train_command`.

#         Exposes exactly how the small (~10-patient) training set was
#         oversampled, so imbalance-driven bias in the served model is
#         auditable from the API, not just from training logs.
#         """
#         metrics_path = Path(MODEL_METRICS_PATH)
#         if not metrics_path.exists():
#             raise HTTPException(
#                 status_code=404,
#                 detail="No saved training report found. Run `python mlout.py train` first.",
#             )
#         results = joblib.load(metrics_path)
#         return {
#             "model_version": results.get("model_version"),
#             "resampling_report": results.get("resampling_report"),
#             "validation": results.get("validation"),
#             "test": {
#                 k: v for k, v in results.get("test", {}).items()
#                 if k not in {"classification_report"}
#             },
#             "test_event_level": results.get("test_event_level"),
#         }

#     @app.post("/predict-features", response_model=FeatureResponse)
#     def predict_features(request: FeatureRequest) -> Dict[str, object]:
#         """Predict from the exact feature vector handed to the future DL stage."""

#         try:
#             raw_features = np.asarray(request.features, dtype=np.float32)
#             if raw_features.ndim != 1 or raw_features.size != len(bundle.feature_names):
#                 raise ValueError(f"Expected exactly {len(bundle.feature_names)} features.")
#             if not np.isfinite(raw_features).all():
#                 raise ValueError("Feature input contains non-finite values.")
#             scaled = bundle.scaler.transform(raw_features[np.newaxis, :]).astype(np.float32)
#             predictions, probabilities = predict_feature_matrix(bundle, scaled)
#             class_id = int(predictions[0])
#             probability = float(probabilities[0])
#             return {
#                 "prediction": bundle.class_names[class_id],
#                 "class_id": class_id,
#                 "seizure_probability": probability,
#                 "confidence": probability if class_id else 1.0 - probability,
#                 "model_version": bundle.model_version,
#                 "feature_count": len(bundle.feature_names),
#             }
#         except (ValueError, TypeError) as exc:
#             raise HTTPException(status_code=422, detail=str(exc)) from exc

#     @app.post("/predict-window", response_model=WindowResponse)
#     def predict_window(request: WindowRequest) -> Dict[str, object]:
#         try:
#             return predict_window_from_preprocessed_array(
#                 bundle=bundle,
#                 eeg_window_uv=np.asarray(request.eeg, dtype=np.float32),
#                 sampling_rate_hz=request.sampling_rate_hz,
#                 channel_names=request.channel_names,
#             )
#         except (ValueError, TypeError) as exc:
#             raise HTTPException(status_code=422, detail=str(exc)) from exc

#     @app.post("/predict-event-sequence")
#     def predict_event_sequence(request: EventSequenceRequest) -> Dict[str, object]:
#         if not request.windows:
#             raise HTTPException(status_code=422, detail="At least one EEG window is required.")

#         metadata: List[dict] = []
#         probabilities: List[float] = []

#         try:
#             windows_sorted = sorted(request.windows, key=lambda w: w.start_seconds)
#             for index, item in enumerate(windows_sorted):
#                 result = predict_window_from_preprocessed_array(
#                     bundle=bundle,
#                     eeg_window_uv=np.asarray(item.eeg, dtype=np.float32),
#                     sampling_rate_hz=item.sampling_rate_hz,
#                     channel_names=item.channel_names,
#                 )
#                 metadata.append(
#                     {
#                         "patient_id": request.patient_id,
#                         "recording_id": request.recording_id,
#                         "window_id": f"api_window_{index:06d}",
#                         "start_seconds": item.start_seconds,
#                         "end_seconds": item.end_seconds,
#                         "label": 0,
#                         "label_name": "unknown",
#                     }
#                 )
#                 probabilities.append(float(result["seizure_probability"]))
#         except (ValueError, TypeError) as exc:
#             raise HTTPException(status_code=422, detail=str(exc)) from exc

#         events = _group_positive_windows(
#             metadata,
#             np.asarray(probabilities, dtype=np.float64),
#             threshold=bundle.decision_threshold,
#         )
#         return {
#             "model_version": bundle.model_version,
#             "decision_threshold": bundle.decision_threshold,
#             "window_predictions": [
#                 {
#                     "start_seconds": meta["start_seconds"],
#                     "end_seconds": meta["end_seconds"],
#                     "seizure_probability": probability,
#                     "prediction": "seizure" if probability >= bundle.decision_threshold else "non_seizure",
#                 }
#                 for meta, probability in zip(metadata, probabilities)
#             ],
#             "detected_events": [event.__dict__ for event in events],
#         }

#     return app


# # -----------------------------------------------------------------------------
# # COMMAND LINE
# # -----------------------------------------------------------------------------


# def train_command() -> None:
#     dataset = load_feature_dataset(ML_DATASET_ARTIFACT_PATH)

#     # Class-balance transparency: show exactly how the (small, ~10-patient)
#     # training set was oversampled BEFORE showing any model performance, so
#     # imbalance-driven bias is visible up front rather than inferred later.
#     print_resampling_report(dataset.resampling_report)

#     bundle = train_model(dataset)
#     save_model_bundle(bundle, MODEL_ARTIFACT_PATH)

#     results = evaluate_bundle(bundle, dataset)
#     if dataset.resampling_report is not None:
#         results["resampling_report"] = asdict(dataset.resampling_report)
#     metrics_path = Path(MODEL_METRICS_PATH)
#     metrics_path.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(results, metrics_path, compress=3)
#     generate_model_report(dataset, results, bundle, MODEL_REPORT_PATH)

#     print(f"\nSelected decision threshold (tuned on validation set): {bundle.decision_threshold:.4f}")
#     print("\n=== TEST CLASSIFICATION METRICS ===")
#     for key, value in results["test"].items():
#         if key not in {"classification_report", "confusion_matrix"}:
#             print(f"{key}: {value}")
#     print("confusion_matrix:", results["test"]["confusion_matrix"])

#     print("\n=== TEST EVENT-LEVEL METRICS ===")
#     event_metrics = results.get("test_event_level")
#     if event_metrics is None:
#         print("Event metrics unavailable because preprocessing annotations were not cached.")
#     else:
#         for key in [
#             "event_sensitivity",
#             "event_precision",
#             "event_f1",
#             "false_alarms_per_hour",
#             "mean_detection_latency_seconds",
#             "median_detection_latency_seconds",
#         ]:
#             print(f"{key}: {event_metrics.get(key)}")


# def serve_command(host: str, port: int) -> None:
#     import uvicorn

#     app = build_app(MODEL_ARTIFACT_PATH)
#     uvicorn.run(app, host=host, port=port)


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Train/evaluate/serve EEG seizure ML model.")
#     subparsers = parser.add_subparsers(dest="command", required=True)

#     train_parser = subparsers.add_parser("train", help="Train and evaluate the model.")
#     train_parser.set_defaults(func=lambda args: train_command())

#     serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server.")
#     serve_parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
#     serve_parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
#     serve_parser.set_defaults(func=lambda args: serve_command(args.host, args.port))

#     args = parser.parse_args()
#     args.func(args)


# if __name__ == "__main__":
#     main()


"""
mlout.py
--------
Train, evaluate, save, load, and serve the classical ML seizure detector.
Stage 1 (this file) of the project. Stage 2 replaces the XGBoost block below
with a 1D CNN -> Bi-LSTM -> Attention deep model; preprocessing.py and the
window/feature contracts in mlprocess.py are designed to carry over unchanged.

Model flow
----------
20-second EEG window
    -> same preprocessing used by preprocessing.py
    -> channel-invariant feature extraction from mlprocess.py
    -> training-fitted StandardScaler
    -> XGBoost classifier (validation-tuned decision threshold)
    -> seizure probability

The API exposes:
    GET  /health
    GET  /model-info
    GET  /training-report        (SMOTE / class-balancing report + metrics)
    POST /predict-window
    POST /predict-event-sequence

The last POST endpoint performs temporal event post-processing over
consecutive 20-second window predictions. This is the project-level
improvement intended to turn independent window predictions into clinically
interpretable seizure events.

Project abstract
-----------------
This project builds a patient-independent, event-level epileptic seizure
detector on the CHB-MIT scalp EEG dataset (~10 patients, 35-50 EDF
recordings for this prototype stage). Raw EDF signals are cleaned with MNE
(band-pass 0.5-40 Hz, 60 Hz notch, resampling to 256 Hz) in preprocessing.py,
then segmented into non-overlapping 20-second windows and converted into a
channel-order-invariant statistical/spectral feature vector (time-domain
statistics, Hjorth parameters, Welch band power per canonical EEG band, and
inter-channel correlation) in mlprocess.py. Because the CHB-MIT class
distribution is heavily skewed toward non-seizure windows, class imbalance
is corrected on the TRAINING split only, after a strict patient-level
train/validation/test split, using a K-Means SMOTE -> SMOTE ->
RandomOverSampler fallback ladder (KMeansSMOTE is the preferred, highest-
quality method, but can fail to find valid clusters on this small a cohort).
XGBoost, chosen for its strength on structured/tabular features, small-data
robustness, and interpretability, is trained on the balanced features with
early stopping and a validation-tuned decision threshold, then evaluated
with both window-level metrics (accuracy, sensitivity, specificity, F1,
ROC/PR-AUC) and clinically-motivated event-level metrics (event sensitivity,
event precision/F1, false alarms per hour, detection latency) produced by
grouping consecutive positive-probability windows into seizure episodes.
This XGBoost pipeline is Stage 1: a scalable, leakage-safe classical ML
baseline. Stage 2 will replace the hand-crafted feature step with an
end-to-end deep model — a 1D CNN over raw EEG windows for automatic local
feature extraction, feeding a Bi-LSTM to capture bidirectional temporal
dependencies across the window sequence, with an attention layer to weight
the most seizure-relevant time steps before the final classification head —
reusing the same preprocessing, patient-level split, and event-level
evaluation built here.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from . import runtime  # noqa: F401
    from .mlprocess import (
        EEGWindow,
        FeatureDataset,
        ResamplingReport,
        build_dl_feature_artifact,
        extract_window_features,
        load_feature_dataset,
        print_resampling_report,
        save_dl_feature_artifact,
        ML_DATASET_ARTIFACT_PATH,
        WINDOW_SECONDS,
    )
    from .preprocessing import (
        CleanEEGRecording,
        SeizureInterval,
        LOW_CUTOFF_HZ,
        HIGH_CUTOFF_HZ,
        NOTCH_FREQUENCY_HZ,
        TARGET_SAMPLING_RATE_HZ,
        WINDOW_DURATION_SECONDS,
        preprocess_raw,
        PREPROCESSED_CACHE_PATH,
        PREPROCESSED_MANIFEST_PATH,
    )
    from .report import generate_model_report
except ImportError:  # supports `python backend/mlout.py`
    import runtime  # type: ignore  # noqa: F401
    from mlprocess import (  # type: ignore
    EEGWindow,
    FeatureDataset,
    ResamplingReport,
    build_dl_feature_artifact,
    extract_window_features,
    load_feature_dataset,
    print_resampling_report,
    save_dl_feature_artifact,
    ML_DATASET_ARTIFACT_PATH,
    WINDOW_SECONDS,
    )
    from preprocessing import (  # type: ignore
    CleanEEGRecording,
    SeizureInterval,
    LOW_CUTOFF_HZ,
    HIGH_CUTOFF_HZ,
    NOTCH_FREQUENCY_HZ,
    TARGET_SAMPLING_RATE_HZ,
    WINDOW_DURATION_SECONDS,
    preprocess_raw,
    PREPROCESSED_CACHE_PATH,
    PREPROCESSED_MANIFEST_PATH,
    )
    from report import generate_model_report  # type: ignore

    # XGBoost is a strong classical baseline for nonlinear EEG feature spaces.
try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - environment-specific dependency guard
    raise ImportError(
        "xgboost is required for mlout.py. Install it with `pip install xgboost`."
    ) from exc

# FastAPI is imported lazily by build_app() so model training can run without
# needing the web server stack in memory.

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACT_PATH: str = str(ARTIFACT_DIR / "seizure_ml_model.joblib")
MODEL_METRICS_PATH: str = str(ARTIFACT_DIR / "ml_metrics.joblib")
MODEL_REPORT_PATH: str = str(ARTIFACT_DIR / "ml_model_report.html")
DL_FEATURE_ARTIFACT_PATH: str = str(ARTIFACT_DIR / "dl_feature_input.joblib")
MODEL_VERSION: str = "ml-baseline-v2"

RANDOM_STATE: int = 42

# Fallback threshold, only used if validation-based threshold tuning below is
# unavailable (e.g. the validation split has zero seizure windows). Prefer
# bundle.decision_threshold (tuned per training run) over this constant.
DEFAULT_SEIZURE_THRESHOLD: float = 0.50

# Strong, but not excessively large, XGBoost baseline for tabular EEG features.
# early_stopping_rounds is set here (not just eval_set) so training actually
# stops once validation logloss stops improving, instead of always fitting
# the full 500 trees and risking overfitting on a small patient sample.
XGB_PARAMS: Dict[str, object] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.90,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "gamma": 0.0,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cpu",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
}

# Threshold selection strategy applied to the validation set after training.
# "f1" maximizes window-level F1. If you care more about false alarms than
# missed windows (usually true for continuous EEG monitoring), consider
# scanning for the smallest threshold that keeps validation recall above a
# clinically-chosen floor instead.
THRESHOLD_SELECTION_METRIC: str = "f1"

# Temporal event-level post-processing.
EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS: int = 1
EVENT_MAX_GAP_WINDOWS: int = 1

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

LOGGER = logging.getLogger("minorproj.mlout")


# -----------------------------------------------------------------------------
# MODEL BUNDLE
# -----------------------------------------------------------------------------


@dataclass
class ModelBundle:
    model: XGBClassifier
    scaler: object
    feature_names: List[str]
    model_version: str
    sampling_rate_hz: float
    window_seconds: float
    low_cutoff_hz: float
    high_cutoff_hz: float
    notch_frequency_hz: float
    class_names: List[str]
    created_at_utc: str
    decision_threshold: float = DEFAULT_SEIZURE_THRESHOLD

    def validate(self) -> None:
        if len(self.class_names) != 2:
            raise ValueError("This project expects binary seizure classification.")
        if not self.feature_names:
            raise ValueError("Feature schema cannot be empty.")
        if self.window_seconds <= 0:
            raise ValueError("Window duration must be positive.")
        if not (0.0 < self.decision_threshold < 1.0):
            raise ValueError("decision_threshold must be strictly between 0 and 1.")


# -----------------------------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------------------------


def make_model() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


def select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = THRESHOLD_SELECTION_METRIC,
    fallback: float = DEFAULT_SEIZURE_THRESHOLD,
) -> float:
    """
    Scan candidate thresholds on a held-out split and pick the one that
    maximizes `metric` (currently only 'f1' is implemented).

    A fixed 0.50 cutoff has no particular reason to be optimal for an
    imbalanced, oversampled-at-train-time model; the operating point should
    be chosen using data the model did not train on. Falls back to
    `fallback` if the split has only one class present (metric undefined).
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if np.unique(y_true).size < 2:
        LOGGER.warning(
            "Cannot tune decision threshold: validation split has only one class. "
            "Falling back to %.2f.",
            fallback,
        )
        return fallback

    candidates = np.unique(np.clip(y_prob, 1e-6, 1 - 1e-6))
    if candidates.size == 0:
        return fallback

    best_threshold = fallback
    best_score = -1.0
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(np.int64)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold selection metric: {metric}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    LOGGER.info(
        "Selected decision threshold=%.4f on validation set (%s=%.4f).",
        best_threshold,
        metric,
        best_score,
    )
    return best_threshold


def train_model(dataset: FeatureDataset) -> ModelBundle:
    dataset.validate()
    model = make_model()

    # Validation is never used as training data. It is used for two things
    # only, both of which stop short of fitting on it: (1) early stopping,
    # so training halts once validation logloss stops improving, and
    # (2) picking the decision threshold below. The tree ensemble itself is
    # fit on X_train/y_train only.
    model.fit(
        dataset.X_train,
        dataset.y_train,
        eval_set=[(dataset.X_validation, dataset.y_validation)],
        verbose=False,
    )

    val_probabilities = model.predict_proba(dataset.X_validation)[:, 1]
    decision_threshold = select_decision_threshold(dataset.y_validation, val_probabilities)

    bundle = ModelBundle(
        model=model,
        scaler=dataset.scaler,
        feature_names=list(dataset.feature_names),
        model_version=MODEL_VERSION,
        sampling_rate_hz=float(TARGET_SAMPLING_RATE_HZ),
        window_seconds=float(WINDOW_DURATION_SECONDS),
        low_cutoff_hz=float(LOW_CUTOFF_HZ),
        high_cutoff_hz=float(HIGH_CUTOFF_HZ),
        notch_frequency_hz=float(NOTCH_FREQUENCY_HZ),
        class_names=["non_seizure", "seizure"],
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        decision_threshold=decision_threshold,
    )
    bundle.validate()
    return bundle


def save_model_bundle(
    bundle: ModelBundle,
    output_path: str | Path = MODEL_ARTIFACT_PATH,
) -> Path:
    bundle.validate()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    LOGGER.info("Saved trained model bundle to %s", output.resolve())
    return output


def load_model_bundle(
    input_path: str | Path = MODEL_ARTIFACT_PATH,
) -> ModelBundle:
    import sys
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        setattr(main_module, "ModelBundle", ModelBundle)
    bundle: ModelBundle = joblib.load(input_path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Unexpected object in model artifact.")
    bundle.validate()
    return bundle


# -----------------------------------------------------------------------------
# WINDOW-LEVEL INFERENCE
# -----------------------------------------------------------------------------


def predict_feature_matrix(
    bundle: ModelBundle,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("Feature input must be 2D.")
    if X.shape[1] != len(bundle.feature_names):
        raise ValueError(
            f"Expected {len(bundle.feature_names)} features, got {X.shape[1]}."
        )
    if not np.isfinite(X).all():
        raise ValueError("Feature input contains non-finite values.")

    probabilities = bundle.model.predict_proba(X)[:, 1].astype(np.float64)
    predictions = (probabilities >= bundle.decision_threshold).astype(np.int64)
    return predictions, probabilities


def predict_window_from_preprocessed_array(
    bundle: ModelBundle,
    eeg_window_uv: np.ndarray,
    sampling_rate_hz: float,
    channel_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    Predict one raw 20-second EEG window.

    The array may be supplied as [channels, samples]. If a caller provides a
    single-channel vector [samples], it is promoted to [1, samples].
    """

    eeg = np.asarray(eeg_window_uv, dtype=np.float32)
    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, :]
    if eeg.ndim != 2:
        raise ValueError("EEG input must have shape [channels, samples].")

    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be a positive finite number.")

    expected_input_samples = int(round(float(sampling_rate_hz) * bundle.window_seconds))
    if eeg.shape[1] != expected_input_samples:
        raise ValueError(
            f"Expected exactly {expected_input_samples} samples ({bundle.window_seconds:.0f}s at "
            f"{sampling_rate_hz:.0f}Hz), got {eeg.shape[1]}."
        )

    # Recreate the exact preprocessing used for training from the raw array.
    # Using MNE here keeps filtering behavior aligned with preprocessing.py.
    import mne

    names = list(channel_names) if channel_names is not None else [f"EEG{i+1:02d}" for i in range(eeg.shape[0])]
    if len(names) != eeg.shape[0]:
        raise ValueError("channel_names length does not match EEG channel count.")

    info = mne.create_info(
        ch_names=names,
        sfreq=float(sampling_rate_hz),
        ch_types=["eeg"] * len(names),
    )
    raw = mne.io.RawArray(eeg.astype(np.float64), info, verbose=False)
    cleaned, cleaned_names, cleaned_sr = preprocess_raw(raw)

    if not np.isclose(cleaned_sr, bundle.sampling_rate_hz):
        raise ValueError("Preprocessing did not produce the model's target sampling rate.")

    window = EEGWindow(
        patient_id="api_patient",
        recording_id="api_recording",
        window_id="api_window",
        start_seconds=0.0,
        end_seconds=bundle.window_seconds,
        label=0,
        label_name="unknown",
        data_uv=cleaned,
        sampling_rate_hz=cleaned_sr,
        channel_names=cleaned_names,
    )

    X_raw, feature_names = extract_window_features(window)
    if feature_names != bundle.feature_names:
        raise ValueError(
            "Feature schema mismatch between API preprocessing and trained model. "
            "Retrain after changing preprocessing or feature extraction."
        )

    X_scaled = bundle.scaler.transform(X_raw[np.newaxis, :]).astype(np.float32)
    predictions, probabilities = predict_feature_matrix(bundle, X_scaled)
    probability = float(probabilities[0])
    prediction = int(predictions[0])

    return {
        "prediction": bundle.class_names[prediction],
        "class_id": prediction,
        "seizure_probability": probability,
        "confidence": probability if prediction == 1 else 1.0 - probability,
        "model_version": bundle.model_version,
        "window_seconds": bundle.window_seconds,
        "sampling_rate_hz": bundle.sampling_rate_hz,
    }


# -----------------------------------------------------------------------------
# CLASSIFICATION METRICS
# -----------------------------------------------------------------------------


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["non_seizure", "seizure"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


# -----------------------------------------------------------------------------
# TEMPORAL EVENT-LEVEL DETECTION
# -----------------------------------------------------------------------------


@dataclass
class PredictedEvent:
    patient_id: str
    recording_id: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    max_probability: float
    mean_probability: float


def _group_positive_windows(
    metadata: Sequence[dict],
    probabilities: np.ndarray,
    threshold: float = DEFAULT_SEIZURE_THRESHOLD,
) -> List[PredictedEvent]:
    grouped: Dict[Tuple[str, str], List[Tuple[dict, float]]] = {}
    for meta, probability in zip(metadata, probabilities):
        key = (meta["patient_id"], meta["recording_id"])
        grouped.setdefault(key, []).append((meta, float(probability)))

    events: List[PredictedEvent] = []
    max_gap_seconds = WINDOW_SECONDS * EVENT_MAX_GAP_WINDOWS

    for (patient_id, recording_id), rows in grouped.items():
        rows = sorted(rows, key=lambda item: float(item[0]["start_seconds"]))
        # Filter for positive windows first to allow gap bridging between positive windows within max_gap_seconds
        positive_rows = [(meta, prob) for meta, prob in rows if prob >= threshold]
        active: List[Tuple[dict, float]] = []
        candidate_groups: List[List[Tuple[dict, float]]] = []

        for meta, probability in positive_rows:
            if not active:
                active = [(meta, probability)]
                continue

            previous_end = float(active[-1][0]["end_seconds"])
            current_start = float(meta["start_seconds"])
            gap = current_start - previous_end
            if gap <= max_gap_seconds + 1e-6:
                active.append((meta, probability))
            else:
                candidate_groups.append(active)
                active = [(meta, probability)]

        if active:
            candidate_groups.append(active)


        for group in candidate_groups:
            if len(group) < EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS:
                continue
            start = float(group[0][0]["start_seconds"])
            end = float(group[-1][0]["end_seconds"])
            probs = np.asarray([prob for _, prob in group], dtype=np.float64)
            events.append(
                PredictedEvent(
                    patient_id=patient_id,
                    recording_id=recording_id,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=end - start,
                    max_probability=float(np.max(probs)),
                    mean_probability=float(np.mean(probs)),
                )
            )

    return events


def _ground_truth_events_from_metadata(
    metadata: Sequence[dict],
    recordings_by_key: Dict[Tuple[str, str], Sequence[Tuple[float, float]]],
) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    result: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    keys = {(m["patient_id"], m["recording_id"]) for m in metadata}
    for key in keys:
        result[key] = list(recordings_by_key.get(key, []))
    return result


def event_level_metrics(
    metadata: Sequence[dict],
    probabilities: np.ndarray,
    ground_truth_events: Dict[Tuple[str, str], Sequence[Tuple[float, float]]],
    threshold: float = DEFAULT_SEIZURE_THRESHOLD,
) -> Dict[str, object]:
    """Calculate event sensitivity, event precision/F1, false alarms/hour and latency."""

    predicted_events = _group_positive_windows(metadata, probabilities, threshold=threshold)

    matched_true = 0
    matched_predicted = 0
    latencies: List[float] = []

    # One prediction may match one true event only; greedy earliest-overlap matching
    # is deterministic and avoids double counting.
    used_predictions: set[int] = set()
    total_true_events = sum(len(events) for events in ground_truth_events.values())

    for true_key, true_events in ground_truth_events.items():
        key_pred_indices = [
            i
            for i, event in enumerate(predicted_events)
            if (event.patient_id, event.recording_id) == true_key
        ]
        for true_start, true_end in true_events:
            candidate_index: Optional[int] = None
            candidate_onset = float("inf")

            for index in key_pred_indices:
                if index in used_predictions:
                    continue
                pred = predicted_events[index]
                overlap = max(
                    0.0,
                    min(pred.end_seconds, true_end)
                    - max(pred.start_seconds, true_start),
                )
                if overlap <= 0:
                    continue
                if pred.start_seconds < candidate_onset:
                    candidate_onset = pred.start_seconds
                    candidate_index = index

            if candidate_index is not None:
                used_predictions.add(candidate_index)
                matched_true += 1
                matched_predicted += 1
                pred = predicted_events[candidate_index]
                latencies.append(max(0.0, pred.end_seconds - true_start))

    false_positive_events = len(predicted_events) - matched_predicted

    # Estimate non-seizure observation hours from the metadata covered by TEST.
    # Each metadata row is a 20-second window, so this remains valid even when
    # multiple recordings/patients have different durations.
    total_test_seconds = 0.0
    seizure_seconds = 0.0
    seen_windows: set[Tuple[str, str, float]] = set()
    for meta in metadata:
        key = (
            meta["patient_id"],
            meta["recording_id"],
            float(meta["start_seconds"]),
        )
        if key in seen_windows:
            continue
        seen_windows.add(key)
        total_test_seconds += max(
            0.0,
            float(meta["end_seconds"]) - float(meta["start_seconds"]),
        )

    for true_events in ground_truth_events.values():
        for start, end in true_events:
            seizure_seconds += max(0.0, end - start)

    non_seizure_hours = max((total_test_seconds - seizure_seconds) / 3600.0, 1e-9)
    false_alarms_per_hour = false_positive_events / non_seizure_hours

    event_sensitivity = matched_true / max(total_true_events, 1)
    event_precision = matched_predicted / max(len(predicted_events), 1)
    event_f1 = (
        2 * event_precision * event_sensitivity / max(event_precision + event_sensitivity, 1e-12)
    )

    return {
        "event_sensitivity": float(event_sensitivity),
        "event_precision": float(event_precision),
        "event_f1": float(event_f1),
        "false_alarm_events": int(false_positive_events),
        "false_alarms_per_hour": float(false_alarms_per_hour),
        "mean_detection_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "median_detection_latency_seconds": float(np.median(latencies)) if latencies else None,
        "detected_events": int(len(predicted_events)),
        "ground_truth_events": int(total_true_events),
        "matched_events": int(matched_true),
        "predicted_event_details": [event.__dict__ for event in predicted_events],
    }


# -----------------------------------------------------------------------------
# FULL EVALUATION
# -----------------------------------------------------------------------------


def evaluate_bundle(
    bundle: ModelBundle,
    dataset: FeatureDataset,
) -> Dict[str, object]:
    y_val_pred, y_val_prob = predict_feature_matrix(bundle, dataset.X_validation)
    y_test_pred, y_test_prob = predict_feature_matrix(bundle, dataset.X_test)

    results: Dict[str, object] = {
        "model_version": bundle.model_version,
        "validation": classification_metrics(
            dataset.y_validation, y_val_pred, y_val_prob
        ),
        "test": classification_metrics(dataset.y_test, y_test_pred, y_test_prob),
        "patient_splits": {
            "train": dataset.train_patients,
            "validation": dataset.validation_patients,
            "test": dataset.test_patients,
        },
    }

    # Event metrics require ground-truth event intervals. Those are supplied by
    # reloading the original recordings rather than duplicating them per window.
    # The artifact itself keeps window-level metadata; the recording intervals are
    # reconstructed from the original preprocessing output.
    try:
        if Path(PREPROCESSED_MANIFEST_PATH).exists():
            try:
                from .preprocessing import iter_preprocessed_recordings
            except ImportError:
                from preprocessing import iter_preprocessed_recordings  # type: ignore
            recordings = iter_preprocessed_recordings(PREPROCESSED_MANIFEST_PATH)
        else:
            try:
                from .preprocessing import load_preprocessed_recordings
            except ImportError:
                from preprocessing import load_preprocessed_recordings  # type: ignore
            recordings = load_preprocessed_recordings(PREPROCESSED_CACHE_PATH)
        event_truth: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        for recording in recordings:
            event_truth[(recording.patient_id, recording.recording_id)] = [
                (interval.start_seconds, interval.end_seconds)
                for interval in recording.seizure_intervals
            ]
            del recording

        results["test_event_level"] = event_level_metrics(
            dataset.test_metadata,
            y_test_prob,
            _ground_truth_events_from_metadata(dataset.test_metadata, event_truth),
            threshold=bundle.decision_threshold,
        )
    except FileNotFoundError:
        LOGGER.warning(
            "Preprocessing cache is unavailable; skipping event-level ground-truth metrics."
        )
        results["test_event_level"] = None

    return results


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------


def build_app(bundle_path: str | Path = MODEL_ARTIFACT_PATH):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    bundle = load_model_bundle(bundle_path)
    app = FastAPI(
        title="EEG Seizure Detection API",
        version=bundle.model_version,
        description="Patient-independent 20-second EEG seizure detector with temporal event post-processing.",
    )

    class WindowRequest(BaseModel):
        eeg: List[List[float]] = Field(..., description="EEG matrix [channels][samples].")
        sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ
        channel_names: Optional[List[str]] = None

    class WindowResponse(BaseModel):
        prediction: str
        class_id: int
        seizure_probability: float
        confidence: float
        model_version: str
        window_seconds: float
        sampling_rate_hz: float

    class FeatureRequest(BaseModel):
        features: List[float] = Field(..., description="ML feature vector in model schema order.")

    class FeatureResponse(BaseModel):
        prediction: str
        class_id: int
        seizure_probability: float
        confidence: float
        model_version: str
        feature_count: int

    class SequenceWindow(BaseModel):
        start_seconds: float = Field(..., ge=0.0)
        end_seconds: float = Field(..., gt=0.0)
        eeg: List[List[float]] = Field(...)
        sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ
        channel_names: Optional[List[str]] = None

    class EventSequenceRequest(BaseModel):
        patient_id: str = "api_patient"
        recording_id: str = "api_recording"
        windows: List[SequenceWindow] = Field(..., min_length=1)

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": bundle.model_version,
        }

    @app.get("/model-info")
    def model_info() -> Dict[str, object]:
        return {
            "model_version": bundle.model_version,
            "created_at_utc": bundle.created_at_utc,
            "sampling_rate_hz": bundle.sampling_rate_hz,
            "window_seconds": bundle.window_seconds,
            "expected_samples": int(bundle.sampling_rate_hz * bundle.window_seconds),
            "classes": bundle.class_names,
            "feature_count": len(bundle.feature_names),
            "decision_threshold": bundle.decision_threshold,
        }

    @app.get("/training-report")
    def training_report() -> Dict[str, object]:
        """
        Class-balancing (SMOTE) and evaluation report saved by `train_command`.

        Exposes exactly how the small (~10-patient) training set was
        oversampled, so imbalance-driven bias in the served model is
        auditable from the API, not just from training logs.
        """
        metrics_path = Path(MODEL_METRICS_PATH)
        if not metrics_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No saved training report found. Run `python mlout.py train` first.",
            )
        results = joblib.load(metrics_path)
        return {
            "model_version": results.get("model_version"),
            "resampling_report": results.get("resampling_report"),
            "validation": results.get("validation"),
            "test": {
                k: v for k, v in results.get("test", {}).items()
                if k not in {"classification_report"}
            },
            "test_event_level": results.get("test_event_level"),
        }

    @app.post("/predict-features", response_model=FeatureResponse)
    def predict_features(request: FeatureRequest) -> Dict[str, object]:
        """Predict from the exact feature vector handed to the future DL stage."""

        try:
            raw_features = np.asarray(request.features, dtype=np.float32)
            if raw_features.ndim != 1 or raw_features.size != len(bundle.feature_names):
                raise ValueError(f"Expected exactly {len(bundle.feature_names)} features.")
            if not np.isfinite(raw_features).all():
                raise ValueError("Feature input contains non-finite values.")
            scaled = bundle.scaler.transform(raw_features[np.newaxis, :]).astype(np.float32)
            predictions, probabilities = predict_feature_matrix(bundle, scaled)
            class_id = int(predictions[0])
            probability = float(probabilities[0])
            return {
                "prediction": bundle.class_names[class_id],
                "class_id": class_id,
                "seizure_probability": probability,
                "confidence": probability if class_id else 1.0 - probability,
                "model_version": bundle.model_version,
                "feature_count": len(bundle.feature_names),
            }
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict-window", response_model=WindowResponse)
    def predict_window(request: WindowRequest) -> Dict[str, object]:
        try:
            return predict_window_from_preprocessed_array(
                bundle=bundle,
                eeg_window_uv=np.asarray(request.eeg, dtype=np.float32),
                sampling_rate_hz=request.sampling_rate_hz,
                channel_names=request.channel_names,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict-event-sequence")
    def predict_event_sequence(request: EventSequenceRequest) -> Dict[str, object]:
        if not request.windows:
            raise HTTPException(status_code=422, detail="At least one EEG window is required.")

        metadata: List[dict] = []
        probabilities: List[float] = []

        try:
            windows_sorted = sorted(request.windows, key=lambda w: w.start_seconds)
            for index, item in enumerate(windows_sorted):
                result = predict_window_from_preprocessed_array(
                    bundle=bundle,
                    eeg_window_uv=np.asarray(item.eeg, dtype=np.float32),
                    sampling_rate_hz=item.sampling_rate_hz,
                    channel_names=item.channel_names,
                )
                metadata.append(
                    {
                        "patient_id": request.patient_id,
                        "recording_id": request.recording_id,
                        "window_id": f"api_window_{index:06d}",
                        "start_seconds": item.start_seconds,
                        "end_seconds": item.end_seconds,
                        "label": 0,
                        "label_name": "unknown",
                    }
                )
                probabilities.append(float(result["seizure_probability"]))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        events = _group_positive_windows(
            metadata,
            np.asarray(probabilities, dtype=np.float64),
            threshold=bundle.decision_threshold,
        )
        return {
            "model_version": bundle.model_version,
            "decision_threshold": bundle.decision_threshold,
            "window_predictions": [
                {
                    "start_seconds": meta["start_seconds"],
                    "end_seconds": meta["end_seconds"],
                    "seizure_probability": probability,
                    "prediction": "seizure" if probability >= bundle.decision_threshold else "non_seizure",
                }
                for meta, probability in zip(metadata, probabilities)
            ],
            "detected_events": [event.__dict__ for event in events],
        }

    return app


# -----------------------------------------------------------------------------
# COMMAND LINE
# -----------------------------------------------------------------------------


def train_command() -> None:
    dataset = load_feature_dataset(ML_DATASET_ARTIFACT_PATH)

    # Class-balance transparency: show exactly how the (small, ~10-patient)
    # training set was oversampled BEFORE showing any model performance, so
    # imbalance-driven bias is visible up front rather than inferred later.
    print_resampling_report(dataset.resampling_report)

    bundle = train_model(dataset)
    save_model_bundle(bundle, MODEL_ARTIFACT_PATH)

    results = evaluate_bundle(bundle, dataset)
    if dataset.resampling_report is not None:
        results["resampling_report"] = asdict(dataset.resampling_report)
    metrics_path = Path(MODEL_METRICS_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, metrics_path, compress=3)
    dl_artifact = build_dl_feature_artifact(dataset)
    save_dl_feature_artifact(dl_artifact, DL_FEATURE_ARTIFACT_PATH)
    generate_model_report(dataset, results, bundle, MODEL_REPORT_PATH)

    print(f"\nSelected decision threshold (tuned on validation set): {bundle.decision_threshold:.4f}")
    print("\n=== TEST CLASSIFICATION METRICS ===")
    for key, value in results["test"].items():
        if key not in {"classification_report", "confusion_matrix"}:
            print(f"{key}: {value}")
    print("confusion_matrix:", results["test"]["confusion_matrix"])

    print("\n=== TEST EVENT-LEVEL METRICS ===")
    event_metrics = results.get("test_event_level")
    if event_metrics is None:
        print("Event metrics unavailable because preprocessing annotations were not cached.")
    else:
        for key in [
            "event_sensitivity",
            "event_precision",
            "event_f1",
            "false_alarms_per_hour",
            "mean_detection_latency_seconds",
            "median_detection_latency_seconds",
        ]:
            print(f"{key}: {event_metrics.get(key)}")


def serve_command(host: str, port: int) -> None:
    import uvicorn

    app = build_app(MODEL_ARTIFACT_PATH)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate/serve EEG seizure ML model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the model.")
    train_parser.set_defaults(func=lambda args: train_command())

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server.")
    serve_parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    serve_parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    serve_parser.set_defaults(func=lambda args: serve_command(args.host, args.port))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()