"""Patient-level OOF threshold and alarm-policy calibration for CNN-BiLSTM.

Run from the repository root:
    uv run python -m backend.dl.threshold_calibration --mode calibrate
    uv run python -m backend.dl.threshold_calibration --mode evaluate-test

Calibration uses only development-patient out-of-fold predictions. The
independent test set is opened only by the separate evaluate-test command.
No model weights are trained or changed by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from backend.preprocessing import *

from . import train as training
from .config import CNN_BILSTM_ARTIFACTS_DIR, DL_ARTIFACTS_DIR, PREPROCESSED_MANIFEST_PATH
from .event_evaluation import evaluate_events
from .model import CNNBiLSTM
from .raw_data import (
    NormalizationCache,
    RawWindow,
    load_ground_truth_events,
    load_manifest_windows,
    load_recording_durations,
)

CALIBRATION_DIR = CNN_BILSTM_ARTIFACTS_DIR / "threshold_calibration"
CALIBRATION_PATH = CALIBRATION_DIR / "threshold_calibration.json"
OOF_PREDICTIONS_PATH = CALIBRATION_DIR / "oof_predictions.npz"
CALIBRATED_TEST_PATH = CALIBRATION_DIR / "calibrated_test_results.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=float), encoding="utf-8")


def _window_metadata(windows: Sequence[RawWindow]) -> list[dict[str, Any]]:
    return [
        {
            "patient_id": str(w.patient_id),
            "recording_id": str(w.recording_id),
            "start_seconds": float(w.start_seconds),
            "end_seconds": float(w.end_seconds),
            "label": int(w.label),
        }
        for w in windows
    ]


def _compact_event_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "actual_seizure_events",
        "detected_events",
        "predicted_alarm_episodes",
        "matched_episodes",
        "false_alarm_episodes",
        "event_sensitivity",
        "event_precision",
        "event_f1",
        "false_alarms_per_hour",
        "mean_detection_latency_seconds",
        "median_detection_latency_seconds",
        "observation_hours",
        "non_seizure_hours",
    )
    return {key: metrics.get(key) for key in keys}


def _evaluate_config(
    metadata: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
    recording_durations: Mapping[tuple[str, str], float],
    *,
    threshold: float,
    min_positive_windows: int,
    max_gap_windows: int,
) -> dict[str, Any]:
    """Evaluate pooled and equal-patient-weighted event performance."""
    aggregate = evaluate_events(
        metadata,
        probabilities,
        ground_truth_events,
        threshold=threshold,
        min_consecutive_windows=min_positive_windows,
        max_gap_windows=max_gap_windows,
        recording_durations=recording_durations,
    )

    by_patient_meta: dict[str, list[Mapping[str, Any]]] = {}
    by_patient_probs: dict[str, list[float]] = {}
    for meta, probability in zip(metadata, probabilities):
        patient_id = str(meta["patient_id"])
        by_patient_meta.setdefault(patient_id, []).append(meta)
        by_patient_probs.setdefault(patient_id, []).append(float(probability))

    patient_metrics: dict[str, dict[str, Any]] = {}
    for patient_id, patient_metadata in sorted(by_patient_meta.items()):
        patient_gt = {
            key: events
            for key, events in ground_truth_events.items()
            if str(key[0]) == patient_id
        }
        patient_durations = {
            key: duration
            for key, duration in recording_durations.items()
            if str(key[0]) == patient_id
        }
        patient_metrics[patient_id] = evaluate_events(
            patient_metadata,
            np.asarray(by_patient_probs[patient_id], dtype=np.float64),
            patient_gt,
            threshold=threshold,
            min_consecutive_windows=min_positive_windows,
            max_gap_windows=max_gap_windows,
            recording_durations=patient_durations,
        )

    seizure_patients = [
        item for item in patient_metrics.values()
        if int(item.get("actual_seizure_events", 0)) > 0
    ]
    # Equal patient weighting prevents a seizure-heavy patient from dominating
    # the operating point. Patients with no events still count for false alarms.
    macro_sensitivity = float(np.mean([x["event_sensitivity"] for x in seizure_patients])) if seizure_patients else 0.0
    macro_precision = float(np.mean([x["event_precision"] for x in seizure_patients])) if seizure_patients else 0.0
    macro_f1 = float(np.mean([x["event_f1"] for x in seizure_patients])) if seizure_patients else 0.0
    max_patient_fa = max(
        (float(x["false_alarms_per_hour"]) for x in patient_metrics.values()),
        default=0.0,
    )

    return {
        "aggregate": _compact_event_metrics(aggregate),
        "macro_event_sensitivity": macro_sensitivity,
        "macro_event_precision": macro_precision,
        "macro_event_f1": macro_f1,
        "max_patient_false_alarms_per_hour": float(max_patient_fa),
        "patients_with_seizures": len(seizure_patients),
        "patients_evaluated": len(patient_metrics),
        "per_patient": {
            patient_id: _compact_event_metrics(metrics)
            for patient_id, metrics in patient_metrics.items()
        },
    }


def _build_threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if not (0.0 < minimum < maximum < 1.0):
        raise ValueError("threshold bounds must satisfy 0 < min < max < 1")
    if step <= 0:
        raise ValueError("threshold step must be positive")
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    values = [round(minimum + i * step, 6) for i in range(count + 1)]
    if values[-1] < maximum - 1e-6:
        values.append(round(maximum, 6))
    return sorted(set(values))


def _load_inputs(manifest_path: Path):
    split_path = training.SPLIT_PATH
    if not split_path.is_file():
        raise FileNotFoundError(f"Frozen patient split not found: {split_path}; run audit first")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    canonical_channels = split["canonical_channels"]
    all_windows = load_manifest_windows(
        manifest_path,
        canonical_channels=canonical_channels,
        quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
    )
    by_patient: dict[str, list[RawWindow]] = {}
    for window in all_windows:
        by_patient.setdefault(window.patient_id, []).append(window)
    gt_events = load_ground_truth_events(
        manifest_path,
        quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
    )
    recording_durations = load_recording_durations(
        manifest_path,
        quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
    )
    return split, canonical_channels, by_patient, gt_events, recording_durations


def _resolve_fold_checkpoint(fold_result: Mapping[str, Any], fold_number: int) -> Path:
    candidate = Path(str(fold_result.get("checkpoint", "")))
    if candidate.is_file():
        return candidate
    fallback = training.CHECKPOINT_DIR / f"fold_{fold_number}.pt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"Fold {fold_number} checkpoint missing. Tried {candidate} and {fallback}. "
        "Calibration needs the saved CV fold checkpoints; it does not retrain them."
    )


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    split, canonical_channels, by_patient, gt_events, recording_durations = _load_inputs(args.manifest)
    cv_path = training.CNN_BILSTM_CV_RESULTS_PATH
    if not cv_path.is_file():
        raise FileNotFoundError(f"CV results not found: {cv_path}; run the existing CV stage first")
    cv_result = json.loads(cv_path.read_text(encoding="utf-8"))
    folds = cv_result.get("folds", [])
    split_folds = split.get("folds", [])
    if cv_result.get("mode") != "cv" or len(folds) != len(split_folds) or not folds:
        raise ValueError("CV results and frozen split do not contain matching completed folds")

    dev_patients = set(map(str, split["dev_patients"]))
    test_patients = set(map(str, split["test_patients"]))
    if dev_patients & test_patients:
        raise ValueError("Patient leakage: dev_patients overlaps test_patients")

    device = training.select_device(args.device)
    all_oof_metadata: list[dict[str, Any]] = []
    all_oof_labels: list[np.ndarray] = []
    all_oof_probabilities: list[np.ndarray] = []
    validation_patient_appearances: dict[str, int] = {}
    normalization_cache: NormalizationCache = {}
    fold_audit: list[dict[str, Any]] = []

    for index, (fold, fold_result) in enumerate(zip(split_folds, folds), start=1):
        fold_number = int(fold.get("fold", index))
        train_patients = set(map(str, fold["training_patients"]))
        validation_patients = set(map(str, fold["validation_patients"]))
        if train_patients & validation_patients:
            raise ValueError(f"Fold {fold_number}: training/validation patient overlap")
        if (train_patients | validation_patients) - dev_patients:
            raise ValueError(f"Fold {fold_number}: fold contains a non-development patient")
        if validation_patients & test_patients:
            raise ValueError(f"Fold {fold_number}: test patient appears in validation")
        for patient_id in validation_patients:
            validation_patient_appearances[patient_id] = validation_patient_appearances.get(patient_id, 0) + 1

        checkpoint_path = _resolve_fold_checkpoint(fold_result, fold_number)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected_epoch = int(fold_result.get("best_epoch", state.get("best_epoch", 0)))
        actual_epoch = int(state.get("best_epoch", expected_epoch))
        if expected_epoch and actual_epoch != expected_epoch:
            raise ValueError(
                f"Fold {fold_number}: checkpoint best_epoch={actual_epoch}, "
                f"but cv_results.json says {expected_epoch}; possible stale checkpoint"
            )
        saved_channels = state.get("canonical_channels")
        if saved_channels is not None and list(saved_channels) != list(canonical_channels):
            raise ValueError(f"Fold {fold_number}: checkpoint channel schema differs from frozen split")
        if "normalization_mean" not in state or "normalization_std" not in state:
            raise ValueError(f"Fold {fold_number}: checkpoint lacks train-fitted normalization")

        validation_windows = [w for patient_id in sorted(validation_patients) for w in by_patient[patient_id]]
        if not validation_windows:
            raise ValueError(f"Fold {fold_number}: validation set is empty")
        loader = training._loader(
            validation_windows,
            np.asarray(state["normalization_mean"], dtype=np.float32),
            np.asarray(state["normalization_std"], dtype=np.float32),
            args.batch_size,
            False,
        )
        model = CNNBiLSTM(len(canonical_channels)).to(device)
        model.load_state_dict(state["model_state"])
        labels, probabilities = training._predict(model, loader, device)
        metadata = _window_metadata(loader.dataset.windows)
        if len(metadata) != len(labels) or not np.array_equal(
            np.asarray([m["label"] for m in metadata], dtype=np.int64), labels
        ):
            raise RuntimeError(f"Fold {fold_number}: prediction/metadata order mismatch")
        if any(str(m["patient_id"]) not in validation_patients for m in metadata):
            raise RuntimeError(f"Fold {fold_number}: a non-validation patient appears in OOF predictions")

        all_oof_metadata.extend(metadata)
        all_oof_labels.append(labels)
        all_oof_probabilities.append(probabilities.astype(np.float64))
        fold_audit.append({
            "fold": fold_number,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "best_epoch": actual_epoch,
            "training_patients": sorted(train_patients),
            "validation_patients": sorted(validation_patients),
            "validation_windows": int(len(labels)),
            "positive_windows": int(labels.sum()),
        })
        del model, loader, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if set(validation_patient_appearances) != dev_patients:
        missing = sorted(dev_patients - set(validation_patient_appearances))
        extra = sorted(set(validation_patient_appearances) - dev_patients)
        raise ValueError(f"OOF coverage mismatch; missing={missing}, extra={extra}")
    repeated = sorted(p for p, count in validation_patient_appearances.items() if count != 1)
    if repeated:
        raise ValueError(f"Each development patient must validate exactly once; invalid={repeated}")

    metadata = all_oof_metadata
    labels = np.concatenate(all_oof_labels)
    probabilities = np.concatenate(all_oof_probabilities)
    observed_patients = {str(m["patient_id"]) for m in metadata}
    if observed_patients != dev_patients:
        raise ValueError("OOF predictions do not cover exactly the frozen development patients")
    if any(str(m["patient_id"]) in test_patients for m in metadata):
        raise ValueError("Independent test patient leaked into OOF predictions")

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    split_hash = _sha256(training.SPLIT_PATH)
    manifest_hash = _sha256(args.manifest)
    np.savez_compressed(
        OOF_PREDICTIONS_PATH,
        labels=labels,
        probabilities=probabilities,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
    )

    thresholds = _build_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    ratio = args.robust_delta / args.threshold_step
    if args.robust_delta > 0 and not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("robust-delta must be an integer multiple of threshold-step")
    neighbor_steps = int(round(ratio))
    if neighbor_steps * 2 >= len(thresholds):
        raise ValueError("robust-delta is too large for the selected threshold grid")

    min_positive_options = sorted(set(args.min_positive_windows))
    gap_options = sorted(set(args.max_gap_windows))
    if not min_positive_options or any(x < 1 for x in min_positive_options):
        raise ValueError("min-positive-windows values must be >= 1")
    if not gap_options or any(x < 0 for x in gap_options):
        raise ValueError("max-gap-windows values must be >= 0")

    # Cache all parameter-grid evaluations. Robustness is assessed by checking
    # the selected threshold and its +/- robust_delta neighbours with the same
    # alarm-merging policy, rather than selecting a brittle single-point peak.
    cache: dict[tuple[float, int, int], dict[str, Any]] = {}
    for min_positive in min_positive_options:
        for gap in gap_options:
            for threshold in thresholds:
                key = (threshold, min_positive, gap)
                cache[key] = _evaluate_config(
                    metadata,
                    probabilities,
                    gt_events,
                    recording_durations,
                    threshold=threshold,
                    min_positive_windows=min_positive,
                    max_gap_windows=gap,
                )

    threshold_to_index = {value: index for index, value in enumerate(thresholds)}
    candidates: list[dict[str, Any]] = []
    for min_positive in min_positive_options:
        for gap in gap_options:
            for threshold in thresholds:
                index = threshold_to_index[threshold]
                neighbor_indices = [index - neighbor_steps, index, index + neighbor_steps]
                if min(neighbor_indices) < 0 or max(neighbor_indices) >= len(thresholds):
                    continue
                neighbours = [cache[(thresholds[i], min_positive, gap)] for i in neighbor_indices]
                centre = cache[(threshold, min_positive, gap)]
                robust_sensitivity = min(float(x["macro_event_sensitivity"]) for x in neighbours)
                robust_f1 = min(float(x["macro_event_f1"]) for x in neighbours)
                robust_precision = min(float(x["macro_event_precision"]) for x in neighbours)
                robust_fa = max(float(x["aggregate"]["false_alarms_per_hour"]) for x in neighbours)
                robust_max_patient_fa = max(float(x["max_patient_false_alarms_per_hour"]) for x in neighbours)
                feasible = (
                    robust_fa <= args.max_fa_per_hour
                    and robust_max_patient_fa <= args.max_patient_fa_per_hour
                )
                violation = (
                    max(0.0, robust_fa - args.max_fa_per_hour) / max(args.max_fa_per_hour, 1e-9)
                    + max(0.0, robust_max_patient_fa - args.max_patient_fa_per_hour)
                    / max(args.max_patient_fa_per_hour, 1e-9)
                )
                latency = centre["aggregate"].get("median_detection_latency_seconds")
                candidates.append({
                    "threshold": float(threshold),
                    "min_positive_windows_per_episode": int(min_positive),
                    "max_gap_windows": int(gap),
                    "constraints_satisfied": bool(feasible),
                    "constraint_violation_score": float(violation),
                    "robust_macro_event_sensitivity": float(robust_sensitivity),
                    "robust_macro_event_f1": float(robust_f1),
                    "robust_macro_event_precision": float(robust_precision),
                    "robust_false_alarms_per_hour": float(robust_fa),
                    "robust_max_patient_false_alarms_per_hour": float(robust_max_patient_fa),
                    "centre_metrics": centre,
                })

    if not candidates:
        raise RuntimeError("Threshold search produced no candidates; widen threshold bounds")

    feasible_candidates = [x for x in candidates if x["constraints_satisfied"]]
    pool = feasible_candidates if feasible_candidates else candidates
    # Under the false-alarm budget, prioritize patient-macro sensitivity. If no
    # candidate meets the configured budget, explicitly return a marked fallback
    # with the smallest constraint violation instead of pretending it is feasible.
    if feasible_candidates:
        pool.sort(key=lambda x: (
            x["robust_macro_event_sensitivity"],
            x["robust_macro_event_f1"],
            x["robust_macro_event_precision"],
            -x["robust_false_alarms_per_hour"],
            -(1e9 if x["centre_metrics"]["aggregate"].get("median_detection_latency_seconds") is None else float(x["centre_metrics"]["aggregate"]["median_detection_latency_seconds"])),
            -abs(x["threshold"] - 0.5),
        ), reverse=True)
    else:
        pool.sort(key=lambda x: (
            -x["constraint_violation_score"],
            x["robust_macro_event_sensitivity"],
            x["robust_macro_event_f1"],
            x["robust_macro_event_precision"],
            -x["robust_false_alarms_per_hour"],
        ), reverse=True)
    selected = pool[0]

    calibration = {
        "schema_version": 1,
        "model": "CNNBiLSTM",
        "method": "patient-level out-of-fold probability and alarm-policy calibration",
        "independent_test_used_for_calibration": False,
        "calibration_limitations": [
            "OOF fold models train on each fold's training patients, while the final model is trained on all development patients; their probability scales may differ.",
            "The existing fold checkpoints do not embed the training-manifest hash. The script verifies fold split membership, checkpoint best_epoch, channel schema, and records checkpoint hashes, but cannot prove the historical manifest used to train each checkpoint.",
        ],
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "split_manifest_path": str(training.SPLIT_PATH),
        "split_manifest_sha256": split_hash,
        "dev_patients": sorted(dev_patients),
        "test_patients": sorted(test_patients),
        "oof_windows": int(len(labels)),
        "oof_positive_windows": int(labels.sum()),
        "oof_event_patients": int(sum(1 for patient_id in dev_patients if any(k[0] == patient_id and len(v) for k, v in gt_events.items()))),
        "fold_audit": fold_audit,
        "search": {
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
            "threshold_candidates": len(thresholds),
            "min_positive_windows_options": min_positive_options,
            "max_gap_windows_options": gap_options,
            "robust_delta": args.robust_delta,
            "robust_neighbor_rule": "worst case at threshold-delta, threshold, threshold+delta",
            "max_false_alarms_per_hour": args.max_fa_per_hour,
            "max_patient_false_alarms_per_hour": args.max_patient_fa_per_hour,
            "selection_policy": "maximize robust equal-patient event sensitivity subject to pooled and per-patient false-alarm constraints; tie-break by robust event F1, precision, false alarms, and latency",
            "candidate_count_evaluated": len(cache),
        },
        "selected": selected,
        "top_candidates": pool[:15],
        "oof_predictions_path": str(OOF_PREDICTIONS_PATH),
    }
    _json_dump(CALIBRATION_PATH, calibration)
    return calibration


def evaluate_independent_test(args: argparse.Namespace) -> dict[str, Any]:
    if not CALIBRATION_PATH.is_file():
        raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_PATH}; run --mode calibrate first")
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    split, canonical_channels, by_patient, gt_events, recording_durations = _load_inputs(args.manifest)
    if calibration.get("manifest_sha256") != _sha256(args.manifest):
        raise ValueError("Manifest hash differs from calibration; recalibrate using current manifest")
    if calibration.get("split_manifest_sha256") != _sha256(training.SPLIT_PATH):
        raise ValueError("Frozen split hash differs from calibration; recalibrate using current split")
    if calibration.get("independent_test_used_for_calibration") is not False:
        raise ValueError("Calibration artifact does not certify development-only calibration")

    selected = calibration["selected"]
    threshold = float(selected["threshold"])
    min_positive = int(selected["min_positive_windows_per_episode"])
    max_gap = int(selected["max_gap_windows"])
    test_patients = set(map(str, split["test_patients"]))
    dev_patients = set(map(str, split["dev_patients"]))
    if test_patients & dev_patients:
        raise ValueError("Frozen split has overlapping dev and test patients")

    final_results_path = training.CNN_BILSTM_FINAL_RESULTS_PATH
    if not final_results_path.is_file():
        raise FileNotFoundError(f"Final training results not found: {final_results_path}")
    final_result = json.loads(final_results_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(final_result["independent_test"]["checkpoint"])
    if not checkpoint_path.is_file():
        fallback = training.CHECKPOINT_DIR / "final.pt"
        if fallback.is_file():
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Final model checkpoint not found: {checkpoint_path}")

    device = training.select_device(args.device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_channels = state.get("canonical_channels")
    if saved_channels is not None and list(saved_channels) != list(canonical_channels):
        raise ValueError("Final checkpoint channel schema differs from the frozen split")
    if "normalization_mean" not in state or "normalization_std" not in state:
        raise ValueError("Final checkpoint lacks train-fitted normalization parameters")
    test_windows = [w for patient_id in sorted(test_patients) for w in by_patient[patient_id]]
    if not test_windows:
        raise ValueError("Independent test set is empty")
    if any(w.patient_id in dev_patients for w in test_windows):
        raise ValueError("Development patient leaked into independent test set")

    loader = training._loader(
        test_windows,
        np.asarray(state["normalization_mean"], dtype=np.float32),
        np.asarray(state["normalization_std"], dtype=np.float32),
        args.batch_size,
        False,
    )
    model = CNNBiLSTM(len(canonical_channels)).to(device)
    model.load_state_dict(state["model_state"])
    labels, probabilities = training._predict(model, loader, device)
    metadata = _window_metadata(loader.dataset.windows)
    if not set(m["patient_id"] for m in metadata).issubset(test_patients):
        raise RuntimeError("Non-test patient appears in independent test predictions")

    # Apply both the frozen calibrated policy and a 0.5 reference using the same
    # alarm grouping parameters, so the comparison isolates the threshold change.
    calibrated = _evaluate_config(
        metadata, probabilities, gt_events, recording_durations,
        threshold=threshold,
        min_positive_windows=min_positive,
        max_gap_windows=max_gap,
    )
    reference = _evaluate_config(
        metadata, probabilities, gt_events, recording_durations,
        threshold=0.5,
        min_positive_windows=min_positive,
        max_gap_windows=max_gap,
    )
    from .event_evaluation import evaluate_window_metrics
    result = {
        "schema_version": 1,
        "model": "CNNBiLSTM",
        "evaluation": "independent patient-level test; threshold and alarm policy frozen from development OOF predictions",
        "calibration_path": str(CALIBRATION_PATH),
        "calibration_manifest_sha256": calibration["manifest_sha256"],
        "final_checkpoint": str(checkpoint_path),
        "final_checkpoint_sha256": _sha256(checkpoint_path),
        "test_patients": sorted(test_patients),
        "test_windows": int(len(labels)),
        "test_positive_windows": int(labels.sum()),
        "selected_policy": {
            "threshold": threshold,
            "min_positive_windows_per_episode": min_positive,
            "max_gap_windows": max_gap,
            "constraints_satisfied_on_development_oof": bool(selected["constraints_satisfied"]),
            "development_robust_false_alarms_per_hour": selected["robust_false_alarms_per_hour"],
            "development_robust_macro_event_sensitivity": selected["robust_macro_event_sensitivity"],
        },
        "calibrated_test": {
            "window_metrics": evaluate_window_metrics(labels, probabilities, threshold=threshold),
            "event_metrics": calibrated,
        },
        "reference_threshold_0_5_same_postprocessing": {
            "window_metrics": evaluate_window_metrics(labels, probabilities, threshold=0.5),
            "event_metrics": reference,
        },
        "warning": "Do not alter the selected policy based on these test results. Any further tuning requires a new independent test cohort.",
    }
    _json_dump(CALIBRATED_TEST_PATH, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate a development-only seizure alarm threshold")
    parser.add_argument("--mode", choices=("calibrate", "evaluate-test"), required=True)
    parser.add_argument("--manifest", type=Path, default=PREPROCESSED_MANIFEST_PATH)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold-min", type=float, default=0.02)
    parser.add_argument("--threshold-max", type=float, default=0.98)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--min-positive-windows", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-gap-windows", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--robust-delta", type=float, default=0.02)
    parser.add_argument("--max-fa-per-hour", type=float, default=0.5,
                        help="Development OOF pooled false-alarm budget; tune to your evaluation objective")
    parser.add_argument("--max-patient-fa-per-hour", type=float, default=2.0,
                        help="Worst single-patient OOF false-alarm budget")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_fa_per_hour <= 0 or args.max_patient_fa_per_hour <= 0:
        parser.error("false-alarm budgets must be positive")
    if args.robust_delta < 0:
        parser.error("--robust-delta must be >= 0")
    if args.mode == "calibrate":
        result = calibrate(args)
    else:
        result = evaluate_independent_test(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))
    print(f"\nSaved: {CALIBRATION_PATH if args.mode == 'calibrate' else CALIBRATED_TEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
