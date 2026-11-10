"""CLI for the PRD Version 1 raw-signal CNN + Bi-LSTM baseline."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

# These names are intentionally re-exported. Existing preprocessing shards were
# serialized while the preprocessing entry point was running as __main__, and
# joblib resolves those legacy class names through the active DL entry point.
from backend.preprocessing import CleanEEGRecording, SeizureInterval

from .audit import audit_and_freeze
from .config import (
    CNN_BILSTM_ARTIFACTS_DIR,
    DEFAULT_DEVICE,
    DL_ARTIFACTS_DIR,
    PREPROCESSED_MANIFEST_PATH,
    SEED,
    USE_CUDA_AMP,
)
from .event_evaluation import evaluate_events, per_patient_and_aggregate_event_evaluation
from .model import CNNBiLSTM
from .raw_data import (
    NormalizationCache,
    RawEEGDataset,
    RawWindow,
    fit_channel_normalization,
    load_ground_truth_events,
    load_manifest_windows,
    load_recording_durations,
    order_windows_for_shard_io,
)

# SPLIT_PATH = DL_ARTIFACTS_DIR / "split_manifest.json"
# RESULTS_PATH = DL_ARTIFACTS_DIR / "training_results.json"
# CHECKPOINT_DIR = DL_ARTIFACTS_DIR / "checkpoints"
SPLIT_PATH = DL_ARTIFACTS_DIR / "split_manifest.json"
RESULTS_PATH = DL_ARTIFACTS_DIR / "training_results.json"
CHECKPOINT_DIR = DL_ARTIFACTS_DIR / "checkpoints"
CNN_BILSTM_CV_RESULTS_PATH = CNN_BILSTM_ARTIFACTS_DIR / "cv_results.json"
CNN_BILSTM_FINAL_RESULTS_PATH = CNN_BILSTM_ARTIFACTS_DIR / "final_results.json"
CNN_BILSTM_CHECKPOINT_DIR = CNN_BILSTM_ARTIFACTS_DIR / "checkpoints"



def select_device(requested: str) -> torch.device:
    if requested not in {"cuda", "cpu", "auto"}:
        raise ValueError(f"unsupported device {requested!r}")
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device("cpu")


def _amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda" and USE_CUDA_AMP


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    result: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "positive_windows": int(labels.sum()),
        "windows": int(labels.size),
        "confusion_matrix": {
            "tn": int(((labels == 0) & (predictions == 0)).sum()),
            "fp": int(((labels == 0) & (predictions == 1)).sum()),
            "fn": int(((labels == 1) & (predictions == 0)).sum()),
            "tp": int(((labels == 1) & (predictions == 1)).sum()),
        },
    }
    if np.unique(labels).size == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["average_precision"] = float(average_precision_score(labels, probabilities))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def _loader(windows: Sequence[RawWindow], mean: np.ndarray, std: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ordered_windows = order_windows_for_shard_io(
        windows,
        shuffle_shards=shuffle,
        seed=SEED,
    )
    dataset = RawEEGDataset(ordered_windows, mean, std)
    if not len(dataset):
        raise ValueError("window construction produced no windows")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _predict(model: CNNBiLSTM, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for signal, target in loader:
            signal = signal.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=_amp_enabled(device)):
                output = model(signal)
            probabilities.append(torch.sigmoid(output).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(labels).astype(np.int64), np.concatenate(probabilities)


# def _fit_fold(
#     train_windows: Sequence[RawWindow],
#     validation_windows: Sequence[RawWindow],
#     *,
#     canonical_channels: Sequence[str],
#     epochs: int,
#     batch_size: int,
#     device: torch.device,
#     checkpoint: Path,
#     use_amp: bool,
#     normalization_cache: NormalizationCache,
# ) -> dict[str, Any]:
def _fit_fold(
    train_windows: Sequence[RawWindow],
    validation_windows: Sequence[RawWindow],
    *,
    canonical_channels: Sequence[str],
    epochs: int,
    batch_size: int,
    device: torch.device,
    checkpoint: Path,
    use_amp: bool,
    normalization_cache: NormalizationCache,
    resume: bool = True,
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]] | None = None,
    recording_durations: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    mean, std = fit_channel_normalization(train_windows, cache=normalization_cache)
    train_loader = _loader(train_windows, mean, std, batch_size, True)
    validation_loader = _loader(validation_windows, mean, std, batch_size, False)
    model = CNNBiLSTM(len(canonical_channels)).to(device)
    positive = max(sum(window.label for window in train_windows), 1)
    negative = max(len(train_windows) - positive, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1

    latest_checkpoint = checkpoint.parent / f"{checkpoint.stem}_latest.pt"

    if resume and latest_checkpoint.is_file():
        saved_latest = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved_latest["model_state"])
        optimizer.load_state_dict(saved_latest["optimizer_state"])
        if "scheduler_state" in saved_latest and saved_latest["scheduler_state"] is not None:
            scheduler.load_state_dict(saved_latest["scheduler_state"])
        if "scaler_state" in saved_latest and saved_latest["scaler_state"] is not None:
            scaler.load_state_dict(saved_latest["scaler_state"])
        if "rng_state" in saved_latest and saved_latest["rng_state"] is not None:
            torch.set_rng_state(saved_latest["rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state" in saved_latest:
                torch.cuda.set_rng_state(saved_latest["cuda_rng_state"].cpu())
        history = list(saved_latest.get("history", []))
        best_loss = float(saved_latest.get("best_loss", float("inf")))
        best_epoch = int(saved_latest.get("best_epoch", 0))
        stale = int(saved_latest.get("stale", 0))
        start_epoch = int(saved_latest.get("epoch", 0)) + 1
    elif resume and checkpoint.is_file():
        saved_best = torch.load(checkpoint, map_location=device, weights_only=False)
        best_loss = float(saved_best.get("best_loss", float("inf")))
        best_epoch = int(saved_best.get("best_epoch", 0))
        history = list(saved_best.get("history", []))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for signal, target in train_loader:
            signal = signal.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(signal), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(target)
        validation_labels, validation_probabilities = _predict(model, validation_loader, device)
        validation_loss = float(
            nn.functional.binary_cross_entropy(
                torch.from_numpy(np.clip(validation_probabilities, 1e-6, 1 - 1e-6)).float(),
                torch.from_numpy(validation_labels.astype(np.float32)),
            )
        )
        scheduler.step(validation_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader.dataset),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "canonical_channels": list(canonical_channels),
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "positive_class_weight": negative / positive,
                    "best_epoch": epoch,
                    "best_loss": best_loss,
                    "history": history,
                },
                checkpoint,
            )
        else:
            stale += 1

        # Always save latest training state for resume functionality
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "canonical_channels": list(canonical_channels),
                "normalization_mean": mean,
                "normalization_std": std,
                "positive_class_weight": negative / positive,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "stale": stale,
                "history": history,
            },
            latest_checkpoint,
        )

        if stale >= 10:
            break

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    labels, probabilities = _predict(model, validation_loader, device)

    # Compute both window-level and comprehensive event-level evaluation
    val_metadata = [
        {
            "patient_id": w.patient_id,
            "recording_id": w.recording_id,
            "start_seconds": w.start_seconds,
            "end_seconds": w.end_seconds,
            "label": w.label,
        }
        for w in validation_loader.dataset.windows
    ]
    gt_events = ground_truth_events or {}
    event_eval = per_patient_and_aggregate_event_evaluation(
        val_metadata,
        probabilities,
        gt_events,
        threshold=0.5,
        recording_durations=recording_durations,
    )

    return {
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "train_positive_windows": positive,
        "validation_metrics": _metrics(labels, probabilities, 0.5),
        "event_evaluation": event_eval,
        "history": history,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
    }

#         "train_positive_windows": positive,
#         "validation_metrics": _metrics(labels, probabilities, 0.5),
#         "history": history,
#         "best_epoch": best_epoch,
#         "checkpoint": str(checkpoint),
#     }



# def _fit_final(
#     train_windows: Sequence[RawWindow],
#     test_windows: Sequence[RawWindow],
#     *,
#     canonical_channels: Sequence[str],
#     epochs: int,
#     batch_size: int,
#     device: torch.device,
#     checkpoint: Path,
#     use_amp: bool,
#     normalization_cache: NormalizationCache,
# ) -> dict[str, Any]:
def _fit_final(
    train_windows: Sequence[RawWindow],
    test_windows: Sequence[RawWindow],
    *,
    canonical_channels: Sequence[str],
    epochs: int,
    batch_size: int,
    device: torch.device,
    checkpoint: Path,
    use_amp: bool,
    normalization_cache: NormalizationCache,
    resume: bool = True,
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]] | None = None,
    recording_durations: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    mean, std = fit_channel_normalization(train_windows, cache=normalization_cache)
    train_loader = _loader(train_windows, mean, std, batch_size, True)
    test_loader = _loader(test_windows, mean, std, batch_size, False)
    model = CNNBiLSTM(len(canonical_channels)).to(device)
    positive = max(sum(window.label for window in train_windows), 1)
    negative = max(len(train_windows) - positive, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history: list[float] = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1

    latest_checkpoint = checkpoint.parent / f"{checkpoint.stem}_latest.pt"

    if resume and latest_checkpoint.is_file():
        saved_latest = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved_latest["model_state"])
        optimizer.load_state_dict(saved_latest["optimizer_state"])
        if "scaler_state" in saved_latest and saved_latest["scaler_state"] is not None:
            scaler.load_state_dict(saved_latest["scaler_state"])
        if "rng_state" in saved_latest and saved_latest["rng_state"] is not None:
            torch.set_rng_state(saved_latest["rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state" in saved_latest:
                torch.cuda.set_rng_state(saved_latest["cuda_rng_state"].cpu())
        history = list(saved_latest.get("history", []))
        start_epoch = int(saved_latest.get("epoch", 0)) + 1
    elif resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        history = list(saved.get("history", []))
        start_epoch = int(saved.get("epochs", len(history))) + 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        for signal, target in train_loader:
            signal = signal.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(signal), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(target)
        epoch_loss = total_loss / len(train_loader.dataset)
        history.append(epoch_loss)

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "canonical_channels": list(canonical_channels),
                "normalization_mean": mean,
                "normalization_std": std,
                "positive_class_weight": negative / positive,
                "epoch": epoch,
                "epochs": epochs,
                "history": history,
            },
            latest_checkpoint,
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "canonical_channels": list(canonical_channels),
            "normalization_mean": mean,
            "normalization_std": std,
            "positive_class_weight": negative / positive,
            "epochs": epochs,
            "history": history,
        },
        checkpoint,
    )
    labels, probabilities = _predict(model, test_loader, device)

    test_metadata = [
        {
            "patient_id": w.patient_id,
            "recording_id": w.recording_id,
            "start_seconds": w.start_seconds,
            "end_seconds": w.end_seconds,
            "label": w.label,
        }
        for w in test_loader.dataset.windows
    ]
    gt_events = ground_truth_events or {}
    event_eval = per_patient_and_aggregate_event_evaluation(
        test_metadata,
        probabilities,
        gt_events,
        threshold=0.5,
        recording_durations=recording_durations,
    )

    return {
        "train_windows": len(train_windows),
        "test_windows": len(test_windows),
        "train_positive_windows": positive,
        "test_metrics": _metrics(labels, probabilities, 0.5),
        "event_evaluation": event_eval,
        "checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
    }



def _load_split() -> dict[str, Any]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"run audit first; missing frozen split: {SPLIT_PATH}")
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Raw EEG CNN + Bi-LSTM baseline")
    parser.add_argument("--mode", required=True, choices=("audit", "cv", "final"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default=DEFAULT_DEVICE,
        help="Training device; CUDA is the default for DL CV and final training.",
    )
    parser.add_argument("--manifest", type=Path, default=PREPROCESSED_MANIFEST_PATH)
    args = parser.parse_args(argv)
    seed_everything()
    DL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        result = audit_and_freeze(args.manifest, output_dir=DL_ARTIFACTS_DIR)
        print(json.dumps(result["report"], indent=2, sort_keys=True))
        return 0

    split = _load_split()
    canonical_channels = split["canonical_channels"]
    device = select_device(args.device)
    use_amp = _amp_enabled(device)
    normalization_cache: NormalizationCache = {}
    all_windows = load_manifest_windows(
        args.manifest,
        canonical_channels=canonical_channels,
        quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
    )
    by_patient: dict[str, list[RawWindow]] = {}
    for window in all_windows:
        by_patient.setdefault(window.patient_id, []).append(window)

    ground_truth_events = load_ground_truth_events(
        args.manifest,
        quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
    )
    recording_durations = load_recording_durations(
        args.manifest,
        quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
    )

    if args.mode == "cv":
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        CNN_BILSTM_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        folds = []
        for fold in split["folds"]:
            train_windows = [w for p in fold["training_patients"] for w in by_patient[p]]
            validation_windows = [w for p in fold["validation_patients"] for w in by_patient[p]]
            fold_res = _fit_fold(
                train_windows,
                validation_windows,
                canonical_channels=canonical_channels,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                checkpoint=CHECKPOINT_DIR / f"fold_{fold['fold']}.pt",
                use_amp=use_amp,
                normalization_cache=normalization_cache,
                ground_truth_events=ground_truth_events,
                recording_durations=recording_durations,
            )
            folds.append(fold_res)
        result = {"mode": "cv", "device": str(device), "folds": folds}
    else:
        # Check both primary cv_results and legacy training_results
        cv_source_path = CNN_BILSTM_CV_RESULTS_PATH if CNN_BILSTM_CV_RESULTS_PATH.is_file() else RESULTS_PATH
        if not cv_source_path.is_file():
            raise FileNotFoundError("run --mode cv before --mode final")
        previous = json.loads(cv_source_path.read_text(encoding="utf-8"))
        if previous.get("mode") != "cv":
            raise ValueError("final training requires a completed CV result")
        # Select final training epochs using the median of the five CV folds' best epochs
        fold_best_epochs = [
            int(fold.get("best_epoch", 1))
            for fold in previous.get("folds", [])
        ]
        # # selected_epochs = max(
        # #     int(fold.get("best_epoch", 1))
        # #     for fold in previous.get("folds", [])
        # # )
        # selected_epochs = int(np.round(np.median(fold_best_epochs))) if fold_best_epochs else args.epochs
        # Deliberately selected final training duration
        selected_epochs = 5
        train_windows = [w for p in split["dev_patients"] for w in by_patient[p]]
        test_windows = [w for p in split["test_patients"] for w in by_patient[p]]
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        CNN_BILSTM_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        result = {
            "mode": "final",
            "device": str(device),
            "selected_epochs_strategy": "fixed_5_epochs",
            "cv_best_epochs": fold_best_epochs,
            "selected_epochs": selected_epochs,
            "independent_test": _fit_final(
                train_windows,
                test_windows,
                canonical_channels=canonical_channels,
                epochs=selected_epochs,
                batch_size=args.batch_size,
                device=device,
                checkpoint=CHECKPOINT_DIR / "final.pt",
                use_amp=use_amp,
                normalization_cache=normalization_cache,
                ground_truth_events=ground_truth_events,
                recording_durations=recording_durations,
            ),
        }
    result["config"] = {
        "seed": SEED,
        "window_seconds": 20,
        "sampling_rate_hz": 256,
        "samples_per_window": 5120,
        "canonical_channels": canonical_channels,
        "model": "CNNBiLSTM",
        "attention": False,
        "requested_device": args.device,
        "amp": use_amp,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    # Save to dedicated separated files so final training never overwrites CV results
    if args.mode == "cv":
        CNN_BILSTM_CV_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CNN_BILSTM_CV_RESULTS_PATH.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        RESULTS_PATH.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    else:
        CNN_BILSTM_FINAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CNN_BILSTM_FINAL_RESULTS_PATH.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        # Backward compatibility for legacy tools expecting training_results.json
        final_legacy_path = DL_ARTIFACTS_DIR / "final_results.json"
        final_legacy_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
