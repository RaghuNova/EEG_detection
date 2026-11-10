# """Two-stage attention experiment using completed CNN-BiLSTM checkpoints."""

# from __future__ import annotations

# import argparse
# import json
# from dataclasses import replace
# from pathlib import Path
# from typing import Any, Sequence

# import numpy as np
# import torch
# from sklearn.metrics import average_precision_score
# from torch import nn
# from torch.utils.data import DataLoader, Dataset

# from .attention import AttentionClassifier
# from .cache import build_cache, verify_cache
# from .config import (
#     ATTENTION_ARTIFACTS_DIR,
#     CNN_BILSTM_ARTIFACTS_DIR,
#     COMPARISON_ARTIFACTS_DIR,
#     DEFAULT_DEVICE,
#     DL_ARTIFACTS_DIR,
#     PREPROCESSED_MANIFEST_PATH,
#     SEED,
#     USE_CUDA_AMP,
# )
# from .comparison import compute_comparison, save_comparison_json
# from .event_evaluation import evaluate_events, per_patient_and_aggregate_event_evaluation
# from .model import CNNBiLSTM
# from .raw_data import (
#     NormalizationCache,
#     RawEEGDataset,
#     RawWindow,
#     fit_channel_normalization,
#     load_ground_truth_events,
#     load_manifest_windows,
#     load_recording_durations,
#     order_windows_for_shard_io,
# )
# from .train import _load_split, select_device, seed_everything

# FEATURE_DIR = DL_ARTIFACTS_DIR / "attention_features"
# # ATTENTION_CHECKPOINT_DIR = DL_ARTIFACTS_DIR / "attention" / "checkpoints"
# # RESULTS_PATH = DL_ARTIFACTS_DIR / "attention" / "cv_results.json"
# # FINAL_RESULTS_PATH = DL_ARTIFACTS_DIR / "attention" / "final_results.json"
# ATTENTION_CHECKPOINT_DIR = ATTENTION_ARTIFACTS_DIR / "checkpoints"
# RESULTS_PATH = ATTENTION_ARTIFACTS_DIR / "cv_results.json"
# FINAL_RESULTS_PATH = ATTENTION_ARTIFACTS_DIR / "final_results.json"
# CNN_BILSTM_CV_RESULTS_PATH = CNN_BILSTM_ARTIFACTS_DIR / "cv_results.json"
# CNN_BILSTM_FINAL_RESULTS_PATH = CNN_BILSTM_ARTIFACTS_DIR / "final_results.json"
# CNN_BILSTM_CHECKPOINT_DIR = CNN_BILSTM_ARTIFACTS_DIR / "checkpoints"
# CV_COMPARISON_PATH = COMPARISON_ARTIFACTS_DIR / "cv_comparison.json"
# FINAL_COMPARISON_PATH = COMPARISON_ARTIFACTS_DIR / "final_comparison.json"



# class FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
#     def __init__(self, feature_path: Path, labels: np.ndarray) -> None:
#         self.features = np.load(feature_path, mmap_mode="r")
#         self.labels = np.asarray(labels, dtype=np.float32)
#         if self.features.ndim != 3 or self.features.shape[0] != len(self.labels):
#             raise ValueError("attention features and labels are misaligned")

#     def __len__(self) -> int:
#         return len(self.labels)

#     def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
#         # return torch.from_numpy(np.asarray(self.features[index])), torch.tensor(
#         #     self.labels[index], dtype=torch.float32
#         # )
#         feat_array = np.array(self.features[index], copy=True)
#         return torch.from_numpy(feat_array), torch.tensor(
#             self.labels[index], dtype=torch.float32
#         )



# def _metadata(window: RawWindow, fold: int) -> dict[str, Any]:
#     return {
#         "patient_id": window.patient_id,
#         "recording_id": window.recording_id,
#         "window_id": window.window_id,
#         "start_seconds": window.start_seconds,
#         "end_seconds": window.end_seconds,
#         "label": window.label,
#         "fold": fold,
#     }


# def extract_features(
#     checkpoint: Path,
#     windows: Sequence[RawWindow],
#     *,
#     canonical_channels: Sequence[str],
#     mean: np.ndarray,
#     std: np.ndarray,
#     output_dir: Path,
#     split_name: str,
#     fold: int,
#     device: torch.device,
#     batch_size: int,
# ) -> tuple[Path, Path, list[RawWindow]]:
#     output_dir.mkdir(parents=True, exist_ok=True)
#     feature_path = output_dir / f"{split_name}_features.npy"
#     metadata_path = output_dir / f"{split_name}_metadata.json"
#     ordered = order_windows_for_shard_io(windows, shuffle_shards=False, seed=SEED)
#     loader = DataLoader(
#         RawEEGDataset(ordered, mean, std),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=device.type == "cuda",
#     )
#     model = CNNBiLSTM(len(canonical_channels)).to(device)
#     state = torch.load(checkpoint, map_location=device, weights_only=False)
#     model.load_state_dict(state["model_state"])
#     model.eval()
#     memmap = np.lib.format.open_memmap(
#         feature_path,
#         mode="w+",
#         dtype=np.float32,
#         shape=(len(ordered), model.recurrent_length, 128),
#     )
#     offset = 0
#     with torch.no_grad():
#         for signals, _ in loader:
#             signals = signals.to(device, non_blocking=device.type == "cuda")
#             with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and USE_CUDA_AMP):
#                 values = model.extract_temporal_features(signals)
#             batch = values.float().cpu().numpy()
#             memmap[offset : offset + len(batch)] = batch
#             offset += len(batch)
#     memmap.flush()
#     metadata_path.write_text(
#         json.dumps(
#             {
#                 "fold": fold,
#                 "split": split_name,
#                 "feature_shape": [len(ordered), model.recurrent_length, 128],
#                 "checkpoint": str(checkpoint),
#                 "canonical_channels": list(canonical_channels),
#                 "metadata": [_metadata(window, fold) for window in ordered],
#             },
#             indent=2,
#         ),
#         encoding="utf-8",
#     )
#     return feature_path, metadata_path, ordered


# # def _fit_attention(
# #     train_features: Path,
# #     validation_features: Path,
# #     train_labels: np.ndarray,
# #     validation_labels: np.ndarray,
# #     *,
# #     device: torch.device,
# #     checkpoint: Path,
# #     epochs: int,
# #     batch_size: int,
# #     resume: bool = True,
# # ) -> dict[str, Any]:
# def _fit_attention(
#     train_features: Path,
#     validation_features: Path,
#     train_labels: np.ndarray,
#     validation_labels: np.ndarray,
#     *,
#     device: torch.device,
#     checkpoint: Path,
#     epochs: int,
#     batch_size: int,
#     resume: bool = True,
#     validation_metadata: Sequence[Mapping[str, Any]] | None = None,
#     ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]] | None = None,
#     recording_durations: Mapping[tuple[str, str], float] | None = None,
# ) -> dict[str, Any]:
#     train_loader = DataLoader(FeatureDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
#     validation_loader = DataLoader(FeatureDataset(validation_features, validation_labels), batch_size=batch_size)
#     model = AttentionClassifier().to(device)
#     positive = max(int(train_labels.sum()), 1)
#     negative = max(len(train_labels) - positive, 1)
#     loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
#     scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and USE_CUDA_AMP)
#     best_loss = float("inf")
#     best_epoch = 0
#     stale = 0
#     history: list[dict[str, float]] = []
#     start_epoch = 1

#     latest_checkpoint = checkpoint.parent / f"{checkpoint.stem}_latest.pt"

#     if resume and latest_checkpoint.is_file():
#         saved = torch.load(latest_checkpoint, map_location=device, weights_only=False)
#         if saved.get("model_state") is not None and saved.get("optimizer_state") is not None:
#             model.load_state_dict(saved["model_state"])
#             optimizer.load_state_dict(saved["optimizer_state"])
#             scaler.load_state_dict(saved.get("scaler_state", {}))
#             if "rng_state" in saved and saved["rng_state"] is not None:
#                 torch.set_rng_state(saved["rng_state"].cpu())
#                 if torch.cuda.is_available() and "cuda_rng_state" in saved:
#                     torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu())
#             history = list(saved.get("history", []))
#             best_loss = float(saved.get("best_loss", best_loss))
#             best_epoch = int(saved.get("best_epoch", 0))
#             stale = int(saved.get("stale", 0))
#             start_epoch = int(saved.get("epoch", 0)) + 1
#     elif resume and checkpoint.is_file():
#         saved = torch.load(checkpoint, map_location=device, weights_only=False)
#         if saved.get("model_state") is not None and saved.get("optimizer_state") is not None:
#             model.load_state_dict(saved["model_state"])
#             optimizer.load_state_dict(saved["optimizer_state"])
#             scaler.load_state_dict(saved.get("scaler_state", {}))
#             history = list(saved.get("history", []))
#             best_loss = float(saved.get("best_loss", best_loss))
#             best_epoch = int(saved.get("best_epoch", int(saved.get("epoch", 0))))
#             stale = int(saved.get("stale", 0))
#             start_epoch = int(saved.get("epoch", 0)) + 1

#     for epoch in range(start_epoch, epochs + 1):
#         model.train()
#         train_loss = 0.0
#         for features, labels in train_loader:
#             features = features.to(device, non_blocking=device.type == "cuda")
#             labels = labels.to(device, non_blocking=device.type == "cuda")
#             optimizer.zero_grad(set_to_none=True)
#             with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and USE_CUDA_AMP):
#                 loss = loss_fn(model(features), labels)
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             train_loss += float(loss.item()) * len(labels)
#         model.eval()
#         probabilities: list[np.ndarray] = []
#         with torch.no_grad():
#             for features, _ in validation_loader:
#                 features = features.to(device, non_blocking=device.type == "cuda")
#                 probabilities.append(torch.sigmoid(model(features)).cpu().numpy())
#         validation_probabilities = np.concatenate(probabilities)
#         validation_loss = float(
#             nn.functional.binary_cross_entropy(
#                 torch.from_numpy(np.clip(validation_probabilities, 1e-6, 1 - 1e-6)).float(),
#                 torch.from_numpy(validation_labels.astype(np.float32)),
#             )
#         )
#         average_precision = float(average_precision_score(validation_labels, validation_probabilities))
#         history.append({
#             "epoch": epoch,
#             "train_loss": train_loss / len(train_labels),
#             "validation_loss": validation_loss,
#             "average_precision": average_precision,
#         })
#         if validation_loss < best_loss:
#             best_loss, best_epoch, stale = validation_loss, epoch, 0
#             torch.save(
#                 {
#                     "model_state": model.state_dict(),
#                     "optimizer_state": optimizer.state_dict(),
#                     "scaler_state": scaler.state_dict(),
#                     "history": history,
#                     "best_loss": best_loss,
#                     "best_epoch": best_epoch,
#                     "epoch": epoch,
#                     "positive_class_weight": negative / positive,
#                 },
#                 checkpoint,
#             )
#         else:
#             stale += 1

#         # Save latest checkpoint for separate resume state
#         torch.save(
#             {
#                 "model_state": model.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "scaler_state": scaler.state_dict(),
#                 "rng_state": torch.get_rng_state(),
#                 "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
#                 "history": history,
#                 "best_loss": best_loss,
#                 "best_epoch": best_epoch,
#                 "stale": stale,
#                 "epoch": epoch,
#                 "positive_class_weight": negative / positive,
#             },
#             latest_checkpoint,
#         )

#         if stale >= 10:
#             break

#     # Restore best checkpoint
#     if checkpoint.is_file():
#         saved_best = torch.load(checkpoint, map_location=device, weights_only=False)
#         model.load_state_dict(saved_best["model_state"])

#     model.eval()
#     probabilities = []
#     with torch.no_grad():
#         for features, _ in validation_loader:
#             features = features.to(device, non_blocking=device.type == "cuda")
#             probabilities.append(torch.sigmoid(model(features)).cpu().numpy())
#     final_val_probs = np.concatenate(probabilities)

#     val_meta = validation_metadata or [
#         {"patient_id": "unknown", "recording_id": "unknown", "start_seconds": i * 20.0, "end_seconds": (i + 1) * 20.0, "label": int(validation_labels[i])}
#         for i in range(len(validation_labels))
#     ]
#     gt_events = ground_truth_events or {}
#     event_eval = per_patient_and_aggregate_event_evaluation(
#         val_meta,
#         final_val_probs,
#         gt_events,
#         threshold=0.5,
#         recording_durations=recording_durations,
#     )

#     return {
#         "best_validation_loss": best_loss,
#         "best_epoch": best_epoch,
#         "history": history,
#         "checkpoint": str(checkpoint),
#         "latest_checkpoint": str(latest_checkpoint),
#         "event_evaluation": event_eval,
#     }



# # def _fit_attention_final(
# #     train_features: Path,
# #     test_features: Path,
# #     train_labels: np.ndarray,
# #     test_labels: np.ndarray,
# #     *,
# #     device: torch.device,
# #     checkpoint: Path,
# #     epochs: int,
# #     batch_size: int,
# # ) -> dict[str, Any]:
# def _fit_attention_final(
#     train_features: Path,
#     test_features: Path,
#     train_labels: np.ndarray,
#     test_labels: np.ndarray,
#     *,
#     device: torch.device,
#     checkpoint: Path,
#     epochs: int,
#     batch_size: int,
#     resume: bool = True,
#     test_metadata: Sequence[Mapping[str, Any]] | None = None,
#     ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]] | None = None,
#     recording_durations: Mapping[tuple[str, str], float] | None = None,
# ) -> dict[str, Any]:
#     train_loader = DataLoader(
#         FeatureDataset(train_features, train_labels), batch_size=batch_size, shuffle=True
#     )
#     test_loader = DataLoader(FeatureDataset(test_features, test_labels), batch_size=batch_size)
#     model = AttentionClassifier().to(device)
#     positive = max(int(train_labels.sum()), 1)
#     negative = max(len(train_labels) - positive, 1)
#     loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
#     scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and USE_CUDA_AMP)
#     history: list[dict[str, float]] = []
#     start_epoch = 1

#     latest_checkpoint = checkpoint.parent / f"{checkpoint.stem}_latest.pt"

#     if resume and latest_checkpoint.is_file():
#         saved = torch.load(latest_checkpoint, map_location=device, weights_only=False)
#         model.load_state_dict(saved["model_state"])
#         optimizer.load_state_dict(saved["optimizer_state"])
#         scaler.load_state_dict(saved.get("scaler_state", {}))
#         if "rng_state" in saved and saved["rng_state"] is not None:
#             torch.set_rng_state(saved["rng_state"].cpu())
#             if torch.cuda.is_available() and "cuda_rng_state" in saved:
#                 torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu())
#         history = list(saved.get("history", []))
#         start_epoch = int(saved.get("epoch", 0)) + 1
#     elif resume and checkpoint.is_file():
#         saved = torch.load(checkpoint, map_location=device, weights_only=False)
#         model.load_state_dict(saved["model_state"])
#         optimizer.load_state_dict(saved["optimizer_state"])
#         scaler.load_state_dict(saved.get("scaler_state", {}))
#         history = list(saved.get("history", []))
#         start_epoch = int(saved.get("epoch", 0)) + 1

#     for epoch in range(start_epoch, epochs + 1):
#         model.train()
#         total_loss = 0.0
#         for features, labels in train_loader:
#             features = features.to(device, non_blocking=device.type == "cuda")
#             labels = labels.to(device, non_blocking=device.type == "cuda")
#             optimizer.zero_grad(set_to_none=True)
#             with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and USE_CUDA_AMP):
#                 loss = loss_fn(model(features), labels)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             total_loss += float(loss.item()) * len(labels)
#         history.append({"epoch": epoch, "train_loss": total_loss / len(train_labels)})

#         # Save latest checkpoint
#         torch.save(
#             {
#                 "model_state": model.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "scaler_state": scaler.state_dict(),
#                 "rng_state": torch.get_rng_state(),
#                 "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
#                 "history": history,
#                 "epoch": epoch,
#                 "positive_class_weight": negative / positive,
#             },
#             latest_checkpoint,
#         )

#     # Save final best checkpoint
#     torch.save(
#         {
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "scaler_state": scaler.state_dict(),
#             "history": history,
#             "epoch": epochs,
#             "positive_class_weight": negative / positive,
#         },
#         checkpoint,
#     )

#     model.eval()
#     probabilities: list[np.ndarray] = []
#     with torch.no_grad():
#         for features, _ in test_loader:
#             features = features.to(device, non_blocking=device.type == "cuda")
#             probabilities.append(torch.sigmoid(model(features)).cpu().numpy())
#     values = np.concatenate(probabilities)

#     t_meta = test_metadata or [
#         {"patient_id": "unknown", "recording_id": "unknown", "start_seconds": i * 20.0, "end_seconds": (i + 1) * 20.0, "label": int(test_labels[i])}
#         for i in range(len(test_labels))
#     ]
#     gt_events = ground_truth_events or {}
#     event_eval = per_patient_and_aggregate_event_evaluation(
#         t_meta,
#         values,
#         gt_events,
#         threshold=0.5,
#         recording_durations=recording_durations,
#     )

#     return {
#         "history": history,
#         "checkpoint": str(checkpoint),
#         "latest_checkpoint": str(latest_checkpoint),
#         "test_average_precision": float(average_precision_score(test_labels, values)),
#         "event_evaluation": event_eval,
#     }



# def main(argv: list[str] | None = None) -> int:
#     parser = argparse.ArgumentParser(description="Stage 2 additive temporal attention")
#     parser.add_argument("--mode", choices=("cache", "verify-cache", "cv", "final"), default="cv")
#     parser.add_argument("--cache-dir", type=Path, default=DL_ARTIFACTS_DIR / "data_cache")
#     parser.add_argument("--rebuild-cache", action="store_true")
#     parser.add_argument("--verify-cache", action="store_true")
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--batch-size", type=int, default=32)
#     parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default=DEFAULT_DEVICE)
#     parser.add_argument("--manifest", type=Path, default=PREPROCESSED_MANIFEST_PATH)
#     args = parser.parse_args(argv)
#     seed_everything()
#     device = select_device(args.device)
#     split = _load_split()
#     if args.mode == "verify-cache" or args.verify_cache:
#         print(json.dumps(verify_cache(args.cache_dir), indent=2))
#         return 0
#     cache_manifest = build_cache(
#         args.manifest,
#         args.cache_dir,
#         canonical_channels=list(split["canonical_channels"]),
#         rebuild=args.rebuild_cache,
#     )
#     if args.mode == "cache":
#         print(json.dumps({"mode": "cache", **verify_cache(args.cache_dir)}, indent=2))
#         return 0
#     all_windows = load_manifest_windows(
#         args.manifest,
#         canonical_channels=split["canonical_channels"],
#         quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
#     )
#     cached_paths = {
#         entry["source_path"]: entry["cache_path"]
#         for entry in cache_manifest["entries"]
#     }
#     all_windows = [
#         replace(
#             window,
#             path=cached_paths.get(window.path, window.path),
#             channel_indices=tuple(range(len(split["canonical_channels"])))
#             if window.path in cached_paths
#             else window.channel_indices,
#         )
#         for window in all_windows
#     ]
#     by_patient: dict[str, list[RawWindow]] = {}
#     for window in all_windows:
#         by_patient.setdefault(window.patient_id, []).append(window)
#     ground_truth_events = load_ground_truth_events(
#         args.manifest,
#         quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
#     )
#     recording_durations = load_recording_durations(
#         args.manifest,
#         quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
#     )

#     if args.mode == "final":
#         baseline_checkpoint = DL_ARTIFACTS_DIR / "checkpoints" / "final.pt"
#         if not baseline_checkpoint.is_file():
#             # Check separated directory
#             alt_baseline = CNN_BILSTM_CHECKPOINT_DIR / "final.pt"
#             if alt_baseline.is_file():
#                 baseline_checkpoint = alt_baseline
#             else:
#                 raise FileNotFoundError("run `backend.dl.train --mode final` before attention final training")
#         previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.is_file() else {}
#         # Select final training epochs using the median of the five CV folds' best epochs
#         fold_best_epochs = [
#             int(item.get("best_epoch", item.get("history", [{}])[-1].get("epoch", 1)))
#             for item in previous.get("folds", [])
#             if item.get("best_epoch") is not None or item.get("history")
#         ]
#         # selected_epochs = max(
#         #     (int(item.get("history", [{}])[-1].get("epoch", 1)) for item in previous.get("folds", [])),
#         #     default=args.epochs,
#         # )
#         selected_epochs = int(np.round(np.median(fold_best_epochs))) if fold_best_epochs else args.epochs
#         train_windows = [w for p in split["dev_patients"] for w in by_patient[p]]
#         test_windows = [w for p in split["test_patients"] for w in by_patient[p]]
#         mean, std = fit_channel_normalization(train_windows)
#         output_dir = FEATURE_DIR / "final"
#         train_path, _, ordered_train = extract_features(
#             baseline_checkpoint, train_windows,
#             canonical_channels=split["canonical_channels"], mean=mean, std=std,
#             output_dir=output_dir, split_name="train", fold=0, device=device,
#             batch_size=args.batch_size,
#         )
#         test_path, _, ordered_test = extract_features(
#             baseline_checkpoint, test_windows,
#             canonical_channels=split["canonical_channels"], mean=mean, std=std,
#             output_dir=output_dir, split_name="test", fold=0, device=device,
#             batch_size=args.batch_size,
#         )
#         test_metadata = [
#             {
#                 "patient_id": w.patient_id,
#                 "recording_id": w.recording_id,
#                 "start_seconds": w.start_seconds,
#                 "end_seconds": w.end_seconds,
#                 "label": w.label,
#             }
#             for w in ordered_test
#         ]
#         ATTENTION_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
#         result = _fit_attention_final(
#             train_path, test_path,
#             np.asarray([w.label for w in ordered_train]),
#             np.asarray([w.label for w in ordered_test]),
#             device=device, checkpoint=ATTENTION_CHECKPOINT_DIR / "final.pt",
#             epochs=selected_epochs, batch_size=args.batch_size,
#             test_metadata=test_metadata,
#             ground_truth_events=ground_truth_events,
#             recording_durations=recording_durations,
#         )
#         result_payload = {
#             "mode": "final",
#             "device": str(device),
#             "selected_epochs_strategy": "median_best_epoch",
#             "cv_best_epochs": fold_best_epochs,
#             "selected_epochs": selected_epochs,
#             **result,
#         }
#         FINAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
#         FINAL_RESULTS_PATH.write_text(json.dumps(result_payload, indent=2, default=float), encoding="utf-8")

#         # Generate final comparison if CNN-BiLSTM final results exist
#         bilstm_final_source = CNN_BILSTM_FINAL_RESULTS_PATH if CNN_BILSTM_FINAL_RESULTS_PATH.is_file() else DL_ARTIFACTS_DIR / "final_results.json"
#         if bilstm_final_source.is_file():
#             bilstm_final_data = json.loads(bilstm_final_source.read_text(encoding="utf-8"))
#             base_metrics = bilstm_final_data.get("independent_test", {}).get("event_evaluation", {})
#             attn_metrics = result.get("event_evaluation", {})
#             comparison = compute_comparison(base_metrics, attn_metrics, comparison_name="Final Independent Test")
#             save_comparison_json(comparison, FINAL_COMPARISON_PATH)

#         print(json.dumps(result_payload, indent=2, default=float))
#         return 0
#     cache: NormalizationCache = {}
#     results = []
#     for fold in split["folds"]:
#         fold_id = int(fold["fold"])
#         train_windows = [w for p in fold["training_patients"] for w in by_patient[p]]
#         validation_windows = [w for p in fold["validation_patients"] for w in by_patient[p]]
#         mean, std = fit_channel_normalization(train_windows, cache=cache)
#         stage1_checkpoint = DL_ARTIFACTS_DIR / "checkpoints" / f"fold_{fold_id}.pt"
#         if not stage1_checkpoint.is_file():
#             alt_s1 = CNN_BILSTM_CHECKPOINT_DIR / f"fold_{fold_id}.pt"
#             if alt_s1.is_file():
#                 stage1_checkpoint = alt_s1
#             else:
#                 raise FileNotFoundError(f"missing Stage 1 checkpoint: {stage1_checkpoint}")
#         fold_dir = FEATURE_DIR / f"fold_{fold_id}"
#         train_path, _, ordered_train = extract_features(stage1_checkpoint, train_windows, canonical_channels=split["canonical_channels"], mean=mean, std=std, output_dir=fold_dir, split_name="train", fold=fold_id, device=device, batch_size=args.batch_size)
#         validation_path, _, ordered_validation = extract_features(stage1_checkpoint, validation_windows, canonical_channels=split["canonical_channels"], mean=mean, std=std, output_dir=fold_dir, split_name="validation", fold=fold_id, device=device, batch_size=args.batch_size)
#         checkpoint = ATTENTION_CHECKPOINT_DIR / f"fold_{fold_id}.pt"
#         checkpoint.parent.mkdir(parents=True, exist_ok=True)
#         val_metadata = [
#             {
#                 "patient_id": w.patient_id,
#                 "recording_id": w.recording_id,
#                 "start_seconds": w.start_seconds,
#                 "end_seconds": w.end_seconds,
#                 "label": w.label,
#             }
#             for w in ordered_validation
#         ]
#         results.append(_fit_attention(
#             train_path, validation_path,
#             np.asarray([w.label for w in ordered_train]),
#             np.asarray([w.label for w in ordered_validation]),
#             device=device, checkpoint=checkpoint,
#             epochs=args.epochs, batch_size=args.batch_size,
#             validation_metadata=val_metadata,
#             ground_truth_events=ground_truth_events,
#             recording_durations=recording_durations,
#         ))
#     RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
#     cv_payload = {"mode": "cv", "device": str(device), "folds": results}
#     RESULTS_PATH.write_text(json.dumps(cv_payload, indent=2, default=float), encoding="utf-8")

#     # Generate CV comparison if CNN-BiLSTM CV results exist
#     bilstm_cv_source = CNN_BILSTM_CV_RESULTS_PATH if CNN_BILSTM_CV_RESULTS_PATH.is_file() else DL_ARTIFACTS_DIR / "training_results.json"
#     if bilstm_cv_source.is_file():
#         bilstm_cv_data = json.loads(bilstm_cv_source.read_text(encoding="utf-8"))
#         # Aggregate comparison across folds if available
#         base_fold_events = [f.get("event_evaluation", {}).get("aggregate", {}) for f in bilstm_cv_data.get("folds", []) if f.get("event_evaluation")]
#         attn_fold_events = [f.get("event_evaluation", {}).get("aggregate", {}) for f in results if f.get("event_evaluation")]
#         if base_fold_events and attn_fold_events:
#             # Average metrics across folds
#             def avg_dict(dicts):
#                 keys = ["event_sensitivity", "event_precision", "event_f1", "false_alarms_per_hour"]
#                 res = {}
#                 for k in keys:
#                     vals = [d[k] for d in dicts if k in d and d[k] is not None]
#                     res[k] = float(np.mean(vals)) if vals else None
#                 # Add window metrics
#                 w_keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision"]
#                 w_res = {}
#                 for wk in w_keys:
#                     w_vals = [d.get("window_metrics", {}).get(wk) for d in dicts if d.get("window_metrics", {}).get(wk) is not None]
#                     w_res[wk] = float(np.mean(w_vals)) if w_vals else None
#                 res["window_metrics"] = w_res
#                 return res

#             cv_comp = compute_comparison(avg_dict(base_fold_events), avg_dict(attn_fold_events), comparison_name="Cross-Validation Average")
#             save_comparison_json(cv_comp, CV_COMPARISON_PATH)

#     print(json.dumps(cv_payload, indent=2, default=float))
#     return 0



# if __name__ == "__main__":
#     raise SystemExit(main())
"""Two-stage attention experiment using completed CNN-BiLSTM checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .attention import AttentionClassifier
from .cache import build_cache, verify_cache
from .config import (
    ATTENTION_ARTIFACTS_DIR,
    CNN_BILSTM_ARTIFACTS_DIR,
    COMPARISON_ARTIFACTS_DIR,
    DEFAULT_DEVICE,
    DL_ARTIFACTS_DIR,
    PREPROCESSED_MANIFEST_PATH,
    SEED,
    USE_CUDA_AMP,
)
from .comparison import compute_comparison, save_comparison_json
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
from .train import _load_split, select_device, seed_everything

FEATURE_DIR = DL_ARTIFACTS_DIR / "attention_features"
FIXED_EPOCHS = 5
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
CALIBRATION_PATH = ATTENTION_ARTIFACTS_DIR / "threshold_calibration.json"
OOF_PATH = ATTENTION_ARTIFACTS_DIR / "oof_predictions.npz"
OOF_METADATA_PATH = ATTENTION_ARTIFACTS_DIR / "oof_metadata.json"
# ATTENTION_CHECKPOINT_DIR = DL_ARTIFACTS_DIR / "attention" / "checkpoints"
# RESULTS_PATH = DL_ARTIFACTS_DIR / "attention" / "cv_results.json"
# FINAL_RESULTS_PATH = DL_ARTIFACTS_DIR / "attention" / "final_results.json"
ATTENTION_CHECKPOINT_DIR = ATTENTION_ARTIFACTS_DIR / "checkpoints"
RESULTS_PATH = ATTENTION_ARTIFACTS_DIR / "cv_results.json"
FINAL_RESULTS_PATH = ATTENTION_ARTIFACTS_DIR / "final_results.json"
CNN_BILSTM_CV_RESULTS_PATH = CNN_BILSTM_ARTIFACTS_DIR / "cv_results.json"
CNN_BILSTM_FINAL_RESULTS_PATH = CNN_BILSTM_ARTIFACTS_DIR / "final_results.json"
CNN_BILSTM_CHECKPOINT_DIR = CNN_BILSTM_ARTIFACTS_DIR / "checkpoints"
CV_COMPARISON_PATH = COMPARISON_ARTIFACTS_DIR / "cv_comparison.json"
FINAL_COMPARISON_PATH = COMPARISON_ARTIFACTS_DIR / "final_comparison.json"



class FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, feature_path: Path, labels: np.ndarray) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.labels = np.asarray(labels, dtype=np.float32)
        if self.features.ndim != 3 or self.features.shape[0] != len(self.labels):
            raise ValueError("attention features and labels are misaligned")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # return torch.from_numpy(np.asarray(self.features[index])), torch.tensor(
        #     self.labels[index], dtype=torch.float32
        # )
        feat_array = np.array(self.features[index], copy=True)
        return torch.from_numpy(feat_array), torch.tensor(
            self.labels[index], dtype=torch.float32
        )



def _metadata(window: RawWindow, fold: int) -> dict[str, Any]:
    return {
        "patient_id": window.patient_id,
        "recording_id": window.recording_id,
        "window_id": window.window_id,
        "start_seconds": window.start_seconds,
        "end_seconds": window.end_seconds,
        "label": window.label,
        "fold": fold,
    }


def extract_features(
    checkpoint: Path,
    windows: Sequence[RawWindow],
    *,
    canonical_channels: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    output_dir: Path,
    split_name: str,
    fold: int,
    device: torch.device,
    batch_size: int,
) -> tuple[Path, Path, list[RawWindow]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{split_name}_features.npy"
    metadata_path = output_dir / f"{split_name}_metadata.json"
    ordered = order_windows_for_shard_io(windows, shuffle_shards=False, seed=SEED)
    loader = DataLoader(
        RawEEGDataset(ordered, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = CNNBiLSTM(len(canonical_channels)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    memmap = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(ordered), model.recurrent_length, 128),
    )
    offset = 0
    with torch.no_grad():
        for signals, _ in loader:
            signals = signals.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and USE_CUDA_AMP):
                values = model.extract_temporal_features(signals)
            batch = values.float().cpu().numpy()
            memmap[offset : offset + len(batch)] = batch
            offset += len(batch)
    memmap.flush()
    metadata_path.write_text(
        json.dumps(
            {
                "fold": fold,
                "split": split_name,
                "feature_shape": [len(ordered), model.recurrent_length, 128],
                "checkpoint": str(checkpoint),
                "canonical_channels": list(canonical_channels),
                "metadata": [_metadata(window, fold) for window in ordered],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return feature_path, metadata_path, ordered


# def _fit_attention(
#     train_features: Path,
#     validation_features: Path,
#     train_labels: np.ndarray,
#     validation_labels: np.ndarray,
#     *,
#     device: torch.device,
#     checkpoint: Path,
#     epochs: int,
#     batch_size: int,
#     resume: bool = False,
# ) -> dict[str, Any]:
def _fit_attention(
    train_features: Path,
    validation_features: Path,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    *,
    device: torch.device,
    checkpoint: Path,
    epochs: int,
    batch_size: int,
    resume: bool = False,
    validation_metadata: Sequence[Mapping[str, Any]] | None = None,
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]] | None = None,
    recording_durations: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    train_loader = DataLoader(FeatureDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(FeatureDataset(validation_features, validation_labels), batch_size=batch_size)
    model = AttentionClassifier().to(device)
    positive = max(int(train_labels.sum()), 1)
    negative = max(len(train_labels) - positive, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and USE_CUDA_AMP)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    start_epoch = 1

    latest_checkpoint = checkpoint.parent / f"{checkpoint.stem}_latest.pt"

    if resume and latest_checkpoint.is_file():
        saved = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        if saved.get("model_state") is not None and saved.get("optimizer_state") is not None:
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            scaler.load_state_dict(saved.get("scaler_state", {}))
            if "rng_state" in saved and saved["rng_state"] is not None:
                torch.set_rng_state(saved["rng_state"].cpu())
                if torch.cuda.is_available() and "cuda_rng_state" in saved:
                    torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu())
            history = list(saved.get("history", []))
            best_loss = float(saved.get("best_loss", best_loss))
            best_epoch = int(saved.get("best_epoch", 0))
            stale = int(saved.get("stale", 0))
            start_epoch = int(saved.get("epoch", 0)) + 1
    elif resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("model_state") is not None and saved.get("optimizer_state") is not None:
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            scaler.load_state_dict(saved.get("scaler_state", {}))
            history = list(saved.get("history", []))
            best_loss = float(saved.get("best_loss", best_loss))
            best_epoch = int(saved.get("best_epoch", int(saved.get("epoch", 0))))
            stale = int(saved.get("stale", 0))
            start_epoch = int(saved.get("epoch", 0)) + 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and USE_CUDA_AMP):
                loss = loss_fn(model(features), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(labels)
        model.eval()
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for features, _ in validation_loader:
                features = features.to(device, non_blocking=device.type == "cuda")
                probabilities.append(torch.sigmoid(model(features)).cpu().numpy())
        validation_probabilities = np.concatenate(probabilities)
        validation_loss = float(
            nn.functional.binary_cross_entropy(
                torch.from_numpy(np.clip(validation_probabilities, 1e-6, 1 - 1e-6)).float(),
                torch.from_numpy(validation_labels.astype(np.float32)),
            )
        )
        average_precision = float(average_precision_score(validation_labels, validation_probabilities))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_labels),
            "validation_loss": validation_loss,
            "average_precision": average_precision,
        })
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "history": history,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "epoch": epoch,
                    "positive_class_weight": negative / positive,
                },
                checkpoint,
            )
        else:
            stale += 1

        # Save latest checkpoint for separate resume state
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "history": history,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "stale": stale,
                "epoch": epoch,
                "positive_class_weight": negative / positive,
            },
            latest_checkpoint,
        )

        if stale >= 10:
            break

    # Restore best checkpoint
    if checkpoint.is_file():
        saved_best = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved_best["model_state"])

    model.eval()
    probabilities = []
    with torch.no_grad():
        for features, _ in validation_loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            probabilities.append(torch.sigmoid(model(features)).cpu().numpy())
    final_val_probs = np.concatenate(probabilities)

    val_meta = validation_metadata or [
        {"patient_id": "unknown", "recording_id": "unknown", "start_seconds": i * 20.0, "end_seconds": (i + 1) * 20.0, "label": int(validation_labels[i])}
        for i in range(len(validation_labels))
    ]
    gt_events = ground_truth_events or {}
    event_eval = per_patient_and_aggregate_event_evaluation(
        val_meta,
        final_val_probs,
        gt_events,
        threshold=0.5,
        recording_durations=recording_durations,
    )

    return {
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "event_evaluation": event_eval,
        "validation_probabilities": final_val_probs.astype(float).tolist(),
        "validation_metadata": list(val_meta),
    }



# def _fit_attention_final(
#     train_features: Path,
#     test_features: Path,
#     train_labels: np.ndarray,
#     test_labels: np.ndarray,
#     *,
#     device: torch.device,
#     checkpoint: Path,
#     epochs: int,
#     batch_size: int,
# ) -> dict[str, Any]:
def _fit_attention_final(
    train_features: Path,
    test_features: Path,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    *,
    device: torch.device,
    checkpoint: Path,
    epochs: int,
    batch_size: int,
    resume: bool = False,
    test_metadata: Sequence[Mapping[str, Any]] | None = None,
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]] | None = None,
    recording_durations: Mapping[tuple[str, str], float] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    train_loader = DataLoader(
        FeatureDataset(train_features, train_labels), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(FeatureDataset(test_features, test_labels), batch_size=batch_size)
    model = AttentionClassifier().to(device)
    positive = max(int(train_labels.sum()), 1)
    negative = max(len(train_labels) - positive, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and USE_CUDA_AMP)
    history: list[dict[str, float]] = []
    start_epoch = 1

    latest_checkpoint = checkpoint.parent / f"{checkpoint.stem}_latest.pt"

    if resume and latest_checkpoint.is_file():
        saved = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved.get("scaler_state", {}))
        if "rng_state" in saved and saved["rng_state"] is not None:
            torch.set_rng_state(saved["rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state" in saved:
                torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu())
        history = list(saved.get("history", []))
        start_epoch = int(saved.get("epoch", 0)) + 1
    elif resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved.get("scaler_state", {}))
        history = list(saved.get("history", []))
        start_epoch = int(saved.get("epoch", 0)) + 1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and USE_CUDA_AMP):
                loss = loss_fn(model(features), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(labels)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_labels)})

        # Save latest checkpoint
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "history": history,
                "epoch": epoch,
                "positive_class_weight": negative / positive,
            },
            latest_checkpoint,
        )

    # Save final best checkpoint
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "epoch": epochs,
            "positive_class_weight": negative / positive,
            "calibrated_threshold": float(threshold),
            "threshold_calibration_source": "development_patient_level_oof",
            "architecture": "four_head_additive_gated_residual_v1",
        },
        checkpoint,
    )

    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            probabilities.append(torch.sigmoid(model(features)).cpu().numpy())
    values = np.concatenate(probabilities)

    t_meta = test_metadata or [
        {"patient_id": "unknown", "recording_id": "unknown", "start_seconds": i * 20.0, "end_seconds": (i + 1) * 20.0, "label": int(test_labels[i])}
        for i in range(len(test_labels))
    ]
    gt_events = ground_truth_events or {}
    event_eval = per_patient_and_aggregate_event_evaluation(
        t_meta,
        values,
        gt_events,
        threshold=threshold,
        recording_durations=recording_durations,
    )

    return {
        "history": history,
        "checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "test_average_precision": float(average_precision_score(test_labels, values)),
        "test_roc_auc": float(roc_auc_score(test_labels, values)) if np.unique(test_labels).size == 2 else None,
        "calibrated_threshold": float(threshold),
        "event_evaluation": event_eval,
    }




# def _calibrate_threshold(
#     metadata: Sequence[Mapping[str, Any]],
#     probabilities: np.ndarray,
#     ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
#     recording_durations: Mapping[tuple[str, str], float],
# ) -> dict[str, Any]:
#     """Select one global operating threshold using development OOF predictions.

#     Priority: retain at least 90% event sensitivity where achievable; among
#     eligible thresholds maximize event precision, then minimize false alarms/hr.
#     If 90% is unattainable, maximize event sensitivity first. The independent
#     test set is never consulted by this function.
#     """
#     candidates = []
#     for threshold in THRESHOLD_GRID:
#         metrics = evaluate_events(
#             metadata, probabilities, ground_truth_events,
#             threshold=float(threshold), recording_durations=recording_durations,
#         )
#         agg = metrics
#         candidates.append({
#             "threshold": float(threshold),
#             "event_sensitivity": float(agg["event_sensitivity"]),
#             "event_precision": float(agg["event_precision"]),
#             "event_f1": float(agg["event_f1"]),
#             "false_alarms_per_hour": float(agg["false_alarms_per_hour"]),
#             "window_precision": float(agg["window_metrics"]["precision"]),
#             "window_recall": float(agg["window_metrics"]["recall"]),
#             "roc_auc": agg["window_metrics"].get("roc_auc"),
#             "average_precision": agg["window_metrics"].get("average_precision"),
#             "detected_events": int(agg["detected_events"]),
#             "actual_seizure_events": int(agg["actual_seizure_events"]),
#         })
#     feasible = [x for x in candidates if x["event_sensitivity"] >= 0.90]
#     if feasible:
#         # Sensitivity is constrained first; then prefer useful precision and fewer alarms.
#         selected = max(feasible, key=lambda x: (
#             x["event_precision"], -x["false_alarms_per_hour"],
#             x["window_precision"], x["event_f1"], x["threshold"],
#         ))
#         policy = "maximize_event_precision_subject_to_event_sensitivity_at_least_0.90"
#     else:
#         selected = max(candidates, key=lambda x: (
#             x["event_sensitivity"], x["event_precision"],
#             -x["false_alarms_per_hour"], x["event_f1"],
#         ))
#         policy = "90_percent_event_sensitivity_unattainable_on_oof; maximize_event_sensitivity_then_precision"
#     return {
#         "method": "global_threshold_from_patient_level_out_of_fold_probabilities",
#         "policy": policy,
#         "target_event_sensitivity": 0.90,
#         "selected": selected,
#         "candidate_count": len(candidates),
#         "candidates": candidates,
#         "warning": "OOF-to-final-model threshold transfer can shift because the final backbone/head is refit on all development patients. This is a development-calibrated operating point, not a clinical guarantee.",
#     }
def _calibrate_threshold(
    metadata: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
    recording_durations: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    """
    Precision-first global threshold calibration using development OOF data.

    Selection:
      1. Exclude candidates with zero detected seizure events.
      2. Maximize event precision.
      3. Minimize false alarms per hour.
      4. Prefer more detected events and higher event sensitivity as tie-breakers.

    Does not access independent test data or modify model weights.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if len(metadata) != len(probabilities):
        raise ValueError("Metadata and probability lengths do not match.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("OOF probabilities contain NaN or infinite values.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("OOF probabilities must be between 0 and 1.")

    represented_keys = {
        (str(m["patient_id"]), str(m["recording_id"]))
        for m in metadata
    }

    matched_gt = {
        key: list(events)
        for key, events in ground_truth_events.items()
        if key in represented_keys
    }

    actual_events = sum(len(events) for events in matched_gt.values())

    if actual_events == 0:
        raise ValueError(
            "Calibration stopped: no ground-truth seizure events match "
            "the attention OOF recordings. Check recording deserialization "
            "and patient/recording IDs."
        )

    # missing_durations = represented_keys - set(recording_durations)
    # invalid_durations = [
    # key for key in represented_keys
    # if not np.isfinite(recording_durations[key])
    # or recording_durations[key] <= 0
    # ]

    # if invalid_durations:
    #     raise ValueError(
    #         f"Calibration stopped: {len(invalid_durations)} OOF recordings "
    #         "have invalid or zero durations."
    #     )
    

    # if missing_durations:
    #     raise ValueError(
    #         f"Calibration stopped: {len(missing_durations)} OOF recordings "
    #         "have no recording duration. False alarms/hour would be unreliable."
    #     )
    missing_durations = represented_keys - set(recording_durations)

    if missing_durations:
        raise ValueError(
            f"Calibration stopped: {len(missing_durations)} OOF recordings "
            "have no recording duration. False alarms/hour would be unreliable."
        )   

    invalid_durations = [
        key for key in represented_keys
            if not np.isfinite(recording_durations[key])
            or recording_durations[key] <= 0
    ]

    if invalid_durations:
        raise ValueError(
            f"Calibration stopped: {len(invalid_durations)} OOF recordings "
            "have invalid or zero durations."
    )
    candidates = []

    for threshold in THRESHOLD_GRID:
        metrics = evaluate_events(
            metadata,
            probabilities,
            matched_gt,
            threshold=float(threshold),
            recording_durations=recording_durations,
        )

        candidates.append({
            "threshold": float(threshold),
            "event_sensitivity": float(metrics["event_sensitivity"]),
            "event_precision": float(metrics["event_precision"]),
            "event_f1": float(metrics["event_f1"]),
            "false_alarms_per_hour": float(
                metrics["false_alarms_per_hour"]
            ),
            "detected_events": int(metrics["detected_events"]),
            "actual_seizure_events": int(
                metrics["actual_seizure_events"]
            ),
            "predicted_alarm_episodes": int(
                metrics["predicted_alarm_episodes"]
            ),
            "false_alarm_episodes": int(
                metrics["false_alarm_episodes"]
            ),
            "mean_detection_latency_seconds": metrics.get(
                "mean_detection_latency_seconds"
            ),
            "median_detection_latency_seconds": metrics.get(
                "median_detection_latency_seconds"
            ),
            "window_precision": float(
                metrics["window_metrics"]["precision"]
            ),
            "window_recall": float(
                metrics["window_metrics"]["recall"]
            ),
        })

    # Do not reward a threshold that produces no true seizure detections.
    feasible = [
        item for item in candidates
        if item["detected_events"] > 0
        and item["predicted_alarm_episodes"] > 0
    ]

    if feasible:
        selected = max(
            feasible,
            key=lambda item: (
                item["event_precision"],
                -item["false_alarms_per_hour"],
                item["detected_events"],
                item["event_sensitivity"],
                item["threshold"],
            ),
        )
        policy = (
            "precision_first_with_at_least_one_detected_event"
        )
    else:
        # No threshold detected a seizure. Report this honestly rather
        # than presenting a zero-detection threshold as successful.
        selected = max(
            candidates,
            key=lambda item: (
                item["event_sensitivity"],
                item["detected_events"],
                item["event_precision"],
                -item["false_alarms_per_hour"],
            ),
        )
        policy = "no_threshold_detected_any_seizure_event"

    return {
        "method": "global_threshold_from_attention_development_oof",
        "policy": policy,
        "selection_priority": [
            "event_precision",
            "minimize_false_alarms_per_hour",
            "detected_events",
            "event_sensitivity",
        ],
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ground_truth_events_in_oof": actual_events,
        "oof_recordings": len(represented_keys),
        "recordings_with_duration": len(represented_keys),
        "warning": (
            "Development OOF calibration only. Threshold transfer to the "
            "final attention model may shift. This is not a clinical guarantee."
        ),
    }

def _load_oof_and_calibrate(
    ground_truth_events: Mapping[tuple[str, str], Sequence[tuple[float, float]]],
    recording_durations: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    if not OOF_PATH.is_file() or not OOF_METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Missing attention OOF predictions. Run backend.dl.attention_train --mode cv first."
        )
    with np.load(OOF_PATH) as data:
        probabilities = data["probabilities"].astype(np.float64)
        labels = data["labels"].astype(np.int64)
    metadata = json.loads(OOF_METADATA_PATH.read_text(encoding="utf-8"))
    if not ground_truth_events:
        raise ValueError(
            "Ground-truth events are empty. Refusing to calibrate attention."
        )

    if not recording_durations:
        raise ValueError(
            "Recording durations are empty. Refusing to calibrate attention."
        )
    if len(metadata) != len(probabilities) or len(labels) != len(probabilities):
        raise ValueError("OOF predictions, labels, and metadata have inconsistent lengths")
    if not np.array_equal(labels, np.asarray([int(m["label"]) for m in metadata])):
        raise ValueError("OOF labels do not match metadata labels")
    result = _calibrate_threshold(metadata, probabilities, ground_truth_events, recording_durations)
    result["oof_windows"] = int(len(labels))
    result["oof_positive_windows"] = int(labels.sum())
    result["oof_roc_auc"] = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None
    result["oof_average_precision"] = float(average_precision_score(labels, probabilities))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 four-head additive temporal attention")
    # parser.add_argument("--mode", choices=("cache", "verify-cache", "cv", "final"), default="cv")
    parser.add_argument(
    "--mode",
    choices=("cache", "verify-cache", "cv", "final", "calibrate-only"),
    default="cv",
    )
    parser.add_argument("--cache-dir", type=Path, default=DL_ARTIFACTS_DIR / "data_cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--verify-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS, choices=(FIXED_EPOCHS,), help="Final and CV attention training are fixed at five epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default=DEFAULT_DEVICE)
    parser.add_argument("--manifest", type=Path, default=PREPROCESSED_MANIFEST_PATH)
    args = parser.parse_args(argv)
    seed_everything()
    device = select_device(args.device)
    split = _load_split()
    if args.mode == "verify-cache" or args.verify_cache:
        print(json.dumps(verify_cache(args.cache_dir), indent=2))
        return 0
    cache_manifest = build_cache(
        args.manifest,
        args.cache_dir,
        canonical_channels=list(split["canonical_channels"]),
        rebuild=args.rebuild_cache,
    )
    if args.mode == "cache":
        print(json.dumps({"mode": "cache", **verify_cache(args.cache_dir)}, indent=2))
        return 0
    all_windows = load_manifest_windows(
        args.manifest,
        canonical_channels=split["canonical_channels"],
        quarantine_report_path=DL_ARTIFACTS_DIR / "audit_report.json",
    )
    cached_paths = {
        entry["source_path"]: entry["cache_path"]
        for entry in cache_manifest["entries"]
    }
    all_windows = [
        replace(
            window,
            path=cached_paths.get(window.path, window.path),
            channel_indices=tuple(range(len(split["canonical_channels"])))
            if window.path in cached_paths
            else window.channel_indices,
        )
        for window in all_windows
    ]
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
    if args.mode == "calibrate-only":
        calibration = _load_oof_and_calibrate(
            ground_truth_events,
            recording_durations,
        )

        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)

        CALIBRATION_PATH.write_text(
            json.dumps(calibration, indent=2, default=float),
            encoding="utf-8",
        )

        print("\nATTENTION THRESHOLD CALIBRATION")
        print(json.dumps(calibration, indent=2, default=float))
        print(f"\nSaved calibration to: {CALIBRATION_PATH}")

        return 0
    if args.mode == "final":
        baseline_checkpoint = DL_ARTIFACTS_DIR / "checkpoints" / "final.pt"
        if not baseline_checkpoint.is_file():
            # Check separated directory
            alt_baseline = CNN_BILSTM_CHECKPOINT_DIR / "final.pt"
            if alt_baseline.is_file():
                baseline_checkpoint = alt_baseline
            else:
                raise FileNotFoundError("run `backend.dl.train --mode final` before attention final training")
        previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.is_file() else {}
        fold_best_epochs = [int(item.get("best_epoch", FIXED_EPOCHS)) for item in previous.get("folds", [])]
        selected_epochs = FIXED_EPOCHS
        calibration = _load_oof_and_calibrate(ground_truth_events, recording_durations)
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2, default=float), encoding="utf-8")
        selected_threshold = float(calibration["selected"]["threshold"])
        train_windows = [w for p in split["dev_patients"] for w in by_patient[p]]
        test_windows = [w for p in split["test_patients"] for w in by_patient[p]]
        mean, std = fit_channel_normalization(train_windows)
        output_dir = FEATURE_DIR / "final"
        train_path, _, ordered_train = extract_features(
            baseline_checkpoint, train_windows,
            canonical_channels=split["canonical_channels"], mean=mean, std=std,
            output_dir=output_dir, split_name="train", fold=0, device=device,
            batch_size=args.batch_size,
        )
        test_path, _, ordered_test = extract_features(
            baseline_checkpoint, test_windows,
            canonical_channels=split["canonical_channels"], mean=mean, std=std,
            output_dir=output_dir, split_name="test", fold=0, device=device,
            batch_size=args.batch_size,
        )
        test_metadata = [
            {
                "patient_id": w.patient_id,
                "recording_id": w.recording_id,
                "start_seconds": w.start_seconds,
                "end_seconds": w.end_seconds,
                "label": w.label,
            }
            for w in ordered_test
        ]
        ATTENTION_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        result = _fit_attention_final(
            train_path, test_path,
            np.asarray([w.label for w in ordered_train]),
            np.asarray([w.label for w in ordered_test]),
            device=device, checkpoint=ATTENTION_CHECKPOINT_DIR / "final.pt",
            epochs=FIXED_EPOCHS, batch_size=args.batch_size, resume=False,
            test_metadata=test_metadata, threshold=selected_threshold,
            ground_truth_events=ground_truth_events,
            recording_durations=recording_durations,
        )
        result_payload = {
            "mode": "final",
            "device": str(device),
            "selected_epochs_strategy": "fixed_5_epochs",
            "cv_best_epochs_reference_only": fold_best_epochs,
            "selected_epochs": FIXED_EPOCHS,
            "threshold_calibration": calibration,
            "threshold_calibration_path": str(CALIBRATION_PATH),
            **result,
        }
        FINAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        FINAL_RESULTS_PATH.write_text(json.dumps(result_payload, indent=2, default=float), encoding="utf-8")

        # Generate final comparison if CNN-BiLSTM final results exist
        bilstm_final_source = CNN_BILSTM_FINAL_RESULTS_PATH if CNN_BILSTM_FINAL_RESULTS_PATH.is_file() else DL_ARTIFACTS_DIR / "final_results.json"
        if bilstm_final_source.is_file():
            bilstm_final_data = json.loads(bilstm_final_source.read_text(encoding="utf-8"))
            base_metrics = bilstm_final_data.get("independent_test", {}).get("event_evaluation", {})
            attn_metrics = result.get("event_evaluation", {})
            comparison = compute_comparison(base_metrics, attn_metrics, comparison_name="Final Independent Test")
            save_comparison_json(comparison, FINAL_COMPARISON_PATH)

        print(json.dumps(result_payload, indent=2, default=float))
        return 0
    cache: NormalizationCache = {}
    results = []
    for fold in split["folds"]:
        fold_id = int(fold["fold"])
        train_windows = [w for p in fold["training_patients"] for w in by_patient[p]]
        validation_windows = [w for p in fold["validation_patients"] for w in by_patient[p]]
        mean, std = fit_channel_normalization(train_windows, cache=cache)
        stage1_checkpoint = DL_ARTIFACTS_DIR / "checkpoints" / f"fold_{fold_id}.pt"
        if not stage1_checkpoint.is_file():
            alt_s1 = CNN_BILSTM_CHECKPOINT_DIR / f"fold_{fold_id}.pt"
            if alt_s1.is_file():
                stage1_checkpoint = alt_s1
            else:
                raise FileNotFoundError(f"missing Stage 1 checkpoint: {stage1_checkpoint}")
        fold_dir = FEATURE_DIR / f"fold_{fold_id}"
        train_path, _, ordered_train = extract_features(stage1_checkpoint, train_windows, canonical_channels=split["canonical_channels"], mean=mean, std=std, output_dir=fold_dir, split_name="train", fold=fold_id, device=device, batch_size=args.batch_size)
        validation_path, _, ordered_validation = extract_features(stage1_checkpoint, validation_windows, canonical_channels=split["canonical_channels"], mean=mean, std=std, output_dir=fold_dir, split_name="validation", fold=fold_id, device=device, batch_size=args.batch_size)
        checkpoint = ATTENTION_CHECKPOINT_DIR / f"fold_{fold_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        val_metadata = [
            {
                "patient_id": w.patient_id,
                "recording_id": w.recording_id,
                "start_seconds": w.start_seconds,
                "end_seconds": w.end_seconds,
                "label": w.label,
            }
            for w in ordered_validation
        ]
        results.append(_fit_attention(
            train_path, validation_path,
            np.asarray([w.label for w in ordered_train]),
            np.asarray([w.label for w in ordered_validation]),
            device=device, checkpoint=checkpoint,
            epochs=args.epochs, batch_size=args.batch_size,
            validation_metadata=val_metadata,
            ground_truth_events=ground_truth_events,
            recording_durations=recording_durations,
        ))
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_probabilities: list[float] = []
    oof_labels: list[int] = []
    oof_metadata: list[dict[str, Any]] = []
    for fold_result in results:
        probs = fold_result.pop("validation_probabilities", [])
        metas = fold_result.pop("validation_metadata", [])
        if len(probs) != len(metas):
            raise RuntimeError("fold OOF probability/metadata mismatch")
        oof_probabilities.extend(float(x) for x in probs)
        oof_metadata.extend(dict(x) for x in metas)
        oof_labels.extend(int(x["label"]) for x in metas)
    if not oof_probabilities:
        raise RuntimeError("No OOF predictions were collected; threshold calibration cannot proceed")
    np.savez_compressed(OOF_PATH, probabilities=np.asarray(oof_probabilities, dtype=np.float32), labels=np.asarray(oof_labels, dtype=np.int8))
    OOF_METADATA_PATH.write_text(json.dumps(oof_metadata, indent=2), encoding="utf-8")
    cv_payload = {
        "mode": "cv", "device": str(device), "fixed_epochs": FIXED_EPOCHS,
        "oof_predictions_path": str(OOF_PATH), "oof_metadata_path": str(OOF_METADATA_PATH),
        "folds": results,
    }
    RESULTS_PATH.write_text(json.dumps(cv_payload, indent=2, default=float), encoding="utf-8")

    # Generate CV comparison if CNN-BiLSTM CV results exist
    bilstm_cv_source = CNN_BILSTM_CV_RESULTS_PATH if CNN_BILSTM_CV_RESULTS_PATH.is_file() else DL_ARTIFACTS_DIR / "training_results.json"
    if bilstm_cv_source.is_file():
        bilstm_cv_data = json.loads(bilstm_cv_source.read_text(encoding="utf-8"))
        # Aggregate comparison across folds if available
        base_fold_events = [f.get("event_evaluation", {}).get("aggregate", {}) for f in bilstm_cv_data.get("folds", []) if f.get("event_evaluation")]
        attn_fold_events = [f.get("event_evaluation", {}).get("aggregate", {}) for f in results if f.get("event_evaluation")]
        if base_fold_events and attn_fold_events:
            # Average metrics across folds
            def avg_dict(dicts):
                keys = ["event_sensitivity", "event_precision", "event_f1", "false_alarms_per_hour"]
                res = {}
                for k in keys:
                    vals = [d[k] for d in dicts if k in d and d[k] is not None]
                    res[k] = float(np.mean(vals)) if vals else None
                # Add window metrics
                w_keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision"]
                w_res = {}
                for wk in w_keys:
                    w_vals = [d.get("window_metrics", {}).get(wk) for d in dicts if d.get("window_metrics", {}).get(wk) is not None]
                    w_res[wk] = float(np.mean(w_vals)) if w_vals else None
                res["window_metrics"] = w_res
                return res

            cv_comp = compute_comparison(avg_dict(base_fold_events), avg_dict(attn_fold_events), comparison_name="Cross-Validation Average")
            save_comparison_json(cv_comp, CV_COMPARISON_PATH)

    print(json.dumps(cv_payload, indent=2, default=float))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
