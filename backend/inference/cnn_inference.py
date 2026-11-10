"""cnn_inference.py
------------------
Model 2: Existing CNN-BiLSTM raw EEG inference adapter.

Requirements:
- Loads checkpoint from models/final.pt (1).zip.
- Instantiates CNNBiLSTM(input_channels=22) from dl/model.py.
- Formats input as [batch, 22, 5120] using 22 canonical channels.
- Normalizes signals using checkpoint's channel-wise normalization_mean and normalization_std.
- Runs extract_temporal_features() to extract recurrent representations [batch, 128, 128].
- Runs model forward pass to compute raw logits and sigmoid probabilities.
- Applies existing CNN decision threshold (0.50).
- Groups positive windows into detected event intervals.
- STRICT RULE: Intermediate temporal features [batch, 128, 128] are preserved for AttentionClassifier.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dl.model import CNNBiLSTM
from backend.inference.preprocessing_wrapper import (
    CANONICAL_CHANNELS,
    ProcessedRecording,
    WindowInfo,
    WINDOW_SAMPLES,
)
from backend.inference.ml_inference import SeizureEvent, group_positive_windows_to_events

LOGGER = logging.getLogger("neuroscan.cnn_inference")


@dataclass
class CNNInferenceResult:
    model_name: str
    decision_threshold: float
    window_probabilities: List[float]
    window_predictions: List[int]
    positive_window_count: int
    total_windows: int
    events: List[SeizureEvent]
    temporal_features: torch.Tensor  # Shape: [batch, 128, 128]
    execution_time_seconds: float
    status: str
    error: Optional[str] = None


class CNNBiLSTMAdapter:
    """Singleton inference adapter for Model 2 (CNN-BiLSTM)."""

    def __init__(
        self,
        checkpoint_path: Path | str = "models/final.pt (1).zip",
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"CNN-BiLSTM checkpoint not found: {self.checkpoint_path}")

        # Determine best available compute device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # On macOS Apple Silicon, PyTorch MPS has an open issue (PyTorch #96056)
            # with non-divisible adaptive_avg_pool1d; CPU is fast, stable, and accurate.
            self.device = torch.device("cpu")

        self.threshold: float = 0.50
        self.model: Optional[CNNBiLSTM] = None
        self.canonical_channels: List[str] = []
        self.norm_mean: np.ndarray = np.array([])
        self.norm_std: np.ndarray = np.array([])
        self._load_model()

    def _load_model(self) -> None:
        """Load checkpoint and initialize CNNBiLSTM."""
        LOGGER.info("Loading CNN-BiLSTM checkpoint from %s on device %s", self.checkpoint_path, self.device)
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)

        self.canonical_channels = list(ckpt.get("canonical_channels", CANONICAL_CHANNELS))
        self.norm_mean = np.asarray(ckpt["normalization_mean"], dtype=np.float32)
        self.norm_std = np.asarray(ckpt["normalization_std"], dtype=np.float32)

        model = CNNBiLSTM(
            input_channels=len(self.canonical_channels),
            lstm_hidden=64,
            dropout=0.30,
            recurrent_length=128,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        self.model = model
        LOGGER.info("CNN-BiLSTM model loaded successfully with %d channels.", len(self.canonical_channels))

    def run_inference(self, processed: ProcessedRecording) -> CNNInferenceResult:
        """Run CNN-BiLSTM inference on all 20-second windows."""
        assert self.model is not None, "Model is not loaded"
        start_time = time.time()

        n_windows = len(processed.windows)
        if n_windows == 0:
            return CNNInferenceResult(
                model_name="CNN-BiLSTM",
                decision_threshold=self.threshold,
                window_probabilities=[],
                window_predictions=[],
                positive_window_count=0,
                total_windows=0,
                events=[],
                temporal_features=torch.empty((0, 128, 128)),
                execution_time_seconds=0.0,
                status="completed_empty",
            )

        # 1. Prepare normalized window tensors [n_windows, 22, 5120]
        window_tensors: List[np.ndarray] = []
        for i in range(n_windows):
            start_idx = i * WINDOW_SAMPLES
            end_idx = start_idx + WINDOW_SAMPLES
            chunk = processed.canonical_matrix[:, start_idx:end_idx]

            # In case the last window is slightly short, pad with zeros
            if chunk.shape[1] < WINDOW_SAMPLES:
                padded = np.zeros((chunk.shape[0], WINDOW_SAMPLES), dtype=np.float32)
                padded[:, :chunk.shape[1]] = chunk
                chunk = padded

            normed = (chunk.astype(np.float32) - self.norm_mean[:, None]) / np.maximum(self.norm_std[:, None], 1e-6)
            window_tensors.append(normed)

        batch_array = np.stack(window_tensors, axis=0)  # Shape: [batch, 22, 5120]
        batch_tensor = torch.from_numpy(batch_array).float().to(self.device)

        # 2. Forward pass with temporal feature extraction
        with torch.no_grad():
            # Intermediate temporal features [batch, 128, 128]
            temporal_features = self.model.extract_temporal_features(batch_tensor)
            # Logits [batch]
            logits = self.model(batch_tensor)
            # Probabilities in [0, 1]
            probs_tensor = torch.sigmoid(logits)

        probabilities = [round(float(p), 5) for p in probs_tensor.cpu().numpy().reshape(-1)]
        predictions = [1 if p >= self.threshold else 0 for p in probabilities]

        # 3. Detect event intervals
        events = group_positive_windows_to_events(probabilities, predictions, processed.window_infos)
        positive_count = int(sum(predictions))
        exec_time = round(time.time() - start_time, 3)

        return CNNInferenceResult(
            model_name="CNN-BiLSTM",
            decision_threshold=self.threshold,
            window_probabilities=probabilities,
            window_predictions=predictions,
            positive_window_count=positive_count,
            total_windows=n_windows,
            events=events,
            temporal_features=temporal_features.cpu(),
            execution_time_seconds=exec_time,
            status="success",
        )
