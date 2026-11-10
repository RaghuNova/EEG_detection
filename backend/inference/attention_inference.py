"""attention_inference.py
------------------------
Model 3: Existing AttentionClassifier with Hybrid Dynamic Calibration.

Requirements:
- Loads checkpoint from models/final.pt.zip.
- Instantiates AttentionClassifier(input_dim=128) from dl/attention.py.
- STRICT RULE: Takes intermediate temporal features [batch, 128, 128] from CNN-BiLSTM.
  Never receives binary predictions.
- Computes logits and 4-head attention weights [batch, 4, 128] via forward_with_attention().
- Computes raw sigmoid probabilities.
- Loads calibration policy from dependencies/hybrid_attention_threshold_calibration.json.
- Computes RECORDING-SPECIFIC dynamic threshold using dl.dynamic_threshold.apply_dynamic_threshold().
- Preserves attention weights, calibrated thresholds, predictions, and detected event episodes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dl.attention import AttentionClassifier
from dl.dynamic_threshold import apply_dynamic_threshold
from backend.inference.preprocessing_wrapper import ProcessedRecording, WindowInfo
from backend.inference.ml_inference import SeizureEvent

LOGGER = logging.getLogger("neuroscan.attention_inference")


@dataclass
class AttentionThresholdInfo:
    global_threshold: float
    quantile: float
    floor: float
    adaptive_blend: float
    quantile_value: float
    calibrated_threshold: float
    method: str


@dataclass
class AttentionInferenceResult:
    model_name: str
    architecture: str
    recording_threshold_info: AttentionThresholdInfo
    window_probabilities: List[float]
    window_thresholds: List[float]
    window_predictions: List[int]
    persistent_window_predictions: List[int]
    positive_window_count: int
    total_windows: int
    events: List[SeizureEvent]
    attention_weights_summary: List[List[float]]  # [n_windows, 128] mean over 4 heads
    execution_time_seconds: float
    status: str
    error: Optional[str] = None


class AttentionModelAdapter:
    """Singleton inference adapter for Model 3 (AttentionClassifier)."""

    def __init__(
        self,
        checkpoint_path: Path | str = "models/final.pt.zip",
        calibration_path: Path | str = "dependencies/hybrid_attention_threshold_calibration.json",
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.calibration_path = Path(calibration_path).resolve()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Attention checkpoint not found: {self.checkpoint_path}")
        if not self.calibration_path.is_file():
            raise FileNotFoundError(f"Calibration JSON not found: {self.calibration_path}")

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model: Optional[AttentionClassifier] = None
        self.calibration_config: Dict[str, Any] = {}
        self.architecture: str = "four_head_additive_gated_residual_v1"
        self._load_resources()

    def _load_resources(self) -> None:
        """Load checkpoint weights and calibration JSON."""
        LOGGER.info("Loading AttentionClassifier from %s", self.checkpoint_path)
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        self.architecture = str(ckpt.get("architecture", "four_head_additive_gated_residual_v1"))

        model = AttentionClassifier(
            input_dim=128,
            num_heads=4,
            head_dim=32,
            attention_dropout=0.15,
            classifier_dropout=0.30,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        self.model = model

        # Load hybrid dynamic calibration JSON
        with open(self.calibration_path, "r", encoding="utf-8") as f:
            self.calibration_config = json.load(f)
        LOGGER.info("AttentionClassifier loaded with dynamic calibration: %s", self.calibration_config.get("method"))

    def run_inference(
        self,
        temporal_features: torch.Tensor,
        processed: ProcessedRecording,
    ) -> AttentionInferenceResult:
        """Run Attention inference given CNN-BiLSTM temporal features and recording metadata."""
        assert self.model is not None, "Model is not loaded"
        start_time = time.time()

        n_windows = len(processed.windows)
        if n_windows == 0 or temporal_features.shape[0] == 0:
            return AttentionInferenceResult(
                model_name="AttentionClassifier",
                architecture=self.architecture,
                recording_threshold_info=AttentionThresholdInfo(
                    global_threshold=0.03,
                    quantile=0.985,
                    floor=0.03,
                    adaptive_blend=1.0,
                    quantile_value=0.03,
                    calibrated_threshold=0.03,
                    method="hybrid_global_recording_quantile",
                ),
                window_probabilities=[],
                window_thresholds=[],
                window_predictions=[],
                persistent_window_predictions=[],
                positive_window_count=0,
                total_windows=0,
                events=[],
                attention_weights_summary=[],
                execution_time_seconds=0.0,
                status="completed_empty",
            )

        # 1. Forward pass with attention
        features_tensor = temporal_features.to(self.device)
        with torch.no_grad():
            logits, weights = self.model.forward_with_attention(features_tensor)
            probs_tensor = torch.sigmoid(logits)

        probs = [float(p) for p in probs_tensor.cpu().numpy().reshape(-1)]
        # Weights shape: [batch, num_heads=4, time=128] -> average over heads for summary
        mean_weights = weights.mean(dim=1).cpu().numpy().tolist()

        # 2. Dynamic Recording-Specific Threshold Calibration
        metadata_list = [
            {
                "patient_id": processed.patient_id,
                "recording_id": processed.recording_id,
                "start_seconds": w.start_seconds,
                "end_seconds": w.end_seconds,
            }
            for w in processed.window_infos
        ]

        calib_output = apply_dynamic_threshold(
            probabilities=probs,
            metadata=metadata_list,
            calibration=self.calibration_config,
            window_seconds=20.0,
        )

        window_thresholds = [round(float(t), 5) for t in calib_output["thresholds"]]
        raw_predictions = [int(p) for p in calib_output["window_predictions"]]
        persistent_predictions = [int(p) for p in calib_output["persistent_window_predictions"]]

        # Extract recording threshold summary
        rec_summary = calib_output["recording_thresholds"][0] if calib_output["recording_thresholds"] else {}
        threshold_info = AttentionThresholdInfo(
            global_threshold=float(rec_summary.get("global_threshold", 0.03)),
            quantile=float(rec_summary.get("quantile", 0.985)),
            floor=float(rec_summary.get("floor", 0.03)),
            adaptive_blend=float(rec_summary.get("adaptive_blend", 1.0)),
            quantile_value=float(rec_summary.get("quantile_value", 0.03)),
            calibrated_threshold=float(rec_summary.get("threshold", 0.03)),
            method=str(self.calibration_config.get("method", "hybrid_global_recording_quantile")),
        )

        # 3. Form events from calibration alarm episodes
        events: List[SeizureEvent] = []
        for idx, ep in enumerate(calib_output.get("alarm_episodes", [])):
            start_sec = float(ep["start_seconds"])
            end_sec = float(ep["end_seconds"])

            # Map start/end seconds to window indices
            start_win = 0
            end_win = 0
            for w_idx, win in enumerate(processed.window_infos):
                if abs(win.start_seconds - start_sec) < 1e-2:
                    start_win = w_idx
                if abs(win.end_seconds - end_sec) < 1e-2:
                    end_win = w_idx

            events.append(
                SeizureEvent(
                    event_id=f"ATTN_EVT_{idx+1:03d}",
                    start_seconds=round(start_sec, 2),
                    end_seconds=round(end_sec, 2),
                    duration_seconds=round(end_sec - start_sec, 2),
                    window_count=int(ep.get("window_count", 1)),
                    mean_probability=round(float(ep.get("mean_probability", 0.0)), 4),
                    max_probability=round(float(ep.get("max_probability", 0.0)), 4),
                    start_window_index=start_win,
                    end_window_index=end_win,
                )
            )

        positive_count = int(sum(persistent_predictions))
        exec_time = round(time.time() - start_time, 3)

        return AttentionInferenceResult(
            model_name="AttentionClassifier",
            architecture=self.architecture,
            recording_threshold_info=threshold_info,
            window_probabilities=[round(p, 5) for p in probs],
            window_thresholds=window_thresholds,
            window_predictions=raw_predictions,
            persistent_window_predictions=persistent_predictions,
            positive_window_count=positive_count,
            total_windows=n_windows,
            events=events,
            attention_weights_summary=mean_weights,
            execution_time_seconds=exec_time,
            status="success",
        )
