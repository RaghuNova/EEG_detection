"""ml_inference.py
-----------------
Model 1: Existing Tabular ML Model (XGBoost + Scaler) inference adapter.

Requirements:
- Reuses existing trained joblib bundle (models/seizure_ml_model.joblib).
- Extracts 122 channel-order-invariant statistical, spectral, and synchrony features.
- Applies the bundle's saved StandardScaler.
- Uses the bundle's saved decision threshold (0.27896...).
- Identifies seizure probability, binary predictions, and grouped seizure event intervals.
- Handles process isolation safely to avoid macOS OpenMP duplicate runtime collisions with PyTorch.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import mlprocess
from backend.inference.preprocessing_wrapper import ProcessedRecording, WindowInfo

LOGGER = logging.getLogger("neuroscan.ml_inference")


@dataclass
class SeizureEvent:
    event_id: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    window_count: int
    mean_probability: float
    max_probability: float
    start_window_index: int
    end_window_index: int


@dataclass
class MLInferenceResult:
    model_name: str
    model_version: str
    decision_threshold: float
    window_probabilities: List[float]
    window_predictions: List[int]
    positive_window_count: int
    total_windows: int
    events: List[SeizureEvent]
    feature_count: int
    execution_time_seconds: float
    status: str
    error: Optional[str] = None


def group_positive_windows_to_events(
    probabilities: List[float],
    predictions: List[int],
    window_infos: List[WindowInfo],
    min_consecutive: int = 1,
    max_gap_seconds: float = 20.0,
) -> List[SeizureEvent]:
    """Group consecutive or near-consecutive positive windows into coherent seizure events."""
    events: List[SeizureEvent] = []
    if not window_infos or len(probabilities) != len(window_infos):
        return events

    runs: List[List[int]] = []
    current_run: List[int] = []

    for i, pred in enumerate(predictions):
        if pred == 1:
            current_run.append(i)
        else:
            if len(current_run) >= min_consecutive:
                runs.append(current_run)
            current_run = []

    if len(current_run) >= min_consecutive:
        runs.append(current_run)

    if not runs:
        return events

    # Merge runs with gap <= max_gap_seconds
    merged_runs: List[List[int]] = [runs[0]]
    for next_run in runs[1:]:
        prev_run = merged_runs[-1]
        prev_end_time = window_infos[prev_run[-1]].end_seconds
        next_start_time = window_infos[next_run[0]].start_seconds
        gap = next_start_time - prev_end_time

        if gap <= max_gap_seconds + 1e-3:
            prev_run.extend(next_run)
        else:
            merged_runs.append(next_run)

    for idx, run_indices in enumerate(merged_runs):
        first_idx = run_indices[0]
        last_idx = run_indices[-1]
        start_sec = window_infos[first_idx].start_seconds
        end_sec = window_infos[last_idx].end_seconds
        run_probs = [probabilities[i] for i in run_indices]

        events.append(
            SeizureEvent(
                event_id=f"ML_EVT_{idx+1:03d}",
                start_seconds=round(start_sec, 2),
                end_seconds=round(end_sec, 2),
                duration_seconds=round(end_sec - start_sec, 2),
                window_count=len(run_indices),
                mean_probability=round(float(np.mean(run_probs)), 4),
                max_probability=round(float(np.max(run_probs)), 4),
                start_window_index=first_idx,
                end_window_index=last_idx,
            )
        )

    return events


def _run_ml_inference_in_subprocess(
    model_path: str,
    feature_matrix: np.ndarray,
) -> Tuple[List[float], List[int], float, str]:
    """Execute XGBoost prediction in a clean subprocess to completely isolate OpenMP runtimes."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_in:
        input_file = temp_in.name
        np.savez_compressed(input_file, X=feature_matrix)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_out:
        output_file = temp_out.name

    worker_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, '{Path('.').resolve()}')
import mlout, numpy as np, json

bundle = mlout.load_model_bundle('{model_path}')
data = np.load('{input_file}')
X = data['X']

if bundle.scaler is not None:
    X_scaled = bundle.scaler.transform(X).astype(np.float32)
else:
    X_scaled = X.astype(np.float32)

preds, probs = mlout.predict_feature_matrix(bundle, X_scaled)

result = {{
    "probabilities": probs.tolist(),
    "predictions": preds.tolist(),
    "decision_threshold": float(bundle.decision_threshold),
    "model_version": str(bundle.model_version),
}}

with open('{output_file}', 'w') as f:
    json.dump(result, f)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        with open(output_file, "r") as f:
            res = json.load(f)
        return (
            res["probabilities"],
            res["predictions"],
            res["decision_threshold"],
            res["model_version"],
        )
    finally:
        Path(input_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def _query_bundle_metadata(model_path: str) -> Dict[str, Any]:
    """Extract model metadata in an isolated subprocess to protect against OpenMP collision."""
    worker_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, '{Path('.').resolve()}')
import mlout, json

bundle = mlout.load_model_bundle('{model_path}')
result = {{
    "model_version": str(bundle.model_version),
    "feature_count": len(bundle.feature_names),
    "feature_names": bundle.feature_names,
    "decision_threshold": float(bundle.decision_threshold),
    "sampling_rate_hz": float(bundle.sampling_rate_hz),
    "window_seconds": float(bundle.window_seconds),
}}
print("___META_START___" + json.dumps(result) + "___META_END___")
"""
    out = subprocess.check_output([sys.executable, "-c", worker_code], text=True)
    raw_json = out.split("___META_START___")[1].split("___META_END___")[0]
    return json.loads(raw_json)


class MLModelAdapter:
    """Singleton inference adapter for Model 1 (XGBoost baseline)."""

    def __init__(self, model_path: Path | str = "models/seizure_ml_model.joblib") -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ML model artifact not found: {self.model_path}")
        self._bundle_info: Dict[str, Any] = _query_bundle_metadata(str(self.model_path))

    @property
    def decision_threshold(self) -> float:
        assert self._bundle_info is not None
        return self._bundle_info["decision_threshold"]

    @property
    def model_version(self) -> str:
        assert self._bundle_info is not None
        return self._bundle_info["model_version"]

    def run_inference(self, processed: ProcessedRecording) -> MLInferenceResult:
        """Run ML feature extraction and inference for all 20-second windows in a recording."""
        import time
        start_time = time.time()

        if not processed.windows:
            return MLInferenceResult(
                model_name="XGBoost Classifier",
                model_version=self.model_version,
                decision_threshold=self.decision_threshold,
                window_probabilities=[],
                window_predictions=[],
                positive_window_count=0,
                total_windows=0,
                events=[],
                feature_count=122,
                execution_time_seconds=0.0,
                status="completed_empty",
            )

        # 1. Extract 122 invariant features for every window using mlprocess
        X, _, _, feature_names = mlprocess.windows_to_features(processed.windows)

        # 2. Run isolated XGBoost inference
        probs, preds, threshold, version = _run_ml_inference_in_subprocess(
            str(self.model_path), X
        )

        # 3. Detect coherent event intervals
        events = group_positive_windows_to_events(probs, preds, processed.window_infos)
        positive_count = int(sum(preds))
        exec_time = round(time.time() - start_time, 3)

        return MLInferenceResult(
            model_name="XGBoost Classifier",
            model_version=version,
            decision_threshold=round(threshold, 5),
            window_probabilities=[round(p, 5) for p in probs],
            window_predictions=preds,
            positive_window_count=positive_count,
            total_windows=len(preds),
            events=events,
            feature_count=len(feature_names),
            execution_time_seconds=exec_time,
            status="success",
        )
