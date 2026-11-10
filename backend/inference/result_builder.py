"""result_builder.py
------------------
Standardized result schema constructor according to NEUROSCAN specifications.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.inference.preprocessing_wrapper import ProcessedRecording
from backend.inference.ml_inference import MLInferenceResult, SeizureEvent
from backend.inference.cnn_inference import CNNInferenceResult
from backend.inference.attention_inference import AttentionInferenceResult


def event_to_dict(event: SeizureEvent) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "start_seconds": event.start_seconds,
        "end_seconds": event.end_seconds,
        "duration_seconds": event.duration_seconds,
        "window_count": event.window_count,
        "mean_probability": event.mean_probability,
        "max_probability": event.max_probability,
        "start_window_index": event.start_window_index,
        "end_window_index": event.end_window_index,
    }


def build_recording_result(
    processed: ProcessedRecording,
    ml_res: MLInferenceResult,
    cnn_res: CNNInferenceResult,
    attn_res: AttentionInferenceResult,
    processing_duration: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct consistent dictionary for one EDF recording."""
    status = "failed" if error else "completed"

    # Merge window-level predictions with timestamps
    windows_data: List[Dict[str, Any]] = []
    n_windows = len(processed.window_infos)

    for i in range(n_windows):
        w_info = processed.window_infos[i]
        ml_prob = ml_res.window_probabilities[i] if i < len(ml_res.window_probabilities) else 0.0
        ml_pred = ml_res.window_predictions[i] if i < len(ml_res.window_predictions) else 0

        cnn_prob = cnn_res.window_probabilities[i] if i < len(cnn_res.window_probabilities) else 0.0
        cnn_pred = cnn_res.window_predictions[i] if i < len(cnn_res.window_predictions) else 0

        attn_prob = attn_res.window_probabilities[i] if i < len(attn_res.window_probabilities) else 0.0
        attn_th = attn_res.window_thresholds[i] if i < len(attn_res.window_thresholds) else 0.03
        attn_pred = attn_res.persistent_window_predictions[i] if i < len(attn_res.persistent_window_predictions) else 0

        windows_data.append(
            {
                "window_index": w_info.window_index,
                "window_id": w_info.window_id,
                "start_seconds": w_info.start_seconds,
                "end_seconds": w_info.end_seconds,
                "ml": {
                    "probability": ml_prob,
                    "prediction": ml_pred,
                },
                "cnn_bilstm": {
                    "probability": cnn_prob,
                    "prediction": cnn_pred,
                },
                "attention": {
                    "probability": attn_prob,
                    "threshold": attn_th,
                    "prediction": attn_pred,
                },
            }
        )

    return {
        "recording_id": processed.recording_id,
        "filename": Path(processed.file_path).name,
        "file_path": processed.file_path,
        "status": status,
        "sampling_frequency": processed.sampling_rate_hz,
        "channel_count": len(processed.channel_names),
        "channel_names": processed.channel_names,
        "duration_seconds": processed.duration_seconds,
        "window_seconds": 20.0,
        "total_windows": n_windows,
        "windows": windows_data,
        "models": {
            "ml": {
                "model_name": ml_res.model_name,
                "model_version": ml_res.model_version,
                "decision_threshold": ml_res.decision_threshold,
                "positive_window_count": ml_res.positive_window_count,
                "probabilities": ml_res.window_probabilities,
                "predictions": ml_res.window_predictions,
                "events": [event_to_dict(e) for e in ml_res.events],
                "execution_time_seconds": ml_res.execution_time_seconds,
                "status": ml_res.status,
            },
            "cnn_bilstm": {
                "model_name": cnn_res.model_name,
                "decision_threshold": cnn_res.decision_threshold,
                "positive_window_count": cnn_res.positive_window_count,
                "probabilities": cnn_res.window_probabilities,
                "predictions": cnn_res.window_predictions,
                "events": [event_to_dict(e) for e in cnn_res.events],
                "execution_time_seconds": cnn_res.execution_time_seconds,
                "status": cnn_res.status,
            },
            "attention": {
                "model_name": attn_res.model_name,
                "architecture": attn_res.architecture,
                "threshold_info": asdict(attn_res.recording_threshold_info),
                "calibrated_threshold": attn_res.recording_threshold_info.calibrated_threshold,
                "positive_window_count": attn_res.positive_window_count,
                "probabilities": attn_res.window_probabilities,
                "thresholds": attn_res.window_thresholds,
                "predictions": attn_res.window_predictions,
                "persistent_predictions": attn_res.persistent_window_predictions,
                "events": [event_to_dict(e) for e in attn_res.events],
                "execution_time_seconds": attn_res.execution_time_seconds,
                "status": attn_res.status,
            },
        },
        "processing": {
            "duration_seconds": round(processing_duration, 3),
            "error": error,
        },
        "preview_signal": processed.raw_signal_sample,
    }


def build_failed_recording_result(
    file_path: str,
    recording_id: str,
    error_message: str,
    processing_duration: float = 0.0,
) -> Dict[str, Any]:
    """Construct result structure for an unreadable or corrupt EDF recording."""
    path = Path(file_path)
    return {
        "recording_id": recording_id or path.stem,
        "filename": path.name,
        "file_path": str(path.resolve()),
        "status": "failed",
        "sampling_frequency": 0.0,
        "channel_count": 0,
        "channel_names": [],
        "duration_seconds": 0.0,
        "window_seconds": 20.0,
        "total_windows": 0,
        "windows": [],
        "models": {
            "ml": {"status": "failed", "error": error_message},
            "cnn_bilstm": {"status": "failed", "error": error_message},
            "attention": {"status": "failed", "error": error_message},
        },
        "processing": {
            "duration_seconds": round(processing_duration, 3),
            "error": error_message,
        },
        "preview_signal": None,
    }


def build_patient_summary(
    patient_id: str,
    analysis_id: str,
    recordings_results: List[Dict[str, Any]],
    total_execution_time: float,
) -> Dict[str, Any]:
    """Aggregate all recording results into a comprehensive patient analysis record."""
    successful = [r for r in recordings_results if r["status"] == "completed"]
    failed = [r for r in recordings_results if r["status"] == "failed"]

    total_duration = sum(r.get("duration_seconds", 0.0) for r in successful)
    total_windows = sum(r.get("total_windows", 0) for r in successful)

    ml_pos = sum(r["models"]["ml"].get("positive_window_count", 0) for r in successful if "models" in r and "ml" in r["models"])
    cnn_pos = sum(r["models"]["cnn_bilstm"].get("positive_window_count", 0) for r in successful if "models" in r and "cnn_bilstm" in r["models"])
    attn_pos = sum(r["models"]["attention"].get("positive_window_count", 0) for r in successful if "models" in r and "attention" in r["models"])

    ml_events = sum(len(r["models"]["ml"].get("events", [])) for r in successful if "models" in r and "ml" in r["models"])
    cnn_events = sum(len(r["models"]["cnn_bilstm"].get("events", [])) for r in successful if "models" in r and "cnn_bilstm" in r["models"])
    attn_events = sum(len(r["models"]["attention"].get("events", [])) for r in successful if "models" in r and "attention" in r["models"])

    return {
        "patient_id": patient_id,
        "analysis_id": analysis_id,
        "status": "completed" if successful else "failed",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_recordings": len(recordings_results),
        "successful_recordings": len(successful),
        "failed_recordings": len(failed),
        "total_duration_seconds": round(total_duration, 2),
        "total_windows": total_windows,
        "execution_time_seconds": round(total_execution_time, 3),
        "summary": {
            "ml_positive_windows": ml_pos,
            "cnn_positive_windows": cnn_pos,
            "attention_positive_windows": attn_pos,
            "ml_total_events": ml_events,
            "cnn_total_events": cnn_events,
            "attention_total_events": attn_events,
        },
        "recordings": recordings_results,
    }
