"""inference_orchestrator.py
--------------------------
Central orchestrator for multi-model EEG inference across patient folders.

Coordinates:
1. Discovery of all EDF recordings in a patient folder.
2. Independent loading and preprocessing of each EDF.
3. Model 1 (ML XGBoost), Model 2 (CNN-BiLSTM), and Model 3 (AttentionClassifier) inference.
4. Intermediate temporal feature pass from CNN-BiLSTM to AttentionClassifier.
5. Dynamic recording-specific calibration.
6. Progress tracking callbacks for background task processing.
7. Independent failure isolation per recording.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.inference.preprocessing_wrapper import (
    preprocess_recording_for_inference,
    get_edf_metadata,
)
from backend.inference.ml_inference import MLModelAdapter
from backend.inference.cnn_inference import CNNBiLSTMAdapter
from backend.inference.attention_inference import AttentionModelAdapter
from backend.inference.result_builder import (
    build_recording_result,
    build_failed_recording_result,
    build_patient_summary,
)

LOGGER = logging.getLogger("neuroscan.inference_orchestrator")


class InferenceOrchestrator:
    """Manages loaded model lifecycles and coordinates patient analysis jobs."""

    def __init__(
        self,
        ml_model_path: Path | str = "models/seizure_ml_model.joblib",
        cnn_checkpoint_path: Path | str = "models/final.pt (1).zip",
        attention_checkpoint_path: Path | str = "models/final.pt.zip",
        calibration_path: Path | str = "dependencies/hybrid_attention_threshold_calibration.json",
        device: Optional[str] = None,
    ) -> None:
        self.ml_model_path = Path(ml_model_path).resolve()
        self.cnn_checkpoint_path = Path(cnn_checkpoint_path).resolve()
        self.attention_checkpoint_path = Path(attention_checkpoint_path).resolve()
        self.calibration_path = Path(calibration_path).resolve()
        self.device = device

        self.ml_adapter: Optional[MLModelAdapter] = None
        self.cnn_adapter: Optional[CNNBiLSTMAdapter] = None
        self.attention_adapter: Optional[AttentionModelAdapter] = None
        self._is_initialized: bool = False

    def initialize_models(self) -> None:
        """Load all model weights once into memory."""
        if self._is_initialized:
            return

        LOGGER.info("Initializing NEUROSCAN inference models...")
        # 1. ML Adapter
        self.ml_adapter = MLModelAdapter(self.ml_model_path)
        LOGGER.info("ML model adapter initialized.")

        # 2. CNN-BiLSTM Adapter
        self.cnn_adapter = CNNBiLSTMAdapter(self.cnn_checkpoint_path, device=self.device)
        LOGGER.info("CNN-BiLSTM adapter initialized.")

        # 3. Attention Adapter
        self.attention_adapter = AttentionModelAdapter(
            self.attention_checkpoint_path,
            self.calibration_path,
            device=self.device,
        )
        LOGGER.info("AttentionClassifier adapter initialized.")

        self._is_initialized = True
        LOGGER.info("All 3 inference models loaded and ready.")

    def get_health_status(self) -> Dict[str, Any]:
        """Verify model files and loaded state."""
        return {
            "ml_model_exists": self.ml_model_path.is_file(),
            "cnn_checkpoint_exists": self.cnn_checkpoint_path.is_file(),
            "attention_checkpoint_exists": self.attention_checkpoint_path.is_file(),
            "calibration_file_exists": self.calibration_path.is_file(),
            "models_loaded": self._is_initialized,
            "device": str(self.cnn_adapter.device) if self.cnn_adapter else "uninitialized",
            "ml_threshold": self.ml_adapter.decision_threshold if self.ml_adapter else None,
            "cnn_threshold": self.cnn_adapter.threshold if self.cnn_adapter else None,
            "attention_calibration_method": (
                self.attention_adapter.calibration_config.get("method")
                if self.attention_adapter else None
            ),
        }

    def process_single_edf(
        self,
        edf_path: Path | str,
        patient_id: str,
        analysis_id: str = "",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        current_index: int = 1,
        total_recordings: int = 1,
    ) -> Dict[str, Any]:
        """Run complete 3-model inference on a single EDF recording."""
        self.initialize_models()
        assert self.ml_adapter is not None
        assert self.cnn_adapter is not None
        assert self.attention_adapter is not None

        path = Path(edf_path).resolve()
        start_rec_time = time.time()
        recording_id = f"{patient_id}_{path.stem.upper()}"

        def stage(name: str, model: Optional[str] = None) -> None:
            LOGGER.info("analysis_id=%s recording_id=%s stage=%s", analysis_id, recording_id, name)
            if progress_callback:
                progress_callback({
                    "stage": name,
                    "current_model": model,
                    "current_index": current_index,
                    "total_recordings": total_recordings,
                    "current_filename": path.name,
                    "percent_complete": int(round((current_index - 1) / max(total_recordings, 1) * 100)),
                })

        try:
            # 1. Preprocess EEG signal
            stage("preprocessing")
            processed = preprocess_recording_for_inference(path, patient_id=patient_id)
            LOGGER.info("analysis_id=%s recording_id=%s stage=preprocessing status=completed", analysis_id, recording_id)

            # 2. Model 1: Tabular ML Inference
            stage("ml_inference", "ml")
            ml_res = self.ml_adapter.run_inference(processed)
            LOGGER.info("analysis_id=%s recording_id=%s stage=ml_inference status=%s", analysis_id, recording_id, ml_res.status)

            # 3. Model 2: CNN-BiLSTM Raw EEG Inference
            stage("cnn_bilstm_inference", "cnn_bilstm")
            cnn_res = self.cnn_adapter.run_inference(processed)
            LOGGER.info("analysis_id=%s recording_id=%s stage=cnn_bilstm_inference status=%s", analysis_id, recording_id, cnn_res.status)

            # 4. Model 3: Attention Classifier Inference
            # Pass intermediate temporal features extracted by CNN-BiLSTM!
            stage("attention_inference", "attention")
            attn_res = self.attention_adapter.run_inference(
                cnn_res.temporal_features,
                processed,
            )
            LOGGER.info("analysis_id=%s recording_id=%s stage=attention_inference status=%s", analysis_id, recording_id, attn_res.status)

            stage("event_detection")
            rec_duration = time.time() - start_rec_time
            result = build_recording_result(
                processed=processed,
                ml_res=ml_res,
                cnn_res=cnn_res,
                attn_res=attn_res,
                processing_duration=rec_duration,
            )
            if progress_callback:
                progress_callback({
                    "stage": "recording_completed",
                    "current_index": current_index,
                    "total_recordings": total_recordings,
                    "current_filename": path.name,
                    "recording_status": "completed",
                    "percent_complete": int(round(current_index / max(total_recordings, 1) * 100)),
                })
            LOGGER.info("analysis_id=%s recording_id=%s status=completed", analysis_id, recording_id)
            return result
        except Exception as exc:
            LOGGER.error("analysis_id=%s recording_id=%s status=failed error=%s", analysis_id, recording_id, exc, exc_info=True)
            rec_duration = time.time() - start_rec_time
            result = build_failed_recording_result(
                file_path=str(path),
                recording_id=recording_id,
                error_message=str(exc),
                processing_duration=rec_duration,
            )
            if progress_callback:
                progress_callback({
                    "stage": "recording_failed",
                    "current_index": current_index,
                    "total_recordings": total_recordings,
                    "current_filename": path.name,
                    "recording_status": "failed",
                    "percent_complete": int(round(current_index / max(total_recordings, 1) * 100)),
                })
            return result

    def analyze_patient_folder(
        self,
        patient_id: str,
        analysis_id: str,
        edf_paths: List[Path | str],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Analyze all EDF recordings for a patient with progress reporting."""
        self.initialize_models()
        start_total = time.time()
        total_files = len(edf_paths)
        results: List[Dict[str, Any]] = []

        for idx, edf_p in enumerate(edf_paths):
            current_path = Path(edf_p).resolve()
            LOGGER.info(
                "[%s] Processing %d/%d: %s",
                analysis_id, idx + 1, total_files, current_path.name
            )

            if progress_callback:
                progress_callback({
                    "stage": "validation",
                    "current_index": idx + 1,
                    "total_recordings": total_files,
                    "current_filename": current_path.name,
                    "percent_complete": int(round((idx / max(total_files, 1)) * 100)),
                })
            LOGGER.info("analysis_id=%s recording_id=%s stage=validation", analysis_id, current_path.stem)

            rec_result = self.process_single_edf(
                current_path,
                patient_id=patient_id,
                analysis_id=analysis_id,
                progress_callback=progress_callback,
                current_index=idx + 1,
                total_recordings=total_files,
            )
            results.append(rec_result)
            if progress_callback:
                completed = len(results)
                successes = sum(r.get("status") == "completed" for r in results)
                progress_callback({
                    "stage": "recording_completed" if rec_result["status"] == "completed" else "recording_failed",
                    "current_index": idx + 1,
                    "total_recordings": total_files,
                    "current_filename": current_path.name,
                    "recording_status": rec_result["status"],
                    "completed_recordings": completed,
                    "successful_recordings": successes,
                    "failed_recordings": completed - successes,
                    "percent_complete": int(round(completed / max(total_files, 1) * 100)),
                })

        total_exec = time.time() - start_total
        patient_summary = build_patient_summary(
            patient_id=patient_id,
            analysis_id=analysis_id,
            recordings_results=results,
            total_execution_time=total_exec,
        )

        if progress_callback:
            progress_callback({
                "stage": "completed",
                "current_index": total_files,
                "total_recordings": total_files,
                "current_filename": None,
                "percent_complete": 100,
            })

        return patient_summary
