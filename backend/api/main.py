"""main.py
-------
FastAPI REST API server for NEUROSCAN EEG Intelligence Platform.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from backend.inference.inference_orchestrator import InferenceOrchestrator
from backend.inference.preprocessing_wrapper import (
    CANONICAL_CHANNELS,
    get_edf_metadata,
)
from backend.reporting.patient_report import generate_patient_pdf_report
from backend.storage.storage_manager import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("neuroscan.api")

app = FastAPI(
    title="NEUROSCAN EEG Intelligence API",
    description="Multi-model EEG seizure detection and analytics platform.",
    version="1.0.0",
)

# Enable CORS for frontend Vite dev server & production builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Singletons
storage_manager = StorageManager()
orchestrator = InferenceOrchestrator()

# In-memory session tracking for active jobs
active_jobs: Dict[str, Dict[str, Any]] = {}
analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neuroscan-analysis")


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-initialize models on server start."""
    try:
        orchestrator.initialize_models()
        LOGGER.info("NEUROSCAN startup: models loaded successfully.")
    except Exception as exc:
        LOGGER.warning("Deferred model initialization: %s", exc)


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Report API and model status."""
    model_health = orchestrator.get_health_status()
    return {
        "status": "healthy",
        "service": "NEUROSCAN API",
        "version": "1.0.0",
        "models": model_health,
        "canonical_channels": CANONICAL_CHANNELS,
    }


@app.post("/api/patients/upload")
async def upload_patient_folder(
    files: List[UploadFile] = File(...),
    relative_paths: Optional[List[str]] = Form(None),
    annotation_files: Optional[List[UploadFile]] = File(None),
    annotation_relative_paths: Optional[List[str]] = Form(None),
    patient_id_override: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload a complete patient EEG folder or multiple EDF files."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were uploaded.",
        )

    analysis_id = f"NS_{uuid.uuid4().hex[:10]}"
    session_dir = storage_manager.create_upload_session_dir(analysis_id)

    detected_patient_id: Optional[str] = patient_id_override
    uploaded_recordings: List[Dict[str, Any]] = []
    rejected_files: List[Dict[str, Any]] = []
    stored_annotations: List[str] = []

    LOGGER.info("Upload received analysis_id=%s files=%d", analysis_id, len(files))

    for idx, file in enumerate(files):
        filename = file.filename or f"recording_{idx}.edf"
        rel_path = (
            relative_paths[idx]
            if relative_paths and idx < len(relative_paths)
            else filename
        )

        # Basic path traversal guard
        clean_rel = os.path.normpath(rel_path.replace("\\", "/")).lstrip("/\\")
        if ".." in Path(clean_rel).parts:
            clean_rel = Path(filename).name

        # Infer patient ID from top-level directory if not already set
        parts = Path(clean_rel).parts
        if len(parts) > 1 and not detected_patient_id:
            detected_patient_id = parts[0].upper()

        if not filename.lower().endswith(".edf"):
            rejected_files.append({
                "filename": filename,
                "reason": "Unsupported format. Only .edf recordings are supported.",
            })
            continue

        # Sidecars and other files are never passed to MNE or inference.
        if not filename.lower().endswith(".edf"):
            rejected_files.append({"filename": filename, "reason": "Only .edf files are inference recordings."})
            continue

        dest_path = session_dir / clean_rel
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Extract basic metadata
            meta = get_edf_metadata(dest_path)
            uploaded_recordings.append(meta)
            LOGGER.info("EDF validation passed analysis_id=%s recording_id=%s", analysis_id, dest_path.stem)
        except Exception as exc:
            LOGGER.error("EDF validation failed (analysis_id=%s, filename=%s): %s", analysis_id, filename, exc)
            dest_path.unlink(missing_ok=True)
            rejected_files.append({
                "filename": filename,
                "reason": f"Corrupted or unreadable EDF: {str(exc)}",
            })

    # Keep CHB annotation sidecars separate from the EDF inference inputs.
    for idx, sidecar in enumerate(annotation_files or []):
        sidecar_name = sidecar.filename or f"annotation_{idx}.edf.seizures"
        if not sidecar_name.lower().endswith(".edf.seizures"):
            continue
        rel_path = (annotation_relative_paths[idx]
                    if annotation_relative_paths and idx < len(annotation_relative_paths)
                    else sidecar_name)
        clean_rel = os.path.normpath(rel_path.replace("\\", "/")).lstrip("/\\")
        if ".." in Path(clean_rel).parts:
            clean_rel = Path(sidecar_name).name
        annotation_path = session_dir / "_annotations" / clean_rel
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        with annotation_path.open("wb") as buffer:
            shutil.copyfileobj(sidecar.file, buffer)
        stored_annotations.append(str(annotation_path))

    if not uploaded_recordings:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "No valid EDF files found in the upload.",
                "rejected_files": rejected_files,
            },
        )

    patient_id = detected_patient_id or "PATIENT_001"

    # Register initial job
    active_jobs[analysis_id] = {
        "analysis_id": analysis_id,
        "patient_id": patient_id,
        "status": "ready",
        "stage": "upload_complete",
        "percent_complete": 0,
        "total_recordings": len(uploaded_recordings),
        "uploaded_recordings": uploaded_recordings,
        "annotation_files": stored_annotations,
        "rejected_files": rejected_files,
        "annotation_count": len(stored_annotations),
        "error": None,
        "current_model": None,
        "completed_recordings": 0,
        "successful_recordings": 0,
        "failed_recordings": 0,
        "recording_statuses": [
            {"filename": item["filename"], "status": "queued", "stage": "queued"}
            for item in uploaded_recordings
        ],
        "stage_events": [],
    }

    return {
        "analysis_id": analysis_id,
        "patient_id": patient_id,
        "total_recordings": len(uploaded_recordings),
        "recordings": uploaded_recordings,
        "rejected_count": len(rejected_files),
        "rejected_files": rejected_files,
        "annotation_count": len(stored_annotations),
        "status": "ready",
    }


def _run_analysis_background(analysis_id: str, patient_id: str, edf_paths: List[str]) -> None:
    """Background worker for multi-recording EEG analysis."""
    job = active_jobs.get(analysis_id)
    if not job:
        return

    job["status"] = "processing"

    def progress_callback(info: Dict[str, Any]) -> None:
        job["stage"] = info.get("stage")
        job["current_index"] = info.get("current_index")
        job["current_filename"] = info.get("current_filename")
        job["current_model"] = info.get("current_model")
        for field in ("percent_complete", "completed_recordings", "successful_recordings", "failed_recordings"):
            if field in info:
                job[field] = info[field]
        rec_idx = int(info.get("current_index", 0)) - 1
        if 0 <= rec_idx < len(job["recording_statuses"]):
            entry = job["recording_statuses"][rec_idx]
            entry["stage"] = info.get("stage", entry["stage"])
            if info.get("recording_status"):
                entry["status"] = info["recording_status"]
        event = {
            "recording": info.get("current_filename"),
            "stage": info.get("stage"),
            "model": info.get("current_model"),
            "time": round(time.time(), 3),
        }
        if not job["stage_events"] or any(
            job["stage_events"][-1].get(key) != event.get(key)
            for key in ("recording", "stage", "model")
        ):
            job["stage_events"].append(event)

    try:
        results = orchestrator.analyze_patient_folder(
            patient_id=patient_id,
            analysis_id=analysis_id,
            edf_paths=edf_paths,
            progress_callback=progress_callback,
        )
        job["stage"] = "results_saved"
        job["current_model"] = None
        storage_manager.save_analysis_result(analysis_id, results)
        LOGGER.info("Analysis results saved analysis_id=%s", analysis_id)
        job["successful_recordings"] = results.get("successful_recordings", 0)
        job["failed_recordings"] = results.get("failed_recordings", 0)
        job["completed_recordings"] = results.get("total_recordings", 0)
        for item, result in zip(job["recording_statuses"], results.get("recordings", [])):
            item["status"] = result.get("status", "failed")
            item["stage"] = "results_saved"

        # Pre-generate PDF report
        try:
            pdf_bytes = generate_patient_pdf_report(results)
            storage_manager.save_report_pdf(analysis_id, pdf_bytes)
        except Exception as pdf_exc:
            LOGGER.error("PDF generation failed analysis_id=%s: %s", analysis_id, pdf_exc)

        job["status"] = "completed"
        job["stage"] = "completed"
        job["percent_complete"] = 100
    except Exception as exc:
        LOGGER.error("Analysis job %s failed: %s", analysis_id, exc, exc_info=True)
        job["status"] = "failed"
        job["error"] = str(exc)


@app.post("/api/analysis/start/{analysis_id}")
async def start_analysis(
    analysis_id: str,
) -> Dict[str, Any]:
    """Begin analysis of uploaded patient recordings."""
    job = active_jobs.get(analysis_id)
    if not job:
        # Check if already processed and stored
        existing_path = storage_manager.analysis_dir / f"{analysis_id}.json"
        if existing_path.is_file():
            return {"analysis_id": analysis_id, "status": "completed"}
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session {analysis_id} not found.",
        )

    if job["status"] in {"processing", "completed"}:
        return {"analysis_id": analysis_id, "status": job["status"]}

    edf_files = [r["file_path"] for r in job["uploaded_recordings"]]
    patient_id = job["patient_id"]

    job["status"] = "processing"
    job["percent_complete"] = 0
    analysis_executor.submit(
        _run_analysis_background,
        analysis_id=analysis_id,
        patient_id=patient_id,
        edf_paths=edf_files,
    )

    return {
        "analysis_id": analysis_id,
        "patient_id": patient_id,
        "status": "processing",
        "total_recordings": len(edf_files),
    }


@app.get("/api/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str) -> Dict[str, Any]:
    """Poll progress status of an active analysis."""
    job = active_jobs.get(analysis_id)
    if job:
        return {
            "analysis_id": analysis_id,
            "patient_id": job.get("patient_id"),
            "status": job.get("status"),
            "stage": job.get("stage"),
            "current_index": job.get("current_index", 0),
            "total_recordings": job.get("total_recordings", 0),
            "current_filename": job.get("current_filename"),
            "percent_complete": job.get("percent_complete", 0),
            "error": job.get("error"),
            "current_model": job.get("current_model"),
            "completed_recordings": job.get("completed_recordings", 0),
            "successful_recordings": job.get("successful_recordings", 0),
            "failed_recordings": job.get("failed_recordings", 0),
            "recording_statuses": job.get("recording_statuses", []),
            "stage_events": job.get("stage_events", [])[-20:],
        }

    # Check stored results
    existing_path = storage_manager.analysis_dir / f"{analysis_id}.json"
    if existing_path.is_file():
        existing = storage_manager._read_analysis_summary(existing_path)
        return {
            "analysis_id": analysis_id,
            "patient_id": existing.get("patient_id"),
            "status": "completed",
            "stage": "completed",
            "percent_complete": 100,
            "total_recordings": existing.get("total_recordings", 0),
            "error": None,
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Analysis session {analysis_id} not found.",
    )


@app.get("/api/analysis/{analysis_id}/results")
async def get_analysis_results(analysis_id: str) -> Dict[str, Any]:
    """Get complete multi-model patient analysis results."""
    result_path = storage_manager.analysis_dir / f"{analysis_id}.json"
    if result_path.is_file() and result_path.stat().st_size > 250 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This older result embeds full EEG traces and is too large to load. Re-run the source EDF analysis for a compact result; the existing JSON was left untouched.",
        )
    existing = storage_manager.get_analysis_result(analysis_id)
    if not existing:
        job = active_jobs.get(analysis_id)
        if job and job["status"] == "processing":
            return {"analysis_id": analysis_id, "status": "processing"}
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis results for {analysis_id} not found.",
        )
    return existing


@app.get("/api/analysis/{analysis_id}/summary")
async def get_analysis_summary(analysis_id: str) -> Dict[str, Any]:
    """Retrieve high-level patient summary and model comparison metrics."""
    res = await get_analysis_results(analysis_id)
    return {
        "patient_id": res.get("patient_id"),
        "analysis_id": res.get("analysis_id"),
        "created_at": res.get("created_at"),
        "total_recordings": res.get("total_recordings"),
        "successful_recordings": res.get("successful_recordings"),
        "total_duration_seconds": res.get("total_duration_seconds"),
        "total_windows": res.get("total_windows"),
        "summary": res.get("summary"),
    }


@app.get("/api/analysis/{analysis_id}/recordings")
async def get_analysis_recordings(analysis_id: str) -> List[Dict[str, Any]]:
    """List recordings analyzed in this session."""
    res = await get_analysis_results(analysis_id)
    recordings = res.get("recordings", [])
    # Return light metadata for recording list
    return [
        {
            "recording_id": r.get("recording_id"),
            "filename": r.get("filename"),
            "status": r.get("status"),
            "duration_seconds": r.get("duration_seconds"),
            "total_windows": r.get("total_windows"),
            "sampling_frequency": r.get("sampling_frequency"),
            "channel_count": r.get("channel_count"),
            "ml_positive": r.get("models", {}).get("ml", {}).get("positive_window_count", 0),
            "cnn_positive": r.get("models", {}).get("cnn_bilstm", {}).get("positive_window_count", 0),
            "attention_positive": r.get("models", {}).get("attention", {}).get("positive_window_count", 0),
            "calibrated_threshold": r.get("models", {}).get("attention", {}).get("calibrated_threshold", 0.03),
        }
        for r in recordings
    ]


@app.get("/api/analysis/{analysis_id}/recordings/{recording_id}")
async def get_single_recording_analysis(analysis_id: str, recording_id: str) -> Dict[str, Any]:
    """Retrieve full detail for one recording."""
    res = await get_analysis_results(analysis_id)
    for rec in res.get("recordings", []):
        if rec.get("recording_id") == recording_id:
            return rec
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Recording {recording_id} not found in analysis {analysis_id}.",
    )


@app.get("/api/analysis/{analysis_id}/recordings/{recording_id}/models/{model_name}")
async def get_model_recording_analysis(
    analysis_id: str, recording_id: str, model_name: str
) -> Dict[str, Any]:
    """Get model-specific results (ml, cnn, or attention) for a recording."""
    rec = await get_single_recording_analysis(analysis_id, recording_id)
    key_map = {"ml": "ml", "cnn": "cnn_bilstm", "attention": "attention"}
    actual_key = key_map.get(model_name.lower())
    if not actual_key or actual_key not in rec.get("models", {}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model name '{model_name}'. Choose ml, cnn, or attention.",
        )
    return {
        "recording_id": recording_id,
        "filename": rec.get("filename"),
        "model_key": actual_key,
        "model_data": rec["models"][actual_key],
        "windows": [
            {
                "window_index": w["window_index"],
                "start_seconds": w["start_seconds"],
                "end_seconds": w["end_seconds"],
                "result": w.get(actual_key),
            }
            for w in rec.get("windows", [])
        ],
    }


@app.get("/api/analysis/{analysis_id}/recordings/{recording_id}/signal")
async def get_eeg_signal_segment(
    analysis_id: str,
    recording_id: str,
    start_seconds: float = Query(0.0, ge=0.0),
    duration_seconds: float = Query(20.0, gt=0.0, le=120.0),
    max_channels: int = Query(16, ge=1, le=24),
) -> Dict[str, Any]:
    """Retrieve EEG raw signal trace for interactive viewer zooming and panning."""
    rec = await get_single_recording_analysis(analysis_id, recording_id)
    file_path = rec.get("file_path")
    if not file_path or not Path(file_path).is_file():
        # Fallback to preview signal if raw file is not locally accessible
        if rec.get("preview_signal"):
            return rec["preview_signal"]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Raw EDF recording is no longer on disk.",
        )

    import mne

    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    start_sample = int(start_seconds * sfreq)
    stop_sample = int((start_seconds + duration_seconds) * sfreq)
    stop_sample = min(stop_sample, raw.n_times)

    picks = mne.pick_types(raw.info, eeg=True)[:max_channels]
    data, times = raw[picks, start_sample:stop_sample]
    ch_names = [raw.ch_names[p] for p in picks]

    # Convert to microvolts
    data_uv = (data * 1e6).tolist()

    return {
        "channel_names": ch_names,
        "sampling_rate_hz": sfreq,
        "start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
        "times": times.tolist(),
        "data_uv": data_uv,
    }


@app.get("/api/analysis/{analysis_id}/report/pdf")
async def download_patient_pdf_report(analysis_id: str) -> Response:
    """Download compiled PDF clinical analysis report."""
    pdf_path = storage_manager.get_report_pdf_path(analysis_id)
    if pdf_path:
        return FileResponse(
            path=str(pdf_path),
            filename=f"NEUROSCAN_Report_{analysis_id}.pdf",
            media_type="application/pdf",
        )

    # If not on disk, compile it on-the-fly
    analysis = storage_manager.get_analysis_result(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found.",
        )

    pdf_bytes = generate_patient_pdf_report(analysis)
    storage_manager.save_report_pdf(analysis_id, pdf_bytes)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=NEUROSCAN_Report_{analysis_id}.pdf"
        },
    )


@app.get("/api/analyses")
async def list_analyses() -> List[Dict[str, Any]]:
    """List historical analysis sessions for Patient Records page."""
    return storage_manager.list_analyses()
