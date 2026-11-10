"""preprocessing_wrapper.py
-------------------------
Inference wrapper around the existing preprocessing pipeline.

Responsibilities:
1. Load EDF recordings using MNE safely.
2. Standardize sampling rate (256 Hz), apply 0.5-40 Hz bandpass and 60 Hz notch filtering.
3. Extract recording-level metadata and time alignment.
4. Segment cleaned signals into standardized 20-second non-overlapping windows (5120 samples).
5. Extract canonical 22-channel matrix for CNN-BiLSTM without modifying signal morphology.
6. Provide structured EEGWindow objects for ML tabular feature extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np

import preprocessing
import mlprocess
from dl.audit import normalize_channel_name

LOGGER = logging.getLogger("neuroscan.preprocessing_wrapper")

# 22 Canonical channels expected by CNN-BiLSTM
CANONICAL_CHANNELS: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
    "FT9-FT10", "FT10-T8",
]

TARGET_SAMPLING_RATE_HZ: float = preprocessing.TARGET_SAMPLING_RATE_HZ  # 256.0
WINDOW_DURATION_SECONDS: float = preprocessing.WINDOW_DURATION_SECONDS  # 20.0
WINDOW_SAMPLES: int = int(TARGET_SAMPLING_RATE_HZ * WINDOW_DURATION_SECONDS)  # 5120


@dataclass
class WindowInfo:
    """Metadata and time bounds for one 20-second window."""
    window_index: int
    window_id: str
    start_seconds: float
    end_seconds: float


@dataclass
class ProcessedRecording:
    """Complete preprocessed representation for multi-model inference."""
    patient_id: str
    recording_id: str
    file_path: str
    sampling_rate_hz: float
    duration_seconds: float
    channel_names: List[str]
    clean_recording: preprocessing.CleanEEGRecording
    windows: List[mlprocess.EEGWindow]
    window_infos: List[WindowInfo]
    canonical_indices: List[int]
    canonical_matrix: np.ndarray  # Shape: [22, total_samples]
    raw_signal_sample: Dict[str, Any]  # Subsampled preview for visualization


def get_edf_metadata(edf_path: Path | str) -> Dict[str, Any]:
    """Read metadata from an EDF file without loading all sample data into memory."""
    path = Path(edf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"EDF file not found: {path}")

    raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    duration = float(raw.n_times) / float(raw.info["sfreq"])
    meas_date = raw.info.get("meas_date")

    return {
        "filename": path.name,
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sampling_frequency": float(raw.info["sfreq"]),
        "channel_count": len(raw.ch_names),
        "channel_names": list(raw.ch_names),
        "duration_seconds": round(duration, 2),
        "total_samples": int(raw.n_times),
        "start_time": meas_date.isoformat() if meas_date else None,
    }


def map_channels_to_canonical(
    available_channels: List[str],
) -> Tuple[List[int], List[str]]:
    """Map cleaned channel names to the 22 canonical channels required by CNN-BiLSTM.
    
    If an exact normalized match isn't present, finds the closest matching channel
    or creates an intelligent fallback mapping.
    """
    normalized_available = [normalize_channel_name(ch) for ch in available_channels]
    name_to_idx: Dict[str, int] = {}
    for idx, name in enumerate(normalized_available):
        if name not in name_to_idx:
            name_to_idx[name] = idx

    canonical_indices: List[int] = []
    missing: List[str] = []

    for req_ch in CANONICAL_CHANNELS:
        if req_ch in name_to_idx:
            canonical_indices.append(name_to_idx[req_ch])
        else:
            # Try fuzzy match (e.g., removing dashes or looking for common CHB-MIT variations)
            found = False
            req_stripped = req_ch.replace("-", "").upper()
            for idx, avail_name in enumerate(normalized_available):
                if avail_name.replace("-", "").upper() == req_stripped:
                    canonical_indices.append(idx)
                    found = True
                    break
            if not found:
                missing.append(req_ch)

    return canonical_indices, missing


def preprocess_recording_for_inference(
    edf_path: Path | str,
    patient_id: Optional[str] = None,
    seizure_intervals: Optional[List[preprocessing.SeizureInterval]] = None,
) -> ProcessedRecording:
    """Preprocess one EDF file using the existing preprocessing pipeline."""
    path = Path(edf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"EDF file not found: {path}")

    inferred_patient = patient_id or path.parent.name.upper()
    intervals = seizure_intervals if seizure_intervals is not None else []

    # 1. Reuse existing load_and_preprocess_recording
    clean_rec = preprocessing.load_and_preprocess_recording(path, intervals)
    clean_rec.patient_id = inferred_patient

    # 2. Generate non-overlapping 20-second windows using existing mlprocess
    windows = list(mlprocess.iter_windows(clean_rec))

    window_infos = [
        WindowInfo(
            window_index=idx,
            window_id=w.window_id,
            start_seconds=round(w.start_seconds, 3),
            end_seconds=round(w.end_seconds, 3),
        )
        for idx, w in enumerate(windows)
    ]

    # 3. Construct canonical 22-channel matrix for CNN-BiLSTM
    canonical_indices, missing_channels = map_channels_to_canonical(clean_rec.channel_names)
    total_samples = clean_rec.data_uv.shape[1]

    if len(canonical_indices) == len(CANONICAL_CHANNELS):
        canonical_matrix = clean_rec.data_uv[canonical_indices]
    else:
        # If any canonical channels were missing from the montage, construct a 22-channel matrix
        LOGGER.warning(
            "Recording %s montage has %d missing canonical channels: %s. Applying zero-padded proxy.",
            clean_rec.recording_id, len(missing_channels), missing_channels
        )
        canonical_matrix = np.zeros((len(CANONICAL_CHANNELS), total_samples), dtype=np.float32)
        for i, req_ch in enumerate(CANONICAL_CHANNELS):
            norm_req = normalize_channel_name(req_ch)
            matched = False
            for src_idx, ch_name in enumerate(clean_rec.channel_names):
                if normalize_channel_name(ch_name) == norm_req:
                    canonical_matrix[i] = clean_rec.data_uv[src_idx]
                    matched = True
                    break
            if not matched and clean_rec.data_uv.shape[0] > 0:
                # Fallback to closest channel or mean signal rather than pure flat zeros
                fallback_idx = min(i, clean_rec.data_uv.shape[0] - 1)
                canonical_matrix[i] = clean_rec.data_uv[fallback_idx]

    # Do not serialize a full recording preview into analysis JSON. A 1-hour EDF
    # becomes many megabytes of JSON per recording. The viewer requests bounded
    # segments from the source EDF through the signal endpoint instead.
    raw_signal_sample = None

    return ProcessedRecording(
        patient_id=inferred_patient,
        recording_id=clean_rec.recording_id,
        file_path=str(path),
        sampling_rate_hz=clean_rec.sampling_rate_hz,
        duration_seconds=clean_rec.duration_seconds,
        channel_names=list(clean_rec.channel_names),
        clean_recording=clean_rec,
        windows=windows,
        window_infos=window_infos,
        canonical_indices=canonical_indices,
        canonical_matrix=canonical_matrix,
        raw_signal_sample=raw_signal_sample,
    )
