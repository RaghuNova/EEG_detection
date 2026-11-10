"""
preprocessing.py
----------------
Raw CHB-MIT EEG -> cleaned, standardized EEG recordings.

Responsibilities
----------------
1. Discover EDF recordings from PATIENT_DATASET_PATH.
2. Parse CHB-MIT seizure annotations from *-summary.txt files.
3. Load EEG channels with MNE.
4. Resample to a fixed sampling rate.
5. Apply band-pass and power-line notch filtering.
6. Perform conservative signal-quality checks without deleting seizure morphology.
7. Return structured CleanEEGRecording objects for mlprocess.py.

Important design rule:
    This module does NOT split data, perform SMOTE, train a model, or fit a scaler.
    Those operations belong to mlprocess.py / mlout.py to avoid data leakage.

For patient-independent experiments, point PATIENT_DATASET_PATH at the directory
containing multiple patient folders (e.g. minorproj/dataset/Chb01, Chb02, ...).
A single Chb01 directory contains only one patient and therefore cannot support a
true patient-independent train/validation/test split by itself.
"""

from __future__ import annotations

import argparse
import logging
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from . import runtime  # noqa: F401
except ImportError:  # supports `python backend/preprocessing.py`
    import runtime  # type: ignore  # noqa: F401
import joblib
import mne
import numpy as np

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------
# FIX: Anchor default paths to this file's location so the script works no
# matter which directory it is launched from. Previously "../minor-proj/..."
# was resolved relative to the current working directory, which produced
# /home/advik-n/minor-proj/minor-proj/dataset/Chb01 (duplicated folder).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# FIX: default now points at the real dataset location (../dataset/Chb01).
# For patient-independent experiments, change this to:
#     str(PROJECT_ROOT / "dataset")
PATIENT_DATASET_PATH: str = str(PROJECT_ROOT / "dataset")

# Shared signal-processing configuration.
TARGET_SAMPLING_RATE_HZ: float = 256.0
LOW_CUTOFF_HZ: float = 0.5
HIGH_CUTOFF_HZ: float = 40.0
NOTCH_FREQUENCY_HZ: float = 60.0

# The 20-second window is defined here as the project-wide contract.
# Actual window generation is intentionally performed in mlprocess.py.
WINDOW_DURATION_SECONDS: float = 20.0

# Conservative quality checks. These do not remove genuine high-amplitude seizure
# activity; they only reject recordings/windows with invalid numeric values or
# channels that are effectively flat.
MIN_CHANNEL_STD_UV: float = 1e-6
MAX_NONFINITE_FRACTION: float = 0.001

# Artifact handling is deliberately conservative. Filtering is the primary noise
# reduction method in this baseline pipeline. Aggressive ICA/component removal is
# intentionally not used because it can remove seizure morphology.

# Output cache used by mlprocess.py if desired.
# FIX: anchored to PROJECT_ROOT instead of a CWD-relative "minorproj/..." path.
PREPROCESSED_CACHE_PATH: str = str(
    PROJECT_ROOT / "artifacts" / "preprocessed_recordings.joblib"
)
PREPROCESSED_CACHE_DIR: Path = PROJECT_ROOT / "artifacts" / "preprocessed_patients"
PREPROCESSED_MANIFEST_PATH: str = str(
    PROJECT_ROOT / "artifacts" / "preprocessed_manifest.json"
)

LOG_LEVEL: int = logging.INFO

# FIX: LOGGER was referenced in load_dataset() but never defined. Configure
# logging once at import time and expose a module-level logger.
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# DATA CONTRACTS
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SeizureInterval:
    """Ground-truth seizure interval in seconds relative to an EDF recording."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("Seizure start time cannot be negative.")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Seizure end time must be greater than start time.")


@dataclass
class CleanEEGRecording:
    """In-memory representation passed from preprocessing to mlprocess."""

    patient_id: str
    recording_id: str
    file_path: str
    channel_names: List[str]
    sampling_rate_hz: float
    data_uv: np.ndarray  # shape: (channels, samples), float32
    duration_seconds: float
    seizure_intervals: List[SeizureInterval]

    def validate(self) -> None:
        if self.data_uv.ndim != 2:
            raise ValueError(
                f"Expected EEG matrix [channels, samples], got {self.data_uv.shape}."
            )
        if self.data_uv.shape[0] != len(self.channel_names):
            raise ValueError("Number of channels does not match channel_names.")
        if self.data_uv.shape[1] == 0:
            raise ValueError("EEG recording contains zero samples.")
        if self.sampling_rate_hz <= 0:
            raise ValueError("Sampling rate must be positive.")
        if not np.isfinite(self.data_uv).all():
            raise ValueError("Preprocessed EEG contains non-finite values.")


# -----------------------------------------------------------------------------
# CHB-MIT ANNOTATION PARSING
# -----------------------------------------------------------------------------

_FILE_RE = re.compile(r"^\s*File Name:\s*(.+?)\s*$", re.IGNORECASE)
_SEIZURE_COUNT_RE = re.compile(
    r"^\s*Number of Seizures in File:\s*(\d+)\s*$", re.IGNORECASE
)
_SEIZURE_START_RE = re.compile(
    r"^\s*Seizure(?:\s+\d+)?\s+Start Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds\s*$",
    re.IGNORECASE,
)
_SEIZURE_END_RE = re.compile(
    r"^\s*Seizure(?:\s+\d+)?\s+End Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds\s*$",
    re.IGNORECASE,
)


def _parse_float_seconds(match: re.Match[str]) -> float:
    return float(match.group(1))


def parse_chb_summary(summary_path: Path) -> Dict[str, List[SeizureInterval]]:
    """Parse a CHB-MIT *-summary.txt into {edf_filename: seizure intervals}."""

    if not summary_path.exists():
        raise FileNotFoundError(f"CHB-MIT summary file not found: {summary_path}")

    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    results: Dict[str, List[SeizureInterval]] = {}
    current_file: Optional[str] = None
    expected_count = 0
    starts: List[float] = []
    ends: List[float] = []

    def flush_current() -> None:
        nonlocal current_file, expected_count, starts, ends
        if current_file is None:
            return
        if expected_count != len(starts) or expected_count != len(ends):
            raise ValueError(
                f"Annotation mismatch for {current_file}: expected {expected_count} "
                f"seizures, got {len(starts)} starts and {len(ends)} ends."
            )
        intervals = [
            SeizureInterval(start_seconds=s, end_seconds=e)
            for s, e in zip(starts, ends)
        ]
        results[current_file] = intervals
        current_file = None
        expected_count = 0
        starts = []
        ends = []

    for line in lines:
        match = _FILE_RE.match(line)
        if match:
            flush_current()
            current_file = match.group(1).strip()
            continue

        match = _SEIZURE_COUNT_RE.match(line)
        if match and current_file is not None:
            expected_count = int(match.group(1))
            continue

        match = _SEIZURE_START_RE.match(line)
        if match and current_file is not None:
            starts.append(_parse_float_seconds(match))
            continue

        match = _SEIZURE_END_RE.match(line)
        if match and current_file is not None:
            ends.append(_parse_float_seconds(match))
            continue

    flush_current()

    return results


# -----------------------------------------------------------------------------
# DATASET DISCOVERY
# -----------------------------------------------------------------------------


def _looks_like_patient_dir(path: Path) -> bool:
    return bool(re.fullmatch(r"chb\d+", path.name, re.IGNORECASE))


def discover_patient_directories(dataset_path: Path) -> List[Path]:
    """Return patient directories from either a single patient folder or a root."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_path.resolve()}"
        )

    if dataset_path.is_file():
        raise ValueError("PATIENT_DATASET_PATH must be a directory, not a file.")

    if _looks_like_patient_dir(dataset_path):
        return [dataset_path]

    patient_dirs = sorted(
        p for p in dataset_path.iterdir() if p.is_dir() and _looks_like_patient_dir(p)
    )
    if not patient_dirs:
        # Accept arbitrary directory names if it directly contains EDF + summary.
        if list(dataset_path.glob("*.edf")):
            return [dataset_path]
        raise FileNotFoundError(
            f"No CHB-MIT patient folders (ChbXX) or EDF recordings found under "
            f"{dataset_path.resolve()}"
        )
    return patient_dirs


def discover_edf_files(patient_dir: Path) -> List[Path]:
    files = sorted(patient_dir.glob("*.edf"), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No EDF files found in {patient_dir.resolve()}")
    return files


def discover_summary_file(patient_dir: Path) -> Path:
    candidates = sorted(patient_dir.glob("*-summary.txt"))
    if candidates:
        return candidates[0]
    candidates = sorted(patient_dir.glob("*summary*.txt"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No CHB-MIT summary annotation file found in {patient_dir.resolve()}"
    )


# -----------------------------------------------------------------------------
# SIGNAL PROCESSING
# -----------------------------------------------------------------------------


def normalize_channel_name(name: str) -> str:
    """Canonicalize whitespace and case while keeping clinically meaningful labels."""

    cleaned = re.sub(r"\s+", " ", name.strip())
    return cleaned.upper()


def _pick_eeg_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Keep EEG channels only, excluding non-EEG auxiliary channels."""

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError("EDF contains no channels recognized as EEG by MNE.")
    raw.pick(picks)
    raw.rename_channels({ch: normalize_channel_name(ch) for ch in raw.ch_names})
    return raw


def preprocess_raw(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, List[str], float]:
    """Apply standard resampling + band-pass + notch filtering to an MNE Raw object."""

    raw = _pick_eeg_channels(raw)

    if not np.isclose(raw.info["sfreq"], TARGET_SAMPLING_RATE_HZ):
        raw.resample(TARGET_SAMPLING_RATE_HZ, npad="auto", verbose=False)

    # Band-pass removes slow drift and very-high-frequency components.
    raw.filter(
        l_freq=LOW_CUTOFF_HZ,
        h_freq=HIGH_CUTOFF_HZ,
        picks="eeg",
        method="fir",
        fir_design="firwin",
        phase="zero-double",
        verbose=False,
    )

    # 60 Hz is the mains frequency used for the standard CHB-MIT configuration.
    # It is outside our 40 Hz upper passband, but the explicit notch remains in
    # the pipeline as a documented project requirement and is harmless.
    if NOTCH_FREQUENCY_HZ < raw.info["sfreq"] / 2.0:
        raw.notch_filter(
            freqs=[NOTCH_FREQUENCY_HZ],
            picks="eeg",
            verbose=False,
        )

    data_uv = raw.get_data(picks="eeg", units="uV").astype(np.float32, copy=False)
    channel_names = [normalize_channel_name(ch) for ch in raw.ch_names]

    nonfinite_fraction = float(np.mean(~np.isfinite(data_uv)))
    if nonfinite_fraction > MAX_NONFINITE_FRACTION:
        raise ValueError(
            f"Too many non-finite samples after preprocessing: "
            f"{nonfinite_fraction:.4%}"
        )

    if not np.isfinite(data_uv).all():
        # Only a tiny number of invalid values are allowed. Interpolate them using
        # nearest finite values per channel rather than silently replacing with 0.
        for channel_index in range(data_uv.shape[0]):
            row = data_uv[channel_index]
            bad = ~np.isfinite(row)
            if bad.any():
                finite_indices = np.flatnonzero(~bad)
                if finite_indices.size == 0:
                    raise ValueError(
                        f"Channel {channel_names[channel_index]} contains no finite values."
                    )
                row[bad] = np.interp(np.flatnonzero(bad), finite_indices, row[~bad])

    channel_std = np.std(data_uv, axis=1)
    flat_channels = channel_std <= MIN_CHANNEL_STD_UV
    if np.all(flat_channels):
        raise ValueError("All EEG channels are effectively flat after preprocessing.")
    if flat_channels.any():
        # Retain the non-flat EEG channels. This avoids injecting synthetic values
        # into the model while still allowing recordings with an isolated dead lead.
        keep = ~flat_channels
        data_uv = data_uv[keep]
        channel_names = [name for name, ok in zip(channel_names, keep) if ok]

    return data_uv, channel_names, float(raw.info["sfreq"])


def load_and_preprocess_recording(
    edf_path: Path,
    seizure_intervals: Sequence[SeizureInterval],
) -> CleanEEGRecording:
    """Load one EDF and return a validated CleanEEGRecording."""

    patient_id = edf_path.parent.name.upper()
    recording_id = f"{edf_path.parent.name.upper()}_{edf_path.stem.upper()}"

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        infer_types=True,
        verbose=False,
    )
    data_uv, channel_names, sampling_rate_hz = preprocess_raw(raw)

    duration_seconds = data_uv.shape[1] / sampling_rate_hz
    cleaned_intervals = [
        interval
        for interval in seizure_intervals
        if interval.start_seconds < duration_seconds
        and interval.end_seconds > 0
    ]

    # Clip annotation intervals to the actual recording boundaries.
    clipped: List[SeizureInterval] = []
    for interval in cleaned_intervals:
        start = max(0.0, interval.start_seconds)
        end = min(duration_seconds, interval.end_seconds)
        if end > start:
            clipped.append(SeizureInterval(start, end))

    recording = CleanEEGRecording(
        patient_id=patient_id,
        recording_id=recording_id,
        file_path=str(edf_path.resolve()),
        channel_names=channel_names,
        sampling_rate_hz=sampling_rate_hz,
        data_uv=np.ascontiguousarray(data_uv, dtype=np.float32),
        duration_seconds=duration_seconds,
        seizure_intervals=clipped,
    )
    recording.validate()
    return recording


def clear_preprocessed_cache(
    cache_dir: str | Path = PREPROCESSED_CACHE_DIR,
    manifest_path: str | Path = PREPROCESSED_MANIFEST_PATH,
) -> None:
    """Delete only the generated preprocessing cache, never the source EDF data."""

    cache = Path(cache_dir)
    if cache.exists():
        for child in sorted(cache.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        cache.rmdir()
    Path(manifest_path).unlink(missing_ok=True)
    Path(PREPROCESSED_CACHE_PATH).unlink(missing_ok=True)


def save_preprocessed_recording(
    recording: CleanEEGRecording,
    output_path: str | Path,
) -> Path:
    """Write one recording shard so no whole-patient matrix stays resident."""

    recording.validate()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    joblib.dump(recording, temporary, compress=3)
    temporary.replace(output)
    return output


def load_preprocessed_recording(input_path: str | Path) -> CleanEEGRecording:
    recording = joblib.load(input_path)
    if not isinstance(recording, CleanEEGRecording):
        raise TypeError(f"Unexpected object in recording cache: {input_path}")
    recording.validate()
    return recording


def preprocess_dataset_to_disk(
    path: Optional[str | Path] = None,
    cache_dir: str | Path = PREPROCESSED_CACHE_DIR,
    manifest_path: str | Path = PREPROCESSED_MANIFEST_PATH,
    clean_cache: bool = True,
) -> Path:
    """Preprocess one EDF at a time and persist patient/recording shards.

    The old implementation retained every patient's full EEG matrix in one
    Python list. This implementation writes each recording immediately, then
    deletes its NumPy arrays and forces garbage collection before reading the
    next file. The manifest is the only object needed by the ML stage.
    """

    if clean_cache:
        clear_preprocessed_cache(cache_dir, manifest_path)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(path if path is not None else PATIENT_DATASET_PATH)
    patient_dirs = discover_patient_directories(dataset_path)
    manifest: Dict[str, object] = {
        "version": 1,
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "config": preprocessing_config(),
        "patients": [],
    }

    for patient_dir in patient_dirs:
        summary_path = discover_summary_file(patient_dir)
        annotations = parse_chb_summary(summary_path)
        edf_files = discover_edf_files(patient_dir)
        patient_id = patient_dir.name.upper()
        patient_cache_dir = cache_root / patient_id
        patient_cache_dir.mkdir(parents=True, exist_ok=True)
        recording_entries: List[dict] = []

        for edf_path in edf_files:
            intervals = annotations.get(edf_path.name, [])
            recording = load_and_preprocess_recording(edf_path, intervals)
            shard_path = patient_cache_dir / f"{recording.recording_id}.joblib"
            save_preprocessed_recording(recording, shard_path)
            recording_entries.append(
                {
                    "recording_id": recording.recording_id,
                    "path": str(shard_path.resolve()),
                }
            )
            del recording
            gc.collect()

        patient_entry = {
            "patient_id": patient_id,
            "source_directory": str(patient_dir.resolve()),
            "recordings": recording_entries,
        }
        cast_patients = manifest["patients"]
        assert isinstance(cast_patients, list)
        cast_patients.append(patient_entry)
        temporary_manifest = Path(manifest_path).with_suffix(".tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_manifest.replace(Path(manifest_path))
        gc.collect()
        LOGGER.info("Cached patient %s (%d recordings) to disk.", patient_id, len(recording_entries))

    if not manifest["patients"]:
        raise RuntimeError("No usable EEG recordings were cached.")
    return Path(manifest_path)


def iter_preprocessed_recordings(
    manifest_path: str | Path = PREPROCESSED_MANIFEST_PATH,
):
    """Yield one cached recording at a time and release it after each yield."""

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Preprocessed manifest not found: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for patient in manifest.get("patients", []):
        for entry in patient.get("recordings", []):
            yield load_preprocessed_recording(entry["path"])


def load_preprocessed_manifest(
    manifest_path: str | Path = PREPROCESSED_MANIFEST_PATH,
) -> dict:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Preprocessed manifest not found: {manifest_file}")
    return json.loads(manifest_file.read_text(encoding="utf-8"))


def load_dataset(path: Optional[str | Path] = None) -> List[CleanEEGRecording]:
    """Compatibility loader; prefer preprocess_dataset_to_disk for full datasets."""

    dataset_path = Path(path if path is not None else PATIENT_DATASET_PATH)
    patient_dirs = discover_patient_directories(dataset_path)
    recordings: List[CleanEEGRecording] = []

    for patient_dir in patient_dirs:
        summary_path = discover_summary_file(patient_dir)
        annotations = parse_chb_summary(summary_path)
        for edf_path in discover_edf_files(patient_dir):
            intervals = annotations.get(edf_path.name, [])
            if edf_path.name not in annotations:
                LOGGER.warning("No summary entry for %s; treating as seizure-free.", edf_path.name)
            recordings.append(load_and_preprocess_recording(edf_path, intervals))

    if not recordings:
        raise RuntimeError("No usable EEG recordings were loaded.")
    return recordings


def save_preprocessed_recordings(
    recordings: Sequence[CleanEEGRecording],
    output_path: str | Path = PREPROCESSED_CACHE_PATH,
) -> Path:
    """Persist the preprocessing output for reproducible downstream processing."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(list(recordings), output, compress=3)
    LOGGER.info("Saved %d recordings to %s", len(recordings), output.resolve())
    return output


def load_preprocessed_recordings(
    input_path: str | Path = PREPROCESSED_CACHE_PATH,
) -> List[CleanEEGRecording]:
    """Load the preprocessing cache created by save_preprocessed_recordings."""

    recordings = joblib.load(input_path)
    if not isinstance(recordings, list):
        raise TypeError(
            "Preprocessed cache must contain a list of CleanEEGRecording objects."
        )
    for recording in recordings:
        if not isinstance(recording, CleanEEGRecording):
            raise TypeError("Preprocessed cache contains an unexpected object type.")
        recording.validate()
    return recordings


def preprocessing_config() -> Dict[str, object]:
    """Return configuration metadata for reproducibility and model versioning."""

    return {
        "dataset_path": PATIENT_DATASET_PATH,
        "target_sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
        "low_cutoff_hz": LOW_CUTOFF_HZ,
        "high_cutoff_hz": HIGH_CUTOFF_HZ,
        "notch_frequency_hz": NOTCH_FREQUENCY_HZ,
        "window_duration_seconds": WINDOW_DURATION_SECONDS,
    }

if __name__ == "__main__":
    manifest = preprocess_dataset_to_disk()
    LOGGER.info("Preprocessing complete. Disk manifest: %s", manifest.resolve())