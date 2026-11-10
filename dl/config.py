"""Central configuration for the CUDA-first independent DL branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DL_ARTIFACTS_DIR = ARTIFACTS_DIR / "dl"
PREPROCESSED_MANIFEST_PATH = ARTIFACTS_DIR / "preprocessed_manifest.json"
SPLIT_MANIFEST_PATH = DL_ARTIFACTS_DIR / "split_manifest.json"
AUDIT_REPORT_PATH = DL_ARTIFACTS_DIR / "audit_report.json"

SEED = 42
EXPECTED_PATIENT_COUNT = 20
SAMPLING_RATE_HZ = 256
WINDOW_SECONDS = 20
WINDOW_STRIDE_SECONDS = 20
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_SECONDS
POSITIVE_OVERLAP_SECONDS = 1.0

# DL training uses the NVIDIA CUDA device by default. Pass --device cpu only
# for environments without CUDA or for small unit-test runs.
DEFAULT_DEVICE = "cuda"
USE_CUDA_AMP = True

# Event evaluation configuration
DEFAULT_SEIZURE_THRESHOLD = 0.50
EVENT_MIN_CONSECUTIVE_POSITIVE_WINDOWS = 1
EVENT_MAX_GAP_WINDOWS = 1
EVENT_MATCH_MIN_OVERLAP_SECONDS = 1.0

# Directory paths for outputs
CNN_BILSTM_ARTIFACTS_DIR = DL_ARTIFACTS_DIR / "cnn_bilstm"
ATTENTION_ARTIFACTS_DIR = DL_ARTIFACTS_DIR / "attention"
COMPARISON_ARTIFACTS_DIR = DL_ARTIFACTS_DIR / "comparison"

