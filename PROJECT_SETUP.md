# NEUROSCAN — Project Setup and Environment Documentation

## 1. System Overview & Architecture

**NEUROSCAN** is a production-grade EEG intelligence and seizure analysis web platform. It integrates:
1. **Existing ML Model**: Tabular feature extraction (122 invariant features) + Trained XGBoost classifier (`models/seizure_ml_model.joblib`, decision threshold ~0.279).
2. **CNN-BiLSTM Deep Learning Model**: Multichannel raw EEG (22 canonical channels, 20s windows at 256 Hz = 5120 samples) -> Conv1D feature extraction -> BiLSTM temporal encoding (`models/final.pt (1).zip`, threshold 0.50).
3. **AttentionClassifier Model**: 4-head additive temporal attention + gated residual fusion operating on intermediate recurrent temporal features (`[batch, 128, 128]`) extracted from the CNN-BiLSTM model (`models/final.pt.zip`).
4. **Dynamic Threshold Calibration**: Recording-specific hybrid quantile calibration loaded from `dependencies/hybrid_attention_threshold_calibration.json` with event episode formation and persistence filtering.

---

## 2. Verified Model Artifacts & Roles

| Artifact Path | Model Component | Architecture / Details | Decision Threshold |
| :--- | :--- | :--- | :--- |
| `models/seizure_ml_model.joblib` | Model 1: Tabular ML | `XGBClassifier` + `StandardScaler` (122 features) | **0.2790** (persisted in bundle) |
| `models/final.pt (1).zip` | Model 2: CNN-BiLSTM | `CNNBiLSTM(input_channels=22, recurrent_length=128)` | **0.50** (standard logistic cutoff) |
| `models/final.pt.zip` | Model 3: AttentionClassifier | `AttentionClassifier(input_dim=128, num_heads=4)` | **Dynamic recording-specific** (loaded from hybrid JSON, baseline floor=0.03, quantile=0.985) |
| `dependencies/hybrid_attention_threshold_calibration.json` | Dynamic Calibration | `hybrid_global_recording_quantile` policy | Adaptive per recording |
| `dependencies/threshold_calibration.json` | Static Global Calibration | Global threshold baseline (0.03) | Reference only |

---

## 3. Environment & Hardware Prerequisites

- **Operating System**: macOS (Apple Silicon or Intel), Linux (Ubuntu 20.04+), or Windows (WSL2 recommended).
- **Python Version**: Python 3.12 or 3.13 (Python 3.13.9 verified).
- **Node.js Version**: Node.js v18.0.0 or higher (v26.9.0 verified).
- **Package Managers**: `uv` (recommended for Python venv & dependency management) and `npm` (v11+).
- **Hardware Acceleration**:
  - Apple Silicon: Metal Performance Shaders (`mps`) supported for PyTorch.
  - NVIDIA: CUDA supported where available.
  - CPU: Fully supported as standard inference fallback.
- **External API Keys**: **NONE**. The application runs 100% locally with zero external network dependencies for inference and analysis.

---

## 4. Installation Instructions

### Step 4.1: Python Backend Environment

```bash
# In the project root directory (<project-root>)

# 1. Create a dedicated virtual environment with uv
uv venv .venv

# 2. Install all required backend dependencies
uv pip install --python .venv/bin/python \
    torch \
    mne \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    imbalanced-learn \
    xgboost \
    pyedflib \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    reportlab \
    matplotlib \
    pytest
```

### Step 4.2: Frontend Application Setup

The frontend is built with React, Vite, Tailwind CSS, Lucide icons, Recharts, and Plotly.js.

```bash
# Navigate to the frontend directory
cd frontend

# Install frontend dependencies
npm install
```

---

## 5. Configuration & Environment Variables

Copy `.env.example` to `.env` in the root and/or frontend directory:

```bash
cp .env.example .env
```

Default environment variables:
```env
# Server Configuration
HOST=127.0.0.1
PORT=8000
EEG_VERBOSE=0

# Model Paths
ML_MODEL_PATH=models/seizure_ml_model.joblib
CNN_MODEL_PATH=models/final.pt (1).zip
ATTENTION_MODEL_PATH=models/final.pt.zip
CALIBRATION_PATH=dependencies/hybrid_attention_threshold_calibration.json

# Storage Configuration
UPLOAD_DIR=storage/uploads
ANALYSIS_DIR=storage/analyses
REPORT_DIR=storage/reports
```

---

## 6. Running the Application

### Option A: Run Backend and Frontend Concurrently

**Terminal 1 — Backend API Server:**
```bash
source .venv/bin/activate
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Frontend Dev Server:**
```bash
cd frontend
npm run dev
```

Visit the dashboard at: **`http://localhost:5173`** (or the port Vite outputs).

---

## 7. Running Verification Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```
