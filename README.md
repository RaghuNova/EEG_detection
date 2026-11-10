# NEUROSCAN

NEUROSCAN is a local EEG analysis application with a React/Vite interface and a FastAPI backend. It processes each EDF recording independently and reports the existing ML, CNN-BiLSTM, and Attention model outputs. It is a research and analysis support tool, not a standalone medical diagnosis.

## Requirements

- Python 3.12 or 3.13
- Node.js 18+ and npm
- The model and calibration files listed below
- No API keys or external inference service

Python dependencies include PyTorch, MNE, NumPy, SciPy, Pandas, scikit-learn, XGBoost, joblib, FastAPI, Uvicorn, python-multipart, and ReportLab. Frontend dependencies are declared in `frontend/package.json`.

## Model files

Keep these existing artifacts in the project root at the following paths:

```text
models/seizure_ml_model.joblib
models/final.pt (1).zip
models/final.pt.zip
dependencies/hybrid_attention_threshold_calibration.json
```

The backend defaults to these paths. The two PyTorch archives are loaded by the existing CNN and Attention adapters; do not rename or replace them. `dependencies/threshold_calibration.json` is also present as a calibration reference.

The publishable hybrid calibration file retains the selected inference parameters unchanged and omits per-patient and per-recording out-of-fold evaluation records. The original calibration file remains in ignored local storage and is not included in Git history.

## Install

Run commands from the project root (`minor_project`). Create a virtual environment and install backend packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch mne pandas numpy scipy scikit-learn imbalanced-learn xgboost joblib pyedflib fastapi 'uvicorn[standard]' python-multipart reportlab matplotlib
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Install frontend packages in a second terminal (or after backend setup):

```bash
cd frontend
npm install
cd ..
```

For the longer environment notes, see [`PROJECT_SETUP.md`](PROJECT_SETUP.md).

## Configure

Optional local configuration is documented in `.env.example`. Copy it if you want a starting point:

```bash
cp .env.example .env
```

The current backend uses project-relative model and storage defaults. Start it from the project root so those model paths resolve correctly. No external API keys are needed. Uploaded EEG files remain on the local machine.

## Run

Open two terminals in the project root.

**Terminal 1 — API:**

```bash
source .venv/bin/activate
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

The API health endpoint is <http://127.0.0.1:8000/api/health> and the interactive API docs are at <http://127.0.0.1:8000/docs>.

**Terminal 2 — web app:**

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite (normally <http://localhost:5173>). The frontend API client defaults to the local backend at port 8000.

## Use

1. Start both the API and frontend as above.
2. Open the app and go to **Patient Upload**.
3. Select an EDF file or a folder containing EDF files. Folder selection is supported by browsers that expose directory upload; individual EDF selection is available as a fallback.
4. Review the detected files and submit the analysis.
5. Follow processing status, then open the resulting patient analysis, model pages, comparison, signal viewer, and PDF report.
6. Prior analyses and reports are stored under `storage/analyses` and `storage/reports`; uploaded EDFs are under `storage/uploads`.

Each recording is processed separately. A failed EDF is reported separately from recordings that complete. Model predictions are inference outputs and are not clinical diagnoses.

## Development and checks

Build the frontend with:

```bash
cd frontend
npm run build
```

Run the existing Python tests from the project root with the virtual environment active:

```bash
pytest
```

## Project layout

```text
backend/api/             FastAPI routes
backend/inference/       Preprocessing and model adapters/orchestration
backend/reporting/       PDF report generation
backend/storage/         Local JSON, upload, and report storage
frontend/src/            React application and pages
dl/                      Existing deep-learning model and calibration code
models/                  Existing trained model artifacts
dependencies/            Existing calibration configuration
test/ and tests/         Existing tests
```

## Limitations and data handling

- Real predictions require the listed artifacts and compatible dependencies.
- No clinical performance claim is made by this application.
- Analysis JSON, uploaded files, and reports are stored locally; manage and remove patient data according to your requirements.
- The API is configured for local development. Do not expose it to an untrusted network without adding appropriate authentication and deployment security.
