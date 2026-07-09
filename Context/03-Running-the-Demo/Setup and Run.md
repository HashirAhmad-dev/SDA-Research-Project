# Setup and Run

## Setup (Windows / PowerShell)

1. **Create and activate a virtual environment (Python >= 3.11 recommended):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. **Install dependencies:**
```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```
*(Note: Use Python 3.11 / 3.12 if pre-built wheels for pandas/numpy are not available for 3.14)*

## Run the Demo

In **two separate terminals**, both with the venv activated:

**Terminal 1 - FastAPI backend:**
```powershell
uvicorn backend.main:app --reload --port 8000
```
This serves the backend at `http://127.0.0.1:8000/docs`.

**Terminal 2 - Streamlit dashboard:**
```powershell
streamlit run frontend/app.py
```
This opens the UI at `http://localhost:8501`.

*(The Streamlit app imports backend modules directly, so it can run even without the FastAPI process, but starting uvicorn enables the REST API).*
