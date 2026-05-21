# HybridSaaS-Sec - Interactive Demo

An interactive, paper-faithful demonstration of the framework introduced in:

> **Mir & Hashir,** _A Scalable Framework for Privacy-Preserving API Monitoring
> and Multimodal Data Leakage Prevention in Enterprise Cloud Environments_,
> Software Design and Architecture Research Paper, 2026.

The framework simulates a Man-in-the-Middle (MITM) proxy that intercepts
SaaS API traffic, runs a **multimodal PII pipeline** (Presidio /
PaddleOCR / PaliGemma-3B VLM), evaluates **hybrid behavioural anomalies**
(LSTM h=128 + Isolation Forest), and orchestrates a unified
**Semantic-Behavioral Risk Score (SBRS)** for enforcement.

All numbers surfaced in the dashboard - **EWMA FPR 42.7 % vs Hybrid FPR
11.2 %**, **SBRS F1 = 0.93**, **overall PII coverage 0.91** - come directly
from the paper and the shipped simulation artefacts (`*.csv`, `*.json`).
Nothing is invented.

---

## Project layout

```
SDA Research Project/
+-- backend/                       # FastAPI + scoring core
|   +-- __init__.py
|   +-- data_loader.py             # Pandas ingestion of CSV/JSON artefacts
|   +-- pii_pipeline.py            # Branch 1/2/3 multimodal PII (simulated)
|   +-- anomaly_engine.py          # LSTM + Isolation Forest (simulated)
|   +-- risk_orchestrator.py       # SBRS = S * (1 + beta * A_hybrid) / 100
|   +-- schemas.py                 # Pydantic models
|   +-- main.py                    # FastAPI service
+-- frontend/
|   +-- app.py                     # Streamlit dashboard (3 tabs)
+-- pii_scan_results.csv
+-- anomaly_detection_comparison.csv
+-- ewma_user_baselines.csv
+-- blocked_events_audit.csv
+-- hybridSaaS_events.json
+-- hybridSaaS_system.log
+-- requirements.txt
+-- README.md
```

---

## Setup (Windows / PowerShell)

```powershell
# 1. Create + activate a virtual environment (Python >= 3.11 recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -U pip
python -m pip install -r requirements.txt
```

> **Note on Python 3.14:** pre-built wheels for pandas / numpy may not be
> available yet for 3.14. If installation tries to build pandas from source,
> use Python 3.11 / 3.12 instead.

---

## Run

In **two separate terminals**, both with the venv activated:

```powershell
# Terminal 1 - FastAPI backend (http://127.0.0.1:8000/docs)
uvicorn backend.main:app --reload --port 8000
```

```powershell
# Terminal 2 - Streamlit dashboard (http://localhost:8501)
streamlit run frontend/app.py
```

The Streamlit app imports the backend modules directly, so it works even
if the FastAPI process isn't running. Start uvicorn only if you want to
exercise the REST surface (e.g. via `curl` or the Swagger UI at `/docs`).

---

## REST endpoints (FastAPI)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`                        | Health + dataset summary |
| `GET`  | `/metrics`                 | Paper-published metrics (FPR, F1, coverage) |
| `GET`  | `/events?limit=N`          | List of intercepted events |
| `GET`  | `/events/{event_id}`       | Raw event payload |
| `GET`  | `/score/{event_id}`        | Full pipeline result (PII + Anomaly + SBRS) |
| `POST` | `/score`                   | Score an arbitrary event payload |
| `GET`  | `/baselines`               | Legacy EWMA per-user baselines (8 windows) |
| `GET`  | `/audit/blocked`           | Audit trail of BLOCK enforcement actions |
| `GET`  | `/comparison`              | Tidy EWMA vs Hybrid per-event evaluation |

Query parameters `alpha` (LSTM weight, default 0.5) and `beta` (enterprise
risk multiplier, default 0.5) on `/score` let you replay the simulation
under different SBRS configurations.

---

## Mathematical core (Section IV of the paper)

```
A_hybrid = alpha * a_LSTM + (1 - alpha) * a_IF
SBRS     = S * (1 + beta * A_hybrid) / 100
```

* `S`         in `[0, 100]` from `min(10*high + 5*medium + 1*low, 100)`.
* `a_LSTM`    in `[0, 1]` - cosine similarity vs user's 24h memory bank.
* `a_IF`      in `[0, 1]` - Isolation Forest outlier score vs global pop.
* `alpha`     dynamic LSTM weighting (default 0.5).
* `beta`      enterprise risk multiplier (default 0.5).

Enforcement bands (calibrated to `anomaly_detection_comparison.csv`):

| SBRS range | Category   | Action |
|------------|------------|--------|
| `< 0.20`   | SAFE       | PERMIT |
| `< 0.60`   | SENSITIVE  | ALERT  |
| `>= 0.60`  | HIGH-RISK  | BLOCK  |

---

## Verifying the data loader

The data loader (`backend/data_loader.py`) ships with a smoke test:

```powershell
python -m backend.data_loader
```

Expected output (counts may differ if the dataset is updated):

```
DATA_DIR : c:\Users\...\SDA Research Project
Events   : 38 | metadata: EVAL-2026-04-13-001
PII scans rows / cols : (40, 21)
Anomaly  rows / cols  : (50, 24)
EWMA     rows / cols  : (80, 11)
Blocked  rows / cols  : (8, 12)
First event id        : EVT-00001
```
