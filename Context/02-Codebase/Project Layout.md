# Project Layout

The HybridSaaS-Sec simulation consists of a backend FastAPI and a frontend Streamlit application.

```text
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
