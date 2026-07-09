# REST Endpoints (FastAPI)

The FastAPI backend exposes the following REST endpoints:

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

You can use query parameters `alpha` (LSTM weight, default 0.5) and `beta` (enterprise risk multiplier, default 0.5) on `/score` to replay the simulation under different SBRS configurations.
