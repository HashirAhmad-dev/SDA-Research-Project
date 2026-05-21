"""
main.py
=======
FastAPI surface that simulates the HybridSaaS-Sec MITM proxy back-end.

Endpoints
---------
GET  /                    Health + dataset summary
GET  /metrics             Headline metrics published in the paper
GET  /events              Lightweight list of intercepted events
GET  /events/{event_id}   Raw intercepted event payload
GET  /score/{event_id}    Full PII + Anomaly + SBRS scoring for one event
POST /score               Score an arbitrary event payload (raw JSON)
GET  /baselines           Legacy EWMA per-user baselines (8 windows)
GET  /audit/blocked       Blocked-event audit trail
GET  /comparison          Tidy EWMA vs Hybrid per-event evaluation rows

Run with:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .anomaly_engine import DEFAULT_ALPHA
from .data_loader import load_all
from .risk_orchestrator import DEFAULT_BETA, score_event
from .schemas import FullScoringResult, PaperMetrics

app = FastAPI(
    title="HybridSaaS-Sec API",
    version=__version__,
    description=(
        "Simulated Man-in-the-Middle proxy back-end for the HybridSaaS-Sec "
        "framework (Mir & Hashir, 2026). Combines a multimodal PII pipeline "
        "with a hybrid LSTM + Isolation Forest anomaly engine and orchestrates "
        "the Semantic-Behavioral Risk Score (SBRS) for enforcement."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    ds = load_all()
    meta = ds.log_metadata
    return {
        "service": "HybridSaaS-Sec API",
        "version": __version__,
        "run_id": meta.get("run_id"),
        "saas_platforms": meta.get("saas_platforms"),
        "users_monitored": meta.get("users_monitored"),
        "total_events": meta.get("total_events"),
        "pii_branches": meta.get("pii_branches"),
        "anomaly_engines": meta.get("anomaly_engines"),
    }


@app.get("/metrics", response_model=PaperMetrics)
def metrics() -> PaperMetrics:
    """Headline empirical metrics reported in Section V of the paper."""
    return PaperMetrics()


# ---------------------------------------------------------------------------
@app.get("/events")
def list_events(limit: int = Query(100, ge=1, le=1000),
                event_type: Optional[str] = None) -> List[Dict[str, Any]]:
    ds = load_all()
    out = []
    for ev in ds.events:
        if event_type and ev.get("event_type") != event_type:
            continue
        req = ev.get("request", {})
        out.append({
            "event_id": ev.get("event_id"),
            "timestamp": ev.get("timestamp"),
            "event_type": ev.get("event_type"),
            "module": ev.get("module"),
            "user_id": req.get("user_id"),
            "user_name": req.get("user_name"),
            "department": req.get("department"),
            "platform": req.get("platform"),
            "file_name": req.get("file_name"),
            "file_type": req.get("file_type"),
            "api_action": req.get("api_action"),
            "enforcement_action": (ev.get("enforcement") or {}).get("action"),
        })
        if len(out) >= limit:
            break
    return out


@app.get("/events/{event_id}")
def get_event(event_id: str) -> Dict[str, Any]:
    ds = load_all()
    ev = ds.get_event(event_id)
    if ev is None:
        raise HTTPException(404, f"Unknown event_id '{event_id}'")
    return ev


# ---------------------------------------------------------------------------
@app.get("/score/{event_id}", response_model=FullScoringResult)
def score(event_id: str,
          alpha: float = Query(DEFAULT_ALPHA, ge=0.0, le=1.0),
          beta: float = Query(DEFAULT_BETA, ge=0.0, le=5.0)) -> FullScoringResult:
    """Run the full MITM-proxy pipeline (PII -> Anomaly -> SBRS) for one event."""
    ds = load_all()
    try:
        return score_event(ds, event_id, alpha=alpha, beta=beta)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/score", response_model=FullScoringResult)
def score_payload(payload: Dict[str, Any] = Body(...),
                  alpha: float = Query(DEFAULT_ALPHA, ge=0.0, le=1.0),
                  beta: float = Query(DEFAULT_BETA, ge=0.0, le=5.0)) -> FullScoringResult:
    """Score an arbitrary event payload following the hybridSaaS_events.json schema."""
    from .anomaly_engine import evaluate as eval_anomaly
    from .pii_pipeline import scan_event
    from .risk_orchestrator import _event_summary, compute_sbrs

    ds = load_all()
    pii = scan_event(payload)
    anomaly = eval_anomaly(ds, payload, alpha=alpha)
    sbrs = compute_sbrs(pii.sensitivity_score, anomaly.hybrid_anomaly_score, beta=beta)
    return FullScoringResult(
        event=_event_summary(payload),
        pii=pii,
        anomaly=anomaly,
        sbrs=sbrs,
        raw_enforcement=payload.get("enforcement"),
    )


# ---------------------------------------------------------------------------
@app.get("/baselines")
def baselines(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    ds = load_all()
    df = ds.ewma_baselines
    if user_id:
        df = df[df["user_id"] == user_id]
    return df.assign(last_updated=df["last_updated"].astype(str)).to_dict("records")


@app.get("/audit/blocked")
def blocked_audit() -> List[Dict[str, Any]]:
    ds = load_all()
    df = ds.blocked_audit
    return df.assign(timestamp=df["timestamp"].astype(str)).to_dict("records")


@app.get("/comparison")
def comparison() -> List[Dict[str, Any]]:
    """EWMA vs Hybrid per-event evaluation - reproduces F1=0.93 / FPR=11.2%."""
    ds = load_all()
    df = ds.anomaly_comparison.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict("records")
