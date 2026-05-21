"""
risk_orchestrator.py
====================
Semantic-Behavioral Risk Score (SBRS) orchestration (Section IV of the paper).

    SBRS = S * (1 + beta * A_hybrid) / 100

  * S         : payload sensitivity in [0, 100]              (pii_pipeline.py)
  * A_hybrid  : fused behavioural anomaly score in [0, 1]    (anomaly_engine.py)
  * beta      : tunable enterprise risk multiplier (paper-default 0.5).

A small S (benign data) keeps SBRS low regardless of anomaly magnitude -
this is exactly what mathematically suppresses the false-positive cascade
that plagues legacy EWMA-only DLP systems. Conversely, even modest anomaly
on highly sensitive content amplifies SBRS into the BLOCK band.

Enforcement bands (calibrated from the simulation dataset):
    SBRS  < 0.20 -> SAFE      -> PERMIT
    SBRS  < 0.60 -> SENSITIVE -> ALERT
    SBRS >= 0.60 -> HIGH-RISK -> BLOCK
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import anomaly_engine, pii_pipeline
from .data_loader import HybridSaaSDataset
from .schemas import (AnomalyResult, EventSummary, FullScoringResult,
                      PIIResult, SBRSResult)


DEFAULT_BETA: float = 0.5

# Calibrated to anomaly_detection_comparison.csv (see EDA in repo README).
SBRS_BANDS = [
    (0.60, "HIGH-RISK", "BLOCK"),
    (0.20, "SENSITIVE", "ALERT"),
    (0.00, "SAFE",      "PERMIT"),
]


def _band(sbrs: float) -> tuple[str, str]:
    for threshold, category, action in SBRS_BANDS:
        if sbrs >= threshold:
            return category, action
    return "SAFE", "PERMIT"


def compute_sbrs(sensitivity_score: int,
                 hybrid_anomaly_score: float,
                 beta: float = DEFAULT_BETA) -> SBRSResult:
    """Pure math entrypoint - no I/O, no dataset coupling."""
    s = max(0, min(100, int(sensitivity_score)))
    a = max(0.0, min(1.0, float(hybrid_anomaly_score)))
    sbrs = s * (1.0 + beta * a) / 100.0
    category, action = _band(sbrs)
    return SBRSResult(
        sensitivity_score=s,
        hybrid_anomaly_score=round(a, 4),
        beta=beta,
        sbrs_value=round(sbrs, 4),
        sbrs_category=category,
        enforcement_action=action,
    )


def _event_summary(event: Dict[str, Any]) -> EventSummary:
    req = event.get("request", {})
    return EventSummary(
        event_id=event.get("event_id", "?"),
        timestamp=str(event.get("timestamp", "")),
        user_id=req.get("user_id", "?"),
        user_name=req.get("user_name", "?"),
        department=req.get("department", "?"),
        platform=req.get("platform", "?"),
        file_name=req.get("file_name", "?"),
        file_type=req.get("file_type", "?"),
        api_action=req.get("api_action", "?"),
        event_type=event.get("event_type", "?"),
    )


def score_event(dataset: HybridSaaSDataset,
                event_id: str,
                alpha: float = anomaly_engine.DEFAULT_ALPHA,
                beta: float = DEFAULT_BETA) -> FullScoringResult:
    """End-to-end MITM-proxy simulation for a single intercepted event."""
    event = dataset.get_event(event_id)
    if event is None:
        raise KeyError(f"Unknown event_id '{event_id}'")

    pii: PIIResult = pii_pipeline.scan_event(event)
    anomaly: AnomalyResult = anomaly_engine.evaluate(dataset, event, alpha=alpha)
    sbrs: SBRSResult = compute_sbrs(
        sensitivity_score=pii.sensitivity_score,
        hybrid_anomaly_score=anomaly.hybrid_anomaly_score,
        beta=beta,
    )

    return FullScoringResult(
        event=_event_summary(event),
        pii=pii,
        anomaly=anomaly,
        sbrs=sbrs,
        raw_enforcement=event.get("enforcement"),
    )
