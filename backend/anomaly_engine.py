"""
anomaly_engine.py
=================
Simulated Hybrid Behavioural Anomaly Engine (Section III.C of the paper).

For each API transaction at time-step t the user's activity is encoded into
the paper's 6-dimensional feature vector x_t:

    [activity_rate, file_type_category, geo_index,
     endpoint_operation, permission_scope_delta, collab_network_density]

Two parallel scorers are then evaluated:

    a_LSTM : 128-d LSTM over a 24h window, cosine-similarity vs the user's
             historical memory bank (temporal anomaly).
    a_IF   : Isolation Forest evaluated on x_t vs the global organisational
             distribution (structural outlier).

They are fused with a dynamic weighting parameter alpha:

    A_hybrid = alpha * a_LSTM + (1 - alpha) * a_IF

This module replays the deterministic per-event scores produced by the
paper's offline simulation (anomaly_detection_comparison.csv +
hybridSaaS_events.json) so the dashboard mirrors the published F1=0.93 /
FPR=11.2% numbers exactly. No real ML model is invoked.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .data_loader import HybridSaaSDataset
from .schemas import AnomalyResult


# Default fusion weight. The CSV's hybrid_anomaly_score == 0.5*lstm + 0.5*IF
# across all 50 rows, so alpha=0.5 reproduces the paper's simulation exactly.
DEFAULT_ALPHA: float = 0.5
HYBRID_FLAG_THRESHOLD: float = 0.5  # row.hybrid_anomaly_flagged uses this


def _feature_vector(event: Dict[str, Any],
                    anomaly_row: Optional[pd.Series]) -> Dict[str, float]:
    """Build the 6-dim x_t for display purposes (paper Sec. III.C)."""
    req = event.get("request", {}) if event else {}
    ba = event.get("behavioral_analysis") or {}

    activity_rate = float(
        (anomaly_row.get("activity_rate_per_hr") if anomaly_row is not None else None)
        or ba.get("activity_rate_per_hr")
        or 0.0
    )

    file_type_map = {
        "text_extractable": 0.2,
        "scanned_image": 0.6,
        "handwritten": 0.9,
        "binary": 0.4,
    }
    file_type_cat = file_type_map.get(str(req.get("file_type", "")).lower(), 0.5)

    # Coarse Pakistan-only geo index for the simulation (Karachi vs. other).
    geo = str(req.get("geo_location", ""))
    geo_index = 0.1 if "Karachi" in geo else (0.7 if geo else 0.5)

    endpoint_map = {"DOWNLOAD": 0.7, "SHARE": 0.9, "UPLOAD": 0.4,
                    "VIEW": 0.1, "DELETE": 0.8}
    endpoint_op = endpoint_map.get(str(req.get("api_action", "")).upper(), 0.3)

    # Permission scope delta / collab density are not stored explicitly in
    # every event -> derive proxies from cross-department access.
    perm_delta = 0.0 if req.get("department") == req.get("file_department") else 0.6
    collab_density = float(ba.get("collab_density", 0.3) or 0.3)

    return {
        "activity_rate": round(activity_rate, 4),
        "file_type_category": file_type_cat,
        "geo_index": geo_index,
        "endpoint_operation": endpoint_op,
        "permission_scope_delta": perm_delta,
        "collab_network_density": collab_density,
    }


def _match_anomaly_row(dataset: HybridSaaSDataset,
                       event: Dict[str, Any]) -> Optional[pd.Series]:
    """Look up the precomputed anomaly evaluation for this event, if any."""
    df = dataset.anomaly_comparison
    if df.empty:
        return None

    user_id = event.get("request", {}).get("user_id")
    file_name = event.get("request", {}).get("file_name")
    ts = pd.to_datetime(event.get("timestamp"), errors="coerce")

    if not user_id or not file_name:
        return None

    cand = df[(df["user_id"] == user_id) & (df["file_accessed"] == file_name)]
    if cand.empty:
        return None

    if pd.notna(ts):
        cand = cand.assign(_dt=(cand["timestamp"] - ts).abs())
        cand = cand.sort_values("_dt")
    return cand.iloc[0]


def evaluate(dataset: HybridSaaSDataset,
             event: Dict[str, Any],
             alpha: float = DEFAULT_ALPHA) -> AnomalyResult:
    """Run the simulated LSTM + IsolationForest evaluation for one event."""
    row = _match_anomaly_row(dataset, event)
    ba = event.get("behavioral_analysis") or {}

    if row is not None:
        a_lstm = float(row["lstm_score"])
        a_if = float(row["isolation_forest_score"])
        ewma_score = float(row["ewma_new"]) if pd.notna(row["ewma_new"]) else None
        ewma_flag = bool(row["ewma_anomaly_flagged"]) \
            if pd.notna(row["ewma_anomaly_flagged"]) else None
    else:
        # No precomputed row -> fall back to behavioural_analysis block in the
        # event payload (some events carry it) or neutral defaults.
        a_lstm = float(ba.get("lstm_score", 0.05))
        a_if = float(ba.get("isolation_forest_score", 0.05))
        ewma_score = ba.get("ewma_score")
        ewma_flag = ba.get("ewma_flagged")

    a_hybrid = alpha * a_lstm + (1.0 - alpha) * a_if
    a_hybrid = max(0.0, min(1.0, a_hybrid))

    return AnomalyResult(
        lstm_score=round(a_lstm, 4),
        isolation_forest_score=round(a_if, 4),
        alpha=alpha,
        hybrid_anomaly_score=round(a_hybrid, 4),
        ewma_score=ewma_score,
        ewma_flagged=ewma_flag,
        hybrid_flagged=a_hybrid >= HYBRID_FLAG_THRESHOLD,
        feature_vector=_feature_vector(event, row),
    )
