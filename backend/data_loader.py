"""
data_loader.py
==============
Step 2 of the HybridSaaS-Sec implementation plan.

Ingests the simulated enterprise telemetry shipped with the paper into typed
Pandas DataFrames / dictionaries. These artefacts act as the dashboard's
"database" and feed every downstream module:

    pii_scan_results.csv            -> Branch 1/2/3 PII scanner output
    anomaly_detection_comparison.csv-> EWMA vs LSTM+IF per-event evaluation
    ewma_user_baselines.csv         -> Legacy EWMA baselines (8 windows, lambda=0.3)
    blocked_events_audit.csv        -> Enforcement audit trail
    hybridSaaS_events.json          -> Full intercepted API events (request + PII +
                                       anomaly + enforcement payloads)
    hybridSaaS_system.log           -> Raw system log (optional, returned as text)

All numeric columns are coerced; timestamps are parsed to pandas datetimes.
Loaders are cached so the FastAPI process and the Streamlit process each
read the files at most once per run.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Project layout:
#   <project_root>/
#       backend/data_loader.py        <- this file
#       *.csv, *.json, *.log          <- simulated dataset (project root)
#
# Override with the HYBRIDSAAS_DATA_DIR environment variable when needed.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
DATA_DIR = Path(os.environ.get("HYBRIDSAAS_DATA_DIR", _PROJECT_ROOT))


FILES = {
    "pii_scans": "pii_scan_results.csv",
    "anomaly_comparison": "anomaly_detection_comparison.csv",
    "ewma_baselines": "ewma_user_baselines.csv",
    "blocked_audit": "blocked_events_audit.csv",
    "events_json": "hybridSaaS_events.json",
    "system_log": "hybridSaaS_system.log",
}


def _path(key: str) -> Path:
    p = DATA_DIR / FILES[key]
    if not p.exists():
        raise FileNotFoundError(
            f"Required dataset '{FILES[key]}' not found at {p}. "
            f"Set HYBRIDSAAS_DATA_DIR or place the file in {DATA_DIR}."
        )
    return p


# ---------------------------------------------------------------------------
# Bundle returned to callers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridSaaSDataset:
    """Typed bundle of every artefact required by the demo."""

    pii_scans: pd.DataFrame
    anomaly_comparison: pd.DataFrame
    ewma_baselines: pd.DataFrame
    blocked_audit: pd.DataFrame
    events: List[Dict[str, Any]]
    log_metadata: Dict[str, Any]
    system_log_path: Path

    # ------------------------------------------------------------------
    # Convenience accessors used by the FastAPI layer
    # ------------------------------------------------------------------
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        for ev in self.events:
            if ev.get("event_id") == event_id:
                return ev
        return None

    def event_ids(self) -> List[str]:
        return [ev["event_id"] for ev in self.events]


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_pii_scans() -> pd.DataFrame:
    """Branch 1/2/3 PII scanner results.

    Numeric columns: ocr_confidence (may be 'N/A'), high/medium/low entity
    counts, sensitivity_score (0-100), base_system_score, processing_ms.
    """
    df = pd.read_csv(_path("pii_scans"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["ocr_confidence"] = pd.to_numeric(df["ocr_confidence"], errors="coerce")
    int_cols = [
        "high_entities", "medium_entities", "low_entities",
        "sensitivity_score", "base_system_score",
    ]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["processing_ms"] = pd.to_numeric(df["processing_ms"], errors="coerce")

    bool_cols = ["slack_alert_sent", "jira_ticket_created"]
    for c in bool_cols:
        df[c] = df[c].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        ).astype("boolean")

    return df


def load_anomaly_comparison() -> pd.DataFrame:
    """Per-event EWMA vs LSTM+IsolationForest evaluation.

    Contains the ground-truth columns used to reproduce the paper's
    EWMA FPR=42.7% vs Hybrid FPR=11.2% / F1=0.93 numbers.
    """
    df = pd.read_csv(_path("anomaly_comparison"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    float_cols = [
        "time_gap_sec", "activity_rate_per_hr",
        "ewma_previous", "ewma_new", "ewma_deviation",
        "lstm_score", "isolation_forest_score", "hybrid_anomaly_score",
        "sbrs_value",
    ]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["pii_sensitivity_score"] = pd.to_numeric(
        df["pii_sensitivity_score"], errors="coerce"
    ).astype("Int64")

    bool_cols = [
        "ewma_anomaly_flagged", "hybrid_anomaly_flagged",
        "is_true_threat", "ewma_correct", "hybrid_correct",
    ]
    for c in bool_cols:
        df[c] = df[c].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        ).astype("boolean")

    return df


def load_ewma_baselines() -> pd.DataFrame:
    """Legacy EWMA baselines per (user, time_window).

    The paper specifies 8 EWMA windows with lambda=0.3 (see anomaly_engines
    metadata in hybridSaaS_events.json). This is the legacy comparator.
    """
    df = pd.read_csv(_path("ewma_baselines"))
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

    for c in ["lambda", "baseline_ewma", "current_ewma",
              "deviation_ratio", "anomaly_threshold"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_blocked_audit() -> pd.DataFrame:
    """Enforcement audit trail (BLOCKED events with SBRS and Jira ticket)."""
    df = pd.read_csv(_path("blocked_audit"), encoding="utf-8")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").astype("Int64")
    df["sbrs"] = pd.to_numeric(df["sbrs"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce").astype("Int64")
    return df


def load_events_json() -> Dict[str, Any]:
    """Raw intercepted API events.

    Returns the full JSON document with two top-level keys:
        - log_metadata: framework run metadata
        - events: list of 38 events, each containing request, pii_detection,
                  behavioral_analysis (where applicable) and enforcement.
    """
    with open(_path("events_json"), "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not {"log_metadata", "events"}.issubset(doc.keys()):
        raise ValueError(
            "hybridSaaS_events.json missing required keys 'log_metadata'/'events'"
        )
    return doc


# ---------------------------------------------------------------------------
# Public entrypoint (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_all() -> HybridSaaSDataset:
    """Load every artefact once and return a frozen bundle."""
    logger.info("Loading HybridSaaS-Sec dataset from %s", DATA_DIR)

    events_doc = load_events_json()
    bundle = HybridSaaSDataset(
        pii_scans=load_pii_scans(),
        anomaly_comparison=load_anomaly_comparison(),
        ewma_baselines=load_ewma_baselines(),
        blocked_audit=load_blocked_audit(),
        events=events_doc["events"],
        log_metadata=events_doc["log_metadata"],
        system_log_path=_path("system_log"),
    )

    logger.info(
        "Loaded %d PII scans | %d anomaly rows | %d baselines | "
        "%d blocked events | %d raw events",
        len(bundle.pii_scans), len(bundle.anomaly_comparison),
        len(bundle.ewma_baselines), len(bundle.blocked_audit),
        len(bundle.events),
    )
    return bundle


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    ds = load_all()
    print("DATA_DIR :", DATA_DIR)
    print("Events   :", len(ds.events), "| metadata:", ds.log_metadata.get("run_id"))
    print("PII scans rows / cols :", ds.pii_scans.shape)
    print("Anomaly  rows / cols  :", ds.anomaly_comparison.shape)
    print("EWMA     rows / cols  :", ds.ewma_baselines.shape)
    print("Blocked  rows / cols  :", ds.blocked_audit.shape)
    print("First event id        :", ds.events[0]["event_id"])
    print("Sample SBRS values    :",
          ds.anomaly_comparison["sbrs_value"].head(5).tolist())
