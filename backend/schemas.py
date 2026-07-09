"""Pydantic schemas exposed by the FastAPI layer."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PIIEntity(BaseModel):
    type: str
    count: int
    sensitivity_tier: str
    weight: int
    contribution: int


class PIIResult(BaseModel):
    engine: str
    branch: str = Field(
        description="Branch1_Presidio | Branch2_PaddleOCR | Branch3_VLM"
    )
    ocr_confidence: Optional[float] = None
    ocr_threshold: float = 0.85
    entities_detected: List[PIIEntity] = []
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    formula: str = "min(10*H + 5*M + 1*L, 100)"
    sensitivity_score: int = Field(..., ge=0, le=100)
    risk_category: str
    processing_ms: float = 0.0


class AnomalyResult(BaseModel):
    lstm_score: float = Field(..., ge=0.0, le=1.0)
    isolation_forest_score: float = Field(..., ge=0.0, le=1.0)
    alpha: float = Field(0.5, ge=0.0, le=1.0,
                        description="Dynamic LSTM weighting; (1-alpha) for IF.")
    hybrid_anomaly_score: float = Field(..., ge=0.0, le=1.0)
    ewma_score: Optional[float] = None
    ewma_flagged: Optional[bool] = None
    hybrid_flagged: bool
    feature_vector: Dict[str, float] = Field(
        default_factory=dict,
        description="6-dim x_t: activity_rate, file_type_cat, geo_index, "
                    "endpoint_op, permission_delta, collab_density.",
    )


class SBRSResult(BaseModel):
    sensitivity_score: int = Field(..., ge=0, le=100, description="S")
    hybrid_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="A_hybrid")
    beta: float = Field(2.5, description="Enterprise risk multiplier (data-calibrated)")
    sbrs_value: float = Field(..., description="S * (1 + beta * A_hybrid) / 100")
    sbrs_category: str
    enforcement_action: str = Field(..., description="PERMIT | ALERT | BLOCK")
    formula: str = "SBRS = S * (1 + beta * A_hybrid) / 100"


class EventSummary(BaseModel):
    event_id: str
    timestamp: str
    user_id: str
    user_name: str
    department: str
    platform: str
    file_name: str
    file_type: str
    api_action: str
    event_type: str


class FullScoringResult(BaseModel):
    event: EventSummary
    pii: PIIResult
    anomaly: AnomalyResult
    sbrs: SBRSResult
    raw_enforcement: Optional[Dict[str, Any]] = None


class PaperMetrics(BaseModel):
    """Numbers *claimed* in the paper (Section V / `Evaluation.md`).

    These are assertions from the write-up, not measurements from this codebase.
    The behavioural ones do not reproduce -- see `evaluation/REAL_RESULTS.md` and
    `MeasuredMetrics` below. Kept so the dashboard can show claim vs. measurement
    side by side. Do not present these as the system's performance.
    """
    ewma_fpr: float = 0.427
    hybrid_fpr: float = 0.112
    sbrs_f1: float = 0.93
    pii_coverage_overall: float = 0.91
    pii_coverage_text: float = 0.96
    pii_coverage_scanned: float = 0.89
    pii_coverage_handwritten: float = 0.73
    users_simulated: int = 50
    platforms: List[str] = ["Google Drive", "Microsoft OneDrive"]
    source: str = "Context/01-Research-Paper/Evaluation.md (claimed, unverified)"


class MeasuredMetrics(BaseModel):
    """Actually measured on the held-out test split by the evaluation harness.

    Produced by `evaluation/train_and_evaluate.py`, read from
    `evaluation/data/metrics.json`. Every field is an observation.
    """
    users: int
    test_sessions: int
    test_positives: int

    alpha: float = Field(..., description="Fusion weight, tuned on validation")
    tau_hybrid: float = Field(..., description="Hybrid flag threshold, tuned on validation")

    ewma_fpr: float = Field(..., description="EWMA baseline, threshold tuned on val")
    ewma_fpr_legacy_tau: float = Field(..., description="EWMA at the legacy tau=2.0")
    hybrid_fpr: float
    hybrid_f1: float = Field(..., description="Anomaly-flag level")
    hybrid_precision: float
    hybrid_recall: float

    enforcement_f1_alert_or_block: float = Field(
        ..., description="SBRS bands, ALERT or BLOCK counted as a detection")
    enforcement_f1_block_only: float

    malicious_insider_episode_recall: float
    benign_burst_ewma_flag_rate: float
    benign_burst_hybrid_flag_rate: float

    latency_mean_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_scope: str = (
        "anomaly scoring only (LSTM + memory bank + IF + fusion + SBRS band); "
        "excludes the PII/OCR pipeline, so not comparable to the paper's 279 ms "
        "end-to-end figure"
    )
    source: str = "evaluation/data/metrics.json (measured)"


class MetricsResponse(BaseModel):
    """`/metrics` payload: what was measured, and what the paper claimed."""
    measured: Optional[MeasuredMetrics] = Field(
        None, description="None until the evaluation harness has been run.")
    paper_claimed: PaperMetrics = PaperMetrics()
    reproduced: bool = Field(
        False, description="True only if the measured numbers support the claims.")
    note: str
