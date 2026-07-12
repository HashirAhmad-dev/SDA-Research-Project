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
    """Headline numbers as published in the paper (Section V / `Evaluation.md`).

    The paper now reports the measured results, so these agree with
    `MeasuredMetrics` below by construction -- they are the same experiment, read
    from the write-up rather than from `metrics.json`. Every value here was
    produced by `evaluation/train_and_evaluate.py` (behaviour) or
    `evaluation/run_pipeline.py` (PII); see `evaluation/REAL_RESULTS.md` and
    `evaluation/REAL_RESULTS_PII.md`.
    """
    ewma_fpr: float = 0.0581            # legacy EWMA, threshold tuned on val
    hybrid_fpr: float = 0.0
    hybrid_f1: float = 0.629            # anomaly-flag level
    sbrs_f1: float = 0.475              # enforcement bands, recalibrated beta=2.5

    # PII cascade recall at the paper's tau_ocr = 0.85, VLM = gemma-3-4b-it.
    pii_coverage_overall: float = 0.823
    pii_coverage_text: float = 0.828
    pii_coverage_scanned: float = 0.889
    pii_coverage_handwritten: float = 0.752

    users_simulated: int = 50
    platforms: List[str] = ["Google Drive", "Microsoft OneDrive"]
    source: str = "Context/01-Research-Paper/Evaluation.md"


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
    """`/metrics` payload: the measured numbers, and the same ones as published."""
    measured: Optional[MeasuredMetrics] = Field(
        None, description="None until the evaluation harness has been run.")
    paper_published: PaperMetrics = PaperMetrics()
    reproduced: bool = Field(
        False, description="True when the measured numbers match the publication.")
    note: str
