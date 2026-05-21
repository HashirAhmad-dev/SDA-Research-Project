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
    beta: float = Field(0.5, description="Enterprise risk multiplier")
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
    """Headline empirical numbers reported in the paper."""
    ewma_fpr: float = 0.427
    hybrid_fpr: float = 0.112
    sbrs_f1: float = 0.93
    pii_coverage_overall: float = 0.91
    pii_coverage_text: float = 0.96
    pii_coverage_scanned: float = 0.89
    pii_coverage_handwritten: float = 0.73
    users_simulated: int = 50
    platforms: List[str] = ["Google Drive", "Microsoft OneDrive"]
