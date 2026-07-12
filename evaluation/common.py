"""
common.py
=========
Shared contracts between `generate_sessions.py` and `train_and_evaluate.py`.

Everything that has to stay byte-compatible with the legacy artefacts
(`anomaly_detection_comparison.csv`, `backend/risk_orchestrator.py`) is defined
exactly once, here.

Reverse-engineered from the legacy CSV (all 50 rows agree):

    ewma_new        = lam * x_t + (1 - lam) * ewma_previous      (lam = 0.3)
    ewma_deviation  = (x_t - ewma_previous) / ewma_previous
    ewma_flagged    = ewma_deviation > tau_ewma                  (legacy tau = 2.0)
    base_action     = BLOCK if ewma_flagged else PERMIT
    hybrid_score    = alpha * a_LSTM + (1 - alpha) * a_IF
    hybrid_flagged  = hybrid_score >= 0.5
    sbrs_value      = S * (1 + beta * hybrid_score) / 100        (beta = 0.5)
    hybrid_action   = band(sbrs_value)                           (0.20 / 0.60)
    ewma_correct    = (ewma_flagged   == is_true_threat)
    hybrid_correct  = (hybrid_flagged == is_true_threat)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
DATA_DIR = EVAL_DIR / "data"

SESSIONS_RAW = DATA_DIR / "sessions_raw.csv"
USER_PROFILES = DATA_DIR / "user_profiles.csv"
GEN_CONFIG = DATA_DIR / "generation_config.json"

# Drop-in replacement target (project root, next to the legacy CSV).
V2_CSV = PROJECT_ROOT / "anomaly_detection_comparison_v2_real.csv"

# ---------------------------------------------------------------------------
# The paper's six-dimensional behavioural feature vector (Sec. III.C)
# ---------------------------------------------------------------------------
FEATURES: Tuple[str, ...] = (
    "activity_rate",
    "file_type_category",
    "geo_index",
    "endpoint_operation",
    "permission_scope_delta",
    "collaboration_density",
)

# Ordinal encodings (paper: "public, internal, confidential, restricted").
FILE_TYPE_CLASSES = ("public", "internal", "confidential", "restricted")
FILE_TYPE_ORDINAL = {c: i / 3.0 for i, c in enumerate(FILE_TYPE_CLASSES)}

# Paper: "read, write, share, delete, permission change".
ENDPOINT_CLASSES = ("read", "write", "share", "delete", "permission_change")
ENDPOINT_ORDINAL = {c: i / 4.0 for i, c in enumerate(ENDPOINT_CLASSES)}

# Payload sensitivity S in [0, 100], keyed by file-type class.
SENSITIVITY_BY_FILE_CLASS = {
    "public": 8.0,
    "internal": 28.0,
    "confidential": 62.0,
    "restricted": 85.0,
}

THREAT_CLASSES = (
    "malicious_insider",
    "compromised_account",
    "negligent_insider",
    "overscoped_thirdparty",
)

# LSTM sequence length: the paper's "24-hour window", one step per hour.
SEQ_LEN = 24

# ---------------------------------------------------------------------------
# Model / enforcement constants (must match backend/)
# ---------------------------------------------------------------------------
EWMA_LAMBDA = 0.3          # ewma_user_baselines.csv, 1h window
LEGACY_EWMA_TAU = 2.0      # ewma_user_baselines.csv, anomaly_threshold
HYBRID_FLAG_THRESHOLD = 0.5
# beta and the band cut-points are data-calibrated on the validation split by
# evaluation/calibrate_sbrs.py; see evaluation/SBRS_RECALIBRATION.md. They must
# stay in step with backend/risk_orchestrator.py, which serves them at runtime.
SBRS_BETA = 2.5

# backend/risk_orchestrator.py::SBRS_BANDS  (threshold, category, action)
SBRS_BANDS = [
    (1.84, "HIGH-RISK", "BLOCK"),
    (1.22, "SENSITIVE", "ALERT"),
    (0.00, "SAFE", "PERMIT"),
]

# The pre-calibration bands (beta=0.5, cut-points 0.50/1.00). Kept only as the
# sensitivity check reported in REAL_RESULTS.md / SBRS_RECALIBRATION.md.
PAPER_DOC_BANDS = [
    (1.00, "HIGH-RISK", "BLOCK"),
    (0.50, "SENSITIVE", "ALERT"),
    (0.00, "SAFE", "PERMIT"),
]

# Exact column names AND order of anomaly_detection_comparison.csv.
CSV_COLUMNS: Tuple[str, ...] = (
    "event_id", "timestamp", "user_id", "user_name", "department", "platform",
    "file_accessed", "time_gap_sec", "activity_rate_per_hr", "ewma_previous",
    "ewma_new", "ewma_deviation", "ewma_anomaly_flagged", "lstm_score",
    "isolation_forest_score", "hybrid_anomaly_score", "hybrid_anomaly_flagged",
    "pii_sensitivity_score", "sbrs_value", "sbrs_category", "base_action",
    "hybrid_action", "is_true_threat", "ewma_correct", "hybrid_correct",
)

DEPARTMENTS = ("Finance", "Legal", "HR", "Engineering", "Sales")
PLATFORMS = ("Google Drive", "Microsoft OneDrive")


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------
def sbrs(sensitivity: float, hybrid_score: float, beta: float = SBRS_BETA) -> float:
    """SBRS = S * (1 + beta * A_hybrid) / 100."""
    s = min(100.0, max(0.0, float(sensitivity)))
    a = min(1.0, max(0.0, float(hybrid_score)))
    return s * (1.0 + beta * a) / 100.0


def band(value: float, bands=SBRS_BANDS) -> Tuple[str, str]:
    """Map an SBRS value onto (category, enforcement_action)."""
    for threshold, category, action in bands:
        if value >= threshold:
            return category, action
    return "SAFE", "PERMIT"


# ---------------------------------------------------------------------------
# EWMA baseline (legacy univariate comparator)
# ---------------------------------------------------------------------------
def ewma_stream(x: Sequence[float], lam: float = EWMA_LAMBDA):
    """Causal EWMA over one user's chronological activity-rate series.

    Returns (previous, new, deviation) arrays. `deviation` is the signed
    relative deviation (x_t - EWMA_{t-1}) / EWMA_{t-1}, matching the legacy CSV.
    The first observation seeds the filter and gets deviation 0 (never flagged).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    prev = np.zeros(n)
    new = np.zeros(n)
    dev = np.zeros(n)
    if n == 0:
        return prev, new, dev

    state = x[0]
    prev[0], new[0], dev[0] = state, state, 0.0
    for i in range(1, n):
        prev[i] = state
        dev[i] = (x[i] - state) / max(state, 1e-6)
        state = lam * x[i] + (1.0 - lam) * state
        new[i] = state
    return prev, new, dev


# ---------------------------------------------------------------------------
# Score calibration
# ---------------------------------------------------------------------------
class RobustCalibrator:
    """Monotone map raw anomaly score -> [0, 1], fitted on TRAIN data only.

    a_LSTM (a cosine distance) and a_IF (a negated isolation depth) live on
    completely different scales. Fusing them with `alpha * a_LSTM + (1-alpha) * a_IF`
    is only meaningful once both share a scale.

    We deliberately do NOT use the empirical CDF (rank) transform: that maps the
    calibration data to a uniform [0, 1], under which a *typical* window scores
    ~0.5 and the paper's fixed `hybrid_flag >= 0.5` would flag half of all
    traffic. Instead we anchor the median of normal traffic at 0 and the
    p99.5 tail at 1, so "normal ~ 0, extreme ~ 1" holds and both the 0.5 flag
    threshold and the SBRS multiplier keep their intended meaning.

        a_cal = clip((raw - p50) / (p99.5 - p50), 0, 1)
    """

    def __init__(self, lo_q: float = 50.0, hi_q: float = 99.5) -> None:
        self.lo_q, self.hi_q = lo_q, hi_q
        self.lo: float = 0.0
        self.hi: float = 1.0

    def fit(self, raw: np.ndarray) -> "RobustCalibrator":
        raw = np.asarray(raw, dtype=float)
        self.lo = float(np.percentile(raw, self.lo_q))
        self.hi = float(np.percentile(raw, self.hi_q))
        if self.hi <= self.lo:
            self.hi = self.lo + 1e-9
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        return np.clip((raw - self.lo) / (self.hi - self.lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval - honest error bars on small per-class counts."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    halfwidth = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion matrix + precision/recall/F1/FPR for boolean arrays."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
    }
