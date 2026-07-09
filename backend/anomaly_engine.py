"""
anomaly_engine.py
=================
Hybrid Behavioural Anomaly Engine (Section III.C of the paper) - REAL inference.

This module used to replay hardcoded per-event scores out of
`anomaly_detection_comparison.csv`. It no longer does. Every score below is
produced by a trained model:

    a_LSTM : a torch LSTM (hidden dim 128) encodes a 24-step window of the
             user's 6-d feature vector in that user's *personal* z-frame; the
             final hidden state is compared by cosine similarity against that
             user's historical memory bank (mean of the top-k neighbours).
    a_IF   : a fitted sklearn IsolationForest scores x_t against the *global*
             organisational distribution.

    A_hybrid = alpha * a_LSTM + (1 - alpha) * a_IF

Both raw scores are pushed through calibrators fitted on held-out training data
so the fusion weight is meaningful. Weights, thresholds, calibrators, memory
banks and per-user statistics all come from `evaluation/artifacts/`, produced by

    python -m evaluation.generate_sessions
    python -m evaluation.train_and_evaluate

Measured behaviour of these artefacts is documented in `evaluation/REAL_RESULTS.md`.
If the artefacts are absent, `evaluate()` raises rather than inventing numbers.

Adapting the legacy demo events
-------------------------------
`hybridSaaS_events.json` predates the trained feature space and does not carry
all six features, nor 24 h of per-user history. The adapter below is explicit
about every substitution it makes:

  * activity_rate           = activity_rate_per_hour / that user's EWMA baseline
                              (from ewma_user_baselines.csv). Missing -> 1.0.
  * file_type_category      = binned from the event's PII sensitivity score.
                              The legacy schema has no public/internal/
                              confidential/restricted label. NOTE: this couples
                              a_LSTM/a_IF weakly to S, which SBRS multiplies
                              again -- a limitation of the legacy event schema,
                              not of the engine.
  * geo_index               = haversine(request origin, that user's modal city
                              across the dataset) / 5000 km.
  * endpoint_operation      = ordinal class of api_action.
  * permission_scope_delta  = cross-department access proxy (0.0 same dept,
                              0.15 otherwise). The legacy events carry no OAuth
                              scope data.
  * collaboration_density   = unavailable -> the user's own baseline (z = 0).

  * history                 = the legacy events are a single day with no hourly
                              telemetry, so the 24-step window is padded with the
                              user's personal baseline and the event occupies the
                              final step. a_LSTM therefore reflects the
                              *instantaneous* deviation from that user's routine,
                              not multi-hour drift. Callers holding real history
                              can pass it via `evaluate(..., history=...)`.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .data_loader import HybridSaaSDataset
from .schemas import AnomalyResult

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get(
    "HYBRIDSAAS_ARTIFACTS_DIR", _PROJECT_ROOT / "evaluation" / "artifacts"))

SEQ_LEN = 24
_FEATURE_KEYS = ("activity_rate", "file_type_category", "geo_index",
                 "endpoint_operation", "permission_scope_delta",
                 "collab_network_density")

# Same city table the evaluation harness uses for geo_index.
_CITIES = {
    "karachi": (24.86, 67.01), "lahore": (31.55, 74.34),
    "islamabad": (33.68, 73.05), "rawalpindi": (33.60, 73.04),
    "faisalabad": (31.42, 73.08), "peshawar": (34.01, 71.58),
    "dubai": (25.20, 55.27), "moscow": (55.75, 37.62), "lagos": (6.52, 3.38),
    "singapore": (1.35, 103.82), "amsterdam": (52.37, 4.90),
    "kyiv": (50.45, 30.52), "sao paulo": (-23.55, -46.63),
}
_GEO_NORM_KM = 5000.0

_ENDPOINT_ORDINAL = {"VIEW": 0.0, "READ": 0.0, "DOWNLOAD": 0.0,
                     "UPLOAD": 0.25, "WRITE": 0.25, "EDIT": 0.25,
                     "SHARE": 0.5, "DELETE": 0.75,
                     "PERMISSION_CHANGE": 1.0, "PERMISSION": 1.0}


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------
class EngineUnavailable(RuntimeError):
    """Raised when the trained artefacts are missing."""


@lru_cache(maxsize=1)
def load_engine() -> Dict[str, Any]:
    """Load the trained LSTM + IsolationForest + calibrators (cached)."""
    pkl, pt = ARTIFACTS_DIR / "engine.pkl", ARTIFACTS_DIR / "lstm.pt"
    if not pkl.exists() or not pt.exists():
        raise EngineUnavailable(
            f"Trained anomaly artefacts not found in {ARTIFACTS_DIR}. "
            "Build them with:\n"
            "    python -m evaluation.generate_sessions\n"
            "    python -m evaluation.train_and_evaluate\n"
            "This module refuses to emit simulated scores."
        )
    import torch  # deferred: keeps `import backend` cheap when unused

    from evaluation.train_and_evaluate import BehaviouralLSTM

    with open(pkl, "rb") as f:
        art = pickle.load(f)
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)

    model = BehaviouralLSTM(n_features=ckpt["n_features"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    art["model"] = model
    art["torch"] = torch

    # Fallback baseline / memory bank for users the engine has never seen.
    mus = np.stack([m for m, _ in art["user_stats"].values()])
    sds = np.stack([s for _, s in art["user_stats"].values()])
    art["global_stats"] = (mus.mean(axis=0), sds.mean(axis=0))
    allb = np.concatenate(list(art["banks"].values()))
    rng = np.random.default_rng(0)
    if len(allb) > 512:
        allb = allb[rng.choice(len(allb), 512, replace=False)]
    art["global_bank"] = allb

    logger.info("Loaded trained anomaly engine: alpha=%.2f tau_hybrid=%.2f "
                "banks=%d", art["alpha"], art["tau_hybrid"], len(art["banks"]))
    return art


def _engine_defaults() -> Tuple[float, float]:
    try:
        art = load_engine()
        return float(art["alpha"]), float(art["tau_hybrid"])
    except Exception:                     # keep `import backend.main` working
        return 0.5, 0.5


# Imported by backend/main.py and frontend/app.py.
DEFAULT_ALPHA, HYBRID_FLAG_THRESHOLD = _engine_defaults()


# ---------------------------------------------------------------------------
# Legacy-event -> feature-vector adapter
# ---------------------------------------------------------------------------
def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _city(geo: Optional[str]) -> Optional[Tuple[float, float]]:
    if not geo:
        return None
    return _CITIES.get(str(geo).split(",")[0].strip().lower())


_HABITUAL_CACHE: Dict[int, Dict[str, str]] = {}


def _habitual(dataset: HybridSaaSDataset) -> Dict[str, str]:
    key = id(dataset)
    if key not in _HABITUAL_CACHE:
        per_user: Dict[str, Counter] = {}
        for ev in dataset.events:
            req = ev.get("request") or {}
            uid, geo = req.get("user_id"), req.get("geo_location")
            if uid and geo:
                per_user.setdefault(uid, Counter())[geo] += 1
        _HABITUAL_CACHE[key] = {u: c.most_common(1)[0][0] for u, c in per_user.items()}
    return _HABITUAL_CACHE[key]


def _ewma_baseline(dataset: HybridSaaSDataset, user_id: str) -> Optional[float]:
    """That user's 1-hour EWMA state from ewma_user_baselines.csv."""
    df = dataset.ewma_baselines
    if df.empty:
        return None
    rows = df[(df.user_id == user_id) & (df.time_window == "1h")]
    if rows.empty:
        rows = df[df.user_id == user_id]
    if rows.empty:
        return None
    val = rows.iloc[-1].get("current_ewma", None)
    if val is None or not np.isfinite(val) or val <= 0:
        val = rows.iloc[-1].get("baseline_ewma", None)
    return float(val) if val is not None and np.isfinite(val) and val > 0 else None


def _sensitivity_to_file_class(s: Optional[float]) -> float:
    """Bin a 0-100 PII sensitivity score onto the paper's 4-level ordinal."""
    if s is None:
        return 1.0 / 3.0                      # 'internal' - the population mode
    if s >= 75:
        return 1.0                            # restricted
    if s >= 50:
        return 2.0 / 3.0                      # confidential
    if s >= 18:
        return 1.0 / 3.0                      # internal
    return 0.0                                # public


def build_feature_vector(dataset: HybridSaaSDataset,
                         event: Dict[str, Any]) -> Dict[str, float]:
    """The paper's 6-dim x_t, in natural units, from a legacy demo event."""
    req = event.get("request") or {}
    uid = req.get("user_id")

    rate = req.get("activity_rate_per_hour")
    base = _ewma_baseline(dataset, uid) if uid else None
    activity_rate = float(rate) / base if (rate and base) else 1.0
    activity_rate = float(np.clip(activity_rate, 0.0, 10.0))

    s = (event.get("pii_detection") or {}).get("sensitivity_score")
    file_type_category = _sensitivity_to_file_class(
        float(s) if s is not None else None)

    origin = _city(req.get("geo_location"))
    home = _city(_habitual(dataset).get(uid)) if uid else None
    if origin and home:
        geo_index = min(_haversine_km(origin, home) / _GEO_NORM_KM, 1.0)
    elif origin or home:
        geo_index = 0.0                       # only one known -> assume habitual
    else:
        geo_index = 0.5                       # genuinely unknown

    endpoint_operation = _ENDPOINT_ORDINAL.get(
        str(req.get("api_action", "")).upper(), 0.0)

    # Cross-department access proxy. 0.15 is the mean of the non-zero scope
    # deltas the engine was trained on; the 0.6 the simulated engine used here
    # is negligent-insider magnitude and, against a ~0.05 per-user sigma,
    # z-scores past the clip and pins a_LSTM at 1.0 for almost every event.
    fdep, dep = req.get("file_department"), req.get("department")
    permission_scope_delta = 0.0 if (fdep is None or fdep == dep) else 0.15

    # No collaboration signal in the legacy schema -> the user's own baseline.
    art = load_engine()
    mu, _ = art["user_stats"].get(uid, art["global_stats"])
    collab = float(mu[5])

    return {
        "activity_rate": round(activity_rate, 4),
        "file_type_category": round(file_type_category, 4),
        "geo_index": round(geo_index, 4),
        "endpoint_operation": round(endpoint_operation, 4),
        "permission_scope_delta": round(permission_scope_delta, 4),
        "collab_network_density": round(collab, 4),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _memory_bank_distance(h: np.ndarray, bank: np.ndarray, k: int) -> float:
    if len(bank) == 0:
        return 0.0
    cos = h @ bank.T
    kk = min(k, cos.shape[-1])
    return float(1.0 - np.partition(cos, -kk, axis=-1)[..., -kk:].mean())


def score_features(x: np.ndarray, user_id: Optional[str],
                   alpha: float,
                   history: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Run the real LSTM + IsolationForest on one 6-d vector.

    `history`: optional (SEQ_LEN-1, 6) array of the user's preceding hourly
    feature vectors in natural units. Absent -> padded with that user's baseline.
    """
    art = load_engine()
    torch = art["torch"]

    mu, sd = art["user_stats"].get(user_id, art["global_stats"])
    bank = art["banks"].get(user_id, art["global_bank"])

    # Personal z-frame window: pad with the user's own baseline (z = 0).
    win = np.zeros((SEQ_LEN, len(x)), dtype=np.float32)
    if history is not None and len(history):
        hz = np.clip((np.asarray(history, dtype=float) - mu) / sd,
                     -art["z_clip"], art["z_clip"])
        win[-1 - len(hz):-1] = hz[-(SEQ_LEN - 1):]
    win[-1] = np.clip((x - mu) / sd, -art["z_clip"], art["z_clip"])

    with torch.no_grad():
        h = art["model"].encode(torch.from_numpy(win[None])).numpy()[0]
    h = h / max(float(np.linalg.norm(h)), 1e-9)

    raw_lstm = _memory_bank_distance(h, bank, int(art["topk"]))
    raw_if = -float(art["iforest"].score_samples(
        art["scaler"].transform(x.reshape(1, -1)))[0])

    a_lstm = float(art["cal_lstm"].transform(np.array([raw_lstm]))[0])
    a_if = float(art["cal_if"].transform(np.array([raw_if]))[0])
    a_hybrid = float(np.clip(alpha * a_lstm + (1.0 - alpha) * a_if, 0.0, 1.0))
    return {"a_lstm": a_lstm, "a_if": a_if, "a_hybrid": a_hybrid}


def _ewma(dataset: HybridSaaSDataset, event: Dict[str, Any]
          ) -> Tuple[Optional[float], Optional[bool]]:
    """Legacy univariate comparator, computed - not replayed."""
    req = event.get("request") or {}
    uid, rate = req.get("user_id"), req.get("activity_rate_per_hour")
    prev = _ewma_baseline(dataset, uid) if uid else None
    if rate is None or prev is None:
        return None, None
    art = load_engine()
    x = float(rate)
    deviation = (x - prev) / max(prev, 1e-6)
    ewma_new = 0.3 * x + 0.7 * prev          # lambda = 0.3, per the paper
    return round(ewma_new, 4), bool(deviation > art["tau_ewma"])


def evaluate(dataset: HybridSaaSDataset,
             event: Dict[str, Any],
             alpha: float = DEFAULT_ALPHA,
             history: Optional[np.ndarray] = None) -> AnomalyResult:
    """Score one intercepted API transaction with the trained hybrid engine."""
    art = load_engine()                       # raises EngineUnavailable if absent
    fv = build_feature_vector(dataset, event)
    x = np.array([fv[k] for k in _FEATURE_KEYS], dtype=float)

    user_id = (event.get("request") or {}).get("user_id")
    scores = score_features(x, user_id, alpha, history=history)
    ewma_new, ewma_flag = _ewma(dataset, event)

    return AnomalyResult(
        lstm_score=round(scores["a_lstm"], 4),
        isolation_forest_score=round(scores["a_if"], 4),
        alpha=alpha,
        hybrid_anomaly_score=round(scores["a_hybrid"], 4),
        ewma_score=ewma_new,
        ewma_flagged=ewma_flag,
        hybrid_flagged=scores["a_hybrid"] >= float(art["tau_hybrid"]),
        feature_vector=fv,
    )
