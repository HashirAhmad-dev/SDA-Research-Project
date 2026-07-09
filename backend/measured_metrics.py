"""
measured_metrics.py
===================
Reads the *measured* headline numbers out of `evaluation/data/metrics.json`,
the artefact written by `evaluation/train_and_evaluate.py`.

Exists so that neither the API nor the dashboard has to hardcode a performance
figure. If the evaluation harness has not been run, `load_measured_metrics()`
returns None and callers must say "not measured" rather than fall back to the
paper's claims. See `evaluation/REAL_RESULTS.md`.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .schemas import MeasuredMetrics

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = Path(os.environ.get(
    "HYBRIDSAAS_METRICS_JSON", _PROJECT_ROOT / "evaluation" / "data" / "metrics.json"))

NOT_MEASURED_NOTE = (
    "No measured metrics available: run `python -m evaluation.generate_sessions` "
    "then `python -m evaluation.train_and_evaluate`. The paper's claimed numbers "
    "are shown for reference only and are known not to reproduce."
)

MEASURED_NOTE = (
    "Measured on a held-out test split by evaluation/train_and_evaluate.py. "
    "The paper's behavioural claims do not reproduce: measured EWMA FPR is far "
    "below the claimed 42.7%, measured hybrid FPR far below the claimed 11.2%, "
    "and enforcement-band F1 is nowhere near the claimed 0.93. See "
    "evaluation/REAL_RESULTS.md for the full comparison and caveats."
)


@lru_cache(maxsize=1)
def load_measured_metrics() -> Optional[MeasuredMetrics]:
    """Parse metrics.json into a typed model, or None if it has not been built."""
    if not METRICS_PATH.exists():
        logger.warning("Measured metrics not found at %s", METRICS_PATH)
        return None
    try:
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        ds, flag = m["dataset"], m["test_flag_level"]
        enf, burst = m["test_enforcement_bands"], m["benign_burst"]
        lat, tune = m["latency"], m["tuning"]
        return MeasuredMetrics(
            users=ds["users"],
            test_sessions=ds["per_split"]["test"]["sessions"],
            test_positives=ds["per_split"]["test"]["positives"],
            alpha=tune["alpha"],
            tau_hybrid=tune["tau_hybrid"],
            ewma_fpr=flag["ewma_tuned"]["fpr"],
            ewma_fpr_legacy_tau=flag["ewma_legacy_tau"]["fpr"],
            hybrid_fpr=flag["hybrid"]["fpr"],
            hybrid_f1=flag["hybrid"]["f1"],
            hybrid_precision=flag["hybrid"]["precision"],
            hybrid_recall=flag["hybrid"]["recall"],
            enforcement_f1_alert_or_block=enf["hybrid_alert_or_block"]["f1"],
            enforcement_f1_block_only=enf["hybrid_block_only"]["f1"],
            malicious_insider_episode_recall=(
                m["per_class_recall"]["malicious_insider"]["hybrid_episode_recall"]),
            benign_burst_ewma_flag_rate=burst["ewma_flag_rate"],
            benign_burst_hybrid_flag_rate=burst["hybrid_flag_rate"],
            latency_mean_ms=lat["mean_ms"],
            latency_p95_ms=lat["p95_ms"],
            latency_p99_ms=lat["p99_ms"],
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("Malformed %s: %s", METRICS_PATH, exc)
        return None
