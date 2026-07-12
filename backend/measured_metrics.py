"""
measured_metrics.py
===================
Reads the *measured* headline numbers out of `evaluation/data/metrics.json`,
the artefact written by `evaluation/train_and_evaluate.py`.

Exists so that neither the API nor the dashboard has to hardcode a performance
figure. If the evaluation harness has not been run, `load_measured_metrics()`
returns None and callers must say "not measured" rather than invent a number.
See `evaluation/REAL_RESULTS.md`.

`load_pii_metrics()` does the same for the multimodal PII cascade
(`evaluation/run_pipeline.py` -> `evaluation/data/pii_metrics_*.json`). The
headline configuration is the paper's tau_ocr = 0.85 with gemma-3-4b-it on
Branch 3 -- the closest routable size to the paper's 3B on-box VLM. The 72B run
is carried alongside as the upper bound. See `evaluation/REAL_RESULTS_PII.md`.
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
    "then `python -m evaluation.train_and_evaluate`."
)

MEASURED_NOTE = (
    "Measured on a held-out test split by evaluation/train_and_evaluate.py "
    "(50 users, chronological 70/15/15 split, nothing tuned on test). These are "
    "the numbers the paper publishes. See evaluation/REAL_RESULTS.md for the "
    "full breakdown, per-class recall and confidence intervals."
)

# The Branch-3 model whose run backs the headline PII figures.
PII_HEADLINE_MODEL = "google-gemma-3-4b-it-deepinfra"
PII_UPPER_BOUND_MODEL = "Qwen-Qwen2-5-VL-72B-Instruct-ovhcloud"


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


@lru_cache(maxsize=4)
def load_pii_metrics(model: str = PII_HEADLINE_MODEL) -> Optional[dict]:
    """Per-category PII precision/recall/F1 for one Branch-3 model, or None.

    Returns the run at the paper's tau_ocr = 0.85, plus the text-only Presidio
    baseline it is measured against and the per-branch latencies.
    """
    path = METRICS_PATH.parent / f"pii_metrics_{model}.json"
    if not path.exists():
        logger.warning("PII metrics not found at %s", path)
        return None
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
        primary = m["results"]["paper_tau_0.85"]
        return {
            "model": m["vlm"]["model"],
            "docs": m["corpus"]["by_category"],
            "gold_entities": m["corpus"]["gold_entities"],
            "tau_ocr": primary["tau"],
            "by_category": primary["by_category"],
            "overall_micro": primary["overall_micro"],
            "overall_weighted_recall": primary["overall_weighted_recall"],
            "baseline": m["baseline_text_only"],
            "latency_ms": m["latency_ms"],
        }
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("Malformed %s: %s", path, exc)
        return None
