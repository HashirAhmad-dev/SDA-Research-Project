"""
pii_common.py
=============
Shared contracts between `build_pii_testset.py` and `run_pipeline.py`.

Everything that must stay byte-compatible with the legacy artefact
(`pii_scan_results.csv`, consumed by `backend/data_loader.py`) is defined once,
here.

Reverse-engineered from the legacy CSV (all 18 rows agree):

    sensitivity_score = min(10*high + 5*medium + 1*low, 100)
    risk_category     = HIGH-RISK if S >= 70 else SENSITIVE if S >= 20 else SAFE
    action_taken      = BLOCK / ALERT / PERMIT for those three bands
    base_system_score = 0 for any image payload (text-only DLP is blind to it)
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
TESTSET_DIR = EVAL_DIR / "pii_testset"
GROUND_TRUTH = TESTSET_DIR / "ground_truth.json"
BUILD_CONFIG = TESTSET_DIR / "build_config.json"

PII_V2_CSV = PROJECT_ROOT / "pii_scan_results_v2_real.csv"
PII_METRICS_JSON = EVAL_DIR / "data" / "pii_metrics.json"
PII_PREDICTIONS = EVAL_DIR / "data" / "pii_predictions.json"

# ---------------------------------------------------------------------------
# Entity schema. Tiers/weights are the paper's weighted entity model, and match
# the tiers already used in hybridSaaS_events.json.
# ---------------------------------------------------------------------------
ENTITY_WEIGHTS = {"HIGH": 10, "MEDIUM": 5, "LOW": 1}

ENTITY_TIER: Dict[str, str] = {
    "CREDIT_CARD": "HIGH",
    "NID": "HIGH",          # Pakistani CNIC, #####-#######-#
    "PASSWORD": "HIGH",
    "EMAIL": "MEDIUM",
    "PHONE": "MEDIUM",
    "ADDRESS": "MEDIUM",    # not in the legacy event log; added, tiered MEDIUM
    "PERSON": "LOW",
    "DOB": "LOW",
}
ENTITY_TYPES: Tuple[str, ...] = tuple(ENTITY_TIER)

CATEGORIES = ("text_extractable", "scanned", "handwritten")

# `file_type` values the legacy CSV uses, per category.
CATEGORY_FILE_TYPE = {
    "text_extractable": "text_extractable",
    "scanned": "scanned_pdf",
    "handwritten": "handwritten",
}

# `branch_used` labels the legacy CSV uses.
BRANCH1 = "Branch1_Presidio"
BRANCH2 = "Branch2_OCR+Presidio"
BRANCH3 = "Branch3_VLM+Presidio"

OCR_CONFIDENCE_THRESHOLD = 0.85   # tau_ocr, paper Section III.B

# Exact column names AND order of pii_scan_results.csv.
CSV_COLUMNS: Tuple[str, ...] = (
    "scan_id", "timestamp", "user_id", "user_name", "department", "platform",
    "file_name", "file_type", "branch_used", "ocr_confidence",
    "high_entities", "medium_entities", "low_entities", "sensitivity_score",
    "risk_category", "base_system_score", "base_system_label", "processing_ms",
    "action_taken", "slack_alert_sent", "jira_ticket_created",
)

DEPARTMENTS = ("Finance", "Legal", "HR", "Engineering", "Sales")
PLATFORMS = ("Google Drive", "Microsoft OneDrive")


# ---------------------------------------------------------------------------
# Scoring (must match backend/pii_pipeline.py)
# ---------------------------------------------------------------------------
def sensitivity_score(high: int, medium: int, low: int) -> int:
    return int(min(10 * high + 5 * medium + 1 * low, 100))


def risk_category(score: int) -> str:
    if score >= 70:
        return "HIGH-RISK"
    if score >= 20:
        return "SENSITIVE"
    return "SAFE"


def action_for(category: str) -> str:
    return {"HIGH-RISK": "BLOCK", "SENSITIVE": "ALERT"}.get(category, "PERMIT")


def tier_counts(entity_types: List[str]) -> Tuple[int, int, int]:
    """(high, medium, low) counts from a list of entity type names."""
    h = m = l = 0
    for t in entity_types:
        tier = ENTITY_TIER.get(t)
        if tier == "HIGH":
            h += 1
        elif tier == "MEDIUM":
            m += 1
        elif tier == "LOW":
            l += 1
    return h, m, l


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------
def normalise(value: str) -> str:
    """Lowercase, strip accents, collapse anything that is not alphanumeric.

    OCR routinely loses punctuation ('sara.malik@x.com' -> 'sara malik@x.com')
    and spacing. Normalising both sides before comparison keeps the metric about
    *entity detection* rather than about punctuation fidelity.
    """
    v = unicodedata.normalize("NFKD", str(value))
    v = "".join(c for c in v if not unicodedata.combining(c))
    v = v.lower()
    return re.sub(r"[^a-z0-9@]+", "", v)


def similarity(a: str, b: str) -> float:
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


FUZZY_THRESHOLD = 0.85


def match_entities(gold: List[dict], pred: List[dict],
                   threshold: float = FUZZY_THRESHOLD,
                   exact: bool = False) -> Tuple[int, int, int, List[dict]]:
    """Greedy one-to-one entity matching within a document.

    A predicted entity matches a gold entity iff the TYPE is identical and the
    normalised values are equal (`exact=True`) or similar enough
    (`exact=False`, SequenceMatcher ratio >= threshold).

    Returns (tp, fp, fn, matched_pairs). Each gold entity can be claimed once,
    so duplicate predictions of the same value count as false positives.
    """
    unmatched_gold = list(range(len(gold)))
    tp = 0
    pairs = []
    for p in pred:
        best_i, best_s = None, 0.0
        for i in unmatched_gold:
            g = gold[i]
            if g["type"] != p["type"]:
                continue
            s = 1.0 if normalise(g["value"]) == normalise(p["value"]) else (
                0.0 if exact else similarity(g["value"], p["value"]))
            if s >= threshold and s > best_s:
                best_i, best_s = i, s
        if best_i is not None:
            unmatched_gold.remove(best_i)
            tp += 1
            pairs.append({"gold": gold[best_i], "pred": p, "similarity": best_s})
    fp = len(pred) - tp
    fn = len(unmatched_gold)
    return tp, fp, fn, pairs


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f}
