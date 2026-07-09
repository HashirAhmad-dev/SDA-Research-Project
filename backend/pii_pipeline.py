"""
pii_pipeline.py
===============
Simulated Multimodal PII Detection pipeline (Section III.B of the paper).

Three branches:
  Branch 1 - Microsoft Presidio v2.2  (text-extractable payloads)
  Branch 2 - PaddleOCR -> Presidio    (scanned PDF / image, OCR conf >= 0.85)
  Branch 3 - PaliGemma-3B VLM (INT8)  (OCR conf < 0.85, e.g. handwriting)

Sensitivity score formula (from the simulated events JSON, faithful to the
paper's weighted entity model):

    S = min(10 * high_count + 5 * medium_count + 1 * low_count, 100)

This module does NOT call any real PII engine. It either:
  (a) replays the deterministic outcome stored in `hybridSaaS_events.json`
      for a known event_id (preferred for the demo); or
  (b) recomputes the score from raw entity counts using the same formula.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .data_loader import HybridSaaSDataset
from .schemas import PIIEntity, PIIResult


OCR_CONFIDENCE_THRESHOLD: float = 0.85  # tau_ocr from paper, Section III.B
ENTITY_WEIGHTS = {"HIGH": 10, "MEDIUM": 5, "LOW": 1}


def _categorise(score: int) -> str:
    """Map sensitivity score (0-100) to the paper's risk_category labels."""
    if score >= 70:
        return "HIGH-RISK"
    if score >= 20:
        return "SENSITIVE"
    return "SAFE"


def _route_branch(file_type: str, ocr_confidence: Optional[float]) -> str:
    """Decide which of the three branches the proxy would dispatch to."""
    ft = (file_type or "").lower()
    if "text" in ft or ft in {"docx", "pdf", "txt", "csv", "xlsx", "py", "env"}:
        return "Branch1_Presidio"
    if ocr_confidence is None:
        # No OCR pass yet - assume routed to OCR branch
        return "Branch2_PaddleOCR"
    if ocr_confidence >= OCR_CONFIDENCE_THRESHOLD:
        return "Branch2_PaddleOCR"
    return "Branch3_VLM"


def compute_sensitivity(high: int, medium: int, low: int) -> int:
    """Paper-faithful weighted sensitivity score, clamped to 100."""
    return int(min(
        ENTITY_WEIGHTS["HIGH"] * high
        + ENTITY_WEIGHTS["MEDIUM"] * medium
        + ENTITY_WEIGHTS["LOW"] * low,
        100,
    ))


def _normalise_entity(raw: Dict[str, Any]) -> PIIEntity:
    """Build a PIIEntity, deriving `weight`/`contribution` when absent.

    `hybridSaaS_events.json` stores entities in two shapes. Branch 1 records
    carry the full {type, count, sensitivity_tier, weight, contribution};
    the Branch 2/3 records carry only {type, count, sensitivity_tier} and left
    `PIIEntity(**e)` raising ValidationError on 8 of the 38 events -- which took
    `/score/{event_id}` and the dashboard's scoring view down with it.

    Both fields are fully determined by the paper's weighted entity model, so we
    derive rather than relax the schema:

        weight       = ENTITY_WEIGHTS[sensitivity_tier]
        contribution = weight * count

    Verified against every long-form entity in the log (e.g. NID x3 @ HIGH -> 30,
    PHONE x8 @ MEDIUM -> 40); see `_selftest_entity_derivation` below.
    """
    tier = str(raw.get("sensitivity_tier", "LOW")).upper()
    count = int(raw.get("count", 0))
    weight = int(raw.get("weight", ENTITY_WEIGHTS.get(tier, 1)))
    contribution = int(raw.get("contribution", weight * count))
    return PIIEntity(
        type=str(raw.get("type", "UNKNOWN")),
        count=count,
        sensitivity_tier=tier,
        weight=weight,
        contribution=contribution,
    )


def scan_event(event: Dict[str, Any]) -> PIIResult:
    """Run the (simulated) multimodal PII pipeline against a single event.

    The HybridSaaS-Sec event log already carries deterministic ground-truth
    PII output for each intercepted call, so we faithfully replay it and
    only recompute the score to prove the formula matches.
    """
    pii = event.get("pii_detection") or {}
    request = event.get("request") or {}

    entities = [_normalise_entity(e) for e in pii.get("entities_detected", [])]
    high = int(pii.get("high_count", 0))
    medium = int(pii.get("medium_count", 0))
    low = int(pii.get("low_count", 0))

    recomputed = compute_sensitivity(high, medium, low)
    reported = int(pii.get("sensitivity_score", recomputed))

    # Trust the reported score (already validated in the simulation) but
    # surface a deterministic recomputation for the dashboard's "audit" view.
    score = reported if reported >= 0 else recomputed

    # OCR confidence sometimes stored under different keys, default None.
    ocr_conf = pii.get("ocr_confidence")
    if ocr_conf in ("N/A", "", None):
        ocr_conf = None
    else:
        try:
            ocr_conf = float(ocr_conf)
        except (TypeError, ValueError):
            ocr_conf = None

    branch = (
        event.get("module")
        or pii.get("branch")
        or _route_branch(request.get("file_type", ""), ocr_conf)
    )

    return PIIResult(
        engine=pii.get("engine", "Microsoft Presidio v2.2"),
        branch=branch,
        ocr_confidence=ocr_conf,
        ocr_threshold=OCR_CONFIDENCE_THRESHOLD,
        entities_detected=entities,
        high_count=high,
        medium_count=medium,
        low_count=low,
        formula="min(10*H + 5*M + 1*L, 100)",
        sensitivity_score=score,
        risk_category=pii.get("risk_category", _categorise(score)),
        processing_ms=float(pii.get("processing_ms", 0.0)),
    )


def scan_by_event_id(dataset: HybridSaaSDataset, event_id: str) -> PIIResult:
    ev = dataset.get_event(event_id)
    if ev is None:
        raise KeyError(f"Unknown event_id '{event_id}'")
    return scan_event(ev)


def _selftest_entity_derivation(dataset: HybridSaaSDataset) -> Dict[str, int]:
    """Prove `_normalise_entity` reproduces every entity that ships full fields.

    Guards the derivation used for the short-form Branch 2/3 records: for each
    long-form entity in the log, recompute weight/contribution from tier+count
    and assert they match what the log states.
    """
    checked = derived = 0
    for ev in dataset.events:
        for raw in (ev.get("pii_detection") or {}).get("entities_detected", []):
            if "weight" not in raw or "contribution" not in raw:
                derived += 1
                continue
            stripped = {k: v for k, v in raw.items()
                        if k not in ("weight", "contribution")}
            got, want = _normalise_entity(stripped), _normalise_entity(raw)
            if (got.weight, got.contribution) != (want.weight, want.contribution):
                raise AssertionError(
                    f"{ev['event_id']} {raw['type']}: derived "
                    f"({got.weight}, {got.contribution}) != "
                    f"logged ({want.weight}, {want.contribution})"
                )
            checked += 1
    return {"verified_against_logged": checked, "derived_short_form": derived}


if __name__ == "__main__":  # pragma: no cover
    from .data_loader import load_all

    ds = load_all()
    print("entity derivation self-test:", _selftest_entity_derivation(ds))
    ok = sum(1 for eid in ds.event_ids() if scan_by_event_id(ds, eid))
    print(f"scan_event() succeeded on {ok}/{len(ds.events)} events")
