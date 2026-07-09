"""
run_pipeline.py
===============
Executes the paper's three-branch multimodal PII cascade (Sec. III.B) for real
against the corpus built by `build_pii_testset.py`, and scores it entity-by-entity
against ground truth. Nothing is replayed; every number is measured.

    Branch 1  Deterministic NLP   Microsoft Presidio on extractable text.
    Branch 2  High-confidence OCR OCR the image; if confidence >= tau_ocr, hand
                                  the transcript to Presidio (same as Branch 1).
    Branch 3  VLM fallback        Below tau_ocr, send the raw image to a
                                  vision-language model and have it emit entities
                                  straight from pixels.

Two substitutions, both forced by the platform, both loud
-----------------------------------------------------------
* **PaddleOCR -> EasyOCR.** `paddlepaddle` publishes no distribution for this
  interpreter (Python 3.14 / Windows), so PaddleOCR cannot run at all here.
  EasyOCR is a real OCR engine with real per-box confidences. Its confidence
  scale is NOT PaddleOCR's: a *pristine, undegraded* render of our print
  documents averages 0.804 char-weighted confidence, already below the paper's
  tau_ocr = 0.85. See REAL_RESULTS_PII.md.
* **Local PaliGemma-3B INT8 -> hosted Qwen2.5-VL-72B-Instruct.** No CUDA is
  available, so `bitsandbytes` 4/8-bit quantisation is impossible, and a 3B
  model in bf16 (~7.1 GB) does not fit in free RAM. Branch 3 therefore calls a
  hosted model through Hugging Face Inference Providers, an OpenAI-compatible
  endpoint. Qwen2.5-VL-**7B** is not served by any provider on the router
  (HTTP 400 model_not_supported); the only Qwen2.5-VL offered is the **72B**.
  Consequently Branch 3 is an UPPER BOUND: a 3B INT8 model on commodity hardware
  would do worse. Its latency is a *network round-trip*, not on-box compute, and
  is not comparable to the paper's figure.

Design: OCR and the VLM are both run over **every** image, and their outputs are
cached. Routing at a given tau is then a post-hoc selection over cached results.
This costs one VLM call per image (once), makes reruns free, lets tau be swept
honestly, and lets us ask a question the cascade cannot: how would the VLM have
done on the documents OCR was confident about?

Outputs
-------
    pii_scan_results_v2_real.csv    (project root, drop-in schema)
    evaluation/data/pii_metrics.json
    evaluation/data/pii_predictions.json
    evaluation/data/vlm_cache/*.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .pii_common import (BRANCH1, BRANCH2, BRANCH3, CSV_COLUMNS,
                         ENTITY_TYPES, GROUND_TRUTH, OCR_CONFIDENCE_THRESHOLD,
                         PII_V2_CSV,
                         TESTSET_DIR, action_for, match_entities, prf,
                         risk_category, sensitivity_score, tier_counts)

DATA_DIR = Path(__file__).resolve().parent / "data"
VLM_CACHE = DATA_DIR / "vlm_cache"

HF_BASE_URL = "https://router.huggingface.co/v1"

# Qwen2.5-VL-7B-Instruct is not served by any provider on the router
# ("model_not_supported", HTTP 400). The only Qwen2.5-VL served is the 72B.
# Same family, same generation, ~10x the parameters -- so Branch 3 below is an
# UPPER BOUND on what the paper's 3B INT8 model would achieve on commodity
# hardware, not a proxy for it. Override with --vlm-model.
VLM_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

# Presidio's entity names -> the paper's schema.
PRESIDIO_MAP = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "DATE_TIME": "DOB",
    "LOCATION": "ADDRESS",
    "NID": "NID",
    "PASSWORD": "PASSWORD",
}
PRESIDIO_MIN_SCORE = 0.35

VLM_PROMPT = """You are a PII detection engine. Look at this document image and extract every piece of personally identifiable information you can read.

Return ONLY a JSON array, no prose, no markdown fences. Each element must be:
  {"type": "<TYPE>", "value": "<exact text as it appears>"}

Allowed TYPE values (use these exactly):
  PERSON        a person's full name
  NID           a national ID number (format #####-#######-#)
  CREDIT_CARD   a payment card number
  PHONE         a telephone number
  EMAIL         an email address
  DOB           a date of birth (YYYY-MM-DD)
  ADDRESS       a street address
  PASSWORD      a password or credential

Do not include field labels (e.g. "Name:") in the value. Do not invent entities.
If the image contains no PII, return [].
"""


# ---------------------------------------------------------------------------
# Branch 1: deterministic NLP (Presidio)
# ---------------------------------------------------------------------------
def build_analyzer():
    """Presidio, plus the two domain recognizers the defaults do not ship."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

    analyzer = AnalyzerEngine()

    # Pakistani CNIC. Tolerant of OCR losing the dashes.
    nid = PatternRecognizer(
        supported_entity="NID",
        patterns=[Pattern("cnic", r"\b\d{5}[-\s]?\d{7}[-\s]?\d\b", 0.9)],
        context=["cnic", "nid", "national", "identity"],
    )
    # A password has no intrinsic pattern, so require mixed case + a digit and
    # lean on Presidio's context boosting. Pure-digit strings (cards, CNICs)
    # cannot match, which keeps this from cannibalising the HIGH-tier numerics.
    pwd = PatternRecognizer(
        supported_entity="PASSWORD",
        patterns=[Pattern(
            "mixed_token",
            r"\b(?=[A-Za-z0-9]{8,20}\b)(?=[^\s]*[a-z])(?=[^\s]*[A-Z])(?=[^\s]*\d)[A-Za-z0-9]+\b",
            0.5)],
        context=["password", "passwd", "pwd", "credential", "temp"],
    )
    analyzer.registry.add_recognizer(nid)
    analyzer.registry.add_recognizer(pwd)
    return analyzer


def _dedupe_overlaps(results) -> list:
    """Presidio emits overlapping spans (PERSON over 'Card 4111...'). Keep the
    highest-scoring span and drop anything that overlaps it."""
    kept = []
    for r in sorted(results, key=lambda x: (-x.score, x.start)):
        if any(r.start < k.end and k.start < r.end for k in kept):
            continue
        kept.append(r)
    return sorted(kept, key=lambda x: x.start)


def presidio_entities(analyzer, text: str) -> List[Dict[str, str]]:
    res = analyzer.analyze(text=text, language="en",
                           entities=list(PRESIDIO_MAP.keys()),
                           score_threshold=PRESIDIO_MIN_SCORE)
    out = []
    for r in _dedupe_overlaps(res):
        t = PRESIDIO_MAP.get(r.entity_type)
        if t:
            out.append({"type": t, "value": text[r.start:r.end].strip()})
    return out


# ---------------------------------------------------------------------------
# Branch 2: OCR
# ---------------------------------------------------------------------------
def ocr_document(reader, path: Path) -> Tuple[str, float, int]:
    """Returns (transcript, char-weighted mean confidence, n_boxes).

    Confidence is averaged over detected text boxes, weighted by the number of
    characters in each box, so a long well-read line counts for more than a
    stray two-character box. This is a *choice*; EasyOCR exposes no document-level
    confidence. Documented in REAL_RESULTS_PII.md.
    """
    res = reader.readtext(np.array(Image.open(path).convert("L")))
    if not res:
        return "", 0.0, 0
    chars = sum(len(t) for _, t, _ in res)
    conf = sum(len(t) * c for _, t, c in res) / max(chars, 1)
    return "\n".join(t for _, t, _ in res), float(conf), len(res)


# ---------------------------------------------------------------------------
# Branch 3: VLM over Hugging Face Inference Providers
# ---------------------------------------------------------------------------
class VLMCapped(RuntimeError):
    """Raised when the provider refuses further calls (402 / persistent 429)."""


def _extract_json_array(raw: str) -> Optional[list]:
    """Pull a JSON array out of a model response that may be fenced or chatty.

    An empty / whitespace-only response is treated as the empty entity list, not
    a parse failure: the prompt instructs the model to return [] when it sees no
    PII, and some models (e.g. gemma-3-4b) satisfy that by emitting nothing at
    all. Scoring an empty response as "malformed" wrongly recorded correct
    zero-PII predictions on the PII-free documents as failures.
    """
    s = (raw or "").strip()
    if not s:
        return []
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else (v.get("entities") if isinstance(v, dict) else None)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, list) else None
        except json.JSONDecodeError:
            return None
    return None


def _clean_vlm_entities(items: list) -> List[Dict[str, str]]:
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type", "")).strip().upper()
        v = str(it.get("value", "")).strip()
        if t in ENTITY_TYPES and v:
            out.append({"type": t, "value": v})
    return out


def model_slug(model: str) -> str:
    """Filesystem-safe slug, provider suffix included (`model:provider`)."""
    return re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")


def cache_path(path: Path, doc_id: str, model: str) -> Path:
    import hashlib
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    return VLM_CACHE / f"{doc_id}-{model_slug(model)}-{digest}.json"


def has_ok_cache(path: Path, doc_id: str, model: str) -> bool:
    """True only if a cached *successful* response exists (failures are retried)."""
    c = cache_path(path, doc_id, model)
    if not c.exists():
        return False
    try:
        return json.loads(c.read_text(encoding="utf-8"))["status"].startswith("ok")
    except (json.JSONDecodeError, KeyError):
        return False


def vlm_document(client, path: Path, doc_id: str, *, model: str = VLM_MODEL,
                 retries: int = 1) -> Tuple[List[Dict[str, str]], float, str, dict]:
    """Call the hosted VLM once (plus one retry on malformed JSON).

    Returns (entities, elapsed_ms, status, usage) where status is one of
    'ok' | 'malformed_json' | 'error:<type>' and usage carries the token counts
    the provider billed. Cached on disk per (doc_id, model, image bytes).
    """
    import openai

    raw = path.read_bytes()
    # Content-addressed AND model-addressed: two different models must never
    # share a cached answer, and rebuilding the corpus must never reuse a
    # response generated for different pixels.
    cache = cache_path(path, doc_id, model)
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        # Only serve *successes* from cache. A cached malformed/error result is a
        # transient failure (or a stale classification from an older parser), so
        # it is re-attempted when credit is available rather than served forever.
        if c["status"].startswith("ok"):
            return c["entities"], c["elapsed_ms"], c["status"] + "+cached", c.get("usage", {})

    b64 = base64.b64encode(raw).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": VLM_PROMPT},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}]

    status, entities, usage, raw_text = "malformed_json", [], {}, ""
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=900,
            )
        except openai.RateLimitError as exc:
            if attempt < retries:
                time.sleep(5.0 * (attempt + 1))
                continue
            raise VLMCapped(f"rate limited: {exc}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code in (402, 403):     # out of credits / not allowed
                raise VLMCapped(f"HTTP {exc.status_code}: {exc}") from exc
            status = f"error:HTTP{exc.status_code}"
            break
        except openai.APIError as exc:
            status = f"error:{type(exc).__name__}"
            break

        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens,
                     "completion_tokens": resp.usage.completion_tokens,
                     "total_tokens": resp.usage.total_tokens}
        raw_text = resp.choices[0].message.content or ""
        parsed = _extract_json_array(raw_text)
        if parsed is not None:
            entities = _clean_vlm_entities(parsed)
            # Distinguish "model listed entities" from "model returned nothing"
            # so an empty response is auditable rather than silently a success.
            status = "ok" if raw_text.strip() else "ok_empty"
            break
        # Malformed -> one retry, nudging harder for bare JSON.
        messages = messages[:1] + [{"role": "user", "content": [
            {"type": "text", "text": VLM_PROMPT + "\nReturn ONLY the JSON array."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}]

    elapsed = (time.perf_counter() - t0) * 1000.0
    VLM_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"entities": entities, "elapsed_ms": elapsed, "status": status,
         "usage": usage, "model": model, "raw": raw_text[:2000]}, indent=1),
        encoding="utf-8")
    return entities, elapsed, status, usage


def make_client():
    """OpenAI-compatible client pointed at HF Inference Providers."""
    import openai

    token = os.environ.get("HF_TOKEN")
    if not token:
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                    token = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Put it in .env (see .env.example) or export it. "
            "Branch 3 calls Hugging Face Inference Providers and needs a token "
            "with 'Make calls to Inference Providers' permission."
        )
    return openai.OpenAI(base_url=HF_BASE_URL, api_key=token, timeout=180.0, max_retries=0)


def get_pricing(client, model: str) -> Optional[Dict[str, float]]:
    """USD-per-1M-token input/output rates for `model[:provider]`, from the router.

    Returns the pinned provider's pricing if `model` carries a `:provider`
    suffix, else the first priced provider. None if the router exposes no price
    (e.g. featherless-ai, which is subscription-billed).
    """
    import urllib.request

    base, _, pinned = model.partition(":")
    try:
        req = urllib.request.Request(
            f"{HF_BASE_URL}/models",
            headers={"Authorization": f"Bearer {client.api_key}"})
        data = json.load(urllib.request.urlopen(req, timeout=60))["data"]
    except Exception:
        return None
    entry = next((m for m in data if m["id"] == base), None)
    if not entry:
        return None
    provs = entry.get("providers", [])
    chosen = next((p for p in provs if p.get("provider") == pinned), None) if pinned else None
    chosen = chosen or next((p for p in provs if p.get("pricing")), None)
    if not chosen or not chosen.get("pricing"):
        return None
    pc = chosen["pricing"]
    return {"provider": chosen["provider"], "input_per_m": pc["input"],
            "output_per_m": pc["output"]}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_group(docs: List[dict], preds: Dict[str, List[dict]], exact: bool = False):
    tp = fp = fn = 0
    for d in docs:
        a, b, c, _ = match_entities(d["entities"], preds[d["doc_id"]], exact=exact)
        tp += a; fp += b; fn += c
    return prf(tp, fp, fn)


def score_by_type(docs: List[dict], preds: Dict[str, List[dict]]):
    per: Dict[str, Dict[str, int]] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in ENTITY_TYPES}
    for d in docs:
        for t in ENTITY_TYPES:
            g = [e for e in d["entities"] if e["type"] == t]
            p = [e for e in preds[d["doc_id"]] if e["type"] == t]
            a, b, c, _ = match_entities(g, p)
            per[t]["tp"] += a; per[t]["fp"] += b; per[t]["fn"] += c
    return {t: prf(**v) for t, v in per.items()}


def route(conf: Optional[float], category: str, tau: float) -> str:
    if category == "text_extractable":
        return BRANCH1
    return BRANCH2 if (conf is not None and conf >= tau) else BRANCH3


def assemble(docs, b1, b2_ocr, b3, tau) -> Dict[str, List[dict]]:
    """Cascade predictions at a given tau, from cached per-branch outputs."""
    out = {}
    for d in docs:
        did, cat = d["doc_id"], d["category"]
        if cat == "text_extractable":
            out[did] = b1[did]["entities"]
        else:
            conf = b2_ocr[did]["confidence"]
            out[did] = (b2_ocr[did]["entities"] if conf >= tau else b3[did]["entities"])
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=OCR_CONFIDENCE_THRESHOLD)
    ap.add_argument("--vlm-model", type=str, default=VLM_MODEL)
    ap.add_argument("--max-vlm-calls", type=int, default=200)
    ap.add_argument("--skip-vlm", action="store_true",
                    help="Run branches 1-2 only (no HF_TOKEN needed).")
    ap.add_argument("--calib-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--write-csv", action="store_true",
                    help="Write the canonical drop-in pii_scan_results_v2_real.csv "
                         "from this run (use for the model you consider primary).")
    args = ap.parse_args()

    slug = model_slug(args.vlm_model)
    metrics_path = DATA_DIR / f"pii_metrics_{slug}.json"
    predictions_path = DATA_DIR / f"pii_predictions_{slug}.json"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["documents"]
    text_docs = [d for d in docs if d["category"] == "text_extractable"]
    img_docs = [d for d in docs if d["category"] != "text_extractable"]
    print(f"[1/5] corpus: {len(docs)} docs "
          f"({len(text_docs)} text, {len(img_docs)} image), "
          f"{sum(len(d['entities']) for d in docs)} gold entities")

    # -- Branch 1 -----------------------------------------------------------
    print("[2/5] Branch 1: Presidio on extractable text ...")
    analyzer = build_analyzer()
    b1: Dict[str, dict] = {}
    for d in docs:
        # Every document also gets the text-only *baseline* treatment: the legacy
        # DLP sees the payload's extractable text, which for an image is nothing.
        t0 = time.perf_counter()
        ents = presidio_entities(analyzer, d["text"]) if d["category"] == "text_extractable" else []
        b1[d["doc_id"]] = {"entities": ents,
                           "elapsed_ms": (time.perf_counter() - t0) * 1000.0}

    # -- Branch 2 -----------------------------------------------------------
    print(f"[3/5] Branch 2: OCR over all {len(img_docs)} images ...")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    b2: Dict[str, dict] = {}
    for i, d in enumerate(img_docs, 1):
        p = TESTSET_DIR / d["path"]
        t0 = time.perf_counter()
        transcript, conf, nbox = ocr_document(reader, p)
        ents = presidio_entities(analyzer, transcript)
        el = (time.perf_counter() - t0) * 1000.0
        b2[d["doc_id"]] = {"entities": ents, "confidence": conf,
                           "transcript": transcript, "boxes": nbox, "elapsed_ms": el}
        if i % 10 == 0:
            print(f"    {i}/{len(img_docs)}  mean_conf so far "
                  f"{np.mean([v['confidence'] for v in b2.values()]):.3f}")

    # -- Branch 3 -----------------------------------------------------------
    # Interleave the two image categories. A provider cap (402) or --max-vlm-calls
    # then truncates *both* categories evenly instead of starving whichever one
    # happens to sort last. (The first run of this script burned all 8 available
    # calls on `scanned` and left `handwritten` completely unmeasured.)
    scanned = [d for d in img_docs if d["category"] == "scanned"]
    handw = [d for d in img_docs if d["category"] == "handwritten"]
    vlm_order = [d for pair in zip(scanned, handw) for d in pair]
    vlm_order += scanned[len(handw):] + handw[len(scanned):]

    b3: Dict[str, dict] = {}
    vlm_status: Dict[str, int] = {}
    capped_after: Optional[int] = None
    if args.skip_vlm:
        print("[4/5] Branch 3: SKIPPED (--skip-vlm)")
        for d in img_docs:
            b3[d["doc_id"]] = {"entities": [], "elapsed_ms": 0.0, "status": "skipped"}
    else:
        print(f"[4/5] Branch 3: {args.vlm_model} via HF Inference Providers "
              f"({len(vlm_order)} images, stratified, cached) ...")
        client = make_client()
        capped = False
        billable = 0
        for i, d in enumerate(vlm_order, 1):
            p = TESTSET_DIR / d["path"]
            # A cache file is only *usable* (free, final) if it holds a success.
            # A cached malformed/error result will be re-attempted, so it does not
            # count as cache here.
            usable_cache = has_ok_cache(p, d["doc_id"], args.vlm_model)
            # Once the provider has refused us, keep walking the corpus but serve
            # only from usable cache. Bailing out would discard paid results.
            if (capped or billable >= args.max_vlm_calls) and not usable_cache:
                b3[d["doc_id"]] = {"entities": [], "elapsed_ms": 0.0, "usage": {},
                                   "billed_now": False,
                                   "status": "not_attempted_capped" if capped
                                   else "not_attempted"}
                continue
            try:
                ents, el, status, usage = vlm_document(client, p, d["doc_id"],
                                                       model=args.vlm_model)
            except VLMCapped as exc:
                capped, capped_after = True, sum(
                    1 for v in b3.values() if v["status"].startswith("ok"))
                print(f"    !! provider refused further calls after "
                      f"{capped_after} completed: {exc}")
                print("    -> continuing from cache only")
                b3[d["doc_id"]] = {"entities": [], "elapsed_ms": 0.0, "usage": {},
                                   "billed_now": False, "status": "not_attempted_capped"}
                continue
            billed_now = not status.endswith("+cached")   # a real call was made
            if billed_now:
                billable += 1
            b3[d["doc_id"]] = {"entities": ents, "elapsed_ms": el, "status": status,
                               "usage": usage, "billed_now": billed_now}
            vlm_status[status] = vlm_status.get(status, 0) + 1
            if i % 10 == 0:
                print(f"    {i}/{len(vlm_order)}  statuses={vlm_status}  billed={billable}")

    # -- Routing + metrics --------------------------------------------------
    print("[5/5] scoring ...")
    confs = {d["doc_id"]: b2[d["doc_id"]]["confidence"] for d in img_docs}

    # tau calibration on a held-out subset, so the tuned tau is not read off the
    # same documents it is reported on.
    rng = np.random.default_rng(args.seed)
    calib_ids = set()
    for cat in ("scanned", "handwritten"):
        ids = [d["doc_id"] for d in docs if d["category"] == cat]
        k = int(round(len(ids) * args.calib_frac))
        calib_ids.update(rng.choice(ids, k, replace=False).tolist())
    calib = [d for d in docs if d["doc_id"] in calib_ids]
    evald = [d for d in docs if d["doc_id"] not in calib_ids]

    best_tau, best_f1 = args.tau, -1.0
    tau_sweep = []
    for t in np.round(np.arange(0.50, 1.001, 0.01), 2):
        m = score_group(calib, assemble(calib, b1, b2, b3, t))
        tau_sweep.append({"tau": float(t), "f1": m["f1"]})
        if m["f1"] > best_f1:
            best_tau, best_f1 = float(t), m["f1"]

    def executed(d: dict, tau: float) -> bool:
        """Did the branch this document routes to actually run?

        A document routed to Branch 3 whose VLM call never happened (provider cap,
        --max-vlm-calls, --skip-vlm) contributes zero predictions. Counting those
        as recall misses would report a budget limit as a model failure. Such
        documents are excluded from the `executed_only` metrics and counted
        separately.
        """
        b = route(confs.get(d["doc_id"]), d["category"], tau)
        return b != BRANCH3 or b3[d["doc_id"]]["status"].startswith("ok")

    results = {}
    for label, tau in (("paper_tau_0.85", args.tau), ("calibrated_tau", best_tau)):
        preds = assemble(docs, b1, b2, b3, tau)
        ran = [d for d in docs if executed(d, tau)]
        by_cat, by_cat_exec = {}, {}
        for cat in ("text_extractable", "scanned", "handwritten"):
            sub = [d for d in docs if d["category"] == cat]
            sub_exec = [d for d in ran if d["category"] == cat]
            by_cat[cat] = score_group(sub, preds) | {
                "n_docs": len(sub),
                "exact_match": score_group(sub, preds, exact=True),
                "routed": {b: sum(route(confs.get(d["doc_id"]), cat, tau) == b for d in sub)
                           for b in (BRANCH1, BRANCH2, BRANCH3)},
            }
            by_cat_exec[cat] = (score_group(sub_exec, preds) | {"n_docs": len(sub_exec)}
                                ) if sub_exec else None
        overall = score_group(docs, preds)
        weighted_recall = sum(by_cat[c]["recall"] * by_cat[c]["n_docs"] for c in by_cat) / len(docs)
        wr_exec = (sum(by_cat_exec[c]["recall"] * by_cat_exec[c]["n_docs"]
                       for c in by_cat_exec if by_cat_exec[c]) / len(ran)) if ran else None
        results[label] = {
            "tau": tau, "by_category": by_cat, "overall_micro": overall,
            "overall_weighted_recall": weighted_recall,
            "by_entity_type": score_by_type(docs, preds),
            "n_docs_branch3_not_executed": len(docs) - len(ran),
            "executed_only": {
                "n_docs": len(ran), "by_category": by_cat_exec,
                "overall_micro": score_group(ran, preds),
                "overall_weighted_recall": wr_exec,
                "by_entity_type": score_by_type(ran, preds),
            },
        }

    # Baseline: text-only DLP. Blind to every image.
    base_preds = {d["doc_id"]: b1[d["doc_id"]]["entities"] for d in docs}
    base_by_cat = {}
    for cat in ("text_extractable", "scanned", "handwritten"):
        sub = [d for d in docs if d["category"] == cat]
        base_by_cat[cat] = score_group(sub, base_preds) | {"n_docs": len(sub)}
    base_weighted = sum(base_by_cat[c]["recall"] * base_by_cat[c]["n_docs"]
                        for c in base_by_cat) / len(docs)

    # Counterfactual: what if the VLM saw every image, regardless of OCR conf?
    vlm_all = {d["doc_id"]: b3[d["doc_id"]]["entities"] for d in img_docs}
    ocr_all = {d["doc_id"]: b2[d["doc_id"]]["entities"] for d in img_docs}
    hi = [d for d in img_docs if confs[d["doc_id"]] >= args.tau
          and b3[d["doc_id"]]["status"].startswith("ok")]

    # -- Cost: actual token usage x the pinned provider's published rate -----
    pricing = get_pricing(client, args.vlm_model) if not args.skip_vlm else None
    billed = [v for v in b3.values() if v.get("billed_now") and v.get("usage")]
    reused = [v for v in b3.values() if not v.get("billed_now")
              and v["status"].startswith("ok")]
    in_tok = sum(v["usage"].get("prompt_tokens", 0) for v in billed)
    out_tok = sum(v["usage"].get("completion_tokens", 0) for v in billed)
    if pricing:
        cost = in_tok / 1e6 * pricing["input_per_m"] + out_tok / 1e6 * pricing["output_per_m"]
    else:
        cost = None
    cost_report = {
        "model": args.vlm_model,
        "pricing_per_1m_tokens": pricing,
        "calls_billed_this_run": len(billed),
        "calls_served_from_cache": len(reused),
        "billed_prompt_tokens": in_tok,
        "billed_completion_tokens": out_tok,
        "actual_cost_usd_this_run": cost,
        "note": "cost covers only the calls actually billed this run; "
                "cache hits (incl. any migrated from earlier sessions) are free. "
                "USD assumes router 'pricing' fields are $/1M tokens.",
    }

    # Head-to-head on exactly the images the VLM actually saw. Both engines get
    # the same pixels, so this is the cleanest OCR-vs-VLM comparison available,
    # and it survives a truncated VLM budget.
    seen = [d for d in img_docs if b3[d["doc_id"]]["status"].startswith("ok")]
    head_to_head = {
        "n_docs": len(seen),
        "by_category": {c: sum(d["category"] == c for d in seen)
                        for c in ("scanned", "handwritten")},
        "gold_entities": sum(len(d["entities"]) for d in seen),
        "ocr_presidio": score_group(seen, ocr_all) if seen else None,
        "vlm": score_group(seen, vlm_all) if seen else None,
        "by_entity_type_vlm": score_by_type(seen, vlm_all) if seen else None,
        "by_entity_type_ocr": score_by_type(seen, ocr_all) if seen else None,
    }

    metrics = {
        "corpus": {
            "documents": len(docs),
            "by_category": {c: sum(d["category"] == c for d in docs)
                            for c in ("text_extractable", "scanned", "handwritten")},
            "gold_entities": sum(len(d["entities"]) for d in docs),
            "pii_free_docs": sum(1 for d in docs if not d["entities"]),
        },
        "matching": {
            "primary": "type must match exactly; value normalised (lowercase, "
                       "accents stripped, non-alphanumeric removed) then compared "
                       "with SequenceMatcher ratio >= 0.85",
            "secondary": "exact_match = normalised values must be identical",
        },
        "ocr": {
            "engine": "EasyOCR (PaddleOCR unavailable: no paddlepaddle wheel)",
            "confidence": "char-weighted mean of per-box confidences",
            "mean_confidence_by_category": {
                cat: float(np.mean([confs[d["doc_id"]] for d in img_docs
                                    if d["category"] == cat]))
                for cat in ("scanned", "handwritten")},
            "pristine_render_confidence_note": 0.804,
        },
        "vlm": {
            "model": args.vlm_model, "endpoint": HF_BASE_URL,
            "substitution_note": "paper specifies PaliGemma-3B INT8 on-box. "
                                 "Qwen2.5-VL-3B/7B/32B are not served by any HF "
                                 "Inference Provider; gemma-3-4b-it is the closest "
                                 "routable size to the paper's 3B, and "
                                 "Qwen2.5-VL-72B is run as a same-family upper bound.",
            "cost": cost_report,
            "statuses": vlm_status,
            "provider_capped": bool(capped_after is not None),
            "cap_reason": "HTTP 402: monthly included credits depleted"
                          if capped_after is not None else None,
            "images_in_corpus": len(img_docs),
            "completed_including_cache": sum(
                1 for v in b3.values() if v["status"].startswith("ok")),
            "not_attempted_due_to_cap": sum(
                1 for v in b3.values() if v["status"] == "not_attempted_capped"),
            "coverage_by_category": {
                c: {"images": sum(1 for d in img_docs if d["category"] == c),
                    "vlm_scored": sum(1 for d in img_docs if d["category"] == c
                                      and b3[d["doc_id"]]["status"].startswith("ok"))}
                for c in ("scanned", "handwritten")},
        },
        "tau_calibration": {
            "calib_docs": len(calib), "eval_docs": len(evald),
            "best_tau_on_calib": best_tau, "calib_f1": best_f1, "sweep": tau_sweep,
        },
        "results": results,
        "baseline_text_only": {"by_category": base_by_cat,
                               "overall_weighted_recall": base_weighted,
                               "overall_micro": score_group(docs, base_preds)},
        "head_to_head_on_images_the_vlm_saw": head_to_head,
        "counterfactual_vlm_on_high_confidence_docs": {
            "n_docs": len(hi),
            "note": "docs OCR was confident about (conf >= tau) AND the VLM also "
                    "scored. Tests whether the cascade's cheap branch was the right "
                    "call on the documents it kept for itself.",
            "ocr": score_group(hi, ocr_all) if hi else None,
            "vlm": score_group(hi, vlm_all) if hi else None,
        },
        "latency_ms": {
            "branch1_mean": float(np.mean([b1[d["doc_id"]]["elapsed_ms"] for d in text_docs])),
            "branch1_p95": float(np.percentile([b1[d["doc_id"]]["elapsed_ms"] for d in text_docs], 95)),
            "branch2_mean": float(np.mean([v["elapsed_ms"] for v in b2.values()])),
            "branch2_p95": float(np.percentile([v["elapsed_ms"] for v in b2.values()], 95)),
            "branch3_mean": (float(np.mean([v["elapsed_ms"] for v in b3.values()
                                            if v["status"].startswith("ok")])) or None)
            if any(v["status"].startswith("ok") for v in b3.values()) else None,
            "branch3_note": "network round-trip to a hosted model, not on-box compute",
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=1, default=float), encoding="utf-8")
    predictions_path.write_text(json.dumps(
        {"branch1": b1, "branch2": b2, "branch3": b3}, indent=1, default=float),
        encoding="utf-8")

    # -- Drop-in CSV (at the paper's tau) -----------------------------------
    tau = args.tau
    rows = []
    for i, d in enumerate(docs, 1):
        did, cat = d["doc_id"], d["category"]
        conf = confs.get(did)
        branch = route(conf, cat, tau)
        if branch == BRANCH1:
            ents, ms = b1[did]["entities"], b1[did]["elapsed_ms"]
        elif branch == BRANCH2:
            ents, ms = b2[did]["entities"], b2[did]["elapsed_ms"]
        else:
            ents = b3[did]["entities"]
            ms = b2[did]["elapsed_ms"] + b3[did]["elapsed_ms"]   # OCR ran first
        h, m, l = tier_counts([e["type"] for e in ents])
        S = sensitivity_score(h, m, l)
        cat_label = risk_category(S)
        bh, bm, bl = tier_counts([e["type"] for e in b1[did]["entities"]])
        bS = sensitivity_score(bh, bm, bl)
        rows.append({
            "scan_id": f"PII-{i:04d}", "timestamp": d["timestamp"],
            "user_id": d["user_id"], "user_name": d["user_name"],
            "department": d["department"], "platform": d["platform"],
            "file_name": d["file_name"], "file_type": d["file_type"],
            "branch_used": branch,
            "ocr_confidence": "N/A" if conf is None else round(conf, 2),
            "high_entities": h, "medium_entities": m, "low_entities": l,
            "sensitivity_score": S, "risk_category": cat_label,
            "base_system_score": bS,
            "base_system_label": (risk_category(bS) if cat == "text_extractable"
                                  else "SAFE (blind to image)"),
            "processing_ms": round(ms, 1), "action_taken": action_for(cat_label),
            "slack_alert_sent": cat_label in ("HIGH-RISK", "SENSITIVE"),
            "jira_ticket_created": cat_label == "HIGH-RISK",
        })
    csv_out = pd.DataFrame(rows)[list(CSV_COLUMNS)]
    csv_out.to_csv(PII_V2_CSV.with_name(f"pii_scan_results_v2_{slug}.csv"), index=False)
    if args.write_csv:
        csv_out.to_csv(PII_V2_CSV, index=False)   # canonical drop-in

    # -- Console ------------------------------------------------------------
    r = results["paper_tau_0.85"]
    print("\n" + "=" * 78)
    print(f"CORPUS {len(docs)} docs | {metrics['corpus']['gold_entities']} gold entities "
          f"| {metrics['corpus']['pii_free_docs']} PII-free")
    print("Matching: fuzzy (normalised, ratio>=0.85). Exact shown in brackets.")
    print("-" * 78)
    print(f"{'category':<20}{'n':>4}{'P':>8}{'R':>8}{'F1':>8}   {'[exact F1]':>10}  routing")
    for cat, m in r["by_category"].items():
        rt = {k.split('_')[0]: v for k, v in m["routed"].items() if v}
        print(f"{cat:<20}{m['n_docs']:>4}{m['precision']:>8.3f}{m['recall']:>8.3f}"
              f"{m['f1']:>8.3f}   {m['exact_match']['f1']:>10.3f}  {rt}")
    n_missing = r["n_docs_branch3_not_executed"]
    if n_missing:
        print("-" * 78)
        print(f"!! {n_missing} docs route to Branch 3 but the VLM never ran "
              f"(provider cap / budget). Above they score 0 recall, which measures "
              f"the budget, not the model.")
        print("   Metrics over the {} docs whose branch DID execute:".format(
            r["executed_only"]["n_docs"]))
        for cat, m in r["executed_only"]["by_category"].items():
            if m:
                print(f"     {cat:<18}{m['n_docs']:>4}{m['precision']:>8.3f}"
                      f"{m['recall']:>8.3f}{m['f1']:>8.3f}")
        eo = r["executed_only"]["overall_micro"]
        print(f"     {'OVERALL micro':<18}{r['executed_only']['n_docs']:>4}"
              f"{eo['precision']:>8.3f}{eo['recall']:>8.3f}{eo['f1']:>8.3f}")
    print("-" * 78)
    print(f"overall micro  P={r['overall_micro']['precision']:.3f} "
          f"R={r['overall_micro']['recall']:.3f} F1={r['overall_micro']['f1']:.3f}")
    print(f"overall weighted recall (hybrid)   {r['overall_weighted_recall']:.3f}")
    print(f"overall weighted recall (baseline) {base_weighted:.3f}")
    print("-" * 78)
    print(f"OCR mean confidence: {metrics['ocr']['mean_confidence_by_category']}")
    print(f"tau calibrated on {len(calib)} held-out docs -> {best_tau:.2f} "
          f"(paper uses {args.tau})")
    rc = results["calibrated_tau"]
    print(f"  at calibrated tau: overall F1={rc['overall_micro']['f1']:.3f} "
          f"weighted recall={rc['overall_weighted_recall']:.3f}")
    print("-" * 78)
    print("per-entity-type F1 (paper tau):")
    for t, m in sorted(r["by_entity_type"].items(), key=lambda x: -x[1]["f1"]):
        gold = m["tp"] + m["fn"]
        print(f"  {t:<14} F1={m['f1']:.3f}  P={m['precision']:.3f} R={m['recall']:.3f}  "
              f"(gold {gold}, fp {m['fp']})")
    if head_to_head["n_docs"]:
        print("-" * 78)
        h = head_to_head
        print(f"HEAD-TO-HEAD on the {h['n_docs']} images the VLM saw "
              f"({h['by_category']}, {h['gold_entities']} gold entities):")
        for name, m in (("OCR+Presidio", h["ocr_presidio"]), ("VLM", h["vlm"])):
            print(f"  {name:<14} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f}  (tp{m['tp']} fp{m['fp']} fn{m['fn']})")
    print("-" * 78)
    lat = metrics["latency_ms"]
    print(f"latency: B1 mean {lat['branch1_mean']:.1f} ms | B2 mean {lat['branch2_mean']:.1f} ms"
          f" | B3 mean {lat['branch3_mean'] if lat['branch3_mean'] else 'n/a'}")
    v = metrics["vlm"]
    print(f"VLM: {v['completed_including_cache']}/{v['images_in_corpus']} images scored "
          f"{v['coverage_by_category']}")
    if v["provider_capped"]:
        print(f"     PROVIDER CAPPED -- {v['cap_reason']}; "
              f"{v['not_attempted_due_to_cap']} images never sent.")
    cr = cost_report
    price = (f"in ${cr['pricing_per_1m_tokens']['input_per_m']}/M "
             f"out ${cr['pricing_per_1m_tokens']['output_per_m']}/M "
             f"({cr['pricing_per_1m_tokens']['provider']})") if cr["pricing_per_1m_tokens"] else "no router price"
    print(f"COST [{args.vlm_model}]: billed {cr['calls_billed_this_run']} calls "
          f"({cr['billed_prompt_tokens']:,} in + {cr['billed_completion_tokens']:,} out tok), "
          f"{cr['calls_served_from_cache']} from cache")
    print(f"     {price} -> ACTUAL THIS RUN = "
          f"{('$%.4f' % cr['actual_cost_usd_this_run']) if cr['actual_cost_usd_this_run'] is not None else 'n/a'}")
    print("=" * 78)
    csv_note = f"\n  -> {PII_V2_CSV} (canonical)" if args.write_csv else ""
    print(f"  -> {metrics_path}\n  -> {predictions_path}"
          f"\n  -> {PII_V2_CSV.with_name(f'pii_scan_results_v2_{slug}.csv')}{csv_note}")


if __name__ == "__main__":
    main()
