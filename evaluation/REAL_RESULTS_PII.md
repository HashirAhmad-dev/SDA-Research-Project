# Real Evaluation Results — Multimodal PII Pipeline (Branch 1/2/3)

Every number here was produced by running the pipeline for real:

```bash
python -m evaluation.build_pii_testset          # synthetic corpus + ground truth
python -m evaluation.run_pipeline --vlm-model "google/gemma-3-4b-it:deepinfra"
python -m evaluation.run_pipeline --vlm-model "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud" --write-csv
```

Branch 1 is Microsoft Presidio, Branch 2 is a real OCR engine, Branch 3 is a
hosted vision-language model. Nothing is replayed from a table. Raw metrics live
in `evaluation/data/pii_metrics_<model>.json`; per-document predictions in
`pii_predictions_<model>.json`.

**Status: COMPLETE.** Both VLMs scored all 80 image documents (40 scanned + 40
handwritten); Branch 1 covers all 40 text documents. 120/120 per model, no gaps.
Reaching this took eight HF-token renewals — the free tier grants only ~7–12
Inference-Provider calls before HTTP 402, and each renewal resumed from cache
(nothing re-billed). Total spend across the whole effort: **~$0.083**.

---

## 1. Corpus (`build_pii_testset.py`)

| Category | Docs | Rendering |
|---|---|---|
| text_extractable | 40 | `.txt` |
| scanned | 40 | print font → image → mild degradation (noise, ≤1.5° rotation, JPEG q40–55) |
| handwritten | 40 | handwriting font → image → harder degradation (blur, low contrast, ≤4° rotation, JPEG q22–37) |
| **Total** | **120** | 368 gold entities; 19 docs deliberately PII-free |

Entities by type: PERSON 101, EMAIL 63, PHONE 44, ADDRESS 44, CREDIT_CARD 34,
DOB 32, PASSWORD 28, NID 22.

**All PII is synthetic, from Faker.** Names, emails (`example.*` reserved
domains), phone numbers, CNIC-format national IDs (random digits), credit-card
numbers, addresses, DOBs, passwords. No real person's data is used. The
generated corpus is gitignored.

Fonts are the OpenType faces bundled with Windows: Arial/Calibri/Times/Georgia/
Tahoma for print; **Segoe Script, Segoe Print, Comic Sans** for handwriting.
Read from `C:/Windows/Fonts` at build time, never redistributed.

**One benchmark bug found and fixed.** The first corpus drew phone numbers from
`numerify('3##')`, producing *unassigned* PK mobile prefixes that libphonenumber
(inside Presidio) correctly rejects — Presidio was being scored as "missing"
numbers it was right to refuse. The generator now uses assigned PK prefixes, and
excludes Faker's 12-digit Maestro cards (no issuer uses them; outside Presidio's
13–16-digit regex). Valid 16-digit Mastercard numbers were **kept** even though
Presidio detects few of them — a real recogniser gap (§6), not an artefact. This
fix alone raised text-extractable F1 from 0.812 to 0.858.

---

## 2. Matching / scoring

- A predicted entity matches a gold entity iff the **type is identical** and the
  **values match after normalisation** (lowercase, strip accents, remove every
  non-alphanumeric character), using `SequenceMatcher` ratio **≥ 0.85** (fuzzy).
  Normalisation is essential because OCR routinely drops punctuation
  (`sara.malik@x.com` → `sara malik@x.com`). Exact-match F1 (normalised values
  identical) is reported alongside and runs ~0.02–0.05 lower.
- Greedy one-to-one within a document; a gold entity is claimed once, so
  duplicate predictions count as false positives.
- An **empty VLM response is scored as zero entities**, not a failure: the prompt
  tells the model to return `[]` when it sees no PII, and gemma-3-4b satisfies
  that by emitting nothing on the 19 PII-free press-release documents. (An
  earlier version mis-recorded these correct zero-PII predictions as
  `malformed_json`; fixed.)

---

## 3. Branch 1 — Presidio on extractable text (n=40)

Real `AnalyzerEngine` with `en_core_web_lg`, plus two custom recognizers (CNIC
regex; a mixed-case+digit password heuristic with context boosting).

| | P | R | F1 |
|---|---|---|---|
| text-extractable (micro over entities) | 0.891 | 0.828 | **0.858** |

Latency **mean 11–12 ms/doc, p95 ~16 ms** — comfortably under the paper's
"< 150 ms" claim for Branch 1. **This claim reproduces.**

---

## 4. Branch 2 — OCR + Presidio (n=80 images)

**PaddleOCR could not be used: `paddlepaddle` publishes no wheel for this
interpreter (Python 3.14 / Windows).** Substituted **EasyOCR**, a real OCR engine
with real per-box confidences (char-weighted mean → document confidence).

### The τ_ocr = 0.85 threshold is not portable across OCR engines

The paper routes to the VLM when OCR confidence < 0.85, a threshold implicitly
calibrated to PaddleOCR. On EasyOCR:

- A **pristine, undegraded** render of our print documents scores **0.804**
  char-weighted confidence — *already below 0.85*.
- Some *degraded* images score **higher** than their clean originals (0.908).
- Mean confidence: scanned **0.844**, handwritten **0.820** — nearly
  indistinguishable, so confidence alone cannot separate the two classes.

**A second, stronger finding now that Branch 3 is complete:** the F1-optimal OCR
threshold, calibrated on a held-out 25% of images, is **τ ≈ 0.98**, not 0.85 —
i.e. route *almost everything* to the VLM. Because the VLM is near-perfect
(§5) and OCR+Presidio is not, the cascade maximises accuracy by using the cheap
branch as little as possible. That **inverts the paper's cost rationale** (the
VLM is supposed to be the rare, expensive fallback). At τ=0.85 the cascade scores
overall F1 0.86–0.88; at the calibrated τ=0.98 it reaches **0.91–0.95** (below).
τ=0.85 is reported as primary for faithfulness to the paper.

Branch-2 latency: **mean ~2.5–2.8 s/doc** (EasyOCR on CPU). Not comparable to the
paper's on-box PaddleOCR figure.

---

## 5. Branch 3 — Vision-Language Model (COMPLETE, both models)

### Model availability (checked, not assumed)

The paper specifies **PaliGemma-3B INT8, on-box**. No CUDA is available here, so
`bitsandbytes` quantisation is impossible and a 3B model in bf16 (~7 GB) does not
fit in free RAM. Branch 3 calls a hosted model via Hugging Face Inference
Providers (OpenAI-compatible endpoint).

Queried the Hub API and probed the router directly:
**`Qwen/Qwen2.5-VL-3B-Instruct` is hosted by no Inference Provider** (empty
mapping; router HTTP 400 `model_not_supported`) — same for 7B and 32B. The only
Qwen2.5-VL served is the 72B. So two models were run to **bracket** the paper's
3B:

| Role | Model | Provider | Rationale |
|---|---|---|---|
| **Proxy for the paper's 3B** | `google/gemma-3-4b-it` | deepinfra | 4B — closest routable size to a 3B on-box model |
| **Upper bound** | `Qwen/Qwen2.5-VL-72B-Instruct` | ovhcloud | ~10× the params |

Each image is sent once as a base64 data URL with a JSON-only extraction prompt;
responses parsed (one retry on malformed JSON) and cached content- and
model-addressed. Provider and model pinned explicitly (`model:provider`).

### Head-to-head — VLM vs OCR+Presidio on all 80 images (same pixels)

| Model | scanned VLM F1 | handwritten VLM F1 | VLM overall F1 | OCR+Presidio F1 | text-only baseline R |
|---|---|---|---|---|---|
| **gemma-3-4b-it** | 0.948 | 0.920 | **0.936** | 0.699 | 0.000 |
| **Qwen2.5-VL-72B** | 1.000 | 1.000 | **1.000** | 0.699 | 0.000 |

The 72B was **perfect: 240/240 entities across all 80 images**, including every
address (which Presidio cannot detect at all). gemma-4B was strong but made real
mistakes (§6).

### Full cascade (Branch 1+2+3), entity-level micro over all 120 docs

At the paper's **τ = 0.85**:

| Category | gemma-3-4b-it F1 | Qwen2.5-VL-72B F1 |
|---|---|---|
| text_extractable | 0.858 | 0.858 |
| scanned | 0.923 | 0.943 |
| handwritten | 0.771 | 0.816 |
| **overall micro** | **0.857** | **0.877** |
| **overall weighted recall** | **0.823** | **0.849** |
| baseline (text-only) weighted recall | 0.276 | 0.276 |

At the **calibrated τ = 0.98** (route almost everything to the VLM):

| | gemma-3-4b-it | Qwen2.5-VL-72B |
|---|---|---|
| overall F1 | 0.908 | **0.949** |
| overall weighted recall | 0.900 | **0.940** |

### Cost (actual, from billed token usage, all sessions)

| Model | Rate ($/1M tok) | Unique calls | Tokens (in/out) | **Total cost** |
|---|---|---|---|---|
| gemma-3-4b-it | in 0.05 / out 0.10 (deepinfra) | 80 | 36,640 / 6,082 | **$0.0024** |
| Qwen2.5-VL-72B | in 1.01 / out 1.01 (ovhcloud) | 80 | ~68,000 / ~4,100 | **~$0.081** |

Grand total ≈ **$0.083**. The dollar cost was never the constraint — the
account's *included monthly credit allowance* was, capping each token at ~7–12
calls. Branch-3 latency: mean ~3.1 s/image, but this is a **network round-trip to
a hosted model**, not on-box compute, and is not comparable to the paper's figure.

---

## 6. Systematic failure modes (useful for the paper's discussion)

1. **Presidio cannot detect street addresses.** ADDRESS recall on clean text is
   **~0.02** over 44 gold addresses; it mislabels street names as PERSON (score
   0.85) or DATE_TIME, producing most of the PERSON false positives. **Both VLMs
   fix this** — the single clearest argument for the VLM branch. (In the full
   cascade ADDRESS still shows F1 0.54, because many address-bearing docs are
   *scanned* and route through OCR at τ=0.85, not the VLM; at τ=0.98 it rises.)

2. **Presidio misses most valid Mastercard numbers** (≈6/20 in isolation) while
   getting Visa/Amex/Discover 20/20 — a gap in its `CreditCardRecognizer`, not
   an OCR problem.

3. **The small VLM (gemma-4B) drops HIGH-tier numerics and hallucinates some
   entities.** Across all 80 images: NID recall **0.53** (8 missed, 4
   hallucinated), CREDIT_CARD 0.79, PASSWORD 0.88, plus 9 spurious EMAILs and 3
   spurious ADDRESSes. These are exactly the long digit-strings a 4B model
   transcribes unreliably — the failure mode you would predict, and a concrete
   reason the paper's 3B claim is optimistic for high-sensitivity fields.
   **The 72B made none of these errors (perfect recall and precision).**

4. **OCR confidence does not rank document difficulty** (§4): degraded images
   sometimes score higher than clean ones, so fixed-threshold routing is
   unreliable on EasyOCR.

---

## 7. Comparison to the paper (Table III / Evaluation.md)

The paper reports **accuracy**; we report **entity-level precision/recall/F1**,
which is stricter and not identical, so treat the comparison as directional.

| Category | Paper: baseline → hybrid | Measured baseline | gemma-3-4b (≈3B proxy) | Qwen2.5-VL-72B |
|---|---|---|---|---|
| text-extractable | 0.96 → 0.96 | — | F1 **0.858** | 0.858 |
| scanned | 0.00 → 0.89 | R **0.000** | VLM F1 **0.948** | **1.000** |
| handwritten | 0.00 → 0.73 | R **0.000** | VLM F1 **0.920** | **1.000** |
| overall weighted | 0.71 → 0.91 | wR **0.276** | wR **0.823** (0.90 @τ0.98) | wR **0.849** (0.94 @τ0.98) |

What holds and what doesn't:

- **The baseline being blind to images reproduces exactly** — text-only DLP has
  0.000 recall on every image, matching the paper's 0.00.
- **The hybrid gain reproduces and exceeds the paper.** Even the ~4B model beats
  the paper's claimed handwritten (0.92 vs 0.73) and scanned (0.95 vs 0.89)
  numbers; the 72B is perfect. The paper is, if anything, conservative for its
  stated model size — **but see the caveat below on synthetic handwriting.**
- **The overall-weighted target (0.91) is reached only at the calibrated τ**
  (0.94 for 72B, 0.90 for gemma). At the paper's τ=0.85 it is 0.82–0.85, because
  τ=0.85 still routes many image docs through the weaker OCR branch.
- **The Branch-1 latency claim (< 150 ms) reproduces** (11–12 ms measured).
- **A dependency the paper does not mention:** the cascade hinges on an OCR
  confidence threshold that is engine-specific and, with a near-perfect VLM,
  optimally set so high (0.98) that the "expensive fallback" becomes the primary
  path. Any reimplementation must recalibrate τ, and should question whether the
  cheap OCR branch earns its place at all.

### Caveats, stated plainly

- **Handwriting is simulated with fonts, not real cursive.** Segoe Script / Print
  and Comic Sans are cleaner and more regular than genuine handwriting or
  whiteboard photos. Real handwritten input would be materially harder, so the
  0.92–1.00 handwritten numbers are optimistic; do not read them as "VLMs solve
  handwriting." They show the *branch routing and extraction* work end-to-end on
  degraded images.
- The corpus is synthetic; degradation is a rough proxy for real scanners.
- The VLM is hosted, not the paper's on-box 3B INT8, and the 72B is deliberately
  an upper bound. The gemma-4B result — with its real NID/CREDIT_CARD failures —
  is the more honest proxy for the paper's claim.
- 240 gold entities across the 80 images; per-category numbers rest on 40 docs
  each — solid, but a synthetic single-corpus result.

---

## 8. Artefacts

| Path | Contents |
|---|---|
| `evaluation/pii_testset/{text,scanned,handwritten}/` | the 120 rendered documents (gitignored) |
| `evaluation/pii_testset/ground_truth.json` | every doc, its text, and its entity list |
| `evaluation/pii_testset/build_config.json` | seed, counts, fonts, degradation params |
| `evaluation/data/pii_metrics_<model>.json` | full metrics per VLM model |
| `evaluation/data/pii_predictions_<model>.json` | per-branch predictions per doc |
| `evaluation/data/vlm_cache/` | cached VLM responses (content+model addressed, incl. raw text) |
| `pii_scan_results_v2_real.csv` | **drop-in** (from the 72B run), legacy 21-col schema |
| `pii_scan_results_v2_<model>.csv` | per-model CSV variants |

`pii_scan_results_v2_real.csv` is a verified drop-in: identical column names and
order, parses through the unmodified `backend/data_loader.py::load_pii_scans()`,
and satisfies the legacy invariants (`S = min(10H+5M+1L,100)`, risk bands, action
mapping, `base_system_score = 0` on images). It reflects the complete 72B run at
τ=0.85.
