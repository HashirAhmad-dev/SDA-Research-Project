# Multimodal PII Detection

Intercepted payloads pass through a three-branch pipeline ordered by cost:

1. **Deterministic NLP:** Native text and text-selectable PDFs go to **Microsoft Presidio**, running regex and NER pattern matching. Measured: **11.3 ms** per document, F1 **0.858**.
2. **High-confidence OCR:** Image payloads go to an OCR engine. If extraction confidence exceeds `tau = 0.85`, the transcript is handed to Presidio. Measured: **2,450 ms** per document.
3. **VLM Fallback:** When OCR confidence is low (cursive handwriting, dense layouts, heavy degradation), the raw image is routed to a Vision-Language Model that infers entities directly from pixels and returns a structured JSON entity list. Measured: **3,468 ms** per document as a hosted API call.

Each branch emits a set of detected entities, reduced to a scalar sensitivity score `S` (0-100) via `S = min(10*H + 5*M + 1*L, 100)` over high/medium/low sensitivity tiers.

## Why this structure?

The expensive VLM only sees payloads the cheaper branches cannot handle, keeping average latency low. Two orders of magnitude separate Branch 1 from the image branches, which is what makes cost-ordering worth the complexity. A text-only scanner scores **0.000** recall on scanned and handwritten documents — it cannot read an image at all — so the cascade is not an optimisation, it is the difference between seeing the data and not.

## Implementation notes (what actually runs)

The reference implementation deviates from the original design in two places, both forced:

- **OCR engine: EasyOCR, not PaddleOCR.** `paddlepaddle` has no distribution for the target Python/OS. Branch 2 uses EasyOCR with a char-weighted mean of per-box confidences.
- **VLM: `google/gemma-3-4b-it`, not PaliGemma-3B INT8 on-box.** Qwen2.5-VL-3B/7B/32B are served by *no* HuggingFace Inference Provider (verified against the Hub API and the router). gemma-3-4b-it is the closest routable size to a 3B on-box model. `Qwen2.5-VL-72B` is run as a same-family **upper bound** — it scores scanned/handwritten F1 0.943 / 0.816 against gemma's 0.923 / 0.771.

Because Branch 3 is a hosted API call, its latency is a network round-trip and must **not** be read as an on-box inference cost.

## The τ = 0.85 threshold is not portable across OCR engines

The routing threshold is implicitly calibrated to *a specific OCR engine's confidence scale*, and that does not transfer:

- A **pristine, undegraded** render of our print documents scores **0.804** char-weighted confidence on EasyOCR — already below 0.85.
- Mean confidence is 0.844 (scanned) vs 0.820 (handwritten) — nearly indistinguishable, so the threshold barely discriminates image quality at all. Some degraded images score *higher* than their clean originals.
- Re-tuned on a held-out 25% of images, the F1-optimal threshold is **τ ≈ 0.98**, not 0.85 — i.e. route almost everything to the VLM. That raises overall weighted recall from 0.823 to **0.90** (gemma) / **0.94** (72B), at the cost of the cascade's economics.

We report τ = 0.85 as primary for faithfulness to the design, but a deployment must calibrate τ against its own OCR engine rather than inherit a literal.

## Systematic failure modes

- **NID (national ID) is the weakest entity type for the smaller VLM** (F1 0.60 vs 1.00 for the 72B): digits are transcribed but grouped wrongly, or hallucinated outright.
- **Presidio's `CREDIT_CARD` recognizer misses valid Mastercards** in plain text (~6/20 detected), a genuine recognizer gap rather than a corpus artefact.
- **PERSON is over-detected in OCR transcripts**: OCR noise turns headings into name-like tokens, producing most of the false positives on image branches.

## Evaluation corpus

120 synthetic documents (40 text-extractable / 40 scanned / 40 handwritten), 368 gold entities, 19 deliberately PII-free. **All PII is generated with Faker — no real person's data is used anywhere.** Scanned documents are rendered then degraded (noise, ≤1.5° rotation, JPEG q40–55); handwritten use a handwriting font and harder degradation (blur, low contrast, ≤4° rotation, JPEG q22–37). Matching is entity-level: type exact, value normalised then compared with `SequenceMatcher >= 0.85`.
