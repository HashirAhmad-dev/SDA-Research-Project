# Evaluation

Every number on this page is **measured** by the harness in `evaluation/`, not asserted. Reproduce with:

```bash
python -m evaluation.generate_sessions      # synthesise the telemetry
python -m evaluation.train_and_evaluate     # train, tune on val, score test once
python -m evaluation.run_pipeline           # the three-branch PII cascade
python -m evaluation.calibrate_sbrs         # enforcement bands
```

Full write-ups: `evaluation/REAL_RESULTS.md`, `evaluation/REAL_RESULTS_PII.md`, `evaluation/SBRS_RECALIBRATION.md`.

## 0. Experimental setup

| | |
|---|---|
| Users | 50 (Faker-generated profiles: department, file types, home geo, collaborators, baseline activity) |
| Window | 3 weeks of hourly sessions, 25,200 grid rows |
| Active sessions | 16,297 (15,461 with enough history to score) |
| Injected threats | **641 = 3.93%** of active sessions, across four classes |
| Benign bursts | 171 labelled *negatives* — the deliberate false-positive trap |
| Split | **Chronological**: train = first 70% of hours, val = next 15%, test = last 15%. TRAIN is further cut into FIT (75%) / CALIB (25%) with a 24 h buffer so a window can never be scored against a memory bank containing itself. |
| Test split | 2,352 sessions, 96 true threats |

Nothing is tuned on test. `alpha`, the flag threshold, the EWMA threshold, `beta` and the enforcement bands are all chosen on validation; test is scored once.

## 1. Multimodal PII extraction

Corpus: **120 documents** (40 text-extractable / 40 scanned / 40 handwritten), **368 gold entities**, 19 documents deliberately PII-free. All PII is synthetic, generated with Faker — no real person's data. Matching is entity-level: type must match exactly, value normalised then compared with `SequenceMatcher >= 0.85` (fuzzy).

Recall against the stored ground truth, at the paper's OCR threshold `tau = 0.85`:

| Category | Baseline (Presidio only) | HybridSaaS-Sec |
|---|---|---|
| Text-extractable | 0.828 | 0.828 |
| Scanned PDF / image | **0.000** | **0.889** |
| Handwritten / low quality | **0.000** | **0.752** |
| **Overall weighted recall** | **0.276** | **0.823** |

The baseline scores 0.000 on both image categories because a text-only scanner cannot read an image at all — that gap is the entire argument for the cascade. Overall micro-F1 is 0.857 (precision 0.887, recall 0.829).

Branch 3 substitutions, and why (see `REAL_RESULTS_PII.md`):
- **PaddleOCR was not installable** (no `paddlepaddle` wheel for this Python/OS). Branch 2 uses **EasyOCR** with a char-weighted confidence.
- **Qwen2.5-VL-3B/7B/32B are hosted by no Inference Provider**, so the paper's ~3B on-box VLM is proxied by **`google/gemma-3-4b-it`** (the closest routable size), with **`Qwen2.5-VL-72B`** run as a same-family upper bound: scanned 0.943 / handwritten 0.816 F1, weighted recall **0.849**.

**`tau_ocr = 0.85` is not portable across OCR engines.** On EasyOCR even a *pristine, undegraded* render of our print documents scores 0.804 char-weighted confidence — already below the threshold — so at 0.85 the cut-point routes on engine calibration as much as on image quality. Re-tuned on a held-out 25% of images it lands near **0.98**, which raises overall weighted recall to **0.90** (gemma) / **0.94** (72B). We report `tau = 0.85` as primary for faithfulness.

## 2. Behavioral analytics (held-out test split)

Fusion weight `alpha = 0.25` and flag threshold `tau = 0.75`, both tuned on validation.

| Engine | FPR | Precision | Recall | F1 |
|---|---|---|---|---|
| EWMA (legacy, threshold tuned on val) | 5.81% | 0.097 | 0.146 | 0.116 |
| EWMA (at the legacy `tau = 2.0`) | 7.54% | 0.076 | 0.146 | 0.100 |
| **Hybrid (LSTM + Isolation Forest)** | **0.00%** | **1.000** | 0.458 | **0.629** |

**The benign-burst trap.** A team bulk-downloading templates before a deadline is a *benign* burst, and it is what rate-based detectors get wrong. Of the 27 benign bursts in the test split, EWMA flags **59.3%**; the hybrid engine flags **0%**. This is the concrete case from the Introduction, and it is why the structural (Isolation Forest) and personal-temporal (LSTM) views are paired.

**Per-class detection.** The engine is not uniformly good:

| Threat class | Window recall | Episode recall |
|---|---|---|
| Compromised account | 1.000 | 1.000 |
| Over-scoped third-party app | 1.000 | 1.000 |
| Negligent insider | 0.889 | 1.000 |
| **Malicious insider (slow drift)** | **0.020** | **0.167** |

Three of the four classes are near-perfect. The slow malicious insider is essentially invisible: mean `A_hybrid` 0.16, against 0.85–0.92 for the other three. This is a genuine architectural limit, not a tuning artefact — a supervised upper bound on the same features reaches AUC 0.922, while the unsupervised memory-bank cosine reaches only 0.74. Drift that stays inside a user's own baseline is, by construction, what a per-user baseline cannot see.

## 3. Enforcement accuracy & latency

The original bands (`beta = 0.5`, cut-points 0.20 / 0.60) were **miscalibrated by construction**: with `A_hybrid` in [0,1], `beta = 0.5` lets behavior move SBRS by at most +50%, so band membership was decided almost entirely by content sensitivity. 85.9% of *benign* sessions escalated to ALERT and 38.9% auto-BLOCKed regardless of behavior. `beta` and the cut-points were therefore re-derived from the joint (S, A) distribution on validation (`evaluation/calibrate_sbrs.py`).

| | old (beta=0.5, 0.20/0.60) | **new (beta=2.5, 1.22/1.84)** |
|---|---|---|
| Enforcement F1 (ALERT or BLOCK) | 0.087 | **0.475** |
| Benign sessions escalated | 85.9% | **4.0%** |
| Benign sessions auto-BLOCKed | 38.9% | **0.4%** |
| Benign bursts flagged | 37% | **0%** |

Enforcement recall by class: compromised account 100%, negligent insider 100%, over-scoped third-party 71%, malicious insider 33%. The 40% of threats that still PERMIT are overwhelmingly the malicious insiders of §2 — no band placement recovers a class the detector cannot separate.

**Latency**, timed with `perf_counter`, not estimated:

| Stage | Mean |
|---|---|
| Anomaly path (LSTM + memory bank + IF + fusion + SBRS), 500 calls | **16.9 ms** (p95 18.2, p99 18.6) |
| PII Branch 1 (Presidio, text) | **11.3 ms** |
| PII Branch 2 (EasyOCR + Presidio) | 2,450 ms |
| PII Branch 3 (VLM, hosted API) | 3,468 ms |

The three PII branches are alternatives, not a sum: a document takes exactly one. The two orders of magnitude between Branch 1 and the image branches is precisely why the cascade is cost-ordered. Branch 3 here is a network round-trip to a hosted model, so it is *not* comparable to an on-box quantized VLM and should not be read as an inference-cost figure.

## 4. Analyst usability

**Not evaluated in this build.** The dashboard exists (`frontend/app.py`), but no analyst study was run, so no time-to-root-cause or false-escalation figures are claimed.
