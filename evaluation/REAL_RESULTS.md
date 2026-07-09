# Real Evaluation Results — Hybrid Anomaly Engine

Every number in this document was produced by running:

```bash
python -m evaluation.generate_sessions      # synthetic telemetry + ground truth
python -m evaluation.train_and_evaluate     # trains, tunes on val, scores test once
```

A real `torch` LSTM (hidden dim 128) and a real `sklearn` IsolationForest are
trained and executed. Nothing is replayed from a table. Raw metrics:
`evaluation/data/metrics.json` (per seed: `metrics_seed{42,7,2024}.json`).

**Headline: the numbers in the paper do not reproduce.** Details in §7.

---

## 1. Sample sizes

| Quantity | Value |
|---|---|
| Users | 50 |
| Simulated span | 21 days, hourly windows |
| Total hourly grid rows | 25,200 (50 × 504) |
| **Active sessions** (≥1 API call) | **16,297** |
| Scoreable sessions (active **and** ≥24 h of history) | 15,461 |
| Dropped: <24 h of history (first day) | 836 |
| **True-threat sessions** | **641 (3.93 % of active sessions)** |
| Benign-burst sessions (labelled **negative**) | 185 |

Injected episodes (an episode spans several consecutive active windows):

| Class | Episodes | Windows |
|---|---|---|
| `malicious_insider` | 39 | 310 |
| `compromised_account` | 60 | 120 |
| `negligent_insider` | 72 | 110 |
| `overscoped_thirdparty` | 34 | 101 |
| `benign_burst` (negative) | 95 | 185 |

### Splits — **by time (chronological)**

Train = first 70 % of the hour grid, val = next 15 %, test = last 15 %. Chosen
over a by-user split because the paper's LSTM scores each window against *that
user's* historical memory bank — a by-user split leaves every test user with no
history at all. Attack episodes are injected wholly inside one split, never
straddling a boundary. Splits are assigned once, in the generator.

Train is further divided by time into **FIT** (first 75 %) and **CALIB** (last
25 %, separated by a 24 h buffer). The scaler, LSTM, IsolationForest and memory
banks are fitted on FIT; the two score calibrators on CALIB. Without the buffer
the memory bank would contain the very windows it is scoring, collapsing train
cosine distances to ~0 and destroying the calibration.

| Split | Sessions scored | Positives | Negatives | Benign bursts |
|---|---|---|---|---|
| train | 10,593 | 434 | 10,159 | 114 |
| val | 2,516 | 95 | 2,421 | 30 |
| **test** | **2,352** | **96** | **2,256** | **27** |

Test positives per class (windows / episodes): `malicious_insider` 51/6,
`compromised_account` 13/9, `negligent_insider` 18/11, `overscoped_thirdparty`
14/5. **These episode counts are small — per-class episode recall carries wide
uncertainty. Wilson 95 % intervals are given throughout; do not over-read them.**

Labels never touch training. The LSTM and forest are fitted on unlabelled FIT
data that still contains the ~3.9 % injected positives, as in deployment.

---

## 2. What was actually built

- **EWMA baseline** — univariate on `activity_rate_per_hr`,
  `EWMA_t = λ·x_t + (1−λ)·EWMA_{t−1}`, λ = 0.3, flagged when
  `(x_t − EWMA_{t−1}) / EWMA_{t−1} > τ_ewma`. τ_ewma **tuned on val** (landed at
  2.50; the legacy constant is 2.0) so the baseline gets the same tuning budget
  as the hybrid engine. Reported at both thresholds.
- **LSTM** — `nn.LSTM(6, 128)`, trained unsupervised by next-step prediction over
  24-hour windows, in each user's **personal z-frame**. Score is the paper's
  memory-bank formulation, `a_LSTM = 1 − mean(top-10 cosine)` against that user's
  FIT-period encodings.
- **Isolation Forest** — `sklearn`, 200 trees, fitted on the FIT split's
  **global** feature distribution.
- **Calibration** — each raw score is mapped to [0,1] by a robust quantile map
  (train p50 → 0, p99.5 → 1) fitted on CALIB. Cosine distance and negated
  isolation depth are not otherwise on a common scale, and `α·a_LSTM +
  (1−α)·a_IF` is meaningless if they are not. A rank/CDF transform was rejected:
  it makes a typical window score 0.5, under which the paper's fixed
  `hybrid_flag ≥ 0.5` would flag half of all traffic.
- **Fusion** — `A_hybrid = α·a_LSTM + (1−α)·a_IF`; α and τ_hybrid grid-searched
  on **val only**. Test scored exactly once.

### Three modelling bugs found by measurement

These are recorded because each changed the result materially.

1. **Both models originally shared one global feature frame.** The paper assigns
   them different reference distributions — LSTM vs *the user's* history, forest
   vs *the organisation*. Globally scaled, "Engineering user opens a `restricted`
   file" and "Legal user does so daily" are identical vectors, so personal drift
   is invisible. Giving the LSTM a per-user z-frame moved α off 0.00 (the val
   search had been discarding the LSTM entirely).
2. **Idle hours were encoded as raw `0.0` on all six dims**, then z-scored — so
   every idle hour became an extreme negative deviation on all axes, including
   collaboration. Since collapsing collaboration is exactly the malicious-insider
   signature, the memory bank filled with "very low collaboration" windows and
   real drift stopped looking unusual. Pinning the five *contextual* dims to the
   user's own mean (z = 0) when there is no transaction — they are undefined, not
   zero — lifted val malicious-insider AUC **0.62 → 0.74**. Single largest fix.
3. **A sequence-autoencoder objective is worse than next-step prediction here**
   (val malicious AUC 0.62 vs 0.70), and its reconstruction error is *worse than
   chance* on that class (**AUC 0.43**): drifting windows are smoother than
   normal ones — collapsed collaboration, steady reads — so an autoencoder
   reconstructs them *more* easily. Embedding centring (−0.05 AUC) and whitening
   (±0.01, hurt other classes) were also tried and rejected. All selection on val.

Scoring-rule check, on test: memory-bank cosine (chosen) vs the other candidate,
next-step prediction error — overall AUC 0.869 vs 0.832; malicious-insider 0.761
vs 0.687. Cosine wins on both, so the paper's formulation is the right call.

---

## 3. Tuned parameters (selected on validation)

| Parameter | Value (seed 42) | Across seeds 42/7/2024 |
|---|---|---|
| `α` (LSTM weight) | **0.25** | 0.35 ± 0.08 |
| `τ_hybrid` | **0.75** | 0.727 ± 0.026 |
| `τ_ewma` | **2.50** | 2.50 ± 0.00 |
| val F1 (hybrid) | 0.690 | 0.685 ± 0.004 |

**`τ_hybrid = 0.75`, not the 0.5 hardcoded in the old `anomaly_engine.py`.** The
backend now reads α and τ from the trained artefacts.

α settles low (0.25–0.35): the forest carries three of four attack classes at
AUC ≈ 1.00, so the F1 objective leans on it. This is itself the finding in §5.

---

## 4. Test-set results (held out, scored once)

Mean ± sd over 3 seeds (42, 7, 2024). Only the LSTM/forest depend on the seed;
EWMA is deterministic.

### Flag level — `A_hybrid ≥ τ` vs `is_true_threat`

| Detector | FPR | Recall | Precision | F1 |
|---|---|---|---|---|
| EWMA (tuned τ=2.5) | 0.0581 | 0.1458 | 0.0966 | 0.1162 |
| EWMA (legacy τ=2.0) | 0.0754 | 0.1458 | 0.0761 | 0.1000 |
| LSTM only | 0.0071 ± 0.0013 | 0.2882 ± 0.0402 | 0.6347 | 0.3948 ± 0.0393 |
| IsolationForest only | 0.0007 ± 0.0004 | 0.4618 ± 0.0049 | 0.9641 | **0.6244 ± 0.0075** |
| **HYBRID** | **0.0004 ± 0.0006** | 0.4549 ± 0.0049 | 0.9787 | **0.6209 ± 0.0056** |

**False positive rate, EWMA vs hybrid: 5.81 % → 0.04 %.** The hybrid engine is a
~145× reduction in false positives. That direction matches the paper's claim;
the magnitudes do not (§7).

Note: **the hybrid does not beat the Isolation Forest alone on F1** (0.621 vs
0.624, within one sd). It buys precision (0.979 vs 0.964) and a lower FPR, and
it is the only configuration with meaningful malicious-insider AUC. But the
fusion is not carrying the result — the forest is.

Threshold-free discrimination (AUC vs all test negatives):
LSTM **0.869 ± 0.016**, IF 0.798 ± 0.003, hybrid 0.841 ± 0.015.
*The fused score has lower AUC than the LSTM alone* — the F1-optimal α trades
ranking quality for precision at the operating point.

### Recall per attack class (hybrid, seed 42; 95 % Wilson CI)

| Class | Windows | Window recall | Episode recall | EWMA window / episode | AUC (LSTM / IF / hybrid) |
|---|---|---|---|---|---|
| `malicious_insider` | 51 (6 ep) | **0.020** [0.003, 0.103] | **0.167** | 0.039 / **0.333** | 0.761 / 0.620 / 0.702 |
| `compromised_account` | 13 (9 ep) | 0.974 ± 0.036 [0.772, 1.0] | 0.963 | 0.462 / 0.667 | 0.985 / 1.000 / 1.000 |
| `negligent_insider` | 18 (11 ep) | 0.889 [0.672, 0.969] | 1.000 | 0.167 / 0.182 | 0.990 / 1.000 / 1.000 |
| `overscoped_thirdparty` | 14 (5 ep) | 1.000 [0.785, 1.0] | 1.000 | 0.214 / 0.400 | 0.994 / 1.000 / 1.000 |

### Enforcement bands (SBRS → PERMIT / ALERT / BLOCK)

`SBRS = S·(1 + 0.5·A_hybrid)/100`, bands from `backend/risk_orchestrator.py`
(SAFE < 0.20 ≤ SENSITIVE < 0.60 ≤ HIGH-RISK).

| Rule | F1 | FPR | Recall | Confusion (seed 42) |
|---|---|---|---|---|
| hybrid, ALERT **or** BLOCK = positive | **0.0866** | 0.8583 | 0.9583 | tp 92, fp 1937, fn 4, tn 319 |
| hybrid, BLOCK only = positive | 0.1537 | 0.3887 | 0.8438 | tp 81, fp 877, fn 15, tn 1379 |
| EWMA `base_action` = BLOCK | 0.1162 | 0.0581 | 0.1458 | tp 14, fp 131, fn 82, tn 2125 |
| paper-doc bands (0.5 / 1.0), ALERT+BLOCK | 0.1327 | 0.4772 | 0.8681 | tp 84, fp 1077, fn 12, tn 1179 |

**The enforcement bands are miscalibrated, and this is a bug in the design, not
in the models.** `A_hybrid ∈ [0,1]` with `β = 0.5` can only move SBRS by ±50 %.
So band membership is decided almost entirely by content sensitivity `S`:

- 85.6 % of test sessions have `S ≥ 20` → at least SENSITIVE → **ALERT even at
  `A_hybrid = 0`**.
- 38.1 % have `S ≥ 60` → **BLOCK even at `A_hybrid = 0`**.

The behavioural engine is nearly decorative at the enforcement layer. Flag-level
F1 is 0.621; enforcement-band F1 is 0.087. The paper's own documented bands
(0.5 / 1.0) are also bad (F1 0.133, FPR 47.7 %). Any honest enforcement F1
requires re-deriving the bands — either raise β substantially or set the
thresholds from the joint (S, A) distribution rather than by hand.

Also: `Context/01-Research-Paper/SBRS.md` specifies bands **0.5 / 1.0** while
`backend/risk_orchestrator.py` implements **0.20 / 0.60**. These are different
systems. The code's bands are used above as primary because the CSV's
`sbrs_category` values depend on them.

### Latency (measured, not estimated)

500 timed scoring calls, `time.perf_counter()`, CPU, 4 torch threads, batch 1,
after warm-up.

| | ms |
|---|---|
| mean | **~17–18** (17.83 / 18.02 / 17.80 across seeds 42 / 7 / 2024) |
| p50 | 18.0–18.3 |
| p95 | 19.4–20.4 |
| p99 | 20.2–24.2 |
| max | 26.8–30.0 |

The ±0.10 across-seed spread of the mean understates the true uncertainty:
re-running the *identical* seed-42 configuration produced 16.95 ms, ~0.9 ms below
the original 17.83 ms. Wall-clock latency varies with machine load, so treat this
as **≈17–18 ms**, not a three-significant-figure quantity. The tail (p99, max) is
noisier still.

This covers the **anomaly path only**: personal z-scale → LSTM forward →
memory-bank cosine → IsolationForest `score_samples` → fusion → SBRS band. It
**excludes the PII/OCR pipeline** and so is **not comparable to the paper's
279 ms end-to-end figure**. For context, the recorded `processing_ms` in the
legacy `pii_scan_results.csv` averages 775 ms (median 178 ms, max 2379 ms) —
those are simulated values, but if they were real the end-to-end budget of
300 ms (design goal G5) would already be blown by the PII branch alone, with the
anomaly engine consuming ~6 % of it.

---

## 5. The benign-burst false positive (asked for explicitly)

The Introduction's motivating example: a team bulk-downloads templates before a
deadline; rate-based detectors cry wolf. Injected as a **labelled negative**:
activity spikes to **7.57× the user's baseline**, while all five other dimensions
stay textbook-normal (habitual geo, low-sensitivity files, read-only, no scope
change, normal collaboration). 27 such sessions in test (14 episodes).

| Detector | Flag rate on benign bursts |
|---|---|
| EWMA (tuned τ=2.5) | **59.26 %** |
| EWMA (legacy τ=2.0) | 66.67 % |
| **HYBRID** | **0.00 %** (all 3 seeds) |

**Yes — the hybrid engine suppresses this false positive, completely, at the
anomaly-flag level.** Mean scores on these sessions: `a_LSTM` 0.309, `a_IF`
0.325, `A_hybrid` 0.316 — elevated (bursts *are* unusual) but far below
τ = 0.75. The multivariate models see five of six dimensions unchanged; the
univariate EWMA sees only the 7.57× rate spike. For contrast, on *ordinary*
negatives (no burst) EWMA still fires at 5.16 % vs the hybrid's 0.04 %.

**But the enforcement layer partially undoes it.** Of those 27 benign bursts,
SBRS bands emit **17 PERMIT and 10 ALERT** — a 37.0 % alert rate — driven purely
by content sensitivity (mean `S` = 16.5, mean SBRS = 0.190, straddling the 0.20
SENSITIVE boundary), not by behaviour. So the claim "SBRS mathematically
suppresses the false-positive cascade" holds for the *behavioural* score and
fails at the *band* threshold. See §4.

---

## 6. If you actually care about malicious insiders

The F1-optimal operating point abandons them: slow drift is 53 % of test
positives but expensive to catch, so the val search picks a threshold that nails
the three loud classes at ~zero FPR. **A single fused threshold cannot serve both
fast attacks and slow drift.**

A second operating point, also selected on val (maximise malicious-insider recall
subject to val FPR ≤ 5 %), gives on test:

| | Value (3 seeds) |
|---|---|
| α | 0.93 ± 0.06 (LSTM-dominated) |
| τ | 0.44 ± 0.05 |
| Test FPR | 3.80 % ± 0.77 |
| Test F1 | 0.482 ± 0.024 |
| Overall recall | 0.597 ± 0.021 |
| **Malicious-insider recall** | **0.248 ± 0.033** (vs 0.020) |
| Benign-burst flag rate | 24.7 % ± 8.7 |

12× the insider recall, for ~95× the false-positive rate and a benign-burst trap
that starts firing again. That is the real trade, and the paper does not mention
it. A production system should run two thresholds (or a per-class head), not one
fused scalar.

Even so, malicious-insider recall tops out around 25 %. A supervised logistic
regression on window statistics, trained *with* labels, reaches **val AUC 0.922**
on this class, and the single feature `collaboration_density` (personal z, last
hour) reaches **0.893** — versus the unsupervised LSTM's 0.761. The signal is
plainly there; the memory-bank cosine leaves most of it on the table. That is a
limitation of the paper's architecture, not of this dataset.

---

## 7. How these compare to the numbers in the paper

Blunt version: **the paper's behavioural numbers do not reproduce, and the CSV
they were taken from does not reproduce them either.**

| Metric | Paper (`Evaluation.md`) | Measured (this harness) | Verdict |
|---|---|---|---|
| EWMA false positive rate | 42.7 % | **5.81 %** (legacy τ: 7.54 %) | Paper overstates ~7× |
| Hybrid false positive rate | 11.2 % | **0.04 %** | Paper overstates ~280× |
| F1 (enforcement) | 0.93 | **0.087** (ALERT+BLOCK) / 0.154 (BLOCK) | Not reproduced |
| F1 (anomaly flag level) | — | 0.621 ± 0.006 | Best honest F1 available |
| "100 % of permission escalations" | 100 % | 100 % (`overscoped_thirdparty`, 14 windows / 5 episodes) | **Holds** (small n) |
| "dangerous mass downloads" caught | 100 % | 97.4 % (`compromised_account`) | Approximately holds |
| "EWMA misses slow insiders; hybrid caught them via LSTM temporal drift" | claimed | **hybrid episode recall 0.167 vs EWMA 0.333** | **Contradicted** |
| Latency | 279 ms end-to-end (<300 ms) | ~17–18 ms anomaly-only | Not comparable; PII path unmeasured |
| Sessions evaluated | 1,200 | 16,297 (2,352 held-out test) | — |

Three things are worth stating plainly.

1. **The hybrid engine is *worse* than EWMA at catching malicious insiders in
   this experiment** (episode recall 0.167 vs 0.333, on 6 test episodes — both
   numbers are terrible and the difference is not significant). The paper's
   central qualitative claim for the LSTM is the one claim that fails. EWMA
   catches 2 of 6 drift episodes essentially by luck, when the drift's mild rate
   increase happens to cross a ratio threshold. The LSTM has real signal on this
   class (AUC 0.761, clearly above the forest's 0.620) but not enough to survive
   a threshold tuned for overall F1.

2. **The FPR improvement is real and larger than claimed, but the baseline is
   weaker than claimed.** EWMA's 42.7 % in the paper is not reachable here even
   at the legacy τ = 2.0 (7.54 %). Reporting a 42.7 % → 11.2 % improvement makes
   the hybrid look worse than it is *and* the baseline worse than it is.

3. **`anomaly_detection_comparison.csv`, the file this work replaces, is not
   consistent with the paper.** Scored directly, its 50 rows give hybrid
   **F1 = 1.000** (tp 6, fp 0, fn 0, tn 44 — perfect separation) and EWMA
   FPR = 79.6 %, F1 = 0.178. So the hardcoded table does not produce the paper's
   own F1 = 0.93 / FPR 42.7 % / 11.2 %. Its hybrid flags land on exactly the six
   true threats and nowhere else — a pattern no trained detector produces on
   50 samples.

### Caveats, stated up front

- **This is synthetic data.** The attack signatures are ones I injected; the
  detector's success on three of four classes partly reflects that they were
  drawn to be separable. The malicious-insider result is the informative one,
  because it was *deliberately* constrained to stay within plausible per-window
  bounds — and that is the class that fails.
- Test episode counts are small (6 / 9 / 11 / 5). Episode-level recalls have wide
  confidence intervals. Window-level counts (51 / 13 / 18 / 14) are better but
  still modest.
- Generalisation across seeds was checked (3 seeds) and is tight for every
  headline number; generalisation across *data* regenerations was not.
- `backend/schemas.py::PaperMetrics` retains the paper's claimed numbers but is
  now explicitly typed as a *claim*. `/metrics` returns
  `MetricsResponse{measured, paper_claimed, reproduced: false, note}`, where
  `measured` is read from `evaluation/data/metrics.json` (None, and labelled "not
  measured", if the harness has not been run). The dashboard shows the measured
  values and surfaces the discrepancy. PII coverage remains a paper claim — this
  repo executes no OCR or VLM and therefore cannot measure it.

---

## 8. Artefacts

| Path | Contents |
|---|---|
| `evaluation/data/sessions_raw.csv` | Every hourly window: features, labels, `split`, `episode_id`. The raw generated dataset, so generation is inspectable. |
| `evaluation/data/user_profiles.csv` | The 50 generated user profiles |
| `evaluation/data/generation_config.json` | Seed, parameters, realised counts |
| `evaluation/data/scored_all_splits.csv` | Every scored session + `split`, `threat_class`, `is_benign_burst`, `episode_id` |
| `evaluation/data/metrics.json` | Everything above, machine-readable |
| `evaluation/artifacts/lstm.pt`, `engine.pkl` | Trained model, forest, calibrators, memory banks — loaded by `backend/anomaly_engine.py` |
| `anomaly_detection_comparison_v2_real.csv` | **Held-out test predictions only** (2,352 rows), byte-compatible with the legacy 25-column schema |

`anomaly_detection_comparison_v2_real.csv` is a verified drop-in: identical
column names and order, parses through the unmodified
`backend/data_loader.py::load_anomaly_comparison()`, and satisfies every
invariant of the legacy file (`sbrs = S(1+0.5A)/100`, `ewma_new = 0.3x + 0.7·prev`,
`base_action = BLOCK ⟺ ewma_anomaly_flagged`, `*_correct = (flag == truth)`).
It contains **test-split rows only** — genuine out-of-sample predictions. It will
not join against `hybridSaaS_events.json`, which describes a different, smaller
event population.
