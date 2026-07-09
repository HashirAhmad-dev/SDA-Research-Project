# SBRS Enforcement Recalibration

Reproduce:

```bash
python -m evaluation.calibrate_sbrs      # reuses evaluation/data/scored_all_splits.csv
```

The SBRS **formula is unchanged** (paper Sec. IV):

    SBRS = S * (1 + beta * A_hybrid) / 100      S in [0,100], A_hybrid in [0,1]

What changed is `beta` and the ALERT/BLOCK band cut-points. Both were re-derived
from real data instead of being hand-picked. No data was regenerated — this
reuses `scored_all_splits.csv` (S, A_hybrid, labels, splits) from the trained
LSTM + IsolationForest anomaly evaluation. **Everything is chosen on the
validation split; the test split is scored exactly once.**

---

## The problem

With `beta = 0.5`, the content term `S` swamps the behavioural term. Because
`A_hybrid in [0,1]` and `beta = 0.5`, behaviour can move SBRS by at most +50% —
so band membership is decided almost entirely by content sensitivity. On the
test split that meant **85.9% of sessions auto-ALERT and ~39% auto-BLOCK**
essentially regardless of behaviour. A restricted file opened perfectly normally
(A=0.05) scored SBRS 0.87 and **BLOCKED**; even a confidential file with no
anomaly BLOCKED. The behavioural engine was decorative at the enforcement layer.

Separately, the code (`0.20 / 0.60`) and `Context/.../SBRS.md` (`0.5 / 1.0`)
documented **different** band thresholds. Neither was grounded in data.

---

## Step 2 — joint (S, A) distribution on validation (2,516 sessions, 95 threats)

| group | n | S mean | A_hybrid mean |
|---|---|---|---|
| normal benign | 2,391 | 47.5 | 0.074 |
| benign burst (labelled negative) | 30 | 9.9 | 0.325 |
| **true threat** | 95 | 66.0 | 0.518 |

The threats are **bimodal in A**, which drives everything below:

| threat class | n | S mean | A_hybrid mean |
|---|---|---|---|
| compromised_account | 15 | 80.5 | **0.924** |
| negligent_insider | 20 | 73.0 | **0.848** |
| overscoped_thirdparty | 14 | 52.7 | **0.876** |
| malicious_insider | 46 | 62.2 | **0.132** |

Three classes are behaviourally loud (A≈0.85–0.92) and cleanly separable. The
malicious insiders are behaviourally **invisible** (A≈0.13, median 0.07) — their
(S, A) is indistinguishable from normal access to sensitive files, of which
there is a lot: **84.6% of normal sessions have S≥20, 39.3% have S≥60.** That
overlap is a hard ceiling no band placement can overcome; it is a detector
limitation (the LSTM misses slow drift), not a calibration bug.

---

## Step 3 — beta re-derived from separation

Objective: the beta that best separates true threats from benign on validation,
measured by ROC-AUC of SBRS vs `is_true_threat`.

| beta | 0.0 | 0.5 | 1.0 | 2.0 | **2.5** | 3.0 | 5.0 | 12.0 |
|---|---|---|---|---|---|---|---|---|
| val AUC | 0.693 | 0.784 | 0.816 | 0.829 | **0.833** | 0.835 | 0.838 | 0.839 |

AUC climbs steeply then **plateaus** — capped at ~0.839 because the malicious
insiders cannot be separated at any beta. The argmax (beta≈12) is a flat, noisy
peak that over-distorts the score scale for <0.7% more AUC. We take the **knee**:
the smallest beta reaching ≥99% of the maximum AUC, which is **beta = 2.5**
(AUC 0.833 = 99.2% of 0.839). beta=0.5 sits at AUC 0.784, well down the slope —
it was leaving most of the achievable separation on the table.

---

## Step 4 — band cut-points re-derived from the score distribution

With beta=2.5, the validation SBRS distributions are:

- benign: p50=0.54, p90=1.02, p99=1.67
- threat: p10=0.44, p50=1.64, p90=2.89

Cut-points are placed in the **sparse upper tail of the benign distribution**, as
an explicit false-positive budget — which is exactly the quantity that was broken:

- **t_alert = 95th percentile of benign val SBRS = 1.22** — the SOC reviews only
  the top ~5% of normal traffic.
- **t_block = 99.5th percentile of benign val SBRS = 1.84** — auto-block fires
  only on the ~0.5% tail that normal traffic essentially never reaches.

An F1-maximising t_alert was tried and **rejected**: on this imbalanced, partly
unseparable data it collapses to a high threshold (1.75) that permits 66% of
threats — useless as a review queue. The percentile budget catches far more
threats at a controlled, analyst-manageable false-positive rate. ALERT is a
review queue (tolerates false positives); BLOCK auto-terminates (must not), so a
tighter budget for BLOCK is appropriate.

---

## Step 5 — code/doc reconciliation

The recalibrated values are grounded in step 4's data, so **the code is now the
authoritative source**:

- `backend/risk_orchestrator.py` — updated: `DEFAULT_BETA = 2.5`,
  `SBRS_BANDS = [(1.84,'HIGH-RISK','BLOCK'), (1.22,'SENSITIVE','ALERT'), (0.00,'SAFE','PERMIT')]`.
- `backend/schemas.py`, `frontend/app.py` (beta slider max 2.0→5.0), `backend/main.py`
  (already `le=5.0`) — updated/verified so nothing breaks at beta=2.5.

`Context/01-Research-Paper/SBRS.md` still shows the old `0.5 / 1.0` tiers, and I
was asked not to modify anything under `Context/`. **To make the doc agree**,
replace its "Default Enforcement Tiers" table with:

| SBRS Range | Action | Effect |
|---|---|---|
| < 1.22 | Allow + log | Transaction proceeds; event recorded for audit |
| 1.22 – 1.84 | Alert | Transaction proceeds; event queued for SOC review |
| >= 1.84 | Block | Proxy terminates request; session flagged for investigation |

…and note beta's default is now **2.5** (data-calibrated), not 0.5.

---

## Step 6 & 7 — before/after on the held-out TEST split (2,352 sessions, 96 threats)

| Config | enforcement F1 | ALERT+BLOCK % | auto-BLOCK % | **benign** ALERT+BLOCK % | benign-burst flag % |
|---|---|---|---|---|---|
| **OLD** — beta=0.5, bands 0.20/0.60 | **0.087** | 86.3% | 40.7% | **85.9%** | 37% |
| DOC — beta=0.5, bands 0.50/1.00 | 0.134 | 49.4% | 2.6% | 47.7% | 0% |
| **NEW** — beta=2.5, bands 1.22/1.84 | **0.479** | 6.2% | 1.5% | **3.9%** | **0%** |

**The "content dominates regardless of behaviour" problem is fixed.** Benign
sessions escalating to ALERT/BLOCK fell from **85.9% to 3.9%**; auto-BLOCK on
benign from **38.9% to 0.4%**. Enforcement F1 rose **5.5×** (0.087 → 0.479).
Detection precision went from 0.045 to 0.397; auto-BLOCK precision from 0.085 to
0.743. Benign bursts (the Introduction's false-positive trap) stay at 0%.

### The honest cost: threat recall

Flagged recall on test is **0.60** (40% of threats now PERMIT), versus 0.96 under
the old config — but the old 0.96 came with 86% of *everything* flagged, i.e. it
"caught" threats only by alerting on nearly all traffic. Per-class test recall
under the new bands makes the trade legible:

| threat class | n | ALERT+BLOCK recall | BLOCK recall |
|---|---|---|---|
| compromised_account | 13 | **100%** | 85% |
| negligent_insider | 18 | **100%** | 56% |
| overscoped_thirdparty | 14 | **71%** | 29% |
| malicious_insider | 51 | **33%** | 2% |

The 40% that slip through are overwhelmingly **malicious insiders** — the class
the behavioural engine cannot see (§ step 2). No SBRS band can recover them
without re-flooding the queue on content sensitivity, because their (S, A) is
identical to normal sensitive-file access. Catching them requires a better
*detector* (the anomaly track's open problem), not a different enforcement band.
The three behaviourally-detectable classes are caught at 71–100%.

---

## Files

| Path | Change |
|---|---|
| `evaluation/calibrate_sbrs.py` | new — the calibration (val) + evaluation (test) |
| `evaluation/data/sbrs_calibration.json` | new — all numbers above, machine-readable |
| `backend/risk_orchestrator.py` | beta 0.5→2.5, bands 0.20/0.60 → 1.22/1.84 |
| `backend/schemas.py` | SBRSResult beta default 0.5→2.5 |
| `frontend/app.py` | beta slider max 2.0→5.0 |
| `Context/.../SBRS.md` | **not modified** (out of bounds); exact replacement text above |

Not touched: the trained models, the anomaly CSVs, or `evaluation/common.py`
(its SBRS constants generated the historical `scored_all_splits.csv` at beta=0.5;
the calibration reads that file's raw S and A columns, not its SBRS column, so it
is unaffected).
