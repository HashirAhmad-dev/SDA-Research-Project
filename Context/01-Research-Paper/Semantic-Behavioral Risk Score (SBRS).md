# Semantic-Behavioral Risk Score (SBRS)

Legacy DLP enforces on content alone or behavior alone. HybridSaaS-Sec computes a single score that correlates payload sensitivity ($S$) with the behavioral context ($A_{hybrid}$):

`SBRS = S * (1 + β * A_hybrid) / 100`

- $S \in [0, 100]$ from the PII pipeline; $A_{hybrid} \in [0, 1]$ from the anomaly engine.
- $\beta$: an enterprise risk multiplier — how far behavior is allowed to amplify content.
- If $S \approx 0$ (public/benign data), even a large behavioral anomaly produces negligible SBRS (no false alerts for bulk benign ops).
- If $S$ is high, structural deviations amplify the score sharply, triggering automated blocking.

## Calibration of β

β is **not** hand-picked. It is chosen on the validation split by the separation it produces between true threats and benign sessions, measured as ROC-AUC of SBRS against the ground-truth label:

| β | 0.0 | 0.5 | 1.0 | 2.0 | **2.5** | 3.0 | 5.0 | 12.0 |
|---|---|---|---|---|---|---|---|---|
| val AUC | 0.693 | 0.784 | 0.816 | 0.829 | **0.833** | 0.835 | 0.838 | 0.839 |

AUC climbs steeply, then plateaus at ~0.839 — capped because the slow malicious insiders cannot be separated at *any* β. The argmax (β≈12) is a flat, noisy peak that distorts the score scale for <0.7% more AUC, so we take the **knee: β = 2.5**, the smallest β reaching ≥99% of the maximum.

**Why β = 0.5 was wrong.** With $A_{hybrid} \in [0,1]$, β = 0.5 lets behavior move SBRS by at most +50%. Band membership was then decided almost entirely by $S$: 85.9% of *benign* sessions escalated to ALERT and 38.9% auto-BLOCKed regardless of how the user behaved. The behavioral engine was decorative at the enforcement layer.

## Default Enforcement Tiers

Cut-points are placed in the sparse upper tail of the **benign** validation SBRS distribution, i.e. as an explicit false-positive budget — the quantity that was broken:

| SBRS Range | Action | Effect | Benign traffic landing here (test) |
|---|---|---|---|
| < 1.22 | Allow + log | Transaction proceeds; event recorded for audit | 96.0% |
| 1.22 – 1.84 | Alert | Transaction proceeds; event queued for SOC review | 3.6% |
| >= 1.84 | Block | Proxy terminates request; session flagged for investigation | 0.4% |

- **1.22** = 95th percentile of benign validation SBRS. The SOC reviews only the top ~5% of normal traffic.
- **1.84** = 99.5th percentile. Auto-block fires only on the tail that normal traffic essentially never reaches. ALERT is a review queue and can tolerate false positives; BLOCK terminates a request and cannot, so it gets the tighter budget.

An F1-maximising ALERT threshold was tried and **rejected**: on this imbalanced, partly unseparable data it collapses to 1.75 and permits 66% of threats — useless as a review queue.

Result on the held-out test split: enforcement F1 **0.087 → 0.475**, benign escalation **85.9% → 4.0%**, benign auto-BLOCK **38.9% → 0.4%**. Derivation in `evaluation/SBRS_RECALIBRATION.md`; the constants live in `backend/risk_orchestrator.py` (`DEFAULT_BETA`, `SBRS_BANDS`) and `evaluation/common.py`, which must agree.
