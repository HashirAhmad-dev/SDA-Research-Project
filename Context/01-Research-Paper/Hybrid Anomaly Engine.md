# Hybrid Anomaly Engine

At each time step $t$, a user's API transaction is encoded as a six-dimensional feature vector.

## Behavioral Feature Vector
| Feature | Encoding |
|---------|----------|
| Activity rate | API calls in current window, normalized against rolling mean |
| File type category | Ordinal class of touched object (public, internal, confidential, restricted) |
| Geographic index | Distance of request origin from habitual locations |
| Endpoint operation | Operation class: read, write, share, delete, permission change |
| Permission scope delta | Change in granted OAuth scope relative to baseline |
| Collaboration density | Degree of user's node in sharing graph over window |

## The Models

The two models deliberately operate in **different reference frames**. This is the load-bearing design decision: given a single shared frame, tuning collapses to `α = 0` and discards the LSTM entirely.

1. **LSTM (hidden dim 128), in each user's *personal* z-frame.** A 24-step window of the feature vector, z-scored against that user's own statistics, is encoded by a next-step predictor; the final hidden state is the behavioral encoding, scored as `1 - mean top-k cosine similarity` against that user's historical memory bank (k = 10). Yields `a_LSTM`. Catches deviations from the user's own routine.
2. **Isolation Forest (200 trees), in the *global* frame.** Scores the vector against the organization-wide distribution. Yields `a_IF`. Catches transactions strange for the organization as a whole, with no user history needed.

Both raw scores are pushed through calibrators fitted on a held-out slice of training data (CALIB, separated from FIT by a 24 h buffer so a window is never scored against a memory bank that contains it), so the fusion weight is meaningful.

**Idle hours must not be encoded as zeros.** Writing a raw `0.0` on all six dimensions for an inactive hour and then z-scoring makes every idle hour an extreme negative deviation on every axis — including collaboration density, which *is* the insider signature. The memory bank then fills with "low collaboration" windows and the drift becomes normal. Pinning the five contextual dimensions to the user's own mean (z = 0) when inactive lifted malicious-insider validation AUC from 0.62 to 0.74 — the single largest correction in the build.

**Objective chosen by measurement, not taste.** A sequence-autoencoder variant of the same encoder scored malicious-insider AUC 0.62 against 0.70 for next-step prediction, and its reconstruction error was *worse than chance* (AUC 0.43) on that class: drifting windows are smoother than normal ones, so an autoencoder reconstructs them more easily. Next-step prediction is retained.

## Fusion

The two scores are fused with a weight $\alpha$, tuned on validation:

`A_hybrid = α * a_LSTM + (1 - α) * a_IF`

Measured: `α = 0.25`, flag threshold `τ = 0.75`. On the held-out test split this gives FPR **0.00%**, precision 1.000, recall 0.458, F1 **0.629**, at **16.9 ms** per scoring call.

## What it catches, and what it does not

| Threat class | Episode recall |
|---|---|
| Compromised account | 1.000 |
| Over-scoped third-party app | 1.000 |
| Negligent insider | 1.000 |
| **Malicious insider (slow drift)** | **0.167** |

The slow malicious insider is the open problem. Its mean `A_hybrid` is 0.16 against 0.85–0.92 for the other three classes — behaviorally invisible. The signal *exists* in the features (a supervised upper bound reaches validation AUC 0.922; `collaboration_density` alone reaches 0.893), but the unsupervised memory-bank cosine only reaches 0.74. Drift that stays inside a user's own baseline is, by construction, what a per-user baseline cannot see. Closing this needs a better detector, not a different threshold.

## Serving

Training uses PyTorch. **Inference does not:** the encoder is a single-layer LSTM, so its weights are exported to `evaluation/artifacts/lstm_weights.npz` and the forward pass is unrolled in numpy (`backend/lstm_infer.py`), verified against torch to 2.4e-7 (float32 noise). The serving path therefore has no deep-learning framework dependency at all.
