# Implementation and Reproducibility

This is the map from the paper to the code. Every headline number in [[Evaluation]] is produced by the scripts below; none is hardcoded.

## Layout

| Path | Role |
|---|---|
| `backend/anomaly_engine.py` | Loads the **trained** LSTM + Isolation Forest and scores events. Raises `EngineUnavailable` if the artefacts are absent — it will not emit a simulated score. |
| `backend/lstm_infer.py` | Torch-free forward pass for the trained LSTM encoder (numpy), so serving needs no deep-learning framework. |
| `backend/pii_pipeline.py` | Sensitivity scoring `S = min(10H + 5M + 1L, 100)` and risk banding. |
| `backend/risk_orchestrator.py` | SBRS fusion. **Authoritative source of `beta` and the enforcement bands.** |
| `backend/main.py` | FastAPI surface (`/score`, `/metrics`, `/comparison`, …). |
| `frontend/app.py` | Streamlit dashboard. |
| `evaluation/generate_sessions.py` | Synthesises the 50-user telemetry and injects the four threat classes + benign bursts. |
| `evaluation/train_and_evaluate.py` | Trains the LSTM + IF, tunes on validation, scores test **once**. |
| `evaluation/run_pipeline.py` | The three-branch PII cascade over the synthetic corpus. |
| `evaluation/build_pii_testset.py` | Builds the 120-document corpus. All PII from Faker. |
| `evaluation/calibrate_sbrs.py` | Re-derives `beta` and the enforcement bands on validation. |
| `evaluation/refresh_sbrs_columns.py` | Re-applies calibrated bands to the artefacts without retraining. |

## Reproducing

```bash
pip install -r requirements.txt -r requirements-eval.txt

python -m evaluation.generate_sessions       # -> evaluation/data/sessions_raw.csv
python -m evaluation.train_and_evaluate      # -> metrics.json, artifacts/, v2 CSV
python -m evaluation.calibrate_sbrs          # -> sbrs_calibration.json
python -m evaluation.build_pii_testset       # -> evaluation/pii_testset/
python -m evaluation.run_pipeline            # -> pii_metrics_*.json, v2 PII CSV
```

Running the app needs only `requirements.txt` — **torch is a training-only dependency.** The trained encoder is exported to `evaluation/artifacts/lstm_weights.npz` and served through a numpy forward pass verified against torch to 2.4e-7. The artefacts are committed, so a fresh checkout can score events without retraining.

## Methodology commitments

- **Chronological split** (train = first 70% of hours, val = next 15%, test = last 15%), not random, so no future information leaks backwards.
- **FIT / CALIB sub-split with a 24 h buffer** inside TRAIN, so a window can never be scored against a memory bank containing itself.
- **Everything is tuned on validation**: `alpha`, the flag threshold, the EWMA threshold, `beta`, the enforcement bands. **Test is scored exactly once.**
- **Labels never touch unsupervised training.** The ~3.9% injected positives remain in the training data as realistic contamination.
- **Latency is timed** (`perf_counter`), never estimated.
- **All PII is synthetic** (Faker). No real person's data is used anywhere in the corpus.

## Deviations from the design, and why

| Design | Built | Reason |
|---|---|---|
| PaddleOCR | **EasyOCR** | No `paddlepaddle` wheel for the target Python/OS. |
| PaliGemma-3B INT8, on-box | **`google/gemma-3-4b-it`** via HF Inference Providers | No CUDA available; and Qwen2.5-VL-3B/7B/32B are served by *no* provider. gemma-3-4b-it is the closest routable size. `Qwen2.5-VL-72B` is run as a same-family upper bound. |
| `beta = 0.5`, bands 0.20/0.60 | **`beta = 2.5`, bands 1.22/1.84** | The originals were miscalibrated by construction — see [[Semantic-Behavioral Risk Score (SBRS)]]. |
| Federated learning | **Not implemented** | Described in [[Federated Learning]] as design; no code, no numbers claimed. |
| Analyst usability study | **Not run** | No time-to-root-cause figures are claimed. |

## Known limitations of this build

1. **Slow malicious insiders are largely missed** (33% enforcement recall). The signal exists in the features but the unsupervised memory-bank cosine cannot reach it. This is the honest weak spot.
2. **`tau_ocr = 0.85` does not transfer across OCR engines** — a deployment must calibrate it against its own engine.
3. **The corpus and the sessions are synthetic.** Absolute numbers will move on real traffic; the relative comparisons (hybrid vs EWMA, cascade vs text-only) are the load-bearing part.
4. **Branch 3 latency is a network round-trip**, not on-box inference cost.
