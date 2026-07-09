# Evaluation

Evaluated in a synthetic enterprise simulation (50 user accounts, Google Drive/OneDrive, 1200 sessions).

## 1. Multimodal PII Extraction
- Text extractable accuracy: 0.96 (Baseline) vs 0.96 (Hybrid)
- Scanned PDF/image: 0.00 (Baseline) vs 0.89 (Hybrid)
- Handwritten/low-quality: 0.00 (Baseline) vs 0.73 (Hybrid)
- Overall weighted accuracy: 0.71 -> 0.91

## 2. Behavioral Analytics
- **False Positive Rate:** EWMA baseline was 42.7% (reacted to every benign burst). Hybrid engine dropped FPR to 11.2%.
- EWMA misses slow, malicious insiders. Hybrid caught them via LSTM temporal drift.

## 3. Enforcement Accuracy & Latency
- **F1 Score:** 0.93.
- Detected 100% of unauthorized permission escalations and dangerous mass downloads.
- Missed some cross-domain cases mostly due to classification threshold tuning.
- **Latency:** End-to-end average 279 ms, well under the 300 ms budget.

## 4. Analyst Usability
- Using the dashboard, mean time to identify root cause dropped from 8.4 mins to 1.9 mins.
- Time to initiate remediation dropped from 12.7 mins to 2.3 mins.
- False escalation rate dropped from 23% to 9%.
