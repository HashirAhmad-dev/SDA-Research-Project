# Threat Model and Design Goals

## Threat Model
We consider an enterprise whose users interact with sanctioned SaaS platforms from corporate-managed endpoints. We consider four adversary classes:
1. **Malicious insiders:** Valid credentials, exfiltrate data they are authorized to read, slowly and within normal patterns.
2. **Compromised accounts:** External attacker with stolen credentials/tokens acting in bursts from unfamiliar endpoints.
3. **Negligent insiders:** Over-share through misconfigured links or send sensitive material without malicious intent.
4. **Over-scoped third-party applications:** Granted OAuth permissions exceed what their function requires.
*(A fifth emerging actor is automated AI agents acting under a user's token).*

Out of scope: Out-of-band exfiltration (photographing screen, removable media, endpoint malware). Proxy host compromise is also out of scope.

## Design Goals

| | Goal | Status in this build |
|---|---|---|
| **G1** | Application-layer visibility into every sanctioned SaaS API transaction. | Architecture only; the proxy is not implemented. |
| **G2** | Low false positive rate so analysts can manage the queue. | **Met.** Anomaly-flag FPR 0.00% (EWMA 5.81%); 4.0% of benign sessions escalate under the calibrated bands, 0.4% auto-block. |
| **G3** | Multimodal content coverage including images, scans, and handwriting. | **Met.** Entity recall 0.000 → 0.889 (scanned) and 0.000 → 0.752 (handwritten). |
| **G4** | Privacy by construction: personal traffic is not intercepted; multi-tenant learning never moves raw logs. | Design only; federated learning is not implemented. |
| **G5** | Fast enforcement (end-to-end budget < 300 ms). | **Partially.** Anomaly scoring is 16.9 ms and the text PII branch 11.3 ms, comfortably inside budget. The image branches (OCR 2.4 s, VLM 3.5 s as a hosted call) are **not** — they would have to run asynchronously or on-box, as the design intends. |

## Detection coverage per adversary class (measured)

| Adversary | Episode recall |
|---|---|
| Compromised accounts | 1.000 |
| Over-scoped third-party applications | 1.000 |
| Negligent insiders | 1.000 |
| **Malicious insiders** | **0.167** |

The threat model's *first* class is the one the engine handles worst. An insider who exfiltrates "slowly and within normal patterns" is, definitionally, the adversary a per-user behavioral baseline is least able to see — the model learns the drift as it happens. This is stated as a limitation rather than smoothed over; see [[Hybrid Anomaly Engine]].
