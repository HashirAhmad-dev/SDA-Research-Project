# Abstract

Enterprise data now moves through the APIs of SaaS platforms such as Google Workspace, Microsoft 365, and Salesforce. Most of this traffic is invisible to perimeter firewalls and to legacy Data Loss Prevention (DLP) tools that depend on static signatures and periodic metadata scans.

This paper presents **HybridSaaS-Sec**, a real-time interception architecture built around three components:
1. A **Long Short-Term Memory (LSTM)** network that models each user's temporal behavior.
2. An **Isolation Forest** that flags structural outliers without labeled data.
3. A **multimodal PII pipeline** that escalates from deterministic NLP to OCR and, when OCR confidence is low, to a Vision-Language Model (VLM).

Their outputs are fused into a single **Semantic-Behavioral Risk Score (SBRS)** that weighs content sensitivity against behavioral anomaly before any enforcement action is taken.

We evaluate on a synthetic 50-user enterprise deployment (16,297 active hourly sessions over three weeks, 3.93% injected threats) under a chronological train/validation/test split; every hyper-parameter is chosen on validation and the held-out test split is scored exactly once. On that split the hybrid engine drives the false-positive rate to **0.00%** against **5.81%** for a tuned EWMA baseline, at an anomaly-flag F1 of **0.629**. It suppresses the benign-burst false positive that defeats rate-based detectors: EWMA flags **59.3%** of benign bulk-download bursts, the hybrid engine **0%**. On a 120-document corpus (368 entities, all PII synthesised with Faker) the multimodal pipeline raises entity recall on scanned and handwritten files from **0.00** — a text-only scanner cannot read an image at all — to **0.889** and **0.752**, lifting overall weighted recall from **0.276** to **0.823**. Anomaly scoring costs **16.9 ms** per call.

We also report where the architecture fails. The engine detects compromised accounts, negligent oversharing, and over-scoped third-party applications at 100%, 100%, and 71% recall, but catches only **33%** of slow malicious insiders: drift gradual enough to stay inside a user's own baseline is close to invisible to a per-user temporal model. Enforcement bands re-derived from the validation distribution reach F1 **0.475**, a 5.5x improvement over hand-picked thresholds, and the residual error is dominated by that one unseparable class. A federated learning module lets multiple tenants train shared models without exchanging raw logs.
