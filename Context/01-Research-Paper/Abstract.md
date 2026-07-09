# Abstract

Enterprise data now moves through the APIs of SaaS platforms such as Google Workspace, Microsoft 365, and Salesforce. Most of this traffic is invisible to perimeter firewalls and to legacy Data Loss Prevention (DLP) tools that depend on static signatures and periodic metadata scans.

This paper presents **HybridSaaS-Sec**, a real-time interception architecture built around three components:
1. A **Long Short-Term Memory (LSTM)** network that models each user's temporal behavior.
2. An **Isolation Forest** that flags structural outliers without labeled data.
3. A **multimodal PII pipeline** that escalates from deterministic NLP to OCR and, when OCR confidence is low, to a quantized Vision-Language Model (VLM).

Their outputs are fused into a single **Semantic-Behavioral Risk Score (SBRS)** that weighs content sensitivity against behavioral anomaly before any enforcement action is taken.

In a simulated 50-user enterprise deployment, the hybrid engine reduced the false positive rate from 42.7% (EWMA baseline) to 11.2%, raised PII detection on scanned and handwritten files from 0.00 to 0.89 and 0.73 respectively, and reached an enforcement F1-score of 0.93 with end-to-end latency under 300 ms. In an analyst usability study, the accompanying dashboard cut mean time to root cause from 8.4 to 1.9 minutes. A federated learning module lets multiple tenants train shared models without exchanging raw logs.
