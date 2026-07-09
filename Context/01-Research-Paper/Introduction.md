# Introduction

Enterprise IT has largely moved onto cloud SaaS platforms like Google Workspace, Microsoft 365, Salesforce, and Dropbox. These hold intellectual property, financial records, and PII. They expose data through permission-driven APIs rather than a network perimeter.

Conventional network intrusion detection systems cannot secure this environment because they only observe encrypted HTTPS traffic. Conventional Cloud Access Security Brokers (CASBs) and Data Security Posture Management (DSPM) tools still rely on static rules, regular expressions, and periodic batch scans, making them slow and easy to evade.

The core difficulty is that malicious and legitimate SaaS activity overlap. Telling them apart requires understanding **what the content is** and **whether this behavior is normal for this account**.

## Limitations of Previous Work
- Existing frameworks (like EWMA) use univariate models, tracking a single dimension (e.g., activity rate). Real user behavior varies across file types, endpoints, locations, etc.
- Text-only PII extraction cannot inspect images at all (scanned documents, whiteboard photos, handwritten notes).

## Contributions
1. A hybrid anomaly detection engine that pairs an **LSTM** (user's temporal baseline) with an **Isolation Forest** (structural outliers against the whole organization).
2. A three-branch **multimodal PII pipeline** (Presidio, PaddleOCR, and a quantized VLM fallback) for visual/unstructured content.
3. The **Semantic-Behavioral Risk Score (SBRS)**, a fusion metric that ties enforcement decisions to content sensitivity and behavioral history.
4. A **federated learning module** for multi-tenant model training without moving raw logs across boundaries.
