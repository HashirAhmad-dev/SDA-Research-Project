# Introduction

Enterprise IT has largely moved onto cloud SaaS platforms like Google Workspace, Microsoft 365, Salesforce, and Dropbox. These hold intellectual property, financial records, and PII. They expose data through permission-driven APIs rather than a network perimeter.

Conventional network intrusion detection systems cannot secure this environment because they only observe encrypted HTTPS traffic. Conventional Cloud Access Security Brokers (CASBs) and Data Security Posture Management (DSPM) tools still rely on static rules, regular expressions, and periodic batch scans, making them slow and easy to evade.

The core difficulty is that malicious and legitimate SaaS activity overlap. Telling them apart requires understanding **what the content is** and **whether this behavior is normal for this account**.

## Limitations of Previous Work
- Existing frameworks (like EWMA) use univariate models, tracking a single dimension (e.g., activity rate). Real user behavior varies across file types, endpoints, locations, etc.
- Text-only PII extraction cannot inspect images at all (scanned documents, whiteboard photos, handwritten notes).

## Contributions
1. A hybrid anomaly detection engine that pairs an **LSTM** (user's temporal baseline) with an **Isolation Forest** (structural outliers against the whole organization).
2. A three-branch **multimodal PII pipeline** (Presidio, then high-confidence OCR, then a VLM fallback) for visual/unstructured content.
3. The **Semantic-Behavioral Risk Score (SBRS)**, a fusion metric that ties enforcement decisions to content sensitivity and behavioral history.
4. A **federated learning module** for multi-tenant model training without moving raw logs across boundaries.
5. A **fully executed evaluation** — trained models, timed calls, and a held-out test split scored once — including the negative results: the engine misses 67% of slow malicious insiders, and the OCR routing threshold does not transfer across OCR engines. See [[Evaluation]] and [[Implementation and Reproducibility]].

## The motivating false positive

A finance team bulk-downloads report templates the day before a quarterly close. Activity rate spikes 7x. A univariate rate model (EWMA) flags it: in our simulation it flags **59.3%** of such benign bursts. The content is low-sensitivity and the burst is structurally ordinary for the organization, so the hybrid engine flags **0%** of them. Every design choice below is in service of telling those two situations apart.
