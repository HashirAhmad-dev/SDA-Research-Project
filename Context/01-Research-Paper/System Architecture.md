# System Architecture

HybridSaaS-Sec sits at the enterprise boundary as a transparent intermediary. Interception, content scanning, and behavioral analytics are separated into independent modules so each can scale on its own.

## 1. Cryptographic Proxy Interception
A Man-in-the-Middle (MITM) proxy acts as a dynamic certificate authority. It negotiates ephemeral Elliptic-Curve Diffie-Hellman (ECDHE) keys on both legs of each connection.
- **Selective Routing:** Based on Server Name Indication (SNI), only sanctioned corporate SaaS domains are inspected. Personal traffic bypasses the proxy entirely.

## 2. Multimodal PII Detection Pipeline
See [[Multimodal PII Detection]]

## 3. Hybrid Anomaly Engine
See [[Hybrid Anomaly Engine]]

## 4. Semantic-Behavioral Risk Score (SBRS)
See [[Semantic-Behavioral Risk Score (SBRS)]]

## 5. How this maps to the code
See [[Implementation and Reproducibility]]
