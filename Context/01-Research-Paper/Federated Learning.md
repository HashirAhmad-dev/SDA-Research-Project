# Federated Learning

Centralizing raw API logs from legally separate organizations violates data sovereignty requirements. HybridSaaS-Sec uses a federated learning module.

- Each tenant trains a local LSTM on its own API telemetry and computes a weight update.
- **Differential privacy noise** is added to clipped gradients before leaving the tenant.
- A central coordinator aggregates updates with Federated Averaging (FedAvg).
- The aggregated model is broadcast back to all participants.

## Benefits
- Collective immunization: an attack pattern in one tenant hardens models for all.
- Softens the cold-start problem for new tenants.
- Raw logs never cross tenant boundaries.
