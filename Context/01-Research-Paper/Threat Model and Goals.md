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
1. **G1: Application-layer visibility** into every sanctioned SaaS API transaction.
2. **G2: Low false positive rate** so analysts can manage the queue.
3. **G3: Multimodal Content coverage** including images, scans, and handwriting.
4. **G4: Privacy by construction**, meaning personal traffic is not intercepted and multi-tenant learning never moves raw logs.
5. **G5: Fast enforcement** to stop a transaction before it completes (end-to-end budget < 300 ms).
