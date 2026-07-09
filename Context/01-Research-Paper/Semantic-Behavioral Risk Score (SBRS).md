# Semantic-Behavioral Risk Score (SBRS)

Legacy DLP enforces on content alone or behavior alone. HybridSaaS-Sec computes a single score that correlates payload sensitivity ($S$) with the behavioral context ($A_{hybrid}$):

`SBRS = S * (1 + β * A_hybrid) / 100`

- $\beta$: a tunable enterprise risk multiplier.
- If $S \approx 0$ (public/benign data), even a large behavioral anomaly produces negligible SBRS (no false alerts for bulk benign ops).
- If $S$ is high, small structural deviations amplify the score sharply, triggering automated blocking.

## Default Enforcement Tiers
| SBRS Range | Action | Effect |
|---|---|---|
| < 0.5 | Allow + log | Transaction proceeds; event recorded for audit |
| 0.5 - 1.0 | Alert | Transaction proceeds; event queued for SOC review |
| >= 1.0 | Block | Proxy terminates request; session flagged for investigation |
