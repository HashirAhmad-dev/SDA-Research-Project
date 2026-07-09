# Hybrid Anomaly Engine

At each time step $t$, a user's API transaction is encoded as a six-dimensional feature vector.

## Behavioral Feature Vector
| Feature | Encoding |
|---------|----------|
| Activity rate | API calls in current window, normalized against rolling mean |
| File type category | Ordinal class of touched object (public, internal, confidential, restricted) |
| Geographic index | Distance of request origin from habitual locations |
| Endpoint operation | Operation class: read, write, share, delete, permission change |
| Permission scope delta | Change in granted OAuth scope relative to baseline |
| Collaboration density | Degree of user's node in sharing graph over window |

## The Models
1. **LSTM:** Processes a 24-hour sequence and produces a behavioral encoding compared against the user's historical memory bank (cosine similarity). Yields `a_LSTM`. Catches deviations from the user's own routine.
2. **Isolation Forest:** Scores the vector against the global organizational distribution. Yields `a_IF`. Catches transactions strange for the organization as a whole (no user history needed).

## Fusion
The two scores are fused with a tunable weight $\alpha$:
`A_hybrid = α * a_LSTM + (1 - α) * a_IF`
