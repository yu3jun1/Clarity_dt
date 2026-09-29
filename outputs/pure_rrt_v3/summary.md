# Pure Stage-wise RRT v3

Upstream CLARITY commit: `dadb82241a24f5ec5e4e4dc994e3116fd4a9da04`.

Primary checkpoint: `best_val_loss.pt`
(`validation_total_loss`).

## Cohort and primary survival unit

Primary survival window rule: `earliest_eligible_four_stage_window_per_patient`.

| Split | Trajectories | Unique patients | Primary survival windows |
|---|---:|---:|---:|
| train | 74 | 38 | 38 |
| validation | 11 | 5 | 5 |
| test | 16 | 8 | 8 |

## Table 1 — Recursive Latent Dynamics

| Group | H1 MSE ↓ | H2 MSE ↓ | H3 MSE ↓ | H1 cosine ↑ | H2 cosine ↑ | H3 cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.1562 ± 0.0000 | 0.1701 ± 0.0000 | 0.2091 ± 0.0000 | 0.9188 ± 0.0000 | 0.9120 ± 0.0000 | 0.8943 ± 0.0000 |
| B | 0.1846 ± 0.0000 | 0.1667 ± 0.0000 | 0.1553 ± 0.0000 | 0.9027 ± 0.0000 | 0.9124 ± 0.0000 | 0.9186 ± 0.0000 |
| C | 0.1461 ± 0.0000 | 0.1447 ± 0.0000 | 0.1556 ± 0.0000 | 0.9240 ± 0.0000 | 0.9247 ± 0.0000 | 0.9196 ± 0.0000 |
| D | 0.1986 ± 0.0000 | 0.1896 ± 0.0000 | 0.1841 ± 0.0000 | 0.8948 ± 0.0000 | 0.8996 ± 0.0000 | 0.9027 ± 0.0000 |

## Table 2 — H3 End-to-End Prognosis (one window per patient)

| Group | H3 C-index ↑ | H3 Brier@365 ↓ |
|---|---:|---:|
| A | 0.4667 ± 0.0000 | 0.4252 ± 0.0000 |
| B | 0.2667 ± 0.0000 | 0.2575 ± 0.0000 |
| C | 0.4000 ± 0.0000 | 0.4419 ± 0.0000 |
| D | 0.3333 ± 0.0000 | 0.3338 ± 0.0000 |

## Table 3 — Ensemble Reliability (Secondary)

| Group | H3 latent disagreement | H3 disagreement-error Pearson | H3 survival-probability disagreement |
|---|---:|---:|---:|
| C | 0.0165 ± 0.0000 | 0.9827 ± 0.0000 | 0.0445 ± 0.0000 |
| D | 0.0147 ± 0.0000 | 0.2335 ± 0.0000 | 0.0246 ± 0.0000 |

## Representation Sanity

| Group | Observed latent variance | Mean adjacent-state L2 | Encoder trainable-parameter RMS Δ |
|---|---:|---:|---:|
| A | 0.0664 ± 0.0000 | 39.2516 ± 0.0000 | 0.0055 ± 0.0000 |
| B | 0.0648 ± 0.0000 | 38.3329 ± 0.0000 | 0.0055 ± 0.0000 |
| C | 0.0601 ± 0.0000 | 36.8590 ± 0.0000 | 0.0056 ± 0.0000 |
| D | 0.0611 ± 0.0000 | 36.8976 ± 0.0000 | 0.0054 ± 0.0000 |

## Appendix — H1/H2 Prognosis (one window per patient)

| Group | H1 C-index ↑ | H2 C-index ↑ | H1 Brier@365 ↓ | H2 Brier@365 ↓ |
|---|---:|---:|---:|---:|
| A | 0.4706 ± 0.0000 | 0.4667 ± 0.0000 | 0.8675 ± 0.0000 | 1.0144 ± 0.0000 |
| B | 0.5882 ± 0.0000 | 0.6000 ± 0.0000 | 0.9063 ± 0.0000 | 0.9129 ± 0.0000 |
| C | 0.4706 ± 0.0000 | 0.4667 ± 0.0000 | 0.9172 ± 0.0000 | 1.0557 ± 0.0000 |
| D | 0.5294 ± 0.0000 | 0.4667 ± 0.0000 | 0.8576 ± 0.0000 | 1.0054 ± 0.0000 |
