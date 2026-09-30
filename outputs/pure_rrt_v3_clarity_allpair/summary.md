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
| A | 0.1659 ± 0.0000 | 0.1605 ± 0.0000 | 0.1595 ± 0.0000 | 0.9137 ± 0.0000 | 0.9165 ± 0.0000 | 0.9173 ± 0.0000 |
| B | 0.5746 ± 0.0000 | 0.6086 ± 0.0000 | 0.6139 ± 0.0000 | 0.6536 ± 0.0000 | 0.6320 ± 0.0000 | 0.6286 ± 0.0000 |
| C | 0.1530 ± 0.0000 | 0.1502 ± 0.0000 | 0.1968 ± 0.0000 | 0.9203 ± 0.0000 | 0.9218 ± 0.0000 | 0.8972 ± 0.0000 |
| D | 0.5772 ± 0.0000 | 0.5967 ± 0.0000 | 0.6005 ± 0.0000 | 0.6479 ± 0.0000 | 0.6346 ± 0.0000 | 0.6319 ± 0.0000 |

## Table 2 — H3 End-to-End Prognosis (one window per patient)

| Group | H3 C-index ↑ | H3 Brier@365 ↓ |
|---|---:|---:|
| A | 0.4667 ± 0.0000 | 0.4291 ± 0.0000 |
| B | 0.4667 ± 0.0000 | 0.3820 ± 0.0000 |
| C | 0.4000 ± 0.0000 | 0.4849 ± 0.0000 |
| D | 0.4000 ± 0.0000 | 0.4133 ± 0.0000 |

## Table 3 — Ensemble Reliability (Secondary)

| Group | H3 latent disagreement | H3 disagreement-error Pearson | H3 survival-probability disagreement |
|---|---:|---:|---:|
| C | 0.0288 ± 0.0000 | 0.8245 ± 0.0000 | 0.0997 ± 0.0000 |
| D | 0.0144 ± 0.0000 | 0.3225 ± 0.0000 | 0.0019 ± 0.0000 |

## Representation Sanity

| Group | Observed latent variance | Mean adjacent-state L2 | Encoder trainable-parameter RMS Δ |
|---|---:|---:|---:|
| A | 0.0710 ± 0.0000 | 41.4536 ± 0.0000 | 0.0063 ± 0.0000 |
| B | 0.0602 ± 0.0000 | 36.6588 ± 0.0000 | 0.0032 ± 0.0000 |
| C | 0.0657 ± 0.0000 | 39.2522 ± 0.0000 | 0.0063 ± 0.0000 |
| D | 0.0592 ± 0.0000 | 36.2962 ± 0.0000 | 0.0031 ± 0.0000 |

## Appendix — H1/H2 Prognosis (one window per patient)

| Group | H1 C-index ↑ | H2 C-index ↑ | H1 Brier@365 ↓ | H2 Brier@365 ↓ |
|---|---:|---:|---:|---:|
| A | 0.3529 ± 0.0000 | 0.2667 ± 0.0000 | 0.7010 ± 0.0000 | 0.7707 ± 0.0000 |
| B | 0.5882 ± 0.0000 | 0.4667 ± 0.0000 | 0.5261 ± 0.0000 | 0.6916 ± 0.0000 |
| C | 0.5294 ± 0.0000 | 0.3333 ± 0.0000 | 0.3502 ± 0.0000 | 0.5249 ± 0.0000 |
| D | 0.5294 ± 0.0000 | 0.4667 ± 0.0000 | 0.5449 ± 0.0000 | 0.9002 ± 0.0000 |
