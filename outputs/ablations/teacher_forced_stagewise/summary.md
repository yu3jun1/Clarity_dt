# Teacher-forced Stage-wise Ablation

Upstream CLARITY commit: `dadb82241a24f5ec5e4e4dc994e3116fd4a9da04`.

Primary checkpoint: `best_val_loss.pt` 
(`validation_total_loss`).

## Training budget

| Group | Optimizer steps | Warmup steps | Samples seen | Effective dataset passes | Selected checkpoint step |
|---|---:|---:|---:|---:|---:|
| E | 2400.0000 ± 0.0000 | 240.0000 ± 0.0000 | 35520.0000 ± 0.0000 | 480.0000 ± 0.0000 | 408.0000 ± 0.0000 |

## Cohort and primary survival unit

Primary survival window rule: `earliest_eligible_four_stage_window_per_patient`.

| Split | Trajectories | Unique patients | Primary survival windows |
|---|---:|---:|---:|
| train | 74 | 38 | 38 |
| validation | 11 | 5 | 5 |
| test | 16 | 8 | 8 |

## Table 1 — Recursive Latent Dynamics

| Group | H1 MSE ↓ | H2 MSE ↓ | H3 MSE ↓ | H3/H1 MSE | H3-H1 MSE | H1 cosine ↑ | H2 cosine ↑ | H3 cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 0.1536 ± 0.0000 | 0.1521 ± 0.0000 | 0.1642 ± 0.0000 | 1.0687 ± 0.0000 | 0.0105 ± 0.0000 | 0.9202 ± 0.0000 | 0.9217 ± 0.0000 | 0.9162 ± 0.0000 |

## Table 2 — H3 End-to-End Prognosis (one window per patient)

| Group | H3 C-index ↑ | H3 Brier@365 ↓ |
|---|---:|---:|
| E | 0.5333 ± 0.0000 | 0.3828 ± 0.0000 |

## Representation Sanity

| Group | Observed latent variance | Mean adjacent-state L2 | Encoder trainable-parameter RMS Δ |
|---|---:|---:|---:|
| E | 0.0619 ± 0.0000 | 36.9635 ± 0.0000 | 0.0060 ± 0.0000 |

## Appendix — H1/H2 Prognosis (one window per patient)

| Group | H1 C-index ↑ | H2 C-index ↑ | H1 Brier@365 ↓ | H2 Brier@365 ↓ |
|---|---:|---:|---:|---:|
| E | 0.5294 ± 0.0000 | 0.7333 ± 0.0000 | 0.4932 ± 0.0000 | 0.5469 ± 0.0000 |
