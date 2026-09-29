# v3 seed42 H3 patient-level survival diagnosis

Source results: `outputs/pure_rrt_v3/primary/{A,B,C,D}_seed42`

The source result directories were read only. This diagnosis is isolated under `outputs/diagnostics/pure_rrt_v3_seed42_survival`. `frozen_source_manifest.csv` records size, mtime, and SHA-256 for every source artifact, including checkpoints. The active experiment config writes to a different output root, so these endpoint-v3 results are not a target of subsequent training.

## H3 summary

| Variant | Checkpoint epoch | C-index | Concordant / comparable | IPCW Brier@365 | Primary-8 latent MSE |
|---|---:|---:|---:|---:|---:|
| A | 35 | 0.4667 | 7/15 | 0.4252 | 0.2367 |
| B | 36 | 0.2667 | 4/15 | 0.2575 | 0.1503 |
| C | 35 | 0.4000 | 6/15 | 0.4419 | 0.1723 |
| D | 33 | 0.3333 | 5/15 | 0.3338 | 0.1816 |

## Why B C-index is lower

Only 6 of the nominal 8 patients enter comparable pairs: the two censored patients (`PatientID_0021` at 126 days and `PatientID_0053` at 91 days) create no comparable pair under the repository C-index implementation. The score therefore has only 15 comparable event-event pairs.

A has 7/15 concordant pairs; B has 4/15. Five pair statuses change from A to B: four losses and one gain, for a net loss of three. The entire net difference is tied to `PatientID_0065`: A ranks this 128-day event above `PatientID_0095`, `PatientID_0074`, and `PatientID_0254`; B reverses all three. Removing `PatientID_0065` makes both A and B C-index exactly 0.3000. The remaining loss (`0095` vs `0254`) and gain (`0254` vs `0083`) cancel.

This is a small-sample ranking reversal, not a broad eight-patient degradation. `PatientID_0014` and `PatientID_0083` remain strongly discordant under both A and B.

## Why B Brier@365 is better

At 365 days, only 6 patients have known status. The same two early-censored patients contribute exactly zero to the IPCW score. Five known patients die by day 365; only `PatientID_0083` survives past day 365 (event at day 398).

B pushes all eight predicted survival probabilities into 0.0250-0.0800. That is poor discrimination but matches the dominant 5-to-1 one-year class balance. The A-to-B Brier change is -0.167611.

`PatientID_0014` explains essentially all improvement: its P(T>365) changes from 0.8032 to 0.0745 despite an observed event at day 295, reducing its Brier contribution by 0.171822. `PatientID_0083` worsens by 0.004818, because B predicts only 0.0250 survival probability for a patient still alive at day 365. All other contributions are tiny.

The opposite metric movement is structurally possible because `risk` and `P(T>365)` come from separate `risk_head` and `survival_head` branches. C-index evaluates continuous-time ordering of the risk head; Brier@365 evaluates a single horizon from the survival head. In this n=8 test set, neither result is stable enough to interpret without the patient table and sensitivity analysis.

## Exports

- `h3_patient_level_long.csv`: requested 32 variant-patient records.
- `h3_patient_level_wide.csv`: eight patients with A/B/C/D side by side.
- `h3_summary.csv`: reproduced aggregate metrics and pair counts.
- `cindex_comparable_pairs.csv`: all 15 comparable pairs and A/B/C/D concordance.
- `leave_one_patient_out_cindex.csv`: patient influence analysis.
- `brier_patient_contributions.csv`: exact IPCW contribution per patient.
- `frozen_source_manifest.csv`: integrity record for the untouched source results.
