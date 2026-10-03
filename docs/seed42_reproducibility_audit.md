# Same-GPU seed42 replicate audit

## Preservation and selection

- Recover the original step2400 A/B/C/D seed42 tracked artifacts from commit `6584f6d` as `historical_rep01`.
- Snapshot the current A/B/C/D seed42 results as `historical_rep02`, and current E seed42 as `historical_rep01`. Include available checkpoints, prediction CSVs and histories, with SHA256 manifests. Leave all original output directories unchanged.
- The old checkpoints, history CSV and patient predictions were overwritten during the earlier rerun and were not tracked by Git. Do not substitute new artifacts for them or reconstruct historical hardware metadata from the current machine.
- New deterministic runs use isolated `A_seed42_rep01`, `A_seed42_rep02`, etc. Keep replicate identifiers distinct from random seeds.
- Preselect `rep01` for the final seed42 comparison; `rep02` checks reproducibility. Never select whichever gives better test results or count two seed42 replicates as two seeds.

## Fixed execution protocol

Run **A42 rep01 → A42 rep02 → B42 rep01 → B42 rep02 → E42 rep01 → E42 rep02**, sequentially on one physical GPU pinned by UUID. Retain the original datasets, split, loss definitions, 2400 optimizer updates, warmup 240, validation interval 24, and `best_val_loss.pt` selected by minimum validation total loss strictly after warmup (earliest tie). Do not switch to the C-index checkpoint.

For new runs: strict `torch.use_deterministic_algorithms(True, warn_only=False)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed startup `PYTHONHASHSEED`, cuDNN deterministic on / benchmark off, TF32 off, math-only SDPA attention, independent seeded DataLoader generators and worker Python/NumPy seeds. Existing launchers retain their historical numerical protocol unless explicitly passed `--deterministic`.

Write `run_metadata.json` and `evaluation_metadata.json` with GPU model/UUID, driver inventory, CUDA/cuDNN/PyTorch/Python/dependency versions, effective numerical settings, configuration/source/data-manifest fingerprints, pretrained paths and initialization fingerprint. Record cumulative sample-order fingerprints at each validation and in each checkpoint. MRI manifest hashing does not hash every cached voxel file; keep cache contents immutable throughout the audit.

These controls follow [PyTorch 2.7 reproducibility guidance](https://docs.pytorch.org/docs/2.7/notes/randomness.html). They can slow execution; they are not a guarantee across library releases or GPU platforms. Unsupported nondeterministic operations fail visibly instead of silently weakening the protocol.

## Predeclared stability gate

Require identical source/configuration/data fingerprints, environment versions, GPU UUID, initial trainable parameters and complete batch order; strict deterministic mode must be enabled in both runs. Require the same primary-checkpoint optimizer step.

- H1/H2/H3 MSE and H3/H1: relative difference ≤1%, using `abs(a-b)/max(abs(a),abs(b),1e-12)`.
- H1/H2/H3 cosine similarity, C-index and Brier365: absolute difference ≤0.001.
- All prediction trajectory keys and endpoints must match; maximum relative difference of within-patient mean MSE ≤5%.

These are operational reproducibility tolerances, not significance thresholds. If a pair fails, report environment/init/order mismatches, first training/validation-history divergence and selected checkpoint steps, retain every run, and hold E43/44 pending diagnosis. Inspect CUDA kernels, hardware, worker/batch order and validation selection using those diagnostics; do not assert a root cause without evidence.

After stable A repeats, the original H3 MSE `0.626819` may be labeled a **candidate anomalous historical replicate** if its relative distance to the stable repeat exceeds 25%. That label never automatically excludes it. Deterministic settings change the execution protocol, and the old run lacks key artifacts; stable new runs alone do not justify performance-based removal.

## Conditional continuation and paired analysis

Only if all A/B/E pairs pass, run E43 and E44 on the same GPU, then compare E42/43/44 against B42/43/44. Use deterministic B42 rep01; B43/44 are the existing historical controls. Explicitly label that execution-protocol mismatch, even though the random seeds and test cohort match. Fully protocol-matched confirmation would require additional B43/44 reruns, which are not automatically added to this request.

Output H1/H2/H3 MSE, cosine, C-index, Brier365, H3/H1 and seed-paired E-minus-B differences. Average windows within each patient before computing patient-level latent MSE/cosine differences; use the fixed primary window for risk/survival probability differences. Average each patient's differences across seeds without treating windows or seeds as extra independent patients. Eight test patients and three seeds support descriptive exploration, not confirmatory claims.

## Running and monitoring

```bash
bash scripts/reproducibility/run_same_gpu.sh 0
```

Outputs: `outputs/reproducibility/seed42_same_gpu/`. Inspect `pipeline_status.json`, `audit_summary.md`, each replicate's `status.json` / `history.csv`, and `A/B/E_seed42_stability.json`. Conditional final outputs are under `comparison/`. Existing directories cannot be overwritten or silently resumed; a failed run stays preserved for diagnosis.
