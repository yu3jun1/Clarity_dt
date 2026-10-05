# Same-GPU reproducibility audit

## Current results and selection

A/B/E seed42 each have independent rep01/rep02 runs, with rep01 chosen before looking at test performance. Rep02 checks reproducibility, not an additional independent random seed. Existing replicate directories cannot be overwritten or silently resumed.

The completed audit retains gate files, resolved configs, run/evaluation metadata, metrics, patient predictions and the E42/43/44-versus-B42/43/44 comparison. Earlier experiment families, old-protocol E42, full historical archives, survival diagnostics and legacy configs/docs were retired only after all gates and final comparison existed. The pre-cleanup tag preserves Git history; cleanup_manifest.json records recoverable local backup paths and retained controls.

The user directed deletion, not archival, of non-resumable incomplete SHM-failure artifacts on 2026-10-04. Their checkpoints lacked optimizer/scheduler/RNG resume state; F42/F44 had not reached checkpoint creation. E42/E43/F42/F44 restart from scratch without changing the frozen training/configuration/DataLoader protocol. E42 rep01/rep02 remain sequential on original GPU0. The recovery governor permits at most three combined E/F tasks and requires >=80 GiB free SHM before launch on an idle assigned GPU. recovery_shm.json retains only the new scheduling state, not failed run contents. Complete numerical-instability results remain retained and gate-controlled.

## Frozen execution and gates

Run A42 twice, B42 twice, then E42 twice sequentially on one GPU pinned by UUID. Keep the original split, models, losses, 2400 updates, warmup240, batch16 and validation interval24. Primary checkpoint is minimum validation total loss strictly after warmup, earliest tie; never replace it with the diagnostic C-index checkpoint.

Strict deterministic algorithms fail visibly on unsupported operators. Use fixed startup PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG=:4096:8, cuDNN deterministic on / benchmark off, TF32 off, math-only SDPA and seeded DataLoader generators/workers. Record effective numerical settings, GPU model/UUID, driver/CUDA/cuDNN/PyTorch/Python/dependency versions, data/source fingerprints, initialization and cumulative batch-order fingerprints. Cache-manifest hashing does not hash every voxel file; keep cached data immutable.

Require equal source/config/data/version/GPU/init/order fingerprints and selected checkpoint step. H1/H2/H3 MSE and H3/H1 relative difference must be ≤1%; cosine/C-index/Brier absolute difference ≤0.001; maximum within-patient-mean MSE relative difference ≤5%. Relative difference is abs(a-b)/max(abs(a),abs(b),1e-12). These are operational reproducibility tolerances, not significance tests.

If any pair fails, retain current runs, report first numerical/order divergence and checkpoint differences, and quarantine speculative E43/44 results: do not produce a formal comparison or perform cleanup. Historical-archive deletion is user-directed housekeeping, not proof that performance-based statistical exclusion is justified.

## Paired comparison

The user approved a scheduling amendment on 2026-10-04: E43/44 start speculatively on same-model GPUs1/2 while E42 rep01/02 remain sequential on GPU0. The frozen training source/configuration and all deterministic settings are unchanged; per-run GPU UUIDs and the amendment are recorded in parallel_schedule.json. Only after all A/B/E gates pass may these results enter the formal E42/43/44-versus-B42 rep01 and retained B43/44 comparison. B43/44 use the historical numerical protocol and lack deterministic environment metadata: this is seed-matched, not fully execution-protocol-matched. Do not silently infer missing hardware versions or claim causal confirmation.

Report H1/H2/H3 MSE, cosine, C-index, Brier365 and H3/H1, with seed-paired E-minus-B differences. Average trajectories within each patient before patient-level MSE/cosine differences; use fixed primary windows for survival predictions. Average each patient's differences across seeds without treating windows/seeds as extra independent patients. Eight patients and three seeds remain exploratory.

## Artifacts and monitoring

Use outputs/reproducibility/seed42_same_gpu/ for current results, with comparison/ for paired reports. Keep B43/44 at their documented control paths while reports reference them. Git ignores outputs/**/*.log and weights; compact prediction and patient-difference CSVs are explicitly retained. Detailed local logs and history CSVs remain available for diagnosis.

The archive command only reads an existing archive index; it never reconstructs deleted experiment families from Git or copies legacy results into a new audit. Audit summaries work without historical/.
