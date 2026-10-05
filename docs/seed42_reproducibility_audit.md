# Historical same-GPU reproducibility audit

## Status: strict training entry retired

This document describes the former strict audit, not the current training protocol. The strict training pipeline and `scripts/reproducibility/run_same_gpu.sh` have been removed. Current A–F training uses one ordinary seeded execution path, following the step2400 behavioral baseline at `9871306`; there is no strict opt-in mode. See the repository README for new-run commands and use a new output root.

Historical configs, numerical settings, metadata, gate decisions and metrics describe their original runs and are not rewritten to claim ordinary execution. Loading an old config ignores its `deterministic` key in memory without writing back the saved file. Read-only gate/comparison tools remain available for existing records; these historical gates are not prerequisites for a new ordinary run or F campaign and cannot establish its reproducibility.

## Historical results and selection

The audit ran A/B/E seed42 with independent rep01/rep02 runs, with rep01 chosen before looking at test performance. Rep02 checked reproducibility, not an additional independent random seed. The audit did not overwrite or silently resume an existing replicate directory.

The completed audit generated gate files, resolved configs, run/evaluation metadata, metrics, patient predictions and the E42/43/44-versus-B42/43/44 comparison. The 2026-10-05 retention correction documented restoration of complete A–F records after an overly broad cleanup, with the first-round A/B/C/D seed42 from git:6584f6d excluded by the user's request, anchored by A H3 MSE=0.6268193917348981. That correction and the earlier cleanup manifest are historical operations, not a guarantee that every directory still exists after later user deletion or migration. Do not reconstruct missing outputs from these documents, Git, inventories or backups without a new request. No Git history rewrite was part of those operations.

The user directed deletion, not archival, of non-resumable incomplete SHM-failure artifacts on 2026-10-04. Their checkpoints lacked optimizer/scheduler/RNG resume state; F42/F44 had not reached checkpoint creation. E42/E43/F42/F44 then restarted from scratch without changing the frozen historical training/configuration/DataLoader protocol. E42 rep01/rep02 remained sequential on original GPU0. That recovery governor permitted at most three combined E/F tasks and required >=80 GiB free SHM before launch on an idle assigned GPU. Its recovery_shm.json recorded the new schedule, not failed run contents. This describes that completed recovery event, not a currently running coordinator.

## Historical frozen execution and gates

The strict audit scheduled A42 twice, B42 twice, then E42 twice sequentially on one GPU pinned by UUID. It kept the original split, models, losses, 2400 updates, warmup240, batch16 and validation interval24. Its primary checkpoint was minimum validation total loss strictly after warmup, earliest tie, not the diagnostic C-index checkpoint. The mathematical design, budget and checkpoint selection remain unchanged in ordinary training.

The former strict mode failed visibly on unsupported operators. It used fixed startup PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG=:4096:8, cuDNN deterministic on / benchmark off, TF32 off, math-only SDPA and seeded DataLoader generators/workers. It recorded effective numerical settings, GPU model/UUID, driver/CUDA/cuDNN/PyTorch/Python/dependency versions, data/source fingerprints, initialization and cumulative batch-order fingerprints. These settings are historical facts, not instructions for the current entry points. Ordinary training still records hardware/software and data/config/source provenance but does not force these strict numerical controls. Neither ordinary seeded training nor historical strict settings guarantee bitwise equality across hardware or library versions. Cache-manifest hashing does not hash every voxel file; cached data must remain immutable during a run.

Historical gates required equal source/config/data/version/GPU/init/order fingerprints and selected checkpoint step. Their thresholds were H1/H2/H3 MSE and H3/H1 relative difference ≤1%; cosine/C-index/Brier absolute difference ≤0.001; maximum within-patient-mean MSE relative difference ≤5%. Relative difference is abs(a-b)/max(abs(a),abs(b),1e-12). These were operational reproducibility tolerances, not significance tests or acceptance criteria for new ordinary runs.

The historical protocol held its formal comparison and deferred cleanup if any pair failed, and required reporting numerical/order divergence and checkpoint differences. That rule applied to that audit's records only. The precisely specified first-round deletion was user-directed record management, not proof that instability was caused by a particular numerical operator or that statistical exclusion was justified.

## Historical paired comparison

The user approved a scheduling amendment on 2026-10-04: E43/44 could start speculatively on same-model GPUs1/2 while E42 rep01/02 remained sequential on GPU0. The frozen training source/configuration and all deterministic settings were unchanged; per-run GPU UUIDs and the amendment were recorded in parallel_schedule.json. The historical formal E42/43/44-versus-B42 rep01 and B43/44 comparison required all A/B/E audit gates to pass. B43/44 used the earlier numerical protocol and lacked deterministic environment metadata: this was seed-matched, not fully execution-protocol-matched. Do not silently infer missing hardware versions or claim causal confirmation.

Report H1/H2/H3 MSE, cosine, C-index, Brier365 and H3/H1, with seed-paired E-minus-B differences. Average trajectories within each patient before patient-level MSE/cosine differences; use fixed primary windows for survival predictions. Average each patient's differences across seeds without treating windows/seeds as extra independent patients. Eight patients and three seeds remain exploratory.

## Reading historical artifacts

The historical output root was outputs/reproducibility/seed42_same_gpu/, with comparison/ for paired reports. Reports recorded their B43/44 control paths. Read these files only where they still exist; do not assume later user-managed output paths are unchanged. Git ignores outputs/**/*.log and weights; compact prediction and patient-difference CSVs have explicit retention rules. Local logs and history CSVs, if available, can support diagnosis.

Historical archive/index tooling only reads an existing index; it never reconstructs removed runs from Git. The retention correction's historical_replicates.json listed four ABCD historical_rep02 copies and the old E42 copy, duplicating their original result directories rather than adding independent replicates. Its inventory was outputs/ablation_inventory.json. These are historical snapshot locations, not a promise of current filesystem contents. Audit summaries can read available current-format results without historical/ and do not restart the retired training pipeline.
