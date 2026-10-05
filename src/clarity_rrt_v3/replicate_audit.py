"""Same-device replicate gates, historical preservation, and paired comparisons."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .reproducibility import source_sha256, utc_now


DEFAULT_ROOT = Path("outputs/reproducibility/seed42_same_gpu")
MAIN_ROOT = Path("outputs/pure_rrt_v3_step2400")
METRIC_NAMES = ("latent_mse", "cosine_similarity", "c_index", "brier365")
# These operational tolerances are fixed before seeing the new replicates.
GATE_POLICY = {
    "aggregate_mse_and_ratio_relative_tolerance": 0.01,
    "cosine_cindex_brier_absolute_tolerance": 0.001,
    "max_patient_mse_relative_tolerance": 0.05,
    "same_selected_checkpoint_step_required": True,
    "same_initial_state_batch_order_device_protocol_required": True,
    "formal_replicate": "rep01 (preselected, never the better-scoring run)",
    "historical_archives_policy": "retired archives are never automatically recreated",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers never see a partially written progress/report file.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("horizon", "primary_survival_window", "event"):
            row[key] = int(row[key])
        for key in ("latent_mse", "cosine_similarity", "risk", "survival365", "survival_time"):
            row[key] = float(row[key])
    return rows


def prediction_key(row: Mapping[str, Any]) -> tuple:
    return tuple(row[key] for key in ("patient", "start", "end", "window_end", "horizon"))


def patient_means(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, dict[str, float]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["patient"], row["horizon"]), []).append(row)
    return {
        key: {metric: float(np.mean([row[metric] for row in values]))
              for metric in ("latent_mse", "cosine_similarity")}
        for key, values in groups.items()
    }


def history_diagnostics(left: Path, right: Path) -> dict[str, Any]:
    if not left.is_file() or not right.is_file():
        return {"available": False}
    with left.open(newline="") as handle:
        rows1 = list(csv.DictReader(handle))
    with right.open(newline="") as handle:
        rows2 = list(csv.DictReader(handle))
    first_loss = first_order = None
    for one, two in zip(rows1, rows2):
        if first_order is None and one.get("batch_order_sha256") != two.get("batch_order_sha256"):
            first_order = int(float(one["optimizer_step"]))
        fields = [key for key in one if key.startswith(("train_", "validation_"))]
        if first_loss is None and any(float(one[key]) != float(two[key]) for key in fields):
            first_loss = int(float(one["optimizer_step"]))
    return {
        "available": True, "row_counts": [len(rows1), len(rows2)],
        "first_numerical_divergence_step": first_loss,
        "first_batch_order_divergence_step": first_order,
    }


def assess_pair(left: Path, right: Path) -> dict[str, Any]:
    one, two = read_json(left / "metrics.json"), read_json(right / "metrics.json")
    meta1, meta2 = read_json(left / "run_metadata.json"), read_json(right / "run_metadata.json")
    failures = []
    required = ("initial_trainable_state_sha256", "batch_order_sha256", "source_sha256",
                "config_sha256", "data_fingerprints", "packages", "cuda_version", "cudnn_version",
                "determinism", "cuda_visible_devices")
    mismatches = [key for key in required if meta1.get(key) != meta2.get(key) or meta1.get(key) is None]
    uuids = [(meta.get("gpu") or {}).get("uuid") for meta in (meta1, meta2)]
    if uuids[0] != uuids[1] or uuids[0] in (None, "unknown"):
        mismatches.append("gpu_uuid")
    for metadata in (meta1, meta2):
        settings = metadata.get("determinism", {})
        if not settings.get("algorithms_enabled") or settings.get("warn_only", True):
            failures.append("strict deterministic algorithms were not enabled")
    if mismatches:
        failures.append("provenance mismatch: " + ", ".join(mismatches))
    if (one["variant"], one["seed"]) != (two["variant"], two["seed"]):
        failures.append("variant/seed mismatch")
    checkpoint_same = one["checkpoint_optimizer_step"] == two["checkpoint_optimizer_step"]
    if not checkpoint_same:
        failures.append("selected checkpoint optimizer step differs")
    differences = {}
    for horizon in ("H1", "H2", "H3"):
        differences[horizon] = {}
        for metric in METRIC_NAMES:
            a, b = float(one["recursive"][horizon][metric]), float(two["recursive"][horizon][metric])
            difference = relative_difference(a, b) if metric == "latent_mse" else abs(a - b)
            limit = 0.01 if metric == "latent_mse" else 0.001
            differences[horizon][metric] = {"left": a, "right": b, "difference": difference}
            if not math.isfinite(difference) or difference > limit:
                failures.append(f"{horizon} {metric} difference exceeds {limit}")
    ratios = [float(run["recursive"]["H3"]["latent_mse"]) / float(run["recursive"]["H1"]["latent_mse"])
              for run in (one, two)]
    ratio_difference = relative_difference(*ratios)
    if not math.isfinite(ratio_difference) or ratio_difference > 0.01:
        failures.append("H3/H1 relative difference exceeds 1%")
    rows1, rows2 = [read_predictions(path / "recursive_predictions.csv") for path in (left, right)]
    keys1, keys2 = [set(prediction_key(row) for row in rows) for rows in (rows1, rows2)]
    patient1, patient2 = patient_means(rows1), patient_means(rows2)
    if keys1 != keys2 or len(keys1) != len(rows1) or len(keys2) != len(rows2):
        failures.append("prediction trajectory keys differ or contain duplicates")
    patient_deltas = [relative_difference(patient1[key]["latent_mse"], patient2[key]["latent_mse"])
                      for key in sorted(patient1.keys() & patient2.keys())]
    max_patient = max(patient_deltas, default=1e300)
    if not math.isfinite(max_patient) or max_patient > 0.05:
        failures.append("max patient-mean MSE relative difference exceeds 5%")
    return {
        "variant": one["variant"], "seed": one["seed"], "stable": not failures,
        "left": str(left), "right": str(right), "policy": GATE_POLICY,
        "failures": failures, "metadata_mismatches": mismatches,
        "checkpoint_steps": [one["checkpoint_optimizer_step"], two["checkpoint_optimizer_step"]],
        "horizon_differences": differences, "h3_h1_ratios": ratios,
        "h3_h1_relative_difference": ratio_difference,
        "max_patient_mse_relative_difference": max_patient,
        "history": history_diagnostics(left / "history.csv", right / "history.csv"),
        "diagnostic_interpretation": (
            "Compare first history divergence with batch-order and initialization fingerprints; "
            "same batches/initialization but divergent losses implicate numerical kernels; "
            "different checkpoint steps require inspecting the validation-loss curve. "
            "Matching metadata alone does not prove which CUDA operator caused divergence."
        ),
    }


def preserve_history(root: Path) -> dict[str, Any]:
    """Read an existing archive index, but never recreate retired experiments."""
    manifest = root / "historical_replicates.json"
    if manifest.exists():
        return read_json(manifest)
    return {"replicates": [], "historical_archives_retired": True}


def seed_matched_comparison(pairs: Sequence[tuple[int, Path, Path]], destination: Path) -> dict[str, Any]:
    patient_rows = []
    summaries = []
    for seed, b_directory, e_directory in pairs:
        b_metrics, e_metrics = [read_json(path / "metrics.json") for path in (b_directory, e_directory)]
        if b_metrics["seed"] != seed or e_metrics["seed"] != seed:
            raise ValueError("Seed mismatch")
        if b_metrics["variant"] != "B" or e_metrics["variant"] != "E":
            raise ValueError("Expected B and E")
        b_rows, e_rows = [read_predictions(path / "recursive_predictions.csv") for path in (b_directory, e_directory)]
        b_keys, e_keys = [set(prediction_key(row) for row in rows) for rows in (b_rows, e_rows)]
        if b_keys != e_keys or len(b_keys) != len(b_rows) or len(e_keys) != len(e_rows):
            raise ValueError("Cannot pair different trajectory cohorts or duplicate keys")
        b_lookup = {prediction_key(row): row for row in b_rows}
        for row in e_rows:
            b_row = b_lookup[prediction_key(row)]
            for field in ("survival_time", "event", "primary_survival_window"):
                if b_row[field] != row[field]:
                    raise ValueError(f"Patient endpoint mismatch: {field}")
        b_patient, e_patient = patient_means(b_rows), patient_means(e_rows)
        primary_b = {(row["patient"], row["horizon"]): row for row in b_rows if row["primary_survival_window"]}
        primary_e = {(row["patient"], row["horizon"]): row for row in e_rows if row["primary_survival_window"]}
        for patient, horizon in sorted(b_patient):
            key = patient, horizon
            b, e = b_patient[key], e_patient[key]
            patient_rows.append({
                "seed": seed, "patient": patient, "horizon": horizon,
                "B_patient_mean_mse": b["latent_mse"], "E_patient_mean_mse": e["latent_mse"],
                "mse_difference_E_minus_B": e["latent_mse"] - b["latent_mse"],
                "B_patient_mean_cosine": b["cosine_similarity"], "E_patient_mean_cosine": e["cosine_similarity"],
                "cosine_difference_E_minus_B": e["cosine_similarity"] - b["cosine_similarity"],
                "risk_difference_E_minus_B": primary_e[key]["risk"] - primary_b[key]["risk"],
                "survival365_difference_E_minus_B": primary_e[key]["survival365"] - primary_b[key]["survival365"],
            })
        b_ratio, e_ratio = [float(m["recursive"]["H3"]["latent_mse"]) / float(m["recursive"]["H1"]["latent_mse"])
                            for m in (b_metrics, e_metrics)]
        summaries.append({
            "seed": seed, "B_path": str(b_directory), "E_path": str(e_directory),
            "B_deterministic_requested": b_metrics.get("deterministic_requested", False),
            "E_deterministic_requested": e_metrics.get("deterministic_requested", False),
            "B": b_metrics["recursive"], "E": e_metrics["recursive"],
            "E_minus_B": {h: {metric: e_metrics["recursive"][h][metric] - b_metrics["recursive"][h][metric]
                               for metric in METRIC_NAMES} for h in ("H1", "H2", "H3")},
            "B_h3_h1": b_ratio, "E_h3_h1": e_ratio, "h3_h1_difference_E_minus_B": e_ratio - b_ratio,
        })
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "patient_level_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_rows[0]))
        writer.writeheader()
        writer.writerows(patient_rows)
    result = {
        "direction": "E minus B; negative MSE/Brier or positive cosine/C-index favors E",
        "formal_seed42_replicate": "rep01 selected before inspecting performance",
        "runs": summaries,
        "limitations": [
            "B seeds43/44 are existing historical controls; deterministic environment metadata is unavailable. "
            "This is seed-matched, not fully execution-protocol-matched; causal attribution needs matched B43/44 reruns.",
            "Patient means average windows within each patient, then give each patient equal weight; windows are not independent patients.",
            "Eight test patients and three seeds are exploratory; no confirmatory significance claim.",
        ],
        "seed_paired_summary": {
            horizon: {metric: {"mean_difference": float(np.mean([run["E_minus_B"][horizon][metric] for run in summaries])),
                               "std_across_seeds": float(np.std([run["E_minus_B"][horizon][metric] for run in summaries], ddof=1)) if len(summaries) > 1 else 0.0}
                      for metric in METRIC_NAMES} for horizon in ("H1", "H2", "H3")
        },
        "h3_h1_mean_seed_paired_difference": float(np.mean([run["h3_h1_difference_E_minus_B"] for run in summaries])),
        "patient_paired_summary": {},
    }
    for horizon in (1, 2, 3):
        selected = [row for row in patient_rows if row["horizon"] == horizon]
        per_patient = {}
        for row in selected:
            per_patient.setdefault(row["patient"], []).append(row["mse_difference_E_minus_B"])
        differences = [float(np.mean(values)) for values in per_patient.values()]
        result["patient_paired_summary"][f"H{horizon}"] = {
            "unique_patient_count": len(per_patient), "patient_seed_pairs": len(selected),
            "mean_patient_difference_averaged_over_seeds": float(np.mean(differences)),
            "patients_with_lower_E_mse": sum(value < 0 for value in differences),
            "patient_differences": dict(zip(sorted(per_patient), [float(np.mean(per_patient[key])) for key in sorted(per_patient)])),
        }
    write_json(destination / "seed_matched_summary.json", result)
    lines = ["# Seed-matched E versus B", "", *[f"- {note}" for note in result["limitations"]], "",
             "|Seed|Group|H1 MSE|H2 MSE|H3 MSE|H3/H1|", "|---|---|---|---|---|---|"]
    for run in summaries:
        for group in ("B", "E"):
            mse = [run[group][h]["latent_mse"] for h in ("H1", "H2", "H3")]
            lines.append(f"|{run['seed']}|{group}|{mse[0]:.6f}|{mse[1]:.6f}|{mse[2]:.6f}|{run[group + '_h3_h1']:.4f}|")
    lines += ["", "Patient-level paired differences are in `patient_level_differences.csv`; all horizon metrics and seed-paired differences are in `seed_matched_summary.json`."]
    (destination / "seed_matched_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def write_audit_summary(root: Path, gates: Mapping[str, Any]) -> None:
    lines = ["# 同卡 seed42 reproducibility audit", "",
             "当前报告只列出确定性复跑。已退役实验可从清理前 Git tag 恢复。", "",
             "|Variant|Replicate|H1 MSE|H2 MSE|H3 MSE|Checkpoint step|",
             "|---|---|---|---|---|---|"]
    for variant in "ABE":
        for replicate in ("rep01", "rep02"):
            path = root / "primary" / f"{variant}_seed42_{replicate}" / "metrics.json"
            if path.exists():
                metrics = read_json(path)
                mse = [metrics["recursive"][h]["latent_mse"] for h in ("H1", "H2", "H3")]
                lines.append(f"|{variant}|{replicate}|{mse[0]:.6f}|{mse[1]:.6f}|{mse[2]:.6f}|{metrics['checkpoint_optimizer_step']}|")
    lines += ["", "## 稳定性 gates", ""]
    for variant in "ABE":
        gate = gates.get(variant)
        lines.append(f"- {variant}: " + ("pending" if gate is None else ("stable" if gate["stable"] else "UNSTABLE — " + "; ".join(gate["failures"]))))
    lines += ["", "正式 seed42 使用预先指定的 rep01；rep02 不计作独立 seed。",
              "2026-10-04 用户批准 E43/44 提前在 GPU1/2 并行启动；只有全部 gate 稳定后纳入正式比较。E42 两次复跑仍在 GPU0 连续进行，调度变更见 parallel_schedule.json。B43/44 为保留的历史对照，最终报告明确执行协议差异。",
              "不可续训的共享内存事故失败记录已按用户要求删除、不归档；E/F 从头恢复的资源限制和调度见 recovery_shm.json。",
              "最终 H1/H2/H3、H3/H1 和患者配对比较见 comparison/。"]
    (root / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(root: Path, variant: str, seed: int, replicate: str, gpu_uuid: str, frozen_source: str) -> Path:
    directory = root / "primary" / f"{variant}_seed{seed}_{replicate}"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite or silently resume {directory}")
    if source_sha256() != frozen_source:
        raise RuntimeError("Source/config changed during audit; stopping instead of mixing protocols")
    directory.mkdir(parents=True)
    config = "configs/ablations/teacher_forced_stagewise.yaml" if variant == "E" else "configs/pure_rrt_v3.yaml"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, PYTHONHASHSEED=str(seed),
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    arguments = ["--config", config, "--variant", variant, "--seed", str(seed), "--device", "cuda:0",
                 "--output-root", str(root), "--replicate", replicate, "--deterministic"]
    status = {"variant": variant, "seed": seed, "replicate_id": replicate, "gpu_uuid": gpu_uuid,
              "state": "training", "train_start_utc": utc_now()}
    write_json(directory / "status.json", status)
    print(f"START {variant} seed={seed} {replicate} GPU={gpu_uuid} {utc_now()}", flush=True)
    try:
        with (directory / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "clarity_rrt_v3.train", *arguments], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status.update(state="evaluating", evaluate_start_utc=utc_now())
        write_json(directory / "status.json", status)
        with (directory / "evaluate.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "clarity_rrt_v3.evaluate", "run", *arguments], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if source_sha256() != frozen_source:
            raise RuntimeError("Source/config changed during run; results require re-audit")
        status.update(state="complete", complete_utc=utc_now())
    except Exception as error:
        status.update(state="failed", error=str(error), failed_utc=utc_now())
        write_json(directory / "status.json", status)
        raise
    write_json(directory / "status.json", status)
    print(f"COMPLETE {variant} seed={seed} {replicate} {utc_now()}", flush=True)
    return directory


def pipeline(root: Path, gpu_id: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_uuid = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=uuid", "--format=csv,noheader"], text=True,
        ).strip()
        # An idle CUDA context can reserve a few hundred MiB without doing work.
        # Reject materially occupied devices; never terminate another user's job.
        usage = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True,
        ).strip()
        memory_used, utilization = [int(value.strip()) for value in usage.split(",")]
        if memory_used > 1024 or utilization > 5:
            raise RuntimeError(f"GPU {gpu_id} is busy ({usage}); choose an idle card")
        preserve_history(root)
        frozen_source = source_sha256()
        write_json(root / "protocol.json", {
            "created_at_utc": utc_now(), "gpu_id": gpu_id, "gpu_uuid": gpu_uuid,
            "source_sha256": frozen_source, "policy": GATE_POLICY,
            "gpu_usage_at_launch": {"memory_used_mib": memory_used, "utilization_percent": utilization},
            "queue": [f"{variant}_seed42_{rep}" for variant in "ABE" for rep in ("rep01", "rep02")],
            "conditional_queue": ["E_seed43_rep01", "E_seed44_rep01"],
            "no_original_outputs_modified": True,
        })
        gates = {}
        write_audit_summary(root, gates)
        try:
            for variant in "ABE":
                write_json(root / "pipeline_status.json", {"state": "running", "active_variant": variant, "updated_at_utc": utc_now(), "gates": gates})
                pair = [run_experiment(root, variant, 42, rep, gpu_uuid, frozen_source) for rep in ("rep01", "rep02")]
                gates[variant] = assess_pair(*pair)
                write_json(root / f"{variant}_seed42_stability.json", gates[variant])
                write_audit_summary(root, gates)
                print(f"GATE {variant}: {'stable' if gates[variant]['stable'] else 'UNSTABLE'}", flush=True)
            if not all(gate["stable"] for gate in gates.values()):
                write_json(root / "pipeline_status.json", {
                    "state": "needs_diagnosis", "gates": gates, "updated_at_utc": utc_now(),
                    "E43_44": "held: at least one replicate pair is unstable", "historical_archives_retired": True,
                })
                return
            for seed in (43, 44):
                write_json(root / "pipeline_status.json", {"state": "running", "active_variant": "E", "active_seed": seed, "updated_at_utc": utc_now(), "gates": gates})
                run_experiment(root, "E", seed, "rep01", gpu_uuid, frozen_source)
            pairs = [(42, root / "primary/B_seed42_rep01", root / "primary/E_seed42_rep01")]
            pairs.extend((seed, MAIN_ROOT / "primary" / f"B_seed{seed}", root / "primary" / f"E_seed{seed}_rep01") for seed in (43, 44))
            seed_matched_comparison(pairs, root / "comparison")
            write_json(root / "pipeline_status.json", {"state": "complete", "updated_at_utc": utc_now(), "gates": gates, "historical_archives_retired": True})
        except Exception as error:
            write_json(root / "pipeline_status.json", {"state": "failed", "updated_at_utc": utc_now(), "error": str(error), "gates": gates})
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("archive", "pipeline"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "archive":
        preserve_history(args.output_root)
        status = args.output_root / "pipeline_status.json"
        gates = read_json(status).get("gates", {}) if status.exists() else {}
        write_audit_summary(args.output_root, gates)
    else:
        pipeline(args.output_root, args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
