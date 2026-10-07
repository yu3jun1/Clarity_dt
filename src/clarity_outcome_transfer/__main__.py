"""Evaluate frozen A/C checkpoints and generate outcome-transfer comparisons."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

from clarity_rrt_v3.evaluate import evaluate_run, training_reference
from clarity_rrt_v3.reproducibility import file_sha256, source_sha256, utc_now
from clarity_rrt_v3.train import PRIMARY_CHECKPOINT_NAME, build_datasets, configure_upstream, load_config
from .analysis import analyze_run, write_json, write_report
from .prediction import recursive_predictions


DEFAULT_CONFIG = "configs/outcome_transfer_ac_v1.yaml"


def worker(config_path, variant, seed):
    study = load_config(config_path)
    config = load_config(study["model_config"])
    config["output_root"] = study["output_root"]
    directory = Path(study["output_root"]) / "runs" / f"{variant}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    status = {"state": "evaluating", "variant": variant, "seed": seed,
              "started_at_utc": utc_now(), "pid": os.getpid(), "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}
    write_json(directory / "status.json", status)
    checkpoint = Path(study["checkpoint_root"]) / f"{variant}_seed{seed}" / PRIMARY_CHECKPOINT_NAME
    metrics, rows, _ = evaluate_run(
        config, variant, seed, "cuda:0", checkpoint, directory, prediction_fn=recursive_predictions,
    )
    reference = training_reference(build_datasets(config)["train"], 3)
    _, _, _, concordance = configure_upstream(config["upstream_root"])
    metrics["outcome_transfer"] = analyze_run(rows, reference, concordance)
    metrics["study"] = "outcome_transfer_ac_v1"
    metrics["source_checkpoint"] = str(checkpoint.resolve())
    write_json(directory / "metrics.json", metrics)
    status.update(state="complete", completed_at_utc=utc_now())
    write_json(directory / "status.json", status)


def campaign(config_path, gpus):
    study = load_config(config_path)
    root = Path(study["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", {
        **study, "started_at_utc": utc_now(), "coordinator_pid": os.getpid(), "gpus": gpus,
        "study_config_sha256": file_sha256(config_path),
        "model_config_sha256": file_sha256(study["model_config"]), "source_sha256": source_sha256(),
        "checkpoint": PRIMARY_CHECKPOINT_NAME, "training": "none; frozen existing A/C checkpoints",
        "true_latent": "Observed endpoint MRI encoded by each checkpoint's own encoder and evaluated by its own head",
        "predicted_latent": "Independent recursive member trajectories; average member risks and sigmoid probabilities",
        "mean_predicted_latent": "Auxiliary head(mean member latent) fusion control",
        "survival_unit": "Earliest eligible four-stage window per patient; H3 endpoint-relative 365 days",
        "censoring_reference": "Training patients, primary H3 window only",
        "paired_error": "C minus A; latent errors averaged within patient/horizon; H3 survival uses primary window",
        "uncertainty_error": "H3 squared survival-probability error among patients with known 365-day status",
    })
    jobs = [(variant, seed) for variant in ("A", "C") for seed in study["seeds"]]
    completed = []
    for offset in range(0, len(jobs), len(gpus)):
        active = jobs[offset:offset + len(gpus)]
        names = [f"{variant}_seed{seed}" for variant, seed in active]
        write_json(root / "campaign_status.json", {
            "state": "running", "active_runs": names, "completed_runs": completed, "updated_at_utc": utc_now(),
        })
        processes = []
        for gpu, (variant, seed), name in zip(gpus, active, names):
            command = [sys.executable, "-m", "clarity_outcome_transfer", "worker", "--config", str(config_path),
                       "--variant", variant, "--seed", str(seed)]
            print(f"START {name} GPU{gpu} {utc_now()}", flush=True)
            with (root / f"{name}.log").open("w") as log:
                process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                           stdout=log, stderr=subprocess.STDOUT)
            processes.append((name, process))
        failed = []
        for name, process in processes:
            if process.wait() != 0:
                failed.append(name)
            else:
                completed.append(name)
                print(f"COMPLETE {name} {utc_now()}", flush=True)
        if failed:
            write_json(root / "campaign_status.json", {"state": "failed", "failed_runs": failed,
                                                       "completed_runs": completed, "updated_at_utc": utc_now()})
            raise RuntimeError(f"Evaluation failed: {failed}")
    write_report(root, study["seeds"])
    write_json(root / "campaign_status.json", {
        "state": "complete", "completed_runs": completed, "completed_at_utc": utc_now(),
    })
    print(f"COMPLETE outcome transfer report {utc_now()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("campaign", "worker"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--variant", choices=("A", "C"))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    if args.command == "campaign":
        campaign(args.config, args.gpus)
    else:
        worker(args.config, args.variant, args.seed)


if __name__ == "__main__":
    main()
