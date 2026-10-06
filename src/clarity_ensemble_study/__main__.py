"""Run the all-pair ensemble reliability study and its size sensitivity study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from clarity_rrt_v3.evaluate import evaluate_run
from clarity_rrt_v3.reproducibility import utc_now
from clarity_rrt_v3.train import PRIMARY_CHECKPOINT_NAME, load_config, train_run
from .analysis import write_report
from .prediction import recursive_predictions


DEFAULT_CONFIG = "configs/ensemble_reliability_size_v1.yaml"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_study(config, size, seed, checkpoint, directory, origin):
    metrics, _, diagnostics = evaluate_run(
        config, f"M{size}", seed, "cuda:0", checkpoint, directory,
        prediction_fn=recursive_predictions,
    )
    metrics.update(
        study="ensemble_reliability_size", ensemble_size=size,
        origin=origin, source_checkpoint=str(Path(checkpoint).resolve()),
        reliability=diagnostics["reliability"],
    )
    write_json(directory / "metrics.json", metrics)
    return metrics


def worker(config_path, size, seed, reference):
    config = load_config(config_path)
    directory = Path(config["output_root"]) / "runs" / f"M{size}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    status = {
        "ensemble_size": size, "seed": seed, "started_at_utc": utc_now(),
        "worker_pid": os.getpid(), "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    if reference:
        variant = {1: "A", 3: "C"}[size]
        checkpoint = Path(config["reference_root"]) / "primary" / f"{variant}_seed{seed}" / PRIMARY_CHECKPOINT_NAME
        status.update(state="evaluating", origin="existing_checkpoint")
        write_json(directory / "status.json", status)
        evaluate_study(config, size, seed, checkpoint, directory, "existing_checkpoint")
    else:
        status.update(state="training", origin="new_training")
        write_json(directory / "status.json", status)
        train_run(config, f"M{size}", seed, "cuda:0", directory)
        status.update(state="evaluating")
        write_json(directory / "status.json", status)
        evaluate_study(config, size, seed, directory / PRIMARY_CHECKPOINT_NAME, directory, "new_training")
    status.update(state="complete", completed_at_utc=utc_now())
    write_json(directory / "status.json", status)


def finalize(config_path):
    config = load_config(config_path)
    root = Path(config["output_root"])
    protocol = json.loads((root / "protocol.json").read_text())
    protocol.update(
        finalizer_pid=os.getpid(),
        scheduling="Independent workers; finalizer waits for all runs before generating Step 2",
    )
    write_json(root / "protocol.json", protocol)
    names = [f"M{size}_seed{seed}" for size in (1, 3, 2, 5)
             for seed in config["training"]["seeds"]]
    while True:
        completed = []
        for name in names:
            status_path = root / "runs" / name / "status.json"
            if status_path.exists() and json.loads(status_path.read_text())["state"] == "complete":
                completed.append(name)
        active = [name for name in names if name not in completed]
        if not active:
            break
        write_json(root / "campaign_status.json", {
            "state": "running", "stage": "step2_size", "active_runs": active,
            "completed_runs": completed, "updated_at_utc": utc_now(),
        })
        time.sleep(30)
    write_report(root, "step2_size", [1, 2, 3, 5])
    write_json(root / "campaign_status.json", {
        "state": "complete", "completed_runs": completed, "completed_at_utc": utc_now(),
    })
    print(f"COMPLETE campaign {utc_now()}", flush=True)


def campaign(config_path):
    config = load_config(config_path)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", {
        "study": "ensemble_reliability_size", "started_at_utc": utc_now(),
        "config": str(Path(config_path).resolve()),
        "coordinator_pid": os.getpid(),
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "seeds": config["training"]["seeds"],
        "reference_models": {"M1": "A", "M3": "C"},
        "new_training_sizes": [2, 5],
        "uncertainty": "Uz=mean_member squared L2 deviation; Us=population variance of sigmoid probability",
        "coverage": [1.0, 0.8, 0.6, 0.4],
        "scheduling": "One idle GPU, sequential jobs; each new training is followed by evaluation",
    })
    completed = []
    stages = [("step1_reliability", [1, 3], True), ("step2_size", [2, 5], False)]
    for stage, sizes, reference in stages:
        for size in sizes:
            for seed in config["training"]["seeds"]:
                name = f"M{size}_seed{seed}"
                write_json(root / "campaign_status.json", {
                    "state": "running", "stage": stage, "active_run": name,
                    "completed_runs": completed, "updated_at_utc": utc_now(),
                })
                print(f"START {stage} {name} {utc_now()}", flush=True)
                command = [sys.executable, "-m", "clarity_ensemble_study", "worker",
                           "--config", str(Path(config_path).resolve()),
                           "--size", str(size), "--seed", str(seed)]
                if reference:
                    command.append("--reference")
                with (root / f"{name}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                completed.append(name)
                print(f"COMPLETE {name} {utc_now()}", flush=True)
        if reference:
            write_report(root, stage, [1, 3])
    finalize(config_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("campaign", "worker", "finalize"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--size", type=int, choices=(1, 2, 3, 5))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "campaign":
        campaign(args.config)
    elif args.command == "finalize":
        finalize(args.config)
    else:
        worker(args.config, args.size, args.seed, args.reference)


if __name__ == "__main__":
    main()
