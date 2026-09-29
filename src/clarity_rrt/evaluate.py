"""Evaluate only recursive H1/H2/H3 deployment for every experiment group."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .model import StagewiseDynamics
from .train import (
    build_loaders,
    configure_upstream,
    encode_mri_stages,
    load_config,
    run_directory,
    seed_everything,
    upstream_args,
)


def training_reference(dataset, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    times = []
    events = []
    for trajectory in dataset.trajectories:
        survival = trajectory.points[horizon]["survival"]
        times.append(float(survival["survival_from_tp_days"]))
        events.append(bool(survival["event_indicator"]))
    return np.asarray(times), np.asarray(events)


def brier365(
    train_time: np.ndarray,
    train_event: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> float:
    from sksurv.metrics import brier_score
    from sksurv.util import Surv

    test_time = np.asarray([row["survival_time"] for row in rows])
    test_event = np.asarray([bool(row["event"]) for row in rows])
    probability = np.asarray([row["survival365"] for row in rows])[:, None]
    _, score = brier_score(
        Surv.from_arrays(train_event, train_time),
        Surv.from_arrays(test_event, test_time),
        probability,
        np.asarray([365.0]),
    )
    return float(score[0])


@torch.no_grad()
def recursive_predictions(
    model: StagewiseDynamics,
    loader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        mri = batch["mri"].to(device)
        deltas = batch["deltas"].to(device)
        states = encode_mri_stages(model, mri, (0, 1, 2, 3))
        targets = states[:, 1:]
        step_conditions = model.encode_conditions(
            batch["step_text"], batch["clinical_text"]
        )
        prefix_conditions = model.encode_conditions(
            batch["prefix_text"], batch["clinical_text"]
        )
        member_states = model.rollout(states[:, 0], step_conditions, deltas)
        risks, logits = model.survival(states[:, 0], member_states, prefix_conditions)
        mean_state = member_states.mean(dim=0)
        mse = ((mean_state - targets) ** 2).mean(dim=(-1, -2))
        cosine = F.cosine_similarity(
            mean_state.flatten(2),
            targets.flatten(2),
            dim=-1,
        )
        uncertainty = member_states.var(dim=0, unbiased=False).mean(dim=(-1, -2))
        mean_risk = risks.mean(dim=0)
        mean_probability = torch.sigmoid(logits).mean(dim=0)
        for sample in range(len(batch["patient"])):
            for horizon in range(3):
                rows.append(
                    {
                        "patient": batch["patient"][sample],
                        "start": batch["timepoints"][sample][0],
                        "end": batch["timepoints"][sample][horizon + 1],
                        "horizon": horizon + 1,
                        "latent_mse": float(mse[sample, horizon].cpu()),
                        "cosine_similarity": float(cosine[sample, horizon].cpu()),
                        "uncertainty": float(uncertainty[sample, horizon].cpu()),
                        "risk": float(mean_risk[sample, horizon].cpu()),
                        "survival365": float(mean_probability[sample, horizon].cpu()),
                        "survival_time": float(batch["survival_time"][sample, horizon]),
                        "event": int(batch["event"][sample, horizon]),
                    }
                )
    return rows


def horizon_metrics(
    rows: Sequence[Mapping[str, Any]],
    reference: tuple[np.ndarray, np.ndarray],
    concordance_index,
) -> dict[str, float]:
    return {
        "samples": float(len(rows)),
        "latent_mse": float(np.mean([row["latent_mse"] for row in rows])),
        "cosine_similarity": float(
            np.mean([row["cosine_similarity"] for row in rows])
        ),
        "c_index": float(
            concordance_index(
                [row["risk"] for row in rows],
                [row["survival_time"] for row in rows],
                [row["event"] for row in rows],
            )
        ),
        "brier365": brier365(*reference, rows),
    }


def uncertainty_metrics(
    rows: Sequence[Mapping[str, Any]],
    coverages: Sequence[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in (1, 2, 3):
        selected = [row for row in rows if row["horizon"] == horizon]
        uncertainty = np.asarray([row["uncertainty"] for row in selected])
        error = np.asarray([row["latent_mse"] for row in selected])
        correlation = float(np.corrcoef(uncertainty, error)[0, 1])
        ordered = sorted(selected, key=lambda row: row["uncertainty"])
        selective = {}
        for coverage in coverages:
            count = max(1, math.ceil(float(coverage) * len(ordered)))
            retained = ordered[:count]
            selective[f"{float(coverage):g}"] = {
                "retained": count,
                "latent_mse": float(
                    np.mean([row["latent_mse"] for row in retained])
                ),
            }
        result[f"H{horizon}"] = {
            "variance_error_pearson": correlation,
            "selective_rollout": selective,
        }
    return result


def evaluate_one(
    config_path: str | Path,
    variant: str,
    seed: int,
    device_name: str,
    rrt_override: float | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(seed)
    device = torch.device(device_name)
    variant_config = config["variants"][variant]
    upstream_train, _, _, concordance = configure_upstream(config["upstream_root"])
    clarity = upstream_train.build_model(upstream_args(config, seed), device)
    model = StagewiseDynamics(
        clarity,
        ensemble_size=int(variant_config["ensemble_size"]),
        seed=seed,
    ).to(device)
    run_dir = run_directory(config, variant, seed, rrt_override)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    loaders, datasets = build_loaders(config, seed)
    rows = recursive_predictions(model, loaders["test"], device)
    with (run_dir / "recursive_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    horizons = {}
    for horizon in (1, 2, 3):
        horizon_rows = [row for row in rows if row["horizon"] == horizon]
        horizons[f"H{horizon}"] = horizon_metrics(
            horizon_rows,
            training_reference(datasets["train"], horizon),
            concordance,
        )
    metrics: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "rrt_weight": float(checkpoint["rrt_weight"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "recursive": horizons,
        "error_accumulation": (
            horizons["H3"]["latent_mse"] - horizons["H1"]["latent_mse"]
        ),
    }
    if int(variant_config["ensemble_size"]) > 1:
        metrics["uncertainty"] = uncertainty_metrics(
            rows, config["evaluation"]["selective_coverages"]
        )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def statistics(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": list(values),
    }


def collect_runs(
    config: Mapping[str, Any],
    variant: str,
    rrt_override: float | None = None,
) -> list[dict[str, Any]]:
    runs = []
    for seed in config["training"]["seeds"]:
        path = run_directory(config, variant, int(seed), rrt_override) / "metrics.json"
        if path.is_file():
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for horizon in (1, 2, 3):
        key = f"H{horizon}"
        for metric in ("latent_mse", "cosine_similarity", "c_index", "brier365"):
            summary[f"{key}_{metric}"] = statistics(
                [float(run["recursive"][key][metric]) for run in runs]
            )
    summary["error_accumulation"] = statistics(
        [float(run["error_accumulation"]) for run in runs]
    )
    if "uncertainty" in runs[0]:
        for horizon in (1, 2, 3):
            key = f"H{horizon}"
            summary[f"{key}_uncertainty_error_pearson"] = statistics(
                [
                    float(run["uncertainty"][key]["variance_error_pearson"])
                    for run in runs
                ]
            )
            coverages = runs[0]["uncertainty"][key]["selective_rollout"]
            for coverage in coverages:
                summary[f"{key}_selective_{coverage}_latent_mse"] = statistics(
                    [
                        float(
                            run["uncertainty"][key]["selective_rollout"][coverage][
                                "latent_mse"
                            ]
                        )
                        for run in runs
                    ]
                )
    return summary


def aggregate(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    primary = {
        variant: summarize_runs(runs)
        for variant in "ABCD"
        if (runs := collect_runs(config, variant))
    }
    ablation: dict[str, Any] = defaultdict(dict)
    for variant in config["ablation"]["variants"]:
        for weight in config["ablation"]["rrt_weights"]:
            runs = collect_runs(config, variant, float(weight))
            if runs:
                ablation[variant][f"{float(weight):g}"] = summarize_runs(runs)
    output = {"primary": primary, "rrt_ablation": dict(ablation)}
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    def value(block: Mapping[str, Any], key: str) -> str:
        metric = block[key]
        return f"{metric['mean']:.4f} ± {metric['std']:.4f}"

    lines = [
        "# Stage-wise recursive experiment",
        "",
        "## Latent rollout",
        "",
        "| Group | H1 MSE ↓ | H2 MSE ↓ | H3 MSE ↓ | ΔMSE H3-H1 ↓ | H1 cosine ↑ | H2 cosine ↑ | H3 cosine ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in "ABCD":
        if variant not in primary:
            continue
        block = primary[variant]
        lines.append(
            f"| {variant} | {value(block, 'H1_latent_mse')} | "
            f"{value(block, 'H2_latent_mse')} | {value(block, 'H3_latent_mse')} | "
            f"{value(block, 'error_accumulation')} | "
            f"{value(block, 'H1_cosine_similarity')} | "
            f"{value(block, 'H2_cosine_similarity')} | "
            f"{value(block, 'H3_cosine_similarity')} |"
        )
    lines.extend(
        [
            "",
            "## Survival",
            "",
            "| Group | H1 C-index ↑ | H2 C-index ↑ | H3 C-index ↑ | H1 Brier ↓ | H2 Brier ↓ | H3 Brier ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in "ABCD":
        if variant not in primary:
            continue
        block = primary[variant]
        lines.append(
            f"| {variant} | {value(block, 'H1_c_index')} | "
            f"{value(block, 'H2_c_index')} | {value(block, 'H3_c_index')} | "
            f"{value(block, 'H1_brier365')} | {value(block, 'H2_brier365')} | "
            f"{value(block, 'H3_brier365')} |"
        )
    lines.extend(
        [
            "",
            "## Ensemble uncertainty",
            "",
            "| Group | H1 corr(var,error) ↑ | H2 corr(var,error) ↑ | H3 corr(var,error) ↑ | H1 low-50% MSE ↓ | H2 low-50% MSE ↓ | H3 low-50% MSE ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in ("C", "D"):
        if variant not in primary:
            continue
        block = primary[variant]
        lines.append(
            f"| {variant} | {value(block, 'H1_uncertainty_error_pearson')} | "
            f"{value(block, 'H2_uncertainty_error_pearson')} | "
            f"{value(block, 'H3_uncertainty_error_pearson')} | "
            f"{value(block, 'H1_selective_0.5_latent_mse')} | "
            f"{value(block, 'H2_selective_0.5_latent_mse')} | "
            f"{value(block, 'H3_selective_0.5_latent_mse')} |"
        )
    lines.extend(
        [
            "",
            "## RRT weight ablation",
            "",
            "| Group | λ_RRT | H3 latent MSE ↓ | H3 C-index ↑ | H3 Brier ↓ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant, weights in ablation.items():
        for weight, block in weights.items():
            lines.append(
                f"| {variant} | {weight} | {value(block, 'H3_latent_mse')} | "
                f"{value(block, 'H3_c_index')} | {value(block, 'H3_brier365')} |"
            )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", default="configs/stagewise_recursive.yaml")
    run.add_argument("--variant", required=True, choices=list("ABCD"))
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--rrt-weight", type=float)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--config", default="configs/stagewise_recursive.yaml")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        metrics = evaluate_one(
            args.config, args.variant, args.seed, args.device, args.rrt_weight
        )
    else:
        metrics = aggregate(args.config)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
