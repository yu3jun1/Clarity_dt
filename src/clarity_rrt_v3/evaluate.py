"""Evaluate only recursive H1/H2/H3 deployment for every experiment group."""

from __future__ import annotations

import argparse
import csv
import json
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


def encoder_trainable_state(model: StagewiseDynamics) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.mri_encoder.named_parameters()
        if parameter.requires_grad
    }


def encoder_rms_delta(
    model: StagewiseDynamics,
    initial: Mapping[str, torch.Tensor],
) -> float:
    squared = 0.0
    count = 0
    for name, parameter in model.mri_encoder.named_parameters():
        if parameter.requires_grad:
            difference = parameter.detach().cpu().float() - initial[name].float()
            squared += float(difference.square().sum())
            count += difference.numel()
    return float(np.sqrt(squared / count))


@torch.no_grad()
def recursive_predictions(
    model: StagewiseDynamics,
    loader,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    observed_states = []
    transition_distances = []
    for batch in loader:
        mri = batch["mri"].to(device)
        deltas = batch["deltas"].to(device)
        states = encode_mri_stages(model, mri, (0, 1, 2, 3))
        observed_states.append(states.detach().float().cpu())
        transition_distances.append(
            torch.linalg.vector_norm(
                states[:, 1:] - states[:, :-1], dim=(-2, -1)
            ).detach().float().cpu()
        )
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
        latent_disagreement = member_states.var(
            dim=0, unbiased=False
        ).mean(dim=(-1, -2))
        mean_risk = risks.mean(dim=0)
        probabilities = torch.sigmoid(logits)
        mean_probability = probabilities.mean(dim=0)
        probability_disagreement = probabilities.std(dim=0, unbiased=False)
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
                        "latent_disagreement": float(
                            latent_disagreement[sample, horizon].cpu()
                        ),
                        "survival_probability_std": float(
                            probability_disagreement[sample, horizon].cpu()
                        ),
                        "risk": float(mean_risk[sample, horizon].cpu()),
                        "survival365": float(mean_probability[sample, horizon].cpu()),
                        "survival_time": float(batch["survival_time"][sample, horizon]),
                        "event": int(batch["event"][sample, horizon]),
                    }
                )
    observed = torch.cat(observed_states).reshape(
        -1, *observed_states[0].shape[2:]
    )
    representation = {
        "observed_latent_variance": float(
            observed.var(dim=0, unbiased=False).mean()
        ),
        "observed_transition_l2_mean": float(
            torch.cat(transition_distances).mean()
        ),
    }
    return rows, representation


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
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in (1, 2, 3):
        selected = [row for row in rows if row["horizon"] == horizon]
        disagreement = np.asarray(
            [row["latent_disagreement"] for row in selected]
        )
        error = np.asarray([row["latent_mse"] for row in selected])
        result[f"H{horizon}"] = {
            "latent_disagreement_mean": float(disagreement.mean()),
            "latent_disagreement_error_pearson": float(
                np.corrcoef(disagreement, error)[0, 1]
            ),
            "survival_probability_disagreement_mean": float(
                np.mean([row["survival_probability_std"] for row in selected])
            ),
        }
    return result


def evaluate_one(
    config_path: str | Path,
    variant: str,
    seed: int,
    device_name: str,
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
    initial_encoder = encoder_trainable_state(model)
    run_dir = run_directory(config, variant, seed)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    loaders, datasets = build_loaders(config, seed)
    rows, representation = recursive_predictions(model, loaders["test"], device)
    representation["encoder_trainable_parameter_rms_delta"] = encoder_rms_delta(
        model, initial_encoder
    )
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
        "training_scheme": checkpoint["training_scheme"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "recursive": horizons,
        "representation_sanity": representation,
        "error_accumulation": (
            horizons["H3"]["latent_mse"] - horizons["H1"]["latent_mse"]
        ),
    }
    if int(variant_config["ensemble_size"]) > 1:
        metrics["uncertainty"] = uncertainty_metrics(rows)
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
) -> list[dict[str, Any]]:
    runs = []
    for seed in config["training"]["seeds"]:
        path = run_directory(config, variant, int(seed)) / "metrics.json"
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
    for metric in (
        "observed_latent_variance",
        "observed_transition_l2_mean",
        "encoder_trainable_parameter_rms_delta",
    ):
        summary[metric] = statistics(
            [float(run["representation_sanity"][metric]) for run in runs]
        )
    if "uncertainty" in runs[0]:
        for horizon in (1, 2, 3):
            key = f"H{horizon}"
            for metric in (
                "latent_disagreement_mean",
                "latent_disagreement_error_pearson",
                "survival_probability_disagreement_mean",
            ):
                summary[f"{key}_{metric}"] = statistics(
                    [
                        float(run["uncertainty"][key][metric])
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
    output = {"primary": primary}
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    def value(block: Mapping[str, Any], key: str) -> str:
        metric = block[key]
        return f"{metric['mean']:.4f} ± {metric['std']:.4f}"

    lines = [
        "# Pure Stage-wise RRT v3",
        "",
        "## Table 1 — Recursive Latent Dynamics",
        "",
        "| Group | H1 MSE ↓ | H2 MSE ↓ | H3 MSE ↓ | H1 cosine ↑ | H2 cosine ↑ | H3 cosine ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in "ABCD":
        if variant not in primary:
            continue
        block = primary[variant]
        lines.append(
            f"| {variant} | {value(block, 'H1_latent_mse')} | "
            f"{value(block, 'H2_latent_mse')} | {value(block, 'H3_latent_mse')} | "
            f"{value(block, 'H1_cosine_similarity')} | "
            f"{value(block, 'H2_cosine_similarity')} | "
            f"{value(block, 'H3_cosine_similarity')} |"
        )
    lines.extend(
        [
            "",
            "## Table 2 — H3 End-to-End Prognosis",
            "",
            "| Group | H3 C-index ↑ | H3 Brier@365 ↓ |",
            "|---|---:|---:|",
        ]
    )
    for variant in "ABCD":
        if variant not in primary:
            continue
        block = primary[variant]
        lines.append(
            f"| {variant} | {value(block, 'H3_c_index')} | "
            f"{value(block, 'H3_brier365')} |"
        )
    lines.extend(
        [
            "",
            "## Table 3 — Ensemble Reliability (Secondary)",
            "",
            "| Group | H3 latent disagreement | H3 disagreement-error Pearson | H3 survival-probability disagreement |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in ("C", "D"):
        if variant not in primary:
            continue
        block = primary[variant]
        lines.append(
            f"| {variant} | {value(block, 'H3_latent_disagreement_mean')} | "
            f"{value(block, 'H3_latent_disagreement_error_pearson')} | "
            f"{value(block, 'H3_survival_probability_disagreement_mean')} |"
        )
    lines.extend(
        [
            "",
            "## Representation Sanity",
            "",
            "| Group | Observed latent variance | Mean adjacent-state L2 | Encoder trainable-parameter RMS Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in "ABCD":
        if variant not in primary:
            continue
        block = primary[variant]
        lines.append(
            f"| {variant} | {value(block, 'observed_latent_variance')} | "
            f"{value(block, 'observed_transition_l2_mean')} | "
            f"{value(block, 'encoder_trainable_parameter_rms_delta')} |"
        )
    lines.extend(
        [
            "",
            "## Appendix — H1/H2 Prognosis",
            "",
            "| Group | H1 C-index ↑ | H2 C-index ↑ | H1 Brier@365 ↓ | H2 Brier@365 ↓ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in "ABCD":
        if variant not in primary:
            continue
        block = primary[variant]
        lines.append(
            f"| {variant} | {value(block, 'H1_c_index')} | "
            f"{value(block, 'H2_c_index')} | "
            f"{value(block, 'H1_brier365')} | "
            f"{value(block, 'H2_brier365')} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", default="configs/pure_rrt_v3.yaml")
    run.add_argument("--variant", required=True, choices=list("ABCD"))
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--device", default="cuda:0")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--config", default="configs/pure_rrt_v3.yaml")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        metrics = evaluate_one(args.config, args.variant, args.seed, args.device)
    else:
        metrics = aggregate(args.config)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
