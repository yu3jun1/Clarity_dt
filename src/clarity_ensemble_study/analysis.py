"""The prescribed ensemble reliability metrics and compact three-seed reports."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


SEEDS = (42, 43, 44)
COVERAGES = (1.0, 0.8, 0.6, 0.4)


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """A constant signal has no defined Pearson correlation."""
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def error_correlations(uncertainty: np.ndarray, error_l2: np.ndarray) -> dict[str, float | None]:
    return {
        "pearson": correlation(uncertainty, error_l2),
        "spearman": correlation(rankdata(uncertainty), rankdata(error_l2)),
    }


def analyze(
    rows: Sequence[Mapping[str, Any]], member_states: np.ndarray,
) -> dict[str, Any]:
    """Analyze window-level H1/H2/H3 predictions, without pooling horizons."""
    member_count = member_states.shape[0]
    result: dict[str, Any] = {}
    for horizon in (1, 2, 3):
        selected = [row for row in rows if row["horizon"] == horizon]
        mse = np.asarray([row["latent_mse"] for row in selected], dtype=float)
        error = np.asarray([row["error_l2"] for row in selected], dtype=float)
        latent_uncertainty = np.asarray(
            [row["latent_uncertainty"] for row in selected], dtype=float,
        )
        survival_uncertainty = np.asarray(
            [row["survival_uncertainty"] for row in selected], dtype=float,
        )
        member_mse = np.asarray(
            [row["member_latent_mse"] for row in selected], dtype=float,
        )
        member_cosine = np.asarray(
            [row["member_cosine_similarity"] for row in selected], dtype=float,
        )
        order = np.argsort(latent_uncertainty, kind="stable")
        coverage = []
        if member_count > 1:
            for fraction in COVERAGES:
                count = int(np.ceil(fraction * len(selected)))
                coverage.append({
                    "coverage": fraction,
                    "samples": count,
                    "latent_mse": float(mse[order[:count]].mean()),
                })
        predictions = member_states[:, :, horizon - 1].reshape(member_count, -1)
        result[f"H{horizon}"] = {
            "latent_uncertainty_mean": float(latent_uncertainty.mean()),
            "survival_uncertainty_mean": float(survival_uncertainty.mean()),
            "latent_error_correlation": error_correlations(latent_uncertainty, error),
            "survival_error_correlation": error_correlations(survival_uncertainty, error),
            "risk_coverage": coverage,
            "members": [
                {"member_index": member,
                 "latent_mse": float(member_mse[:, member].mean()),
                 "cosine_similarity": float(member_cosine[:, member].mean())}
                for member in range(member_count)
            ],
            "member_pairs": [
                {"member_i": first, "member_j": second,
                 "prediction_pearson": correlation(predictions[first], predictions[second])}
                for first, second in combinations(range(member_count), 2)
            ],
            "averaging_gain_mse": float(member_mse.mean() - mse.mean()),
        }
    return result


def statistics(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)),
            "values": list(values)}


def mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def summarize(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average per-seed reliability measures; never pool rows across seeds."""
    performance = {}
    reliability = {}
    for horizon in ("H1", "H2", "H3"):
        performance[horizon] = {
            metric: statistics([run["recursive"][horizon][metric] for run in runs])
            for metric in ("latent_mse", "cosine_similarity")
        }
        blocks = [run["reliability"][horizon] for run in runs]
        reliability[horizon] = {
            key: float(np.mean([block[key] for block in blocks]))
            for key in ("latent_uncertainty_mean", "survival_uncertainty_mean", "averaging_gain_mse")
        }
        for key in ("latent_error_correlation", "survival_error_correlation"):
            reliability[horizon][key] = {
                metric: mean_defined([block[key][metric] for block in blocks])
                for metric in ("pearson", "spearman")
            }
        reliability[horizon]["risk_coverage"] = [
            {"coverage": point["coverage"], "samples": point["samples"],
             "latent_mse": float(np.mean([
                 block["risk_coverage"][index]["latent_mse"] for block in blocks
             ]))}
            for index, point in enumerate(blocks[0]["risk_coverage"])
        ]
        reliability[horizon]["members"] = [
            {"member_index": member["member_index"], **{
                metric: float(np.mean([block["members"][index][metric] for block in blocks]))
                for metric in ("latent_mse", "cosine_similarity")
            }}
            for index, member in enumerate(blocks[0]["members"])
        ]
        reliability[horizon]["member_pairs"] = [
            {"member_i": pair["member_i"], "member_j": pair["member_j"],
             "prediction_pearson": mean_defined([
                 block["member_pairs"][index]["prediction_pearson"] for block in blocks
             ])}
            for index, pair in enumerate(blocks[0]["member_pairs"])
        ]
    return {"ensemble_size": runs[0]["ensemble_size"],
            "performance": performance, "reliability": reliability,
            "h3_stability_std": performance["H3"]["latent_mse"]["std"]}


def plot_results(root: Path, destination: Path, models: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for seed in SEEDS:
        with (root / "runs" / f"M3_seed{seed}" / "recursive_predictions.csv").open(
            newline="", encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for horizon, axis in enumerate(axes, start=1):
            selected = [row for row in rows if int(row["horizon"]) == horizon]
            axis.scatter([float(row["latent_uncertainty"]) for row in selected],
                         [float(row["error_l2"]) for row in selected], s=20, alpha=0.75)
            axis.set(title=f"M3 seed {seed}: H{horizon}", xlabel="Latent uncertainty Uz",
                     ylabel="Latent error (L2)")
        figure.tight_layout()
        figure.savefig(destination / f"uncertainty_error_M3_seed{seed}.png", dpi=160)
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for horizon, axis in enumerate(axes, start=1):
        for name, model in models.items():
            points = model["reliability"][f"H{horizon}"]["risk_coverage"]
            if points:
                ordered = sorted(points, key=lambda point: point["coverage"])
                axis.plot([point["coverage"] for point in ordered],
                          [point["latent_mse"] for point in ordered], marker="o", label=name)
        axis.set(title=f"H{horizon}", xlabel="Coverage", ylabel="Latent MSE",
                 xticks=(0.4, 0.6, 0.8, 1.0))
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "risk_coverage.png", dpi=160)
    plt.close(figure)


def write_report(root: str | Path, stage: str, sizes: Sequence[int]) -> dict[str, Any]:
    root = Path(root)
    destination = root / "reports" / stage
    destination.mkdir(parents=True, exist_ok=True)
    models = {}
    for size in sizes:
        runs = [json.loads((root / "runs" / f"M{size}_seed{seed}" / "metrics.json").read_text(
            encoding="utf-8",
        )) for seed in SEEDS]
        models[f"M{size}"] = summarize(runs)
    summary = {
        "stage": stage, "seeds": list(SEEDS), "models": models,
        "evaluation_unit": "test trajectory-window at each horizon",
        "correlation_aggregation": "mean of within-seed correlations; no cross-seed pooling",
        "survival_uncertainty_interpretation": "probability disagreement vs latent error, not survival calibration",
        "uncertainty_definitions": {
            "latent": "mean_m sum_latent((z_m - mean_m(z)) ** 2)",
            "survival": "population variance of member sigmoid(logit)",
            "error": "L2 norm of ensemble-mean latent error",
        },
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )

    def number(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    def mean_std(block: Mapping[str, Any]) -> str:
        return f"{block['mean']:.4f} ± {block['std']:.4f}"

    lines = [
        f"# Ensemble Reliability and Size Analysis: {stage}", "",
        "Seeds: 42, 43, 44. M1/M3 re-evaluate existing A/C checkpoints; M2/M5 are newly trained.", "",
        "Errors are measured per test trajectory-window and horizon. Values are seed means;",
        "± denotes sample standard deviation across the three seeds. H3 MSE standard deviation is the stability measure.", "",
        "## Prediction performance", "",
        "| Model | H1 MSE | H2 MSE | H3 MSE | H1 Cosine | H2 Cosine | H3 Cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in models.items():
        values = [mean_std(model["performance"][horizon][metric])
                  for metric in ("latent_mse", "cosine_similarity")
                  for horizon in ("H1", "H2", "H3")]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.extend([
        "", "## Uncertainty–latent-error correlation", "",
        "Correlations are computed within each seed against L2 error, then averaged.",
        "Survival disagreement is an auxiliary latent-error signal, not a survival correctness/calibration metric.", "",
        "| Model | Horizon | Uz Pearson | Uz Spearman | Us Pearson | Us Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for name, model in models.items():
        for horizon, block in model["reliability"].items():
            values = [number(block[key][metric])
                      for key in ("latent_error_correlation", "survival_error_correlation")
                      for metric in ("pearson", "spearman")]
            lines.append(f"| {name} | {horizon} | " + " | ".join(values) + " |")
    lines.extend([
        "", "M1 has constant zero disagreement: correlations are undefined and it has no uncertainty-based coverage curve.",
        "", "## Four-point risk–coverage", "",
        "Lowest-Uz windows are retained, using ceil(coverage × N) and stable ordering for ties.", "",
        "| Model | Coverage | Retained windows / seed | H1 MSE | H2 MSE | H3 MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, model in models.items():
        for index, point in enumerate(model["reliability"]["H1"]["risk_coverage"]):
            values = [number(model["reliability"][horizon]["risk_coverage"][index]["latent_mse"])
                      for horizon in ("H1", "H2", "H3")]
            lines.append(f"| {name} | {point['coverage']:.0%} | {point['samples']} | "
                         + " | ".join(values) + " |")
    lines.extend([
        "", "Member MSE/Cosine, raw-prediction pairwise Pearson correlations, and averaging gains are in [summary.json](summary.json).", "",
        "Latent-space comparisons retain the original jointly trained encoder setup; different model encoders can differ in scale.", "",
        "## Figures", "",
        "![Risk–coverage](risk_coverage.png)", "",
    ])
    for seed in SEEDS:
        lines.extend([f"![M3 seed {seed}: uncertainty vs L2 error](uncertainty_error_M3_seed{seed}.png)", ""])
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    plot_results(root, destination, models)
    return summary
