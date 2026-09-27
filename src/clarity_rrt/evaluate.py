"""Direct, recursive, and uncertainty evaluation for trained variants."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import _valid_endpoint, load_split, parse_tp_num, timepoint_id
from .model import ClarityDynamicsEnsemble
from .train import (
    build_datasets,
    build_loaders,
    configure_upstream,
    encode_pair,
    load_config,
    seed_everything,
    upstream_args,
)


def _training_reference(dataset) -> tuple[np.ndarray, np.ndarray]:
    times, events = [], []
    for base_index in dataset._base_indices:
        item = dataset.base.index[base_index]
        times.append(float(item["survival_time"]))
        events.append(bool(item["event_indicator"]))
    return np.asarray(times, dtype=float), np.asarray(events, dtype=bool)


def ipcw_brier(
    train_time: np.ndarray,
    train_event: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    evaluation_time: float = 365.0,
) -> tuple[float | None, str | None]:
    if not rows:
        return None, "empty evaluation set"
    try:
        from sksurv.metrics import brier_score
        from sksurv.util import Surv
    except ImportError:
        return None, "scikit-survival is not installed"
    test_time = np.asarray([row["time"] for row in rows], dtype=float)
    test_event = np.asarray([bool(row["event"]) for row in rows], dtype=bool)
    estimate = np.asarray([row["survival365"] for row in rows], dtype=float)[:, None]
    try:
        train_y = Surv.from_arrays(train_event, train_time)
        test_y = Surv.from_arrays(test_event, test_time)
        _, score = brier_score(train_y, test_y, estimate, np.asarray([evaluation_time]))
        return float(score[0]), None
    except ValueError as error:
        return None, str(error)


def metric_block(
    rows: Sequence[Mapping[str, Any]],
    train_reference: tuple[np.ndarray, np.ndarray],
    concordance_index,
    brier_time: float,
) -> dict[str, Any]:
    patients = {str(row["patient"]) for row in rows}
    events = int(sum(bool(row["event"]) for row in rows))
    if rows:
        try:
            c_index = float(
                concordance_index(
                    np.asarray([row["risk"] for row in rows]),
                    np.asarray([row["time"] for row in rows]),
                    np.asarray([row["event"] for row in rows]),
                )
            )
        except (ValueError, ZeroDivisionError):
            c_index = None
    else:
        c_index = None
    brier, reason = ipcw_brier(*train_reference, rows, evaluation_time=brier_time)
    return {
        "patients": len(patients),
        "samples": len(rows),
        "events": events,
        "c_index": c_index,
        "brier365": brier,
        "brier_na_reason": reason,
    }


@torch.no_grad()
def direct_predictions(model, loader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        pre, target = encode_pair(model, batch, device)
        delta = batch["time_delta"].to(device)
        outputs = model.forward_pair(pre, batch["drugs_text"], delta, batch["clinical_text"])
        member_prob = torch.sigmoid(outputs["member_logit"])
        mean_prob = member_prob.mean(dim=0)
        mean_risk = outputs["member_risk"].mean(dim=0)
        latent_mae = (outputs["member_post"] - target[None]).abs().mean(dim=(0, 2, 3))
        latent_mse = ((outputs["member_post"] - target[None]) ** 2).mean(dim=(0, 2, 3))
        for index, meta in enumerate(batch["meta"]):
            probabilities = member_prob[:, index].cpu().tolist()
            rows.append(
                {
                    "patient": meta["pid"],
                    "start": meta["pre_tp"],
                    "end": meta["post_tp"],
                    "mode": "direct",
                    "horizon": 0,
                    "time": float(batch["survival_time"][index]),
                    "event": int(batch["event_indicator"][index]),
                    "risk": float(mean_risk[index].cpu()),
                    "survival365": float(mean_prob[index].cpu()),
                    "uncertainty": float(member_prob[:, index].std(unbiased=False).cpu()),
                    "member_probabilities": json.dumps(probabilities),
                    "latent_mae": float(latent_mae[index].cpu()),
                    "latent_mse": float(latent_mse[index].cpu()),
                }
            )
    return rows


def _recursive_windows(base, patient_ids: set[str], horizon: int) -> list[dict[str, Any]]:
    records = []
    for patient_id in sorted(patient_ids):
        patient = base.patients["patients"].get(patient_id, {})
        timeline = sorted(
            patient.get("timeline", []), key=lambda tp: parse_tp_num(tp.get("tp_id", ""))
        )
        for start in range(0, len(timeline) - horizon):
            points = timeline[start : start + horizon + 1]
            days = [float(point.get("mri_day", 0)) for point in points]
            ids = [timepoint_id(patient_id, point["tp_id"]) for point in points]
            if any(right <= left for left, right in zip(days, days[1:])):
                continue
            if any(not base.mri_loader.has(mri_id) for mri_id in ids):
                continue
            if not _valid_endpoint(points[-1]):
                continue
            step_text = []
            for pre, post in zip(points, points[1:]):
                step_text.append(
                    base._pack_drugs_json(
                        {
                            "pre_tp": pre["tp_id"],
                            "post_tp": post["tp_id"],
                            "pre_actions": pre.get("actions", {}),
                            "post_actions": post.get("actions", {}),
                        }
                    )
                )
            endpoint_text = base._pack_drugs_json(
                {
                    "pre_tp": points[0]["tp_id"],
                    "post_tp": points[-1]["tp_id"],
                    "pre_actions": points[0].get("actions", {}),
                    "post_actions": points[-1].get("actions", {}),
                }
            )
            records.append(
                {
                    "patient": patient_id,
                    "points": points,
                    "ids": ids,
                    "days": days,
                    "step_text": step_text,
                    "endpoint_text": endpoint_text,
                    "clinical_text": base._format_clinical_text(patient.get("context_static", {})),
                }
            )
            break  # pre-registered rule: first legal window per patient and H
    return records


@torch.no_grad()
def recursive_predictions(
    model,
    base,
    patient_ids: set[str],
    horizons: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        windows = _recursive_windows(base, patient_ids, int(horizon))
        for offset in range(0, len(windows), batch_size):
            chunk = windows[offset : offset + batch_size]
            mri = torch.stack(
                [torch.stack([base.mri_loader.load(mri_id) for mri_id in row["ids"]]) for row in chunk]
            ).to(device)
            initial = model.mri_encoder(mri[:, 0])
            target = model.mri_encoder(mri[:, -1]).detach()
            delta = torch.tensor(
                [[right - left for left, right in zip(row["days"], row["days"][1:])] for row in chunk],
                dtype=torch.float32,
                device=device,
            )
            step_text = [
                [row["step_text"][step] for row in chunk] for step in range(int(horizon))
            ]
            clinical = [row["clinical_text"] for row in chunk]
            step_conditions = model.encode_step_conditions(step_text, clinical)
            states = model.rollout_encoded(initial, step_conditions, delta)
            terminal = states[:, :, -1]
            endpoint_condition = model.encode_condition(
                [row["endpoint_text"] for row in chunk], clinical
            )
            member_risk, member_logit = model.endpoint_predictions(
                initial, terminal, endpoint_condition
            )
            member_prob = torch.sigmoid(member_logit)
            mean_risk = member_risk.mean(dim=0)
            mean_prob = member_prob.mean(dim=0)
            latent_mae = (terminal - target[None]).abs().mean(dim=(0, 2, 3))
            latent_mse = ((terminal - target[None]) ** 2).mean(dim=(0, 2, 3))
            for index, record in enumerate(chunk):
                endpoint = record["points"][-1]
                survival = endpoint["survival"]
                probabilities = member_prob[:, index].cpu().tolist()
                rows.append(
                    {
                        "patient": record["patient"],
                        "start": record["points"][0]["tp_id"],
                        "end": endpoint["tp_id"],
                        "mode": "recursive",
                        "horizon": int(horizon),
                        "time": float(survival["survival_from_tp_days"]),
                        "event": int(survival["event_indicator"]),
                        "risk": float(mean_risk[index].cpu()),
                        "survival365": float(mean_prob[index].cpu()),
                        "uncertainty": float(member_prob[:, index].std(unbiased=False).cpu()),
                        "member_probabilities": json.dumps(probabilities),
                        "latent_mae": float(latent_mae[index].cpu()),
                        "latent_mse": float(latent_mse[index].cpu()),
                    }
                )
    return rows


def uncertainty_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    train_reference: tuple[np.ndarray, np.ndarray],
    brier_time: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rows = list(rows)
    keep = max(1, int(round(0.8 * len(rows)))) if rows else 0
    selected = sorted(rows, key=lambda row: (row["uncertainty"], row["patient"]))[:keep]
    selected_brier, selected_reason = ipcw_brier(
        *train_reference, selected, evaluation_time=brier_time
    )
    rng = np.random.default_rng(seed)
    random_scores = []
    for _ in range(repeats):
        indices = rng.choice(len(rows), size=keep, replace=False) if keep else []
        sample = [rows[int(index)] for index in indices]
        score, _ = ipcw_brier(*train_reference, sample, evaluation_time=brier_time)
        if score is not None:
            random_scores.append(score)
    return {
        "total_patients": len({row["patient"] for row in rows}),
        "retained": keep,
        "actual_coverage": keep / len(rows) if rows else None,
        "retained_events": int(sum(row["event"] for row in selected)),
        "low_uncertainty_brier365": selected_brier,
        "low_uncertainty_brier_na_reason": selected_reason,
        "random_same_size_brier365_mean": float(np.mean(random_scores)) if random_scores else None,
        "random_repeats_completed": len(random_scores),
    }


def evaluate_one(
    config_path: str | Path,
    variant: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(seed)
    device = torch.device(device_name)
    upstream_train, _, _, _, concordance = configure_upstream(config["upstream_root"])
    official = upstream_train.build_model(upstream_args(config, seed), device)
    model = ClarityDynamicsEnsemble(
        official,
        int(config["variants"][variant]["ensemble_size"]),
        seed,
    ).to(device)
    run_dir = Path(config["run_root"]) / f"{variant}_seed{seed}"
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["upstream_commit"] != config["upstream_commit"]:
        raise ValueError("checkpoint upstream commit does not match experiment config")
    model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    loaders, datasets, split = build_loaders(config, seed)
    direct = direct_predictions(model, loaders["test"], device)
    base, _, _ = build_datasets(config)
    test_ids = set(split["splits"]["test"])
    recursive = recursive_predictions(
        model,
        base,
        test_ids,
        config["evaluation"]["recursive_horizons"],
        int(config["training"]["val_batch_size"]),
        device,
    )
    rows = direct + recursive
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    train_ref = _training_reference(datasets["train"])
    brier_time = float(config["evaluation"]["brier_time_days"])
    history_path = run_dir / "history.csv"
    resource_rows = []
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            resource_rows = list(csv.DictReader(handle))
    metrics: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "upstream_commit": config["upstream_commit"],
        "split_sha256": split["split_sha256"],
        "resources": {
            "training_seconds": sum(float(row["seconds"]) for row in resource_rows),
            "peak_gpu_gb": max(
                (float(row.get("peak_gpu_gb", 0.0)) for row in resource_rows),
                default=None,
            ),
        },
        "direct": metric_block(direct, train_ref, concordance, brier_time),
        "recursive": {},
    }
    for horizon in config["evaluation"]["recursive_horizons"]:
        horizon_rows = [row for row in recursive if row["horizon"] == int(horizon)]
        metrics["recursive"][f"H{horizon}"] = metric_block(
            horizon_rows, train_ref, concordance, brier_time
        )
    if model.ensemble_size > 1:
        h2 = [row for row in recursive if row["horizon"] == 2]
        metrics["uncertainty_h2"] = uncertainty_diagnostic(
            h2,
            train_ref,
            brier_time,
            int(config["evaluation"]["uncertainty_random_repeats"]),
            seed,
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def aggregate_results(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    collected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in config["variants"]:
        for seed in config["training"]["seeds"]:
            path = Path(config["run_root"]) / f"{variant}_seed{seed}" / "metrics.json"
            if path.is_file():
                collected[variant].append(json.loads(path.read_text(encoding="utf-8")))

    def stats(values: Sequence[float | None]) -> dict[str, Any]:
        valid = np.asarray([value for value in values if value is not None], dtype=float)
        return {
            "runs": int(valid.size),
            "mean": float(valid.mean()) if valid.size else None,
            "std": float(valid.std(ddof=1)) if valid.size > 1 else (0.0 if valid.size == 1 else None),
            "values": list(values),
        }

    summary: dict[str, Any] = {}
    for variant, runs in collected.items():
        summary[variant] = {
            "direct_c_index": stats([run["direct"]["c_index"] for run in runs]),
            "direct_brier365": stats([run["direct"]["brier365"] for run in runs]),
        }
        for horizon in config["evaluation"]["recursive_horizons"]:
            summary[variant][f"H{horizon}_c_index"] = stats(
                [run["recursive"][f"H{horizon}"]["c_index"] for run in runs]
            )
        summary[variant]["H2_brier365"] = stats(
            [run["recursive"]["H2"]["brier365"] for run in runs]
        )
        summary[variant]["training_seconds"] = stats(
            [run.get("resources", {}).get("training_seconds") for run in runs]
        )
        summary[variant]["peak_gpu_gb"] = stats(
            [run.get("resources", {}).get("peak_gpu_gb") for run in runs]
        )
        if runs and "uncertainty_h2" in runs[0]:
            summary[variant]["uncertainty_low80_brier365"] = stats(
                [run["uncertainty_h2"]["low_uncertainty_brier365"] for run in runs]
            )
            summary[variant]["uncertainty_random80_brier365"] = stats(
                [run["uncertainty_h2"]["random_same_size_brier365_mean"] for run in runs]
            )
    output = {"expected_runs_per_variant": len(config["training"]["seeds"]), "variants": summary}
    root = Path(config["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    def formatted(block: Mapping[str, Any], digits: int = 4) -> str:
        if not block or block.get("mean") is None:
            return "NA"
        return f"{block['mean']:.{digits}f} ± {block['std']:.{digits}f}"

    lines = [
        "# CLARITY RRT/Ensemble results",
        "",
        "## Direct official-pair task",
        "",
        "| Method | C-index ↑ | Brier@365 ↓ | Train time (s) | Peak GPU (GB) | Runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in "ABCD":
        block = summary.get(variant, {})
        lines.append(
            f"| {variant} | {formatted(block.get('direct_c_index', {}))} | "
            f"{formatted(block.get('direct_brier365', {}))} | "
            f"{formatted(block.get('training_seconds', {}), 1)} | "
            f"{formatted(block.get('peak_gpu_gb', {}), 2)} | "
            f"{block.get('direct_c_index', {}).get('runs', 0)} |"
        )
    lines.extend([
        "",
        "## Common recursive deployment",
        "",
        "| Method | C@H1 ↑ | C@H2 ↑ | C@H3 ↑ | Brier@H2 ↓ |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant in "ABCD":
        block = summary.get(variant, {})
        lines.append(
            f"| {variant} | {formatted(block.get('H1_c_index', {}))} | "
            f"{formatted(block.get('H2_c_index', {}))} | "
            f"{formatted(block.get('H3_c_index', {}))} | "
            f"{formatted(block.get('H2_brier365', {}))} |"
        )
    lines.extend([
        "",
        "## H2 uncertainty ranking (ensemble only)",
        "",
        "| Method | Low-uncertainty 80% Brier ↓ | Random same-size Brier ↓ |",
        "|---|---:|---:|",
    ])
    for variant in ("C", "D"):
        block = summary.get(variant, {})
        lines.append(
            f"| {variant} | {formatted(block.get('uncertainty_low80_brier365', {}))} | "
            f"{formatted(block.get('uncertainty_random80_brier365', {}))} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run")
    one.add_argument("--config", default="configs/experiment.yaml")
    one.add_argument("--variant", required=True, choices=list("ABCD"))
    one.add_argument("--seed", required=True, type=int)
    one.add_argument("--checkpoint")
    one.add_argument("--device", default="cuda:0")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--config", default="configs/experiment.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        print(json.dumps(evaluate_one(args.config, args.variant, args.seed, args.checkpoint, args.device), indent=2))
    else:
        print(json.dumps(aggregate_results(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
