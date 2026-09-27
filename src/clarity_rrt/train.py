"""Training entry point for the four CLARITY comparison variants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import FixedSplitPairDataset, load_split
from .model import ClarityDynamicsEnsemble


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    return config


def configure_upstream(upstream_root: str | Path):
    root = Path(upstream_root).resolve()
    predictor_root = root / "Predictor"
    for path in (str(predictor_root), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import train as upstream_train  # type: ignore
    from dataset.dataset_glioma_all_pairs_text import (  # type: ignore
        Config,
        GliomaAllPairsTextDataset,
    )
    from models.full_model import extract_drug_category  # type: ignore
    from utils.metrics import concordance_index  # type: ignore

    return upstream_train, Config, GliomaAllPairsTextDataset, extract_drug_category, concordance_index


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def upstream_args(config: Mapping[str, Any], seed: int) -> Namespace:
    data = config["data"]
    model = config["model"]
    training = config["training"]
    return Namespace(
        text_encoder_name=model["text_encoder_name"],
        freeze_text_encoder=bool(model.get("freeze_text_encoder", False)),
        latent_dim=int(model["latent_dim"]),
        num_modalities=int(model["num_modalities"]),
        lambda_l1=float(training["lambda_l1"]),
        lambda_cox=float(training["lambda_cox"]),
        lambda_bce=float(training["lambda_bce"]),
        dropout=float(model["dropout"]),
        brainiac_ckpt=model.get("brainiac_ckpt"),
        brainiac_tokens=int(model.get("brainiac_tokens", 8)),
        brainiac_lora_r=int(model.get("brainiac_lora_r", 8)),
        num_epochs=int(training["epochs"]),
        lr=float(training["lr"]),
        text_lr=float(training["text_lr"]),
        text_proj_lr=float(training["text_proj_lr"]),
        vision_lr=float(training.get("vision_lr", 5e-5)),
        survival_wd=float(training["survival_weight_decay"]),
        timeline_json=data["timeline_json"],
        mri_data_dir=data["mri_data_dir"],
        features_csv=data.get("features_csv"),
        take_dims=int(data.get("take_dims", 767)),
        seed=seed,
    )


def build_datasets(config: Mapping[str, Any]):
    _, upstream_config, upstream_dataset, _, _ = configure_upstream(config["upstream_root"])
    data_config = upstream_config(
        timeline_json=config["data"]["timeline_json"],
        features_csv=config["data"].get("features_csv"),
        take_dims=int(config["data"].get("take_dims", 767)),
        mri_data_dir=config["data"]["mri_data_dir"],
    )
    base = upstream_dataset(data_config, include_between=False)
    split = load_split(config["data"]["split_file"])
    horizons = config["rrt"]["horizons"]
    datasets = {
        name: FixedSplitPairDataset(base, split["splits"][split_name], horizons=horizons)
        for name, split_name in (("train", "train"), ("validation", "validation"), ("test", "test"))
    }
    return base, datasets, split


def build_loaders(config: Mapping[str, Any], seed: int):
    _, datasets, split = build_datasets(config)
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)

    def init_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    max_chains = int(config["rrt"]["max_chains_per_pair_batch"])
    common = {
        "num_workers": int(training["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": init_worker,
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(training["pair_batch_size"]),
            shuffle=True,
            generator=generator,
            collate_fn=datasets["train"].collate_fn(max_chains=max_chains),
            **common,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=int(training["val_batch_size"]),
            shuffle=False,
            collate_fn=datasets["validation"].collate_fn(max_chains=0),
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(training["val_batch_size"]),
            shuffle=False,
            collate_fn=datasets["test"].collate_fn(max_chains=0),
            **common,
        ),
    }
    return loaders, datasets, split


def _add_group(groups: list[dict[str, Any]], params, lr: float, weight_decay: float) -> None:
    selected = [parameter for parameter in params if parameter.requires_grad]
    if selected:
        groups.append({"params": selected, "lr": lr, "weight_decay": weight_decay})


def build_optimizer(
    config: Mapping[str, Any], model: ClarityDynamicsEnsemble
) -> tuple[AdamW, CosineAnnealingLR]:
    training = config["training"]
    groups: list[dict[str, Any]] = []
    for predictor in model.predictors:
        _add_group(groups, predictor.time_proj.parameters(), float(training["lr"]), 5e-4)
        _add_group(
            groups,
            (parameter for name, parameter in predictor.named_parameters() if "time_proj" not in name),
            float(training["lr"]) * 0.1,
            1e-5,
        )
    _add_group(
        groups,
        model.survival_module.parameters(),
        float(training["lr"]),
        float(training["survival_weight_decay"]),
    )
    text_lora, text_projection, text_other = [], [], []
    for name, parameter in model.shared_text_encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            text_lora.append(parameter)
        elif "final_proj" in name:
            text_projection.append(parameter)
        else:
            text_other.append(parameter)
    _add_group(groups, text_lora, float(training["text_lr"]), 0.0)
    _add_group(groups, text_projection, float(training["text_proj_lr"]), 1e-4)
    _add_group(groups, text_other, float(training["text_lr"]), 0.0)
    if model.mri_encoder is not None:
        _add_group(
            groups,
            model.mri_encoder.parameters(),
            float(training.get("vision_lr", 5e-5)),
            0.0,
        )
    if not groups:
        raise ValueError("no trainable parameters were registered")
    optimizer = AdamW(groups)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"]), eta_min=1e-6
    )
    return optimizer, scheduler


def encode_pair(
    model: ClarityDynamicsEnsemble, batch: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.mri_encoder is not None and "pre_mri" in batch:
        pre_mri = batch["pre_mri"].to(device, non_blocking=True)
        post_mri = batch["post_mri"].to(device, non_blocking=True)
        pre_latent = model.mri_encoder(pre_mri)
        post_latent = model.mri_encoder(post_mri).detach()
        return pre_latent, post_latent
    return (
        batch["pre_latent"].to(device, non_blocking=True),
        batch["post_latent"].to(device, non_blocking=True),
    )


def recursive_latent_loss(
    model: ClarityDynamicsEnsemble,
    chain_groups: Sequence[Mapping[str, Any]],
    device: torch.device,
    supervise_from_step: int = 2,
) -> tuple[torch.Tensor, int]:
    if not chain_groups:
        parameter = next(model.parameters())
        return parameter.new_zeros(()), 0
    weighted_losses = []
    chain_counts = []
    for group in chain_groups:
        mri = group["mri"].to(device, non_blocking=True)
        delta = group["delta"].to(device, non_blocking=True)
        chain_count, horizon = delta.shape
        initial = model.mri_encoder(mri[:, 0])
        targets = torch.stack(
            [model.mri_encoder(mri[:, step]).detach() for step in range(1, horizon + 1)],
            dim=1,
        )
        conditions = model.encode_step_conditions(
            group["drugs_text_steps"], group["clinical_text"]
        )
        member_states = model.rollout_encoded(initial, conditions, delta)
        first_index = supervise_from_step - 1
        if first_index >= horizon:
            continue
        loss = F.l1_loss(
            member_states[:, :, first_index:],
            targets[None, :, first_index:].expand(model.ensemble_size, -1, -1, -1, -1),
        )
        weighted_losses.append(loss * chain_count)
        chain_counts.append(chain_count)
    if not chain_counts:
        parameter = next(model.parameters())
        return parameter.new_zeros(()), 0
    return sum(weighted_losses) / sum(chain_counts), sum(chain_counts)


class ExperimentTrainer:
    def __init__(
        self,
        config: Mapping[str, Any],
        variant: str,
        seed: int,
        model: ClarityDynamicsEnsemble,
        loaders: Mapping[str, DataLoader],
        optimizer: AdamW,
        scheduler: CosineAnnealingLR,
        device: torch.device,
        run_dir: Path,
        extract_drug_category,
        concordance_index,
    ) -> None:
        self.config = config
        self.variant = variant
        self.seed = seed
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = run_dir
        self.extract_drug_category = extract_drug_category
        self.concordance_index = concordance_index
        self.history: list[dict[str, Any]] = []
        self.best_c_index = -float("inf")
        self.start_epoch = 1

    def _set_phase(self, epoch: int) -> None:
        training = self.config["training"]
        if epoch <= int(training["warmup_epochs"]):
            weights = (1.0, 0.0, 0.0)
        else:
            weights = (
                float(training["lambda_l1"]),
                float(training["lambda_cox"]),
                float(training["lambda_bce"]),
            )
        self.model.base.lambda_l1.fill_(weights[0])
        self.model.base.lambda_cox.fill_(weights[1])
        self.model.base.lambda_bce.fill_(weights[2])

    def _member_base_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        post_target: torch.Tensor,
        survival_time: torch.Tensor,
        event: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses = []
        logs: dict[str, list[float]] = {"l1": [], "cox": [], "bce": []}
        for member in range(self.model.ensemble_size):
            loss, details = self.model.base.compute_loss(
                outputs["member_post"][member],
                outputs["member_risk"][member, :, None],
                outputs["member_logit"][member, :, None],
                post_target,
                survival_time,
                event,
            )
            losses.append(loss)
            for name in logs:
                logs[name].append(float(details[name]))
        return torch.stack(losses).mean(), {
            name: float(np.mean(values)) for name, values in logs.items()
        }

    def _auxiliary_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        pre_latent: torch.Tensor,
        time_delta: torch.Tensor,
        drugs_text: Sequence[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        training = self.config["training"]
        cf_weight = float(training.get("cf_weight", 0.0))
        contrastive_weight = float(training.get("contrastive_weight", 0.0))
        variance_weight = float(training.get("variance_weight", 0.0))
        categories = [self.extract_drug_category(value) for value in drugs_text]
        cf_losses, contrastive_losses, variance_losses = [], [], []
        for member, predictor in enumerate(self.model.predictors):
            prediction = outputs["member_post"][member]
            if cf_weight:
                cf_losses.append(
                    self.model.base.drug_swap_diversity_loss(
                        predictor,
                        pre_latent,
                        outputs["condition"],
                        time_delta,
                        prediction,
                        categories,
                        float(training.get("cf_cos_margin", 0.9)),
                    )
                )
            if contrastive_weight:
                contrastive_losses.append(
                    self.model.base.drug_category_contrastive_loss(
                        categories, prediction, temperature=0.1
                    )
                )
            if variance_weight:
                variance_losses.append(self.model.base.variance_loss(prediction, gamma=0.05))
        zero = pre_latent.new_zeros(())
        cf = torch.stack(cf_losses).mean() if cf_losses else zero
        contrastive = torch.stack(contrastive_losses).mean() if contrastive_losses else zero
        variance = torch.stack(variance_losses).mean() if variance_losses else zero
        total = cf_weight * cf + contrastive_weight * contrastive + variance_weight * variance
        return total, {
            "cf": float(cf.detach()),
            "contrastive": float(contrastive.detach()),
            "variance": float(variance.detach()),
        }

    def run_epoch(self, epoch: int, training_mode: bool) -> dict[str, float]:
        loader = self.loaders["train" if training_mode else "validation"]
        self.model.train(training_mode)
        records: dict[str, list[float]] = {
            name: [] for name in ("total", "base", "l1", "cox", "bce", "cf", "contrastive", "variance", "rrt")
        }
        risks, times, events = [], [], []
        actual_chains = 0
        rrt_batches = 0
        variant_cfg = self.config["variants"][self.variant]
        rrt_enabled = (
            training_mode
            and float(variant_cfg["rrt_weight"]) > 0
            and epoch >= int(self.config["rrt"]["start_epoch"])
        )
        context = torch.enable_grad() if training_mode else torch.no_grad()
        with context:
            for batch in loader:
                survival_time = batch["survival_time"].to(self.device, non_blocking=True)
                event = batch["event_indicator"].to(self.device, non_blocking=True)
                time_delta = batch["time_delta"].to(self.device, non_blocking=True)
                pre_latent, post_target = encode_pair(self.model, batch, self.device)
                outputs = self.model.forward_pair(
                    pre_latent,
                    batch["drugs_text"],
                    time_delta,
                    batch["clinical_text"],
                )
                base_loss, base_logs = self._member_base_loss(
                    outputs, post_target, survival_time, event
                )
                aux_loss, aux_logs = self._auxiliary_loss(
                    outputs, pre_latent, time_delta, batch["drugs_text"]
                ) if training_mode else (pre_latent.new_zeros(()), {"cf": 0.0, "contrastive": 0.0, "variance": 0.0})
                rrt_loss = pre_latent.new_zeros(())
                chain_count = 0
                if rrt_enabled:
                    rrt_loss, chain_count = recursive_latent_loss(
                        self.model,
                        batch["chain_groups"],
                        self.device,
                        int(self.config["rrt"]["supervise_from_step"]),
                    )
                    if chain_count:
                        rrt_batches += 1
                        actual_chains += chain_count
                total = base_loss + aux_loss + float(variant_cfg["rrt_weight"]) * rrt_loss
                if training_mode:
                    self.optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(self.config["training"]["grad_clip_norm"]),
                    )
                    self.optimizer.step()
                records["total"].append(float(total.detach()))
                records["base"].append(float(base_loss.detach()))
                for name, value in base_logs.items():
                    records[name].append(value)
                for name, value in aux_logs.items():
                    records[name].append(value)
                records["rrt"].append(float(rrt_loss.detach()))
                risks.append(outputs["member_risk"].mean(dim=0).detach().cpu())
                times.append(survival_time.detach().cpu())
                events.append(event.detach().cpu())
        risk_array = torch.cat(risks).numpy()
        time_array = torch.cat(times).numpy()
        event_array = torch.cat(events).numpy()
        result = {name: float(np.mean(values)) if values else 0.0 for name, values in records.items()}
        result["c_index"] = float(self.concordance_index(risk_array, time_array, event_array))
        result["eligible_chains"] = float(actual_chains)
        result["rrt_batches"] = float(rrt_batches)
        return result

    def fit(self, epochs: int) -> None:
        for epoch in range(self.start_epoch, epochs + 1):
            self._set_phase(epoch)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            train_metrics = self.run_epoch(epoch, training_mode=True)
            val_metrics = self.run_epoch(epoch, training_mode=False)
            self.scheduler.step()
            row = {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "peak_gpu_gb": (
                    torch.cuda.max_memory_allocated(self.device) / 1e9
                    if self.device.type == "cuda"
                    else 0.0
                ),
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            self.history.append(row)
            self._write_history()
            is_best = (
                epoch > int(self.config["training"]["warmup_epochs"])
                and val_metrics["c_index"] > self.best_c_index
            )
            if is_best:
                self.best_c_index = val_metrics["c_index"]
                self.save_checkpoint(epoch, "best.pt")
            self.save_checkpoint(epoch, "last.pt")
            print(json.dumps(row, sort_keys=True))

    def _write_history(self) -> None:
        path = self.run_dir / "history.csv"
        if not self.history:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)

    def save_checkpoint(self, epoch: int, filename: str) -> None:
        trainable = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        state = {
            name: tensor.detach().cpu()
            for name, tensor in self.model.state_dict().items()
            if name in trainable or name.endswith(("lambda_l1", "lambda_cox", "lambda_bce"))
        }
        torch.save(
            {
                "schema_version": 1,
                "epoch": epoch,
                "variant": self.variant,
                "seed": self.seed,
                "ensemble_size": self.model.ensemble_size,
                "trainable_state_dict": state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_c_index": self.best_c_index,
                "history": self.history,
                "upstream_commit": self.config["upstream_commit"],
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.random.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            self.run_dir / filename,
        )

    def resume(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["variant"] != self.variant or int(checkpoint["seed"]) != self.seed:
            raise ValueError("checkpoint variant/seed does not match requested run")
        self.model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_c_index = float(checkpoint.get("best_c_index", -float("inf")))
        self.history = list(checkpoint.get("history", []))
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        self.start_epoch = int(checkpoint["epoch"]) + 1


def train_one(
    config_path: str | Path,
    variant: str,
    seed: int,
    device_name: str = "cuda:0",
    resume: str | None = None,
    epochs_override: int | None = None,
) -> Path:
    config = load_config(config_path)
    if variant not in config["variants"]:
        raise ValueError(f"unknown variant {variant!r}")
    seed_everything(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(device_name)
    upstream_train, _, _, extract_category, concordance = configure_upstream(config["upstream_root"])
    args = upstream_args(config, seed)
    official_model = upstream_train.build_model(args, device)
    model = ClarityDynamicsEnsemble(
        official_model,
        ensemble_size=int(config["variants"][variant]["ensemble_size"]),
        seed=seed,
    ).to(device)
    model.assert_independent_members()
    loaders, datasets, split = build_loaders(config, seed)
    optimizer, scheduler = build_optimizer(config, model)
    run_dir = Path(config["run_root"]) / f"{variant}_seed{seed}"
    if not resume and (run_dir / "last.pt").exists():
        raise FileExistsError(
            f"run already exists at {run_dir}; pass --resume or move the existing run"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    resolved["active_variant"] = variant
    resolved["active_seed"] = seed
    resolved["split_sha256"] = split["split_sha256"]
    resolved["dataset_counts"] = {
        name: {"pairs": len(dataset), "chains": dataset.chain_count}
        for name, dataset in datasets.items()
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    trainer = ExperimentTrainer(
        config,
        variant,
        seed,
        model,
        loaders,
        optimizer,
        scheduler,
        device,
        run_dir,
        extract_category,
        concordance,
    )
    if resume:
        trainer.resume(resume)
    trainer.fit(epochs_override or int(config["training"]["epochs"]))
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--variant", required=True, choices=list("ABCD"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, help="override only for smoke/debug runs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_one(args.config, args.variant, args.seed, args.device, args.resume, args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
