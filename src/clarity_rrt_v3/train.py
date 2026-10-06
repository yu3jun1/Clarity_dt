"""Train the frozen CLARITY all-pair vs pure recursive 2x2 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
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

from .data import (
    PRIMARY_SURVIVAL_WINDOW_RULE,
    AllPairDataset,
    CachedMRIVolumeLoader,
    StagewiseTrajectoryDataset,
    extract_treatment_category,
    load_split,
    load_timeline,
)
from .model import StagewiseDynamics
from .reproducibility import (
    collect_metadata, state_sha256, utc_now,
)


PRIMARY_CHECKPOINT_NAME = "best_val_loss.pt"
CINDEX_CHECKPOINT_NAME = "best_val_cindex.pt"
MAIN_FACTORIAL_VARIANTS = {
    "A": {"training_scheme": "clarity_all_pair", "ensemble_size": 1},
    "B": {"training_scheme": "pure_recursive", "ensemble_size": 1},
    "C": {"training_scheme": "clarity_all_pair", "ensemble_size": 3},
    "D": {"training_scheme": "pure_recursive", "ensemble_size": 3},
}
TF_ABLATION_VARIANTS = {
    "E": {"training_scheme": "teacher_forced_stagewise", "ensemble_size": 1},
}
# Backward-compatible public name used by the frozen main-factorial tests.
FACTORIAL_VARIANTS = MAIN_FACTORIAL_VARIANTS


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def assert_factorial_design(config: Mapping[str, Any]) -> None:
    assert (
        config.get("experiment_kind", "main_factorial") == "main_factorial"
        and config["variants"] == MAIN_FACTORIAL_VARIANTS
    ), (
        "The v3 experiment definition must remain the frozen 2x2 "
        "all-pair/recursive x single/ensemble design"
    )


def assert_experiment_design(config: Mapping[str, Any]) -> None:
    kind = config.get("experiment_kind", "main_factorial")
    if kind == "main_factorial":
        assert_factorial_design(config)
    elif kind == "teacher_forced_stagewise_ablation":
        assert config["variants"] == TF_ABLATION_VARIANTS, (
            "The teacher-forced stage-wise ablation must contain only the "
            "single-predictor E variant"
        )
    else:
        raise AssertionError(f"Unknown experiment kind: {kind}")


def assert_upstream_commit(config: Mapping[str, Any]) -> str:
    root = Path(config["upstream_root"]).resolve()
    expected = str(config["upstream_commit"])
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert actual == expected, (
        f"CLARITY commit mismatch: expected {expected}, found {actual} in {root}"
    )
    return actual


def configure_upstream(upstream_root: str | Path):
    root = Path(upstream_root).resolve()
    for path in (root / "Predictor", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import train as upstream_train  # type: ignore
    from dataset.dataset_glioma_all_pairs_text import (  # type: ignore
        GliomaAllPairsTextDataset,
        MRIVolumeLoader,
    )
    from utils.metrics import concordance_index  # type: ignore

    return upstream_train, GliomaAllPairsTextDataset, MRIVolumeLoader, concordance_index


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def upstream_args(config: Mapping[str, Any], seed: int) -> Namespace:
    model = config["model"]
    training = config["training"]
    return Namespace(
        text_encoder_name=model["text_encoder_name"],
        freeze_text_encoder=bool(model["freeze_text_encoder"]),
        latent_dim=int(model["latent_dim"]),
        num_modalities=int(model["num_modalities"]),
        lambda_l1=float(training["lambda_l1"]),
        lambda_cox=float(training["lambda_cox"]),
        lambda_bce=float(training["lambda_bce"]),
        dropout=float(model["dropout"]),
        brainiac_ckpt=model["brainiac_ckpt"],
        brainiac_tokens=int(model["brainiac_tokens"]),
        brainiac_lora_r=int(model["brainiac_lora_r"]),
        lr=float(training["lr"]),
        text_lr=float(training["text_lr"]),
        text_proj_lr=float(training["text_proj_lr"]),
        vision_lr=float(training["vision_lr"]),
        survival_wd=float(training["survival_weight_decay"]),
        timeline_json=config["data"]["timeline_json"],
        mri_data_dir=config["data"]["mri_data_dir"],
        features_csv=None,
        take_dims=767,
        seed=seed,
    )


def build_datasets(config: Mapping[str, Any]) -> dict[str, StagewiseTrajectoryDataset]:
    _, upstream_dataset, _, _ = configure_upstream(config["upstream_root"])
    patients = load_timeline(config["data"]["timeline_json"])
    mri_loader = CachedMRIVolumeLoader(config["data"]["mri_cache_dir"])
    split = load_split(config["data"]["split_file"])["splits"]
    return {
        name: StagewiseTrajectoryDataset(
            patients,
            mri_loader,
            upstream_dataset._format_clinical_text,
            split[name],
        )
        for name in ("train", "validation", "test")
    }


def build_loaders(
    config: Mapping[str, Any],
    seed: int,
    variant: str,
) -> tuple[dict[str, DataLoader], dict[str, StagewiseTrajectoryDataset]]:
    datasets = build_datasets(config)
    loader_datasets: dict[str, Any] = dict(datasets)
    if config["variants"][variant]["training_scheme"] == "clarity_all_pair":
        for name in ("train", "validation"):
            trajectory_dataset = datasets[name]
            eligible_patient_ids = {
                trajectory.patient_id
                for trajectory in trajectory_dataset.trajectories
            }
            loader_datasets[name] = AllPairDataset(
                trajectory_dataset.patients,
                trajectory_dataset.mri_loader,
                trajectory_dataset.format_clinical,
                eligible_patient_ids,
                trajectory_dataset.primary_window_keys(),
            )

    training = config["training"]
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(
                training["batch_size"]
                if name == "train"
                else training["evaluation_batch_size"]
            ),
            shuffle=name == "train",
            generator=(
                torch.Generator().manual_seed(seed) if name == "train" else None
            ),
            num_workers=int(training["num_workers"]),
            collate_fn=dataset.collate,
        )
        for name, dataset in loader_datasets.items()
    }
    return loaders, datasets


def add_parameters(groups: list[dict[str, Any]], parameters, lr: float, weight_decay: float) -> None:
    selected = [parameter for parameter in parameters if parameter.requires_grad]
    if selected:
        groups.append({"params": selected, "lr": lr, "weight_decay": weight_decay})


def build_optimizer(
    config: Mapping[str, Any], model: StagewiseDynamics
) -> tuple[AdamW, CosineAnnealingLR]:
    training = config["training"]
    groups: list[dict[str, Any]] = []
    for predictor in model.predictors:
        add_parameters(groups, predictor.time_proj.parameters(), float(training["lr"]), 5e-4)
        add_parameters(
            groups,
            (
                parameter
                for name, parameter in predictor.named_parameters()
                if "time_proj" not in name
            ),
            float(training["lr"]) * 0.1,
            1e-5,
        )
    add_parameters(
        groups,
        model.survival_module.parameters(),
        float(training["lr"]),
        float(training["survival_weight_decay"]),
    )
    text_lora = []
    text_projection = []
    text_other = []
    for name, parameter in model.text_encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            text_lora.append(parameter)
        elif "final_proj" in name:
            text_projection.append(parameter)
        else:
            text_other.append(parameter)
    add_parameters(groups, text_lora, float(training["text_lr"]), 0.0)
    add_parameters(groups, text_projection, float(training["text_proj_lr"]), 1e-4)
    add_parameters(groups, text_other, float(training["text_lr"]), 0.0)
    add_parameters(groups, model.mri_encoder.parameters(), float(training["vision_lr"]), 0.0)
    optimizer = AdamW(groups)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=int(training["total_steps"]), eta_min=1e-6
    )
    return optimizer, scheduler


def encode_mri_stages(
    model: StagewiseDynamics,
    mri: torch.Tensor,
    stages: Sequence[int],
) -> torch.Tensor:
    return torch.stack([model.mri_encoder(mri[:, stage]) for stage in stages], dim=1)


def mean_horizon_l1(
    member_states: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    losses = [
        F.l1_loss(
            member_states[:, :, horizon],
            targets[:, horizon].unsqueeze(0).expand_as(member_states[:, :, horizon]),
        )
        for horizon in range(targets.shape[1])
    ]
    return torch.stack(losses).mean()


def counterfactual_loss(
    model: StagewiseDynamics,
    initial: torch.Tensor,
    member_states: torch.Tensor,
    conditions: torch.Tensor,
    deltas: torch.Tensor,
    treatment_texts: Sequence[Sequence[str]],
    margin: float,
    pre_states: torch.Tensor | None = None,
) -> torch.Tensor:
    categories = [
        [extract_treatment_category(text) for text in stage_texts]
        for stage_texts in treatment_texts
    ]
    losses = []
    for member, predictor in enumerate(model.predictors):
        for stage, stage_categories in enumerate(categories):
            if pre_states is None:
                previous = (
                    initial
                    if stage == 0
                    else member_states[member, :, stage - 1]
                )
            else:
                previous = pre_states[:, stage]
            prediction = member_states[member, :, stage]
            losses.append(
                model.clarity.drug_swap_diversity_loss(
                    predictor=predictor,
                    pre_latent=previous,
                    condition_emb=conditions[:, stage],
                    time_delta=deltas[:, stage],
                    pred_latent=prediction,
                    drug_categories=stage_categories,
                    cos_margin=margin,
                )
            )
    return torch.stack(losses).mean()


def one_year_bce(
    logits: torch.Tensor,
    survival_time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    label = (survival_time > 365).float()
    mask = (survival_time > 365) | ((survival_time <= 365) & (event == 1))
    if not mask.any():
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits[mask], label[mask])


def outcome_loss(
    model: StagewiseDynamics,
    risks: torch.Tensor,
    logits: torch.Tensor,
    survival_time: torch.Tensor,
    event: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cox = torch.stack(
        [
            model.clarity._simple_cox_loss(risks[member], survival_time, event)
            for member in range(model.ensemble_size)
        ]
    ).mean()
    bce = torch.stack(
        [
            one_year_bce(logits[member], survival_time, event)
            for member in range(model.ensemble_size)
        ]
    ).mean()
    return cox, bce


def run_directory(
    config: Mapping[str, Any],
    variant: str,
    seed: int,
) -> Path:
    suffix = f"_{config['replicate_id']}" if config.get("replicate_id") else ""
    if suffix and (not suffix[1:].replace("-", "").replace("_", "").isalnum()):
        raise ValueError("replicate_id must contain only letters, numbers, '-' or '_'")
    return Path(config["output_root"]) / "primary" / f"{variant}_seed{seed}{suffix}"


def resolve_run_config(config_path, output_root=None, replicate_id=None):
    """Resolve a step2400 run without inheriting retired execution switches."""
    config = load_config(config_path)
    if output_root is not None:
        config["output_root"] = str(output_root)
    if replicate_id is not None:
        config["replicate_id"] = replicate_id
    config.pop("deterministic", None)
    return config


def compute_training_budget(
    config: Mapping[str, Any],
    train_loader: DataLoader,
) -> dict[str, int | float]:
    training = config["training"]
    optimizer_steps = int(training["total_steps"])
    dataset_size = len(train_loader.dataset)
    batch_size = int(train_loader.batch_size)
    steps_per_pass = len(train_loader)
    complete_passes, remaining_steps = divmod(optimizer_steps, steps_per_pass)
    samples_seen = (
        complete_passes * dataset_size
        + min(remaining_steps * batch_size, dataset_size)
    )
    return {
        "optimizer_steps": optimizer_steps,
        "warmup_steps": int(training["warmup_steps"]),
        "validation_interval_steps": int(training["validation_interval_steps"]),
        "samples_seen": samples_seen,
        "effective_dataset_passes": samples_seen / dataset_size,
    }


class Trainer:
    metric_names = ("loss", "latent", "cox", "bce", "cf")

    def __init__(
        self,
        config: Mapping[str, Any],
        variant: str,
        model: StagewiseDynamics,
        loaders: Mapping[str, DataLoader],
        optimizer: AdamW,
        scheduler: CosineAnnealingLR,
        concordance_index,
        device: torch.device,
        run_dir: Path,
    ) -> None:
        self.config = config
        self.variant = variant
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.concordance_index = concordance_index
        self.device = device
        self.run_dir = run_dir
        self.history: list[dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.best_c_index = -float("inf")
        self.optimizer_steps = 0
        self.samples_seen = 0
        self.training_budget = compute_training_budget(config, loaders["train"])
        self.training_dataset_size = len(loaders["train"].dataset)
        self.batch_order_digest = hashlib.sha256()

    @property
    def clarity_all_pair(self) -> bool:
        return self.training_scheme == "clarity_all_pair"

    @property
    def training_scheme(self) -> str:
        return str(self.config["variants"][self.variant]["training_scheme"])

    def _all_pair_step(
        self,
        batch: Mapping[str, Any],
        optimizer_step: int,
        compute_cf: bool,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        initial = self.model.mri_encoder(batch["pre_mri"].to(self.device))
        target = self.model.mri_encoder(batch["post_mri"].to(self.device)).detach()
        condition = self.model.encode_conditions(
            [batch["treatment_text"]],
            batch["clinical_text"],
        )
        delta = batch["time_delta"].to(self.device)
        direct = self.model.direct(initial, condition[:, 0], delta)
        latent = F.l1_loss(
            direct,
            target.unsqueeze(0).expand_as(direct),
        )
        terminal = direct.unsqueeze(2)
        risks, logits = self.model.survival(initial, terminal, condition)
        survival_time = batch["survival_time"].to(self.device)
        event = batch["event"].to(self.device)
        cox, bce = outcome_loss(
            self.model,
            risks[:, :, 0],
            logits[:, :, 0],
            survival_time,
            event,
        )

        training = self.config["training"]
        cf = (
            counterfactual_loss(
                self.model,
                initial,
                terminal,
                condition,
                delta.unsqueeze(1),
                [batch["treatment_text"]],
                float(training["cf_cos_margin"]),
            )
            if compute_cf
            else latent.new_zeros(())
        )
        if optimizer_step <= int(training["warmup_steps"]):
            total = latent
        else:
            total = (
                float(training["lambda_l1"]) * latent
                + float(training["lambda_cox"]) * cox
                + float(training["lambda_bce"]) * bce
            )
        total = total + float(training["cf_weight"]) * cf
        values = {
            "loss": float(total.detach()),
            "latent": float(latent.detach()),
            "cox": float(cox.detach()),
            "bce": float(bce.detach()),
            "cf": float(cf.detach()),
        }
        return total, values, risks[:, :, 0].mean(dim=0).detach()

    def step(
        self,
        batch: Mapping[str, Any],
        optimizer_step: int,
        compute_cf: bool,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        if self.clarity_all_pair:
            return self._all_pair_step(batch, optimizer_step, compute_cf)

        mri = batch["mri"].to(self.device)
        deltas = batch["deltas"].to(self.device)
        terminal_time = batch["survival_time"][:, 2].to(self.device)
        terminal_event = batch["event"][:, 2].to(self.device)
        full_condition = self.model.encode_conditions(
            [batch["full_text"]], batch["clinical_text"]
        )

        states = encode_mri_stages(self.model, mri, (0, 1, 2, 3))
        targets = states[:, 1:].detach()
        step_conditions = self.model.encode_conditions(
            batch["step_text"], batch["clinical_text"]
        )
        if self.training_scheme == "pure_recursive":
            predictions = self.model.rollout(states[:, 0], step_conditions, deltas)
            cf_pre_states = None
        elif self.training_scheme == "teacher_forced_stagewise":
            teacher_inputs = torch.stack(
                (
                    states[:, 0],
                    states[:, 1].detach(),
                    states[:, 2].detach(),
                ),
                dim=1,
            )
            predictions = self.model.teacher_forced(
                teacher_inputs,
                step_conditions,
                deltas,
            )
            cf_pre_states = teacher_inputs
        else:
            raise AssertionError(f"Unknown training scheme: {self.training_scheme}")

        latent = mean_horizon_l1(predictions, targets)
        cf_states = predictions
        cf_conditions = step_conditions
        cf_deltas = deltas
        cf_texts = batch["step_text"]
        terminal = predictions[:, :, -1:]
        initial = states[:, 0]

        risks, logits = self.model.survival(initial, terminal, full_condition)
        cox, bce = outcome_loss(
            self.model,
            risks[:, :, 0],
            logits[:, :, 0],
            terminal_time,
            terminal_event,
        )
        training = self.config["training"]
        cf = (
            counterfactual_loss(
                self.model,
                initial,
                cf_states,
                cf_conditions,
                cf_deltas,
                cf_texts,
                float(training["cf_cos_margin"]),
                pre_states=cf_pre_states,
            )
            if compute_cf
            else latent.new_zeros(())
        )
        if optimizer_step <= int(training["warmup_steps"]):
            total = latent
        else:
            total = (
                float(training["lambda_l1"]) * latent
                + float(training["lambda_cox"]) * cox
                + float(training["lambda_bce"]) * bce
            )
        total = total + float(training["cf_weight"]) * cf
        values = {
            "loss": float(total.detach()),
            "latent": float(latent.detach()),
            "cox": float(cox.detach()),
            "bce": float(bce.detach()),
            "cf": float(cf.detach()),
        }
        return total, values, risks[:, :, 0].mean(dim=0).detach()

    def training_update(
        self,
        batch: Mapping[str, Any],
        optimizer_step: int,
    ) -> dict[str, float]:
        self.model.train(True)
        total, values, _ = self.step(batch, optimizer_step, compute_cf=True)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            float(self.config["training"]["grad_clip_norm"]),
        )
        self.optimizer.step()
        return values

    def validation(self, optimizer_step: int) -> dict[str, float]:
        self.model.train(False)
        records = {name: [] for name in self.metric_names}
        batch_sizes = []
        risks = []
        times = []
        events = []
        primary_windows = []
        with torch.no_grad():
            for batch in self.loaders["validation"]:
                _, values, risk = self.step(
                    batch,
                    optimizer_step,
                    compute_cf=False,
                )
                for name, value in values.items():
                    records[name].append(value)
                batch_sizes.append(len(batch["patient"]))
                risks.append(risk.cpu())
                if self.clarity_all_pair:
                    times.append(batch["survival_time"])
                    events.append(batch["event"])
                else:
                    times.append(batch["survival_time"][:, 2])
                    events.append(batch["event"][:, 2])
                primary_windows.append(batch["primary_survival_window"])
        result = {
            name: float(np.average(values, weights=batch_sizes))
            for name, values in records.items()
        }
        primary = torch.cat(primary_windows)
        result["c_index"] = float(
            self.concordance_index(
                torch.cat(risks)[primary].numpy(),
                torch.cat(times)[primary].numpy(),
                torch.cat(events)[primary].numpy(),
            )
        )
        return result

    def fit(self) -> None:
        self.batch_order_digest = hashlib.sha256()
        training = self.config["training"]
        total_steps = int(training["total_steps"])
        warmup_steps = int(training["warmup_steps"])
        validation_interval = int(training["validation_interval_steps"])
        train_loader = self.loaders["train"]
        train_iterator = iter(train_loader)
        records = {name: [] for name in self.metric_names}
        batch_sizes: list[int] = []

        for optimizer_step in range(1, total_steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            values = self.training_update(batch, optimizer_step)
            self.batch_order_digest.update(json.dumps(
                [batch["patient"], batch.get("timepoints")], separators=(",", ":")
            ).encode())
            self.scheduler.step()
            self.optimizer_steps = optimizer_step
            batch_size = len(batch["patient"])
            self.samples_seen += batch_size
            batch_sizes.append(batch_size)
            for name, value in values.items():
                records[name].append(value)

            if optimizer_step % validation_interval != 0 and optimizer_step < total_steps:
                continue

            train = {
                name: float(np.average(values, weights=batch_sizes))
                for name, values in records.items()
            }
            validation = self.validation(optimizer_step)
            row: dict[str, Any] = {
                "optimizer_step": float(optimizer_step),
                "samples_seen": float(self.samples_seen),
                "effective_dataset_passes": (
                    self.samples_seen / self.training_dataset_size
                ),
                "batch_order_sha256": self.batch_order_digest.hexdigest(),
            }
            row.update({f"train_{name}": value for name, value in train.items()})
            row.update(
                {f"validation_{name}": value for name, value in validation.items()}
            )
            self.history.append(row)
            self.write_history()

            if optimizer_step > warmup_steps:
                if validation["loss"] < self.best_val_loss:
                    self.best_val_loss = validation["loss"]
                    self.save(
                        optimizer_step,
                        PRIMARY_CHECKPOINT_NAME,
                        "validation_total_loss",
                        validation,
                    )
                if validation["c_index"] > self.best_c_index:
                    self.best_c_index = validation["c_index"]
                    self.save(
                        optimizer_step,
                        CINDEX_CHECKPOINT_NAME,
                        "validation_patient_level_c_index",
                        validation,
                    )
            print(json.dumps(row, sort_keys=True), flush=True)
            records = {name: [] for name in self.metric_names}
            batch_sizes = []

        assert self.optimizer_steps == int(self.training_budget["optimizer_steps"])
        assert self.samples_seen == int(self.training_budget["samples_seen"])

    def write_history(self) -> None:
        with (self.run_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)

    def save(
        self,
        optimizer_step: int,
        filename: str,
        selection_criterion: str,
        validation: Mapping[str, float],
    ) -> None:
        trainable = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        state = {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
            if name in trainable
        }
        torch.save(
            {
                "optimizer_step": optimizer_step,
                "samples_seen": self.samples_seen,
                "effective_dataset_passes": (
                    self.samples_seen / self.training_dataset_size
                ),
                "training_budget": dict(self.training_budget),
                "variant": self.variant,
                "seed": self.config.get("active_seed"),
                "replicate_id": self.config.get("replicate_id"),
                "batch_order_sha256": self.batch_order_digest.hexdigest(),
                "training_scheme": self.config["variants"][self.variant][
                    "training_scheme"
                ],
                "upstream_commit": self.config["upstream_commit"],
                "selection_criterion": selection_criterion,
                "validation_metrics": dict(validation),
                "state_dict": state,
            },
            self.run_dir / filename,
        )


def train_one(
    config_path: str | Path,
    variant: str,
    seed: int,
    device_name: str,
    output_root: str | Path | None = None,
    replicate_id: str | None = None,
) -> Path:
    config = resolve_run_config(config_path, output_root, replicate_id)
    assert_experiment_design(config)
    return train_run(config, variant, seed, device_name, run_directory(config, variant, seed))


def train_run(
    config: Mapping[str, Any],
    variant: str,
    seed: int,
    device_name: str,
    run_dir: str | Path,
) -> Path:
    """Train the configured dynamics using the shared step2400 implementation."""
    config = dict(config)
    config["active_seed"] = seed
    upstream_commit = assert_upstream_commit(config)
    seed_everything(seed)
    device = torch.device(device_name)
    run_dir = Path(run_dir)
    if config.get("replicate_id") and any(
        (run_dir / name).exists() for name in ("config.yaml", "history.csv", PRIMARY_CHECKPOINT_NAME)
    ):
        raise FileExistsError(f"Refusing to overwrite replicate: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    variant_config = config["variants"][variant]
    upstream_train, _, _, concordance = configure_upstream(config["upstream_root"])
    clarity = upstream_train.build_model(upstream_args(config, seed), device)
    model = StagewiseDynamics(
        clarity,
        ensemble_size=int(variant_config["ensemble_size"]),
        seed=seed,
    ).to(device)
    loaders, datasets = build_loaders(config, seed, variant)
    optimizer, scheduler = build_optimizer(config, model)
    # Match 9871306's initialization order: metadata must not initialize CUDA
    # ahead of the historical model -> loaders -> optimizer construction.
    metadata = collect_metadata(config, variant, seed, device)
    metadata["initial_trainable_state_sha256"] = state_sha256(model)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    resolved = dict(config)
    resolved["upstream_commit"] = upstream_commit
    resolved["active_variant"] = variant
    resolved["active_seed"] = seed
    resolved["active_training_scheme"] = variant_config["training_scheme"]
    resolved["training_budget"] = compute_training_budget(config, loaders["train"])
    resolved["primary_checkpoint"] = PRIMARY_CHECKPOINT_NAME
    resolved["secondary_checkpoint"] = CINDEX_CHECKPOINT_NAME
    resolved["primary_survival_window_rule"] = PRIMARY_SURVIVAL_WINDOW_RULE
    resolved["cohort_counts"] = {
        name: dataset.cohort_counts() for name, dataset in datasets.items()
    }
    if variant_config["training_scheme"] == "clarity_all_pair":
        resolved["all_pair_counts"] = {
            name: loaders[name].dataset.cohort_counts()
            for name in ("train", "validation")
        }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    trainer = Trainer(
        config,
        variant,
        model,
        loaders,
        optimizer,
        scheduler,
        concordance,
        device,
        run_dir,
    )
    trainer.fit()
    metadata.update({
        "training_complete_at_utc": utc_now(),
        "batch_order_sha256": trainer.batch_order_digest.hexdigest(),
        "optimizer_steps": trainer.optimizer_steps, "samples_seen": trainer.samples_seen,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return run_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/pure_rrt_v3.yaml")
    result.add_argument("--variant", required=True, choices=list("ABCDE"))
    result.add_argument("--seed", required=True, type=int)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--output-root")
    result.add_argument("--replicate", dest="replicate_id")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    train_one(args.config, args.variant, args.seed, args.device,
              args.output_root, args.replicate_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
