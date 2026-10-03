"""Deterministic execution controls and auditable run provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_worker(worker_id: int) -> None:
    # DataLoader has already assigned a reproducible torch seed to this worker.
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    if not deterministic:
        return  # Preserve the historical training protocol for old launchers.
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(f"Deterministic runs require PYTHONHASHSEED={seed} at launch")
    if torch.cuda.is_initialized():
        raise RuntimeError("Configure determinism before initializing CUDA")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Fused attention backward can be nondeterministic. Pin the math backend.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256()
    for directory in ("src", "configs", "scripts", "third_party/CLARITY/Predictor"):
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".sh"):
                digest.update(path.as_posix().encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def collect_metadata(
    config: Mapping[str, Any], variant: str, seed: int, device: torch.device,
) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "numpy", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "logical_index": index, "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "unknown")),
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    try:
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        inventory = []
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "diff", "HEAD", "--", "src", "scripts", "configs"], text=True))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    fingerprints = {}
    for key in ("timeline_json", "split_file"):
        path = config["data"][key]
        fingerprints[key] = {"path": str(path), "sha256": file_sha256(path)}
    manifest = Path(config["data"]["mri_cache_dir"]) / "manifest.json"
    fingerprints["mri_cache_manifest"] = {"path": str(manifest), "sha256": file_sha256(manifest)}
    return {
        "created_at_utc": utc_now(), "variant": variant, "seed": seed,
        "replicate_id": config.get("replicate_id"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "pytorch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "gpu": gpu, "nvidia_smi_inventory": inventory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": {key: os.environ.get(key) for key in (
            "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        )},
        "determinism": {
            "requested": bool(config.get("deterministic", False)),
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled(),
        },
        "dataloader": {
            "num_workers": int(config["training"]["num_workers"]),
            "batch_size": int(config["training"]["batch_size"]),
            "worker_init_fn": "seed_worker" if config.get("deterministic") else None,
            "independent_validation_generator": bool(config.get("deterministic")),
            "shuffle_seed": seed, "drop_last": False, "persistent_workers": False,
        },
        "git_commit": commit, "tracked_source_dirty": dirty,
        "source_sha256": source_sha256(), "data_fingerprints": fingerprints,
        "upstream_commit": config["upstream_commit"],
        "config_sha256": hashlib.sha256(json.dumps({
            key: value for key, value in config.items()
            if key not in ("replicate_id", "output_root")
        }, sort_keys=True).encode()).hexdigest(),
        "pretrained_paths": {key: config["model"][key] for key in ("brainiac_ckpt", "text_encoder_name")},
        "checkpoint_selection": "minimum validation_total_loss strictly after warmup; earliest tie",
    }
