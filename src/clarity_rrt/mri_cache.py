"""Build, validate, use, and safely remove the shared raw MRI cache."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CACHE_MAGIC = "clarity_rrt_raw_mri_cache"
CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path("/dev/shm/clarity_mri_cache")
MANIFEST_NAME = "manifest.json"
SAFE_ID = re.compile(r"^PatientID_[A-Za-z0-9-]+_Timepoint_[0-9]+$")


def _filename(mri_id: str) -> str:
    if not SAFE_ID.fullmatch(mri_id):
        raise ValueError(f"unsafe MRI identifier: {mri_id!r}")
    return f"{mri_id}.npy"


def _manifest_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / MANIFEST_NAME


def load_cache_manifest(
    cache_dir: str | Path, *, require_complete: bool = True
) -> dict[str, Any]:
    path = _manifest_path(cache_dir)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"MRI cache manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("magic") != CACHE_MAGIC:
        raise ValueError(f"invalid MRI cache magic in {path}")
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported MRI cache schema in {path}")
    if require_complete and not manifest.get("complete", False):
        raise ValueError(f"MRI cache is incomplete: {path}")
    if not isinstance(manifest.get("entries"), dict):
        raise ValueError(f"MRI cache entries are invalid: {path}")
    return manifest


class CachedMRIVolumeLoader:
    """Read exact float32 raw volumes from a shared, manifest-validated cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_source_dir: str | Path | None = None,
        expected_split_sha256: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        manifest = load_cache_manifest(self.cache_dir)
        if expected_source_dir is not None:
            expected = str(Path(expected_source_dir).resolve())
            if manifest.get("source_dir") != expected:
                raise ValueError(
                    f"MRI cache source mismatch: {manifest.get('source_dir')} != {expected}"
                )
        if (
            expected_split_sha256 is not None
            and manifest.get("split_sha256") != expected_split_sha256
        ):
            raise ValueError("MRI cache split hash does not match the configured split")
        self.entries: dict[str, dict[str, Any]] = manifest["entries"]

    def has(self, mri_id: str) -> bool:
        entry = self.entries.get(mri_id)
        return bool(
            entry
            and (self.cache_dir / str(entry.get("file", ""))).is_file()
        )

    def load(self, mri_id: str) -> torch.Tensor:
        entry = self.entries.get(mri_id)
        if entry is None:
            raise KeyError(f"MRI timepoint is absent from cache: {mri_id}")
        path = self.cache_dir / str(entry["file"])
        array = np.load(path, mmap_mode="c", allow_pickle=False)
        expected_shape = tuple(map(int, entry["shape"]))
        if array.dtype != np.float32 or array.shape != expected_shape:
            raise ValueError(
                f"corrupt MRI cache entry {path}: "
                f"dtype={array.dtype}, shape={array.shape}"
            )
        return torch.from_numpy(array)


def _raw_dataset_context(config_path: str | Path):
    from .train import build_datasets, load_config

    config = load_config(config_path)
    raw_config = copy.deepcopy(config)
    raw_config["data"].pop("mri_cache_dir", None)
    base, datasets, split = build_datasets(raw_config)
    if base.mri_loader is None:
        raise ValueError("configured upstream dataset has no raw MRI loader")
    return config, base, datasets, split


def collect_required_mri_ids(datasets: Mapping[str, Any], base: Any) -> list[str]:
    ids: set[str] = set()
    for dataset in datasets.values():
        for base_index in dataset._base_indices:
            item = base.index[base_index]
            ids.update((str(item["pre_id"]), str(item["post_id"])))
        for spec in dataset._chain_by_base_index.values():
            ids.update(map(str, spec.mri_ids))
    return sorted(ids)


def _valid_array(path: Path) -> tuple[bool, list[int], int]:
    if not path.is_file() or path.is_symlink():
        return False, [], 0
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = array.dtype == np.float32 and array.ndim == 4 and array.shape[0] == 4
        return valid, list(array.shape), int(array.nbytes)
    except (OSError, ValueError):
        return False, [], 0


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def build_cache(
    config_path: str | Path,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    verify_samples: int = 8,
) -> dict[str, Any]:
    config, base, datasets, split = _raw_dataset_context(config_path)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    if cache_path.is_symlink():
        raise ValueError(f"MRI cache directory must not be a symlink: {cache_path}")
    ids = collect_required_mri_ids(datasets, base)

    # The configured cohort is about 59.3 GiB. Include existing partial files
    # when checking whether the 252 GiB tmpfs has enough capacity to resume.
    existing_bytes = sum(
        child.stat().st_size
        for child in cache_path.glob("*.npy")
        if child.is_file() and not child.is_symlink()
    )
    free_bytes = shutil.disk_usage(cache_path).free
    required_capacity = 64 * 1024**3
    if free_bytes + existing_bytes < required_capacity:
        raise OSError(
            f"insufficient cache capacity: free+existing="
            f"{(free_bytes + existing_bytes) / 1024**3:.2f} GiB, "
            f"required={required_capacity / 1024**3:.2f} GiB"
        )

    entries: dict[str, dict[str, Any]] = {}
    total = len(ids)
    for index, mri_id in enumerate(ids, start=1):
        filename = _filename(mri_id)
        target = cache_path / filename
        valid, shape, nbytes = _valid_array(target)
        if not valid:
            if target.exists():
                target.unlink()
            tensor = base.mri_loader.load(mri_id)
            array = tensor.detach().cpu().numpy()
            if array.dtype != np.float32 or array.ndim != 4 or array.shape[0] != 4:
                raise ValueError(
                    f"unexpected raw MRI for {mri_id}: "
                    f"dtype={array.dtype}, shape={array.shape}"
                )
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_path,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.save(handle, array, allow_pickle=False)
            os.replace(temporary, target)
            valid, shape, nbytes = _valid_array(target)
            if not valid:
                raise ValueError(f"failed to validate newly cached MRI: {target}")
        entries[mri_id] = {
            "file": filename,
            "shape": shape,
            "dtype": "float32",
            "nbytes": nbytes,
        }
        if index == 1 or index % 10 == 0 or index == total:
            print(
                json.dumps(
                    {
                        "cache": str(cache_path),
                        "cached": index,
                        "total": total,
                        "gib": sum(entry["nbytes"] for entry in entries.values())
                        / 1024**3,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest: dict[str, Any] = {
        "magic": CACHE_MAGIC,
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(Path(config["data"]["mri_data_dir"]).resolve()),
        "split_sha256": split["split_sha256"],
        "entry_count": len(entries),
        "total_bytes": sum(entry["nbytes"] for entry in entries.values()),
        "entries": entries,
    }
    _write_json_atomic(_manifest_path(cache_path), manifest)

    sample_count = min(max(0, int(verify_samples)), len(ids))
    samples = random.Random(20260927).sample(ids, sample_count)
    cached_loader = CachedMRIVolumeLoader(
        cache_path,
        expected_source_dir=config["data"]["mri_data_dir"],
        expected_split_sha256=split["split_sha256"],
    )
    for mri_id in samples:
        source = base.mri_loader.load(mri_id)
        cached = cached_loader.load(mri_id)
        if not torch.equal(source, cached):
            raise AssertionError(f"cached MRI is not bitwise equal: {mri_id}")
    print(
        json.dumps(
            {
                "MRI_CACHE_READY": True,
                "cache": str(cache_path),
                "entries": len(entries),
                "gib": manifest["total_bytes"] / 1024**3,
                "bitwise_verified_samples": sample_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest


def cleanup_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
    requested = Path(os.path.abspath(os.fspath(cache_dir)))
    expected = DEFAULT_CACHE_DIR
    if requested != expected:
        raise ValueError(
            f"refusing to delete non-dedicated path: {requested}; expected {expected}"
        )
    if not requested.exists():
        print(json.dumps({"MRI_CACHE_REMOVED": False, "reason": "already absent"}))
        return
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError(f"refusing to delete unsafe cache target: {requested}")
    load_cache_manifest(requested)
    shutil.rmtree(requested)
    print(json.dumps({"MRI_CACHE_REMOVED": True, "cache": str(requested)}))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", default="configs/experiment.yaml")
    build.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    build.add_argument("--verify-samples", type=int, default=8)
    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args(argv)
    if args.command == "build":
        build_cache(args.config, args.cache_dir, verify_samples=args.verify_samples)
    else:
        cleanup_cache(args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
