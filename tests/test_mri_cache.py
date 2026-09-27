from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import clarity_rrt.mri_cache as mri_cache
from clarity_rrt.mri_cache import (
    CACHE_MAGIC,
    CACHE_SCHEMA_VERSION,
    CachedMRIVolumeLoader,
    cleanup_cache,
)


def _make_cache(root: Path) -> tuple[Path, torch.Tensor]:
    cache = root / "cache"
    cache.mkdir()
    mri_id = "PatientID_0001_Timepoint_2"
    array = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
    np.save(cache / f"{mri_id}.npy", array, allow_pickle=False)
    manifest = {
        "magic": CACHE_MAGIC,
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": True,
        "source_dir": str((root / "source").resolve()),
        "split_sha256": "abc123",
        "entries": {
            mri_id: {
                "file": f"{mri_id}.npy",
                "shape": list(array.shape),
                "dtype": "float32",
                "nbytes": array.nbytes,
            }
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache, torch.from_numpy(array)


def test_cached_loader_returns_exact_tensor(tmp_path):
    cache, expected = _make_cache(tmp_path)
    loader = CachedMRIVolumeLoader(
        cache,
        expected_source_dir=tmp_path / "source",
        expected_split_sha256="abc123",
    )
    assert loader.has("PatientID_0001_Timepoint_2")
    assert not loader.has("PatientID_0001_Timepoint_3")
    assert torch.equal(loader.load("PatientID_0001_Timepoint_2"), expected)


def test_cleanup_refuses_every_non_dedicated_path(tmp_path):
    cache, _ = _make_cache(tmp_path)
    with pytest.raises(ValueError, match="refusing to delete non-dedicated path"):
        cleanup_cache(cache)
    assert cache.is_dir()


def test_cleanup_removes_only_the_validated_dedicated_directory(
    tmp_path, monkeypatch
):
    cache, _ = _make_cache(tmp_path)
    sibling = tmp_path / "must_remain"
    sibling.mkdir()
    monkeypatch.setattr(mri_cache, "DEFAULT_CACHE_DIR", cache)
    cleanup_cache(cache)
    assert not cache.exists()
    assert sibling.is_dir()
