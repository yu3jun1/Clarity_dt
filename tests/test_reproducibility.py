from __future__ import annotations

import csv
import json
import random

import numpy as np
import pytest
import torch

from clarity_rrt_v3.replicate_audit import (
    assess_pair, history_diagnostics, patient_means, seed_matched_comparison,
)
from clarity_rrt_v3.reproducibility import configure_reproducibility, seed_worker, state_sha256
from clarity_rrt_v3.train import load_config, resolve_run_config, run_directory


def test_replicate_output_isolation():
    config = load_config("configs/pure_rrt_v3.yaml")
    assert str(run_directory(config, "A", 42)).endswith("primary/A_seed42")
    resolved = resolve_run_config("configs/pure_rrt_v3.yaml", "/tmp/audit", "rep01", True)
    assert run_directory(resolved, "A", 42).as_posix() == "/tmp/audit/primary/A_seed42_rep01"
    assert resolved["deterministic"]
    assert config["output_root"] == "outputs/pure_rrt_v3_step2400"
    resolved["replicate_id"] = "../../overwrite"
    with pytest.raises(ValueError):
        run_directory(resolved, "A", 42)


def test_strict_determinism_requires_startup_hash_seed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=42"):
        configure_reproducibility(42, True)


def test_strict_determinism_controls(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    algorithm, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    previous = {
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        configure_reproducibility(42, True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        assert torch.backends.cuda.math_sdp_enabled()
    finally:
        torch.use_deterministic_algorithms(algorithm, warn_only=warn)
        torch.backends.cudnn.benchmark = previous["benchmark"]
        torch.backends.cudnn.deterministic = previous["deterministic"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
        torch.backends.cuda.enable_flash_sdp(previous["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(previous["efficient"])
        torch.backends.cuda.enable_cudnn_sdp(previous["cudnn"])
        torch.backends.cuda.enable_math_sdp(previous["math"])


def test_worker_seeds_and_parameter_fingerprint(monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: 456)
    seed_worker(0)
    one = random.random(), np.random.random()
    seed_worker(1)
    assert one == (random.random(), np.random.random())
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    initial = state_sha256(model)
    assert initial == state_sha256(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert state_sha256(model) != initial


def make_run(path, variant="B", seed=42, mse=0.1, checkpoint=480):
    path.mkdir()
    metrics = {
        "variant": variant, "seed": seed, "checkpoint_optimizer_step": checkpoint,
        "deterministic_requested": True,
        "recursive": {f"H{h}": {"latent_mse": mse * h, "cosine_similarity": 0.9,
                                 "c_index": 0.7, "brier365": 0.2} for h in (1, 2, 3)},
    }
    metadata = {key: "same" for key in ("initial_trainable_state_sha256", "batch_order_sha256", "source_sha256",
                                         "config_sha256", "data_fingerprints", "packages", "cuda_version", "cudnn_version", "cuda_visible_devices")}
    metadata.update(gpu={"uuid": "GPU-test"}, determinism={"algorithms_enabled": True, "warn_only": False})
    (path / "metrics.json").write_text(json.dumps(metrics))
    (path / "run_metadata.json").write_text(json.dumps(metadata))
    rows = [{"patient": patient, "start": "tp0", "end": f"tp{h}", "window_end": "tp3", "horizon": h,
             "latent_mse": mse * h, "cosine_similarity": 0.9, "primary_survival_window": 1,
             "risk": 0.1, "survival365": 0.7, "survival_time": 500, "event": 1}
            for patient in ("p1", "p2") for h in (1, 2, 3)]
    with (path / "recursive_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_pair_gate_stable_and_rejects_checkpoint_or_metric_drift(tmp_path):
    left = make_run(tmp_path / "one")
    right = make_run(tmp_path / "two")
    assert assess_pair(left, right)["stable"]
    changed = json.loads((right / "metrics.json").read_text())
    changed["checkpoint_optimizer_step"] = 504
    (right / "metrics.json").write_text(json.dumps(changed))
    assert not assess_pair(left, right)["stable"]
    changed["checkpoint_optimizer_step"] = 480
    changed["recursive"]["H3"]["latent_mse"] = 0.6
    (right / "metrics.json").write_text(json.dumps(changed))
    assert not assess_pair(left, right)["stable"]


def test_pair_gate_rejects_wrong_gpu_and_order(tmp_path):
    left, right = make_run(tmp_path / "one"), make_run(tmp_path / "two")
    meta = json.loads((right / "run_metadata.json").read_text())
    meta["gpu"]["uuid"] = "GPU-other"
    meta["batch_order_sha256"] = "different"
    (right / "run_metadata.json").write_text(json.dumps(meta))
    result = assess_pair(left, right)
    assert not result["stable"]
    assert "gpu_uuid" in result["metadata_mismatches"]
    assert "batch_order_sha256" in result["metadata_mismatches"]


def test_patient_level_averages_windows_not_pseudoreplicated():
    means = patient_means([
        {"patient": "p1", "horizon": 1, "latent_mse": 1.0, "cosine_similarity": 0.8},
        {"patient": "p1", "horizon": 1, "latent_mse": 3.0, "cosine_similarity": 0.6},
        {"patient": "p2", "horizon": 1, "latent_mse": 9.0, "cosine_similarity": 0.2},
    ])
    assert means[("p1", 1)]["latent_mse"] == 2.0
    assert means[("p2", 1)]["latent_mse"] == 9.0


def test_seed_matched_direction_and_pairing(tmp_path):
    b = make_run(tmp_path / "b", mse=0.2)
    e = make_run(tmp_path / "e", variant="E", mse=0.1)
    result = seed_matched_comparison([(42, b, e)], tmp_path / "comparison")
    assert result["runs"][0]["E_minus_B"]["H3"]["latent_mse"] == pytest.approx(-0.3)
    assert result["patient_paired_summary"]["H1"]["unique_patient_count"] == 2
    assert result["patient_paired_summary"]["H1"]["patients_with_lower_E_mse"] == 2
    with pytest.raises(ValueError, match="Seed mismatch"):
        seed_matched_comparison([(43, b, e)], tmp_path / "bad")


def test_history_diagnostics_first_divergence(tmp_path):
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    left.write_text("optimizer_step,train_loss,batch_order_sha256\n24,1,a\n48,0.8,b\n")
    right.write_text("optimizer_step,train_loss,batch_order_sha256\n24,1,a\n48,0.9,c\n")
    result = history_diagnostics(left, right)
    assert result["first_numerical_divergence_step"] == 48
    assert result["first_batch_order_divergence_step"] == 48


def test_compact_audit_summary_does_not_require_or_recreate_history(tmp_path):
    from clarity_rrt_v3.replicate_audit import preserve_history, write_audit_summary

    assert preserve_history(tmp_path)["replicates"] == []
    write_audit_summary(tmp_path, {})
    assert (tmp_path / "audit_summary.md").is_file()
    assert not (tmp_path / "historical").exists()
    assert not (tmp_path / "historical_replicates.json").exists()
    assert not (tmp_path / "old_A_replicate_classification.json").exists()
