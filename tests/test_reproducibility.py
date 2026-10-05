from __future__ import annotations

import csv
import inspect
import json

import pytest
import torch

from clarity_rrt_v3.replicate_audit import (
    assess_pair, history_diagnostics, patient_means, seed_matched_comparison,
)
from clarity_rrt_v3 import evaluate, replicate_audit, reproducibility, train
from clarity_rrt_v3.reproducibility import state_sha256
from clarity_rrt_v3.train import load_config, resolve_run_config, run_directory


def test_replicate_output_isolation():
    config = load_config("configs/pure_rrt_v3.yaml")
    assert str(run_directory(config, "A", 42)).endswith("primary/A_seed42")
    resolved = resolve_run_config("configs/pure_rrt_v3.yaml", "/tmp/audit", "rep01")
    assert run_directory(resolved, "A", 42).as_posix() == "/tmp/audit/primary/A_seed42_rep01"
    assert "deterministic" not in resolved
    assert config["output_root"] == "outputs/pure_rrt_v3_step2400"
    resolved["replicate_id"] = "../../overwrite"
    with pytest.raises(ValueError):
        run_directory(resolved, "A", 42)


def test_step2400_discards_retired_switch_without_rewriting_saved_config(tmp_path):
    config = load_config("configs/pure_rrt_v3.yaml")
    config["deterministic"] = True
    saved = tmp_path / "saved_audit_config.yaml"
    saved.write_text(json.dumps(config))
    ordinary = resolve_run_config(saved)
    assert "deterministic" not in ordinary
    assert load_config(saved)["deterministic"] is True


@pytest.mark.parametrize("module,arguments", [
    (train, ["--variant", "B", "--seed", "42"]),
    (evaluate, ["run", "--variant", "B", "--seed", "42"]),
])
def test_retired_training_flag_is_rejected(module, arguments):
    assert not hasattr(module.parser().parse_args(arguments), "deterministic")
    with pytest.raises(SystemExit) as error:
        module.parser().parse_args([*arguments, "--deterministic"])
    assert error.value.code == 2


def test_retired_training_api_is_removed():
    for function in (resolve_run_config, train.train_one, evaluate.evaluate_one):
        assert "deterministic" not in inspect.signature(function).parameters
    assert not hasattr(reproducibility, "configure_reproducibility")
    assert not hasattr(reproducibility, "seed_worker")
    assert not hasattr(replicate_audit, "run_experiment")
    assert not hasattr(replicate_audit, "pipeline")


def test_step2400_loader_protocol_has_no_execution_switch(monkeypatch):
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return index

        @staticmethod
        def collate(items):
            return items

    monkeypatch.setattr(train, "build_datasets", lambda config: {
        name: DummyDataset() for name in ("train", "validation", "test")
    })
    config = load_config("configs/pure_rrt_v3.yaml")
    # Even an unresolved historical config cannot activate a separate path.
    config["deterministic"] = True
    loaders, _ = train.build_loaders(config, seed=42, variant="B")
    assert loaders["train"].generator.initial_seed() == 42
    for name in ("train", "validation", "test"):
        assert loaders[name].worker_init_fn is None
    for name in ("validation", "test"):
        assert loaders[name].generator is None


def test_step2400_metadata_does_not_initialize_cuda_before_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    events = []
    seeds = []

    class Model:
        def to(self, device):
            return self

    class StopBeforeTraining(Exception):
        pass

    def build_model(args, device):
        events.append("model")
        return Model()

    def build_loaders(*args):
        events.append("loaders")
        return {}, {}

    def build_optimizer(*args):
        events.append("optimizer")
        return None, None

    def collect_metadata(*args):
        assert events == ["model", "loaders", "optimizer"]
        raise StopBeforeTraining

    monkeypatch.setattr(train, "assert_upstream_commit", lambda config: config["upstream_commit"])
    monkeypatch.setattr(train, "seed_everything", seeds.append)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def forbidden_control(*args, **kwargs):
        raise AssertionError("training must not change numerical backend settings")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", forbidden_control)
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp", "enable_math_sdp"):
        monkeypatch.setattr(torch.backends.cuda, name, forbidden_control)
    monkeypatch.setattr(train, "configure_upstream", lambda root: (SimpleNamespace(build_model=build_model), None, None, None))
    monkeypatch.setattr(train, "StagewiseDynamics", lambda clarity, **kwargs: clarity)
    monkeypatch.setattr(train, "build_loaders", build_loaders)
    monkeypatch.setattr(train, "build_optimizer", build_optimizer)
    monkeypatch.setattr(train, "collect_metadata", collect_metadata)
    with pytest.raises(StopBeforeTraining):
        train.train_one("configs/pure_rrt_v3.yaml", "B", 42, "cpu", output_root=tmp_path)
    assert seeds == [42]


def test_metadata_records_versions_without_changing_execution(monkeypatch):
    config = resolve_run_config("configs/pure_rrt_v3.yaml")
    monkeypatch.setattr(reproducibility.importlib.metadata, "version", lambda name: "package-version")
    monkeypatch.setattr(torch.version, "cuda", "cuda-version")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9000)
    monkeypatch.setattr(reproducibility, "file_sha256", lambda path: "data-hash")
    monkeypatch.setattr(reproducibility, "source_sha256", lambda: "source-hash")
    monkeypatch.setattr(reproducibility.subprocess, "check_output", lambda *args, **kwargs: "")

    def forbidden_control(*args, **kwargs):
        raise AssertionError("metadata must only observe numerical backend settings")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", forbidden_control)
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp", "enable_math_sdp"):
        monkeypatch.setattr(torch.backends.cuda, name, forbidden_control)
    metadata = reproducibility.collect_metadata(config, "B", 42, torch.device("cpu"))
    assert metadata["pytorch_version"] == str(torch.__version__)
    assert metadata["cuda_version"] == "cuda-version"
    assert metadata["cudnn_version"] == 9000
    assert metadata["behavioral_baseline_commit"] == "9871306"
    assert "execution_profile" not in metadata
    assert "requested" not in metadata["determinism"]
    assert metadata["dataloader"]["worker_init_fn"] is None
    assert metadata["dataloader"]["independent_validation_generator"] is False


def test_parameter_fingerprint():
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
    summary = (tmp_path / "audit_summary.md").read_text()
    assert "历史记录与精确删除范围见实验记录索引" in summary
    assert "已退役实验" not in summary
    assert not (tmp_path / "historical").exists()
    assert not (tmp_path / "historical_replicates.json").exists()
    assert not (tmp_path / "old_A_replicate_classification.json").exists()


def test_history_retention_status_uses_existing_filtered_index(tmp_path):
    from clarity_rrt_v3.replicate_audit import history_retention_status

    assert history_retention_status(tmp_path) == {
        "historical_archives_retired": True,
        "historical_auto_reconstruction_disabled": True,
    }
    index = tmp_path / "historical_replicates.json"
    index.write_text(json.dumps({"replicates": [{"variant": "A", "replicate_id": "historical_rep02"}]}))
    before = index.read_bytes()
    assert history_retention_status(tmp_path) == {
        "historical_archives_retired": False,
        "historical_auto_reconstruction_disabled": True,
    }
    assert index.read_bytes() == before
    assert not (tmp_path / "historical").exists()
    assert not (tmp_path / "old_A_replicate_classification.json").exists()
