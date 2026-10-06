from __future__ import annotations

import copy
import json
from pathlib import Path

from clarity_ensemble_study import __main__ as study
from clarity_rrt_v3.train import load_config


def test_study_changes_only_member_count_from_all_pair_baseline():
    baseline = load_config("configs/pure_rrt_v3.yaml")
    config = load_config(study.DEFAULT_CONFIG)
    for name in ("upstream_root", "upstream_commit", "data", "model", "training"):
        assert config[name] == baseline[name]
    assert config["variants"] == {
        f"M{size}": {"training_scheme": "clarity_all_pair", "ensemble_size": size}
        for size in (1, 2, 3, 5)
    }
    assert config["output_root"] != baseline["output_root"]


def config_in(tmp_path):
    config = copy.deepcopy(load_config(study.DEFAULT_CONFIG))
    config["output_root"] = str(tmp_path / "new_study")
    config["reference_root"] = str(tmp_path / "historical")
    return config


def test_campaign_runs_reference_evaluation_before_six_new_train_evaluations(tmp_path, monkeypatch):
    config = config_in(tmp_path)
    monkeypatch.setattr(study, "load_config", lambda path: config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    commands, reports = [], []

    def subprocess_run(command, **kwargs):
        assert kwargs["check"] is True
        commands.append(command)
        size = int(command[command.index("--size") + 1])
        seed = int(command[command.index("--seed") + 1])
        directory = Path(config["output_root"]) / "runs" / f"M{size}_seed{seed}"
        directory.mkdir(parents=True)
        study.write_json(directory / "status.json", {"state": "complete"})

    monkeypatch.setattr(study.subprocess, "run", subprocess_run)
    monkeypatch.setattr(study, "write_report", lambda root, stage, sizes: reports.append(
        (stage, sizes, len(commands))
    ))
    study.campaign("study.yaml")
    assert len(commands) == 12
    assert [int(command[command.index("--size") + 1]) for command in commands] == [
        1, 1, 1, 3, 3, 3, 2, 2, 2, 5, 5, 5,
    ]
    assert [int(command[command.index("--seed") + 1]) for command in commands] == [42, 43, 44] * 4
    assert all("--reference" in command for command in commands[:6])
    assert all("--reference" not in command for command in commands[6:])
    assert all(command[2] == "clarity_ensemble_study" for command in commands)
    assert reports == [("step1_reliability", [1, 3], 6), ("step2_size", [1, 2, 3, 5], 12)]
    status = json.loads((Path(config["output_root"]) / "campaign_status.json").read_text())
    assert status["state"] == "complete"
    assert len(status["completed_runs"]) == 12


def fake_evaluation(calls):
    def evaluate(config, variant, seed, device, checkpoint, directory, prediction_fn):
        calls.append((variant, seed, Path(checkpoint), Path(directory), prediction_fn))
        return {"seed": seed, "variant": variant}, [], {"reliability": {"H1": {}}}
    return evaluate


def test_reference_worker_reads_existing_checkpoint_and_writes_only_new_directory(tmp_path, monkeypatch):
    config = config_in(tmp_path)
    original = Path(config["reference_root"]) / "primary/C_seed42/best_val_loss.pt"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original checkpoint")
    monkeypatch.setattr(study, "load_config", lambda path: config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    calls = []
    monkeypatch.setattr(study, "evaluate_run", fake_evaluation(calls))

    def forbidden_training(*args, **kwargs):
        raise AssertionError("M1/M3 must not be retrained")

    monkeypatch.setattr(study, "train_run", forbidden_training)
    study.worker("study.yaml", 3, 42, reference=True)
    directory = Path(config["output_root"]) / "runs/M3_seed42"
    assert calls == [("M3", 42, original, directory, study.recursive_predictions)]
    assert original.read_bytes() == b"original checkpoint"
    assert list(original.parent.iterdir()) == [original]
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["origin"] == "existing_checkpoint"
    assert metrics["ensemble_size"] == 3
    assert metrics["reliability"] == {"H1": {}}


def test_new_worker_evaluates_its_primary_checkpoint_after_training(tmp_path, monkeypatch):
    config = config_in(tmp_path)
    monkeypatch.setattr(study, "load_config", lambda path: config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    order, calls = [], []
    directory = Path(config["output_root"]) / "runs/M5_seed44"

    def train(config, variant, seed, device, run_dir):
        assert (variant, seed, device, run_dir) == ("M5", 44, "cuda:0", directory)
        order.append("train")

    evaluation = fake_evaluation(calls)

    def evaluate(*args, **kwargs):
        order.append("evaluate")
        return evaluation(*args, **kwargs)

    monkeypatch.setattr(study, "train_run", train)
    monkeypatch.setattr(study, "evaluate_run", evaluate)
    study.worker("study.yaml", 5, 44, reference=False)
    assert order == ["train", "evaluate"]
    assert calls[0][2] == directory / "best_val_loss.pt"
    status = json.loads((directory / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["origin"] == "new_training"
    assert status["gpu"] == "4"
    assert status["worker_pid"] > 0
