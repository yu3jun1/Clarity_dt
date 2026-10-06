from __future__ import annotations

import json
import os

from clarity_ensemble_study import __main__ as study


def study_statuses(tmp_path, monkeypatch, pending=()):
    root = tmp_path / "new_study"
    root.mkdir()
    config = {
        "output_root": str(root),
        "variants": {f"M{size}": {"ensemble_size": size} for size in (1, 2, 3, 5)},
        "training": {"seeds": [42, 43, 44]},
    }
    monkeypatch.setattr(study, "load_config", lambda path: config)
    (root / "protocol.json").write_text(json.dumps({"coordinator_pid": 123, "study": "ensemble_reliability_size"}))
    names = []
    for size in (1, 2, 3, 5):
        for seed in (42, 43, 44):
            name = f"M{size}_seed{seed}"
            names.append(name)
            directory = root / "runs" / name
            directory.mkdir(parents=True)
            (directory / "status.json").write_text(json.dumps({
                "ensemble_size": size, "seed": seed,
                "state": "training" if name in pending else "complete",
            }))
    return root, names


def test_finalizer_waits_for_parallel_workers_before_writing_report(tmp_path, monkeypatch):
    pending = {"M5_seed43", "M5_seed44"}
    root, names = study_statuses(tmp_path, monkeypatch, pending)
    reports, waits = [], []

    def sleep(seconds):
        assert not reports
        assert seconds == 30
        status = json.loads((root / "campaign_status.json").read_text())
        assert status["state"] == "running"
        assert status["stage"] == "step2_size"
        assert set(status["active_runs"]) == pending
        assert set(status["completed_runs"]) == set(names) - pending
        waits.append(seconds)
        for name in pending:
            path = root / "runs" / name / "status.json"
            worker_status = json.loads(path.read_text())
            worker_status["state"] = "complete"
            path.write_text(json.dumps(worker_status))

    def report(destination, stage, sizes):
        assert all(json.loads((root / "runs" / name / "status.json").read_text())["state"] == "complete"
                   for name in names)
        reports.append((destination, stage, sizes))

    monkeypatch.setattr(study.time, "sleep", sleep)
    monkeypatch.setattr(study, "write_report", report)
    study.finalize("study.yaml")
    assert waits == [30]
    assert reports == [(root, "step2_size", [1, 2, 3, 5])]
    status = json.loads((root / "campaign_status.json").read_text())
    assert status["state"] == "complete"
    assert set(status["completed_runs"]) == set(names)
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["finalizer_pid"] == os.getpid()
    assert protocol["coordinator_pid"] == 123


def test_finalizer_writes_report_immediately_when_all_runs_complete(tmp_path, monkeypatch):
    root, names = study_statuses(tmp_path, monkeypatch)
    reports = []

    def forbidden_sleep(seconds):
        raise AssertionError("Completed experiments must not wait")

    monkeypatch.setattr(study.time, "sleep", forbidden_sleep)
    monkeypatch.setattr(study, "write_report", lambda destination, stage, sizes:
                        reports.append((destination, stage, sizes)))
    study.finalize("study.yaml")
    assert reports == [(root, "step2_size", [1, 2, 3, 5])]
    status = json.loads((root / "campaign_status.json").read_text())
    assert status["state"] == "complete"
    assert set(status["completed_runs"]) == set(names)
