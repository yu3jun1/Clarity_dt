from __future__ import annotations

import json

import numpy as np
import torch

from clarity_rrt.data import (
    CachedMRIVolumeLoader,
    StagewiseTrajectoryDataset,
)


class FakeMRI:
    def has(self, identifier):
        return True

    def load(self, identifier):
        number = int(identifier.rsplit("_", 1)[-1])
        return torch.full((1, 2, 2, 2), float(number))


class FakeBase:
    def __init__(self):
        timeline = []
        for index in range(1, 6):
            timeline.append(
                {
                    "tp_id": f"TP{index}",
                    "mri_day": index * 30,
                    "actions": {"drug": [{"agent": f"d{index}"}]},
                    "survival": {
                        "survival_from_tp_days": 600 - index * 30,
                        "event_indicator": 1,
                    },
                }
            )
        self.patients = {
            "patients": {
                "P1": {"context_static": {"age": 50}, "timeline": timeline}
            }
        }
        self.mri_loader = FakeMRI()

    @staticmethod
    def _pack_drugs_json(item):
        return json.dumps(item, sort_keys=True)

    @staticmethod
    def _format_clinical_text(context):
        return "clinical"


def dataset():
    base = FakeBase()
    return StagewiseTrajectoryDataset(
        base.patients,
        base.mri_loader,
        base._pack_drugs_json,
        base._format_clinical_text,
        ["P1"],
    )


def test_dataset_builds_consecutive_four_stage_windows():
    trajectories = dataset()
    assert len(trajectories) == 2
    first = trajectories[0]
    assert first["timepoints"] == ["TP1", "TP2", "TP3", "TP4"]
    assert first["deltas"].tolist() == [30.0, 30.0, 30.0]
    assert len(first["step_text"]) == 3
    full_plan = json.loads(first["full_text"])
    assert len(full_plan["between_actions"]) == 2


def test_collate_keeps_step_and_prefix_text_separate():
    trajectories = dataset()
    batch = trajectories.collate([trajectories[0], trajectories[1]])
    assert batch["mri"].shape[:2] == (2, 4)
    assert len(batch["step_text"]) == 3
    assert len(batch["prefix_text"]) == 3
    assert all(len(texts) == 2 for texts in batch["step_text"])


def test_cached_mri_loader_reads_manifest_entry(tmp_path):
    identifier = "PatientID_0001_Timepoint_1"
    array = np.arange(32, dtype=np.float32).reshape(4, 2, 2, 2)
    np.save(tmp_path / f"{identifier}.npy", array)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"entries": {identifier: {"file": f"{identifier}.npy"}}}
        ),
        encoding="utf-8",
    )
    loader = CachedMRIVolumeLoader(tmp_path)
    assert loader.has(identifier)
    assert torch.equal(loader.load(identifier), torch.from_numpy(array))
