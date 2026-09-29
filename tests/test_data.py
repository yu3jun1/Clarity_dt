from __future__ import annotations

import json

import numpy as np
import torch

from clarity_rrt_v3.data import (
    AllPairDataset,
    CachedMRIVolumeLoader,
    StagewiseTrajectoryDataset,
    extract_treatment_category,
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
                    "actions": {
                        "drug": [
                            {
                                "agent": f"d{index}",
                                "interval_start_day": (index - 1) * 30 + 1,
                                "interval_end_day": index * 30,
                            }
                        ]
                    },
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
    def _format_clinical_text(context):
        return "clinical"


def dataset():
    base = FakeBase()
    return StagewiseTrajectoryDataset(
        base.patients,
        base.mri_loader,
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
    assert len(full_plan["intervals"]) == 3
    first_step = json.loads(first["step_text"][0])
    assert first_step == {
        "intervals": [
            {
                "start_tp": "TP1",
                "end_tp": "TP2",
                "actions": {
                    "drug": [{"agent": "d2", "start_offset_days": 1, "end_offset_days": 30}]
                },
            }
        ]
    }
    assert trajectories.cohort_counts() == {
        "trajectory_count": 2,
        "unique_patient_count": 1,
        "primary_survival_window_count": 1,
    }
    assert trajectories.primary_trajectories()[0].points == trajectories.trajectories[0].points


def test_all_pair_dataset_uses_each_patient_pair_once():
    base = FakeBase()
    trajectories = StagewiseTrajectoryDataset(
        base.patients,
        base.mri_loader,
        base._format_clinical_text,
        ["P1"],
    )
    pairs = AllPairDataset(
        base.patients,
        base.mri_loader,
        base._format_clinical_text,
        ["P1"],
        trajectories.primary_window_keys(),
    )

    assert len(pairs) == 10
    assert pairs.cohort_counts() == {
        "pair_count": 10,
        "unique_patient_count": 1,
        "primary_survival_pair_count": 1,
    }
    assert pairs[2]["timepoints"] == ["TP1", "TP4"]
    assert pairs[2]["primary_survival_window"] is True
    assert pairs[3]["timepoints"] == ["TP1", "TP5"]
    assert len(json.loads(pairs[3]["treatment_text"])["intervals"]) == 4

    batch = pairs.collate([pairs[0], pairs[3]])
    assert batch["pre_mri"].shape[:2] == (2, 1)
    assert batch["post_mri"].shape[:2] == (2, 1)
    assert batch["time_delta"].tolist() == [30.0, 120.0]
    assert batch["primary_survival_window"].tolist() == [False, False]


def test_interval_treatment_excludes_future_and_course_wide_information():
    trajectories = dataset()
    points = trajectories.trajectories[0].points
    points[1]["actions"]["drug"].extend(
        [
            {
                "agent": "future",
                "start_day": 70,
                "end_day": 80,
                "assigned_reason": "beyond_last_tp",
            },
            {
                "agent": "spanning",
                "start_day": 50,
                "end_day": 90,
                "cycle_length_days": 10,
                "num_cycles": 5,
            },
        ]
    )

    actions = json.loads(trajectories[0]["step_text"][0])["intervals"][0]["actions"]
    assert actions["drug"] == [
        {"agent": "d2", "start_offset_days": 1, "end_offset_days": 30},
        {
            "agent": "spanning",
            "cycle_length_days": 10,
            "start_offset_days": 20,
            "end_offset_days": 30,
        },
    ]


def test_interval_treatment_category_uses_interval_actions():
    text = json.dumps(
        {
            "intervals": [
                {
                    "actions": {
                        "radiation": [{}],
                        "chemotherapy": [{"agent": "Temozolomide"}],
                        "additional_2": [{"agent": "Avastin"}],
                    }
                }
            ]
        }
    )
    assert extract_treatment_category(text) == "BEV+RT+TMZ"


def test_collate_keeps_step_and_prefix_text_separate():
    trajectories = dataset()
    batch = trajectories.collate([trajectories[0], trajectories[1]])
    assert batch["mri"].shape[:2] == (2, 4)
    assert len(batch["step_text"]) == 3
    assert len(batch["prefix_text"]) == 3
    assert all(len(texts) == 2 for texts in batch["step_text"])
    assert batch["primary_survival_window"].tolist() == [True, False]


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
