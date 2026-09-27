from __future__ import annotations

import json
from pathlib import Path

from clarity_rrt.data import create_fixed_split, eligible_patient_summary, load_split


def _make_cohort(root: Path, patient_count: int = 12):
    mri_root = root / "mri"
    patients = {}
    for patient_index in range(patient_count):
        patient = f"PatientID_{patient_index:04d}"
        timeline = []
        for tp in range(1, 4):
            directory = mri_root / patient / f"Timepoint_{tp}"
            directory.mkdir(parents=True)
            for modality in ("t1c", "t2w", "t1n", "t2f"):
                (directory / f"{patient}_Timepoint_{tp}_brain_{modality}.nii.gz").touch()
            timeline.append(
                {
                    "tp_id": f"TP{tp}",
                    "mri_day": tp * 30,
                    "actions": {},
                    "survival": {
                        "survival_from_tp_days": 500 - tp * 20,
                        "event_indicator": patient_index % 2,
                    },
                }
            )
        patients[patient] = {"context_static": {}, "timeline": timeline}
    timeline_path = root / "timeline.json"
    timeline_path.write_text(json.dumps({"patients": patients}), encoding="utf-8")
    return timeline_path, mri_root


def test_split_is_deterministic_complete_and_patient_disjoint(tmp_path):
    timeline, mri_root = _make_cohort(tmp_path)
    first = create_fixed_split(timeline, mri_root, tmp_path / "split1.json", seed=7)
    second = create_fixed_split(timeline, mri_root, tmp_path / "split2.json", seed=7)
    assert first["split_sha256"] == second["split_sha256"]
    loaded = load_split(tmp_path / "split1.json")
    split_sets = [set(loaded["splits"][name]) for name in ("train", "validation", "test")]
    assert not (split_sets[0] & split_sets[1])
    assert not (split_sets[0] & split_sets[2])
    assert not (split_sets[1] & split_sets[2])
    assert set().union(*split_sets) == set(eligible_patient_summary(timeline, mri_root))


def test_incomplete_mri_timepoint_only_removes_affected_pairs(tmp_path):
    timeline, mri_root = _make_cohort(tmp_path, patient_count=3)
    missing = mri_root / "PatientID_0000" / "Timepoint_2" / "PatientID_0000_Timepoint_2_brain_t1c.nii.gz"
    missing.unlink()
    summary = eligible_patient_summary(timeline, mri_root)
    assert summary["PatientID_0000"]["pairs"] == 1
