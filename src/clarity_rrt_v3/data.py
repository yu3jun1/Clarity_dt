"""Four-stage trajectories for the stage-wise recursive experiment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_timepoint(value: str) -> int:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else -1


def mri_id(patient_id: str, timepoint: str) -> str:
    return f"{patient_id}_Timepoint_{parse_timepoint(timepoint)}"


def load_timeline(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class CachedMRIVolumeLoader:
    """Load raw float32 MRI volumes from the configured shared-memory cache."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        with (self.cache_dir / "manifest.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.entries = json.load(handle)["entries"]

    def has(self, identifier: str) -> bool:
        entry = self.entries.get(identifier)
        return entry is not None and (self.cache_dir / entry["file"]).is_file()

    def load(self, identifier: str) -> torch.Tensor:
        path = self.cache_dir / self.entries[identifier]["file"]
        return torch.from_numpy(np.load(path, mmap_mode="c", allow_pickle=False))


def valid_survival(point: Mapping[str, Any]) -> bool:
    survival = point.get("survival", {})
    time = float(survival.get("survival_from_tp_days", -1))
    event = int(survival.get("event_indicator", 0))
    return time >= 0 and not (event == 0 and time <= 0)


@dataclass(frozen=True)
class Trajectory:
    patient_id: str
    points: tuple[Mapping[str, Any], ...]


def treatment_text(
    pack: Callable[[dict[str, Any]], str],
    points: Sequence[Mapping[str, Any]],
) -> str:
    item: dict[str, Any] = {
        "pre_tp": points[0]["tp_id"],
        "post_tp": points[-1]["tp_id"],
        "pre_actions": points[0].get("actions", {}),
        "post_actions": points[-1].get("actions", {}),
    }
    if len(points) > 2:
        item["between_actions"] = [point.get("actions", {}) for point in points[1:-1]]
    return pack(item)


class StagewiseTrajectoryDataset(Dataset):
    """All consecutive four-observation windows from the fixed patient split."""

    def __init__(
        self,
        patients: Mapping[str, Any],
        mri_loader: Any,
        pack_treatment: Callable[[dict[str, Any]], str],
        format_clinical: Callable[[dict[str, Any]], str],
        patient_ids: Iterable[str],
    ) -> None:
        self.patients = patients
        self.mri_loader = mri_loader
        self.pack_treatment = pack_treatment
        self.format_clinical = format_clinical
        selected = set(map(str, patient_ids))
        self.trajectories: list[Trajectory] = []
        for patient_id in sorted(selected):
            patient = patients["patients"].get(patient_id, {})
            timeline = sorted(
                patient.get("timeline", []),
                key=lambda point: parse_timepoint(point.get("tp_id", "")),
            )
            for start in range(len(timeline) - 3):
                points = tuple(timeline[start : start + 4])
                days = [float(point.get("mri_day", 0)) for point in points]
                ids = [mri_id(patient_id, point["tp_id"]) for point in points]
                if not all(right > left for left, right in zip(days, days[1:])):
                    continue
                if not all(mri_loader.has(identifier) for identifier in ids):
                    continue
                if not all(valid_survival(point) for point in points[1:]):
                    continue
                self.trajectories.append(Trajectory(patient_id, points))

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory = self.trajectories[index]
        points = trajectory.points
        ids = [mri_id(trajectory.patient_id, point["tp_id"]) for point in points]
        days = torch.tensor([float(point["mri_day"]) for point in points])
        survival = [point["survival"] for point in points[1:]]
        prefixes = [
            treatment_text(self.pack_treatment, points[: horizon + 1])
            for horizon in range(1, 4)
        ]
        return {
            "mri": torch.stack([self.mri_loader.load(identifier) for identifier in ids]),
            "step_text": [
                treatment_text(self.pack_treatment, points[step : step + 2])
                for step in range(3)
            ],
            "prefix_text": prefixes,
            "full_text": prefixes[-1],
            "clinical_text": self.format_clinical(
                self.patients["patients"][trajectory.patient_id].get("context_static", {})
            ),
            "deltas": days[1:] - days[:-1],
            "survival_time": torch.tensor(
                [float(item["survival_from_tp_days"]) for item in survival]
            ),
            "event": torch.tensor([float(item["event_indicator"]) for item in survival]),
            "patient": trajectory.patient_id,
            "timepoints": [point["tp_id"] for point in points],
        }

    @staticmethod
    def collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "mri": torch.stack([sample["mri"] for sample in samples]),
            "step_text": [
                [sample["step_text"][step] for sample in samples] for step in range(3)
            ],
            "prefix_text": [
                [sample["prefix_text"][horizon] for sample in samples]
                for horizon in range(3)
            ],
            "full_text": [sample["full_text"] for sample in samples],
            "clinical_text": [sample["clinical_text"] for sample in samples],
            "deltas": torch.stack([sample["deltas"] for sample in samples]),
            "survival_time": torch.stack([sample["survival_time"] for sample in samples]),
            "event": torch.stack([sample["event"] for sample in samples]),
            "patient": [sample["patient"] for sample in samples],
            "timepoints": [sample["timepoints"] for sample in samples],
        }
