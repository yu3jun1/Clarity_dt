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


PRIMARY_SURVIVAL_WINDOW_RULE = "earliest_eligible_four_stage_window_per_patient"


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


def _action_interval(
    action: Mapping[str, Any], start_day: int, end_day: int
) -> tuple[int, int] | None:
    if "interval_start_day" in action and "interval_end_day" in action:
        action_start = int(action["interval_start_day"])
        action_end = int(action["interval_end_day"])
    elif action.get("day_of_insertion") is not None:
        action_start = action_end = int(action["day_of_insertion"])
    elif action.get("start_day") is not None:
        action_start = int(action["start_day"])
        action_end = int(action.get("end_day") or end_day)
    else:
        return None

    clipped_start = max(action_start, start_day + 1)
    clipped_end = min(action_end, end_day)
    if clipped_start > clipped_end:
        return None
    return clipped_start, clipped_end


def _interval_action(
    action: Mapping[str, Any], start_day: int, end_day: int
) -> dict[str, Any] | None:
    interval = _action_interval(action, start_day, end_day)
    if interval is None:
        return None

    clipped_start, clipped_end = interval
    result = {
        key: action[key]
        for key in ("agent", "type", "cycle_length_days")
        if action.get(key) is not None
    }
    result["start_offset_days"] = clipped_start - start_day
    result["end_offset_days"] = clipped_end - start_day

    course_start = action.get("start_day")
    course_end = action.get("end_day")
    course_is_inside = (
        course_start is not None
        and course_end is not None
        and start_day < int(course_start) <= int(course_end) <= end_day
    )
    if course_is_inside:
        for key in ("dose", "num_cycles", "dose_gy", "fractions"):
            if action.get(key) is not None:
                result[key] = action[key]
    return result


def interval_actions(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    start_day = int(start["mri_day"])
    end_day = int(end["mri_day"])
    actions: dict[str, list[dict[str, Any]]] = {}
    for category, items in end.get("actions", {}).items():
        selected = [
            interval_action
            for action in items
            if (interval_action := _interval_action(action, start_day, end_day))
            is not None
        ]
        if selected:
            actions[category] = selected
    return actions


def treatment_text(points: Sequence[Mapping[str, Any]]) -> str:
    intervals = [
        {
            "start_tp": start["tp_id"],
            "end_tp": end["tp_id"],
            "actions": interval_actions(start, end),
        }
        for start, end in zip(points, points[1:])
    ]
    return json.dumps(
        {"intervals": intervals}, ensure_ascii=False, separators=(",", ":")
    )


def extract_treatment_category(text: str) -> str:
    payload = json.loads(text)
    categories: set[str] = set()
    for interval in payload["intervals"]:
        for category, items in interval["actions"].items():
            if category == "radiation":
                categories.add("RT")
            elif category in ("chemotherapy", "additional_1", "additional_2"):
                for item in items:
                    agent = str(item.get("agent", "")).lower()
                    if "temozolomide" in agent:
                        categories.add("TMZ")
                    elif "bevacizumab" in agent or "avastin" in agent:
                        categories.add("BEV")
                    elif agent:
                        categories.add("OTHER")
    return "+".join(sorted(categories)) if categories else "no_treatment"


class StagewiseTrajectoryDataset(Dataset):
    """All consecutive four-observation windows from the fixed patient split."""

    def __init__(
        self,
        patients: Mapping[str, Any],
        mri_loader: Any,
        format_clinical: Callable[[dict[str, Any]], str],
        patient_ids: Iterable[str],
    ) -> None:
        self.patients = patients
        self.mri_loader = mri_loader
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
        self._primary_window_keys = self.primary_window_keys()

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def unique_patient_count(self) -> int:
        return len({trajectory.patient_id for trajectory in self.trajectories})

    def primary_trajectories(self) -> tuple[Trajectory, ...]:
        first: dict[str, tuple[tuple[float, int], Trajectory]] = {}
        for trajectory in self.trajectories:
            start = trajectory.points[0]
            order = (float(start["mri_day"]), parse_timepoint(start["tp_id"]))
            current = first.get(trajectory.patient_id)
            if current is None or order < current[0]:
                first[trajectory.patient_id] = (order, trajectory)
        return tuple(first[patient_id][1] for patient_id in sorted(first))

    def primary_window_keys(self) -> set[tuple[str, str, str]]:
        return {
            (
                trajectory.patient_id,
                trajectory.points[0]["tp_id"],
                trajectory.points[-1]["tp_id"],
            )
            for trajectory in self.primary_trajectories()
        }

    def cohort_counts(self) -> dict[str, int]:
        return {
            "trajectory_count": len(self.trajectories),
            "unique_patient_count": self.unique_patient_count,
            "primary_survival_window_count": len(self.primary_trajectories()),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory = self.trajectories[index]
        points = trajectory.points
        ids = [mri_id(trajectory.patient_id, point["tp_id"]) for point in points]
        days = torch.tensor([float(point["mri_day"]) for point in points])
        survival = [point["survival"] for point in points[1:]]
        prefixes = [
            treatment_text(points[: horizon + 1])
            for horizon in range(1, 4)
        ]
        return {
            "mri": torch.stack([self.mri_loader.load(identifier) for identifier in ids]),
            "step_text": [
                treatment_text(points[step : step + 2])
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
            "primary_survival_window": (
                trajectory.patient_id,
                points[0]["tp_id"],
                points[-1]["tp_id"],
            )
            in self._primary_window_keys,
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
            "primary_survival_window": torch.tensor(
                [sample["primary_survival_window"] for sample in samples],
                dtype=torch.bool,
            ),
        }
