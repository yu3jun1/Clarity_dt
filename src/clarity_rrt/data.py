"""Fixed patient splits and chain-aware wrappers around CLARITY's dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODALITIES = ("t1c", "t2w", "t1n", "t2f")


def parse_tp_num(value: str) -> int:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else -1


def timepoint_id(patient_id: str, tp: str) -> str:
    return f"{patient_id}_Timepoint_{parse_tp_num(tp)}"


def mri_paths(mri_root: str | Path, patient_id: str, tp: str) -> list[Path]:
    number = parse_tp_num(tp)
    directory = Path(mri_root) / patient_id / f"Timepoint_{number}"
    stem = f"{patient_id}_Timepoint_{number}_brain_"
    return [directory / f"{stem}{modality}.nii.gz" for modality in MODALITIES]


def has_complete_mri(mri_root: str | Path, patient_id: str, tp: str) -> bool:
    return all(path.is_file() for path in mri_paths(mri_root, patient_id, tp))


def load_timeline(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("patients"), dict):
        raise ValueError("timeline JSON must contain a 'patients' object")
    return payload


def _valid_endpoint(tp: Mapping[str, Any]) -> bool:
    survival = tp.get("survival") or {}
    try:
        time = float(survival.get("survival_from_tp_days", -1.0))
        event = int(survival.get("event_indicator", 0))
    except (TypeError, ValueError):
        return False
    return time >= 0 and not (event == 0 and time <= 0)


def eligible_patient_summary(
    timeline_json: str | Path,
    mri_root: str | Path,
) -> dict[str, dict[str, int]]:
    """Return patients with at least one official-compatible MRI pair.

    The filtering mirrors the upstream all-pairs dataset and additionally enforces
    that all four raw MRI modalities exist at both endpoints.
    """
    patients = load_timeline(timeline_json)["patients"]
    summary: dict[str, dict[str, int]] = {}
    for patient_id, patient in patients.items():
        timeline = sorted(
            patient.get("timeline", []), key=lambda item: parse_tp_num(item.get("tp_id", ""))
        )
        pair_count = 0
        event_pairs = 0
        for i, pre in enumerate(timeline[:-1]):
            if not has_complete_mri(mri_root, patient_id, pre.get("tp_id", "")):
                continue
            for post in timeline[i + 1 :]:
                if not has_complete_mri(mri_root, patient_id, post.get("tp_id", "")):
                    continue
                if float(post.get("mri_day", 0)) <= float(pre.get("mri_day", 0)):
                    continue
                if not _valid_endpoint(post):
                    continue
                pair_count += 1
                event_pairs += int((post.get("survival") or {}).get("event_indicator", 0) == 1)
        if pair_count:
            summary[str(patient_id)] = {
                "pairs": pair_count,
                "event_pairs": event_pairs,
                "timepoints": len(timeline),
            }
    return summary


def _allocation_counts(total: int, ratios: Sequence[float]) -> list[int]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split_ratios must contain three positive values")
    norm = [value / sum(ratios) for value in ratios]
    exact = [total * value for value in norm]
    counts = [math.floor(value) for value in exact]
    for index in sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    return counts


def create_fixed_split(
    timeline_json: str | Path,
    mri_root: str | Path,
    output: str | Path,
    seed: int = 20260927,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    upstream_commit: str = "dadb82241a24f5ec5e4e4dc994e3116fd4a9da04",
) -> dict[str, Any]:
    summary = eligible_patient_summary(timeline_json, mri_root)
    patient_ids = sorted(summary)
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    counts = _allocation_counts(len(patient_ids), ratios)
    train_end = counts[0]
    val_end = train_end + counts[1]
    splits = {
        "train": sorted(patient_ids[:train_end]),
        "validation": sorted(patient_ids[train_end:val_end]),
        "test": sorted(patient_ids[val_end:]),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "ratios": list(ratios),
        "upstream_commit": upstream_commit,
        "timeline_json": str(Path(timeline_json).resolve()),
        "mri_data_dir": str(Path(mri_root).resolve()),
        "eligibility": "official all-pairs labels + four raw MRI modalities at both endpoints",
        "splits": splits,
        "counts": {},
    }
    for name, ids in splits.items():
        payload["counts"][name] = {
            "patients": len(ids),
            "pairs": sum(summary[pid]["pairs"] for pid in ids),
            "event_pairs": sum(summary[pid]["event_pairs"] for pid in ids),
        }
    canonical = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    payload["split_sha256"] = hashlib.sha256(canonical).hexdigest()
    validate_split_payload(payload, eligible_patients=set(summary))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_split(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_split_payload(payload)
    return payload


def validate_split_payload(
    payload: Mapping[str, Any], eligible_patients: set[str] | None = None
) -> None:
    splits = payload.get("splits", {})
    expected = ("train", "validation", "test")
    if any(name not in splits for name in expected):
        raise ValueError("split file must define train, validation, and test")
    sets = {name: set(map(str, splits[name])) for name in expected}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"patient leakage between {left} and {right}: {sorted(overlap)[:3]}")
    union = set().union(*sets.values())
    if eligible_patients is not None and union != eligible_patients:
        missing = eligible_patients - union
        extra = union - eligible_patients
        raise ValueError(f"split does not cover eligible cohort; missing={len(missing)}, extra={len(extra)}")


@dataclass(frozen=True)
class ChainSpec:
    patient_id: str
    timepoints: tuple[str, ...]
    mri_ids: tuple[str, ...]
    mri_days: tuple[float, ...]
    drugs_text: tuple[str, ...]
    clinical_text: str

    @property
    def horizon(self) -> int:
        return len(self.timepoints) - 1

    @property
    def key(self) -> str:
        return f"{self.patient_id}:{'->'.join(self.timepoints)}"


class FixedSplitPairDataset:
    """Filter upstream all-pairs samples and attach valid consecutive chains."""

    def __init__(
        self,
        base_dataset: Any,
        patient_ids: Iterable[str],
        horizons: Sequence[int] = (2, 3),
        require_mri: bool = True,
    ) -> None:
        self.base = base_dataset
        self.patient_ids = set(map(str, patient_ids))
        self.horizons = set(map(int, horizons))
        self.require_mri = require_mri
        self._base_indices = [
            index
            for index, item in enumerate(base_dataset.index)
            if str(item["pid"]) in self.patient_ids and self._pair_available(item)
        ]
        self._chain_by_base_index = self._build_chains()

    def _pair_available(self, item: Mapping[str, Any]) -> bool:
        if not self.require_mri or self.base.mri_loader is None:
            return True
        return self.base.mri_loader.has(item["pre_id"]) and self.base.mri_loader.has(item["post_id"])

    def _build_chains(self) -> dict[int, ChainSpec]:
        patients = self.base.patients["patients"]
        result: dict[int, ChainSpec] = {}
        for base_index in self._base_indices:
            item = self.base.index[base_index]
            horizon = int(item["j"] - item["i"])
            if horizon not in self.horizons:
                continue
            raw_timeline = sorted(
                patients[item["pid"]].get("timeline", []),
                key=lambda tp: parse_tp_num(tp.get("tp_id", "")),
            )
            points = raw_timeline[item["i"] : item["j"] + 1]
            if len(points) != horizon + 1:
                continue
            days = tuple(float(point.get("mri_day", 0)) for point in points)
            if any(right <= left for left, right in zip(days, days[1:])):
                continue
            ids = tuple(timepoint_id(item["pid"], point["tp_id"]) for point in points)
            if self.require_mri and self.base.mri_loader is not None:
                if any(not self.base.mri_loader.has(mri_id) for mri_id in ids):
                    continue
            step_text: list[str] = []
            for pre, post in zip(points, points[1:]):
                step_text.append(
                    self.base._pack_drugs_json(
                        {
                            "pre_tp": pre["tp_id"],
                            "post_tp": post["tp_id"],
                            "pre_actions": pre.get("actions", {}),
                            "post_actions": post.get("actions", {}),
                        }
                    )
                )
            clinical = self.base._format_clinical_text(
                patients[item["pid"]].get("context_static", {})
            )
            result[base_index] = ChainSpec(
                patient_id=item["pid"],
                timepoints=tuple(point["tp_id"] for point in points),
                mri_ids=ids,
                mri_days=days,
                drugs_text=tuple(step_text),
                clinical_text=clinical,
            )
        return result

    def __len__(self) -> int:
        return len(self._base_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = self._base_indices[index]
        sample = self.base[base_index]
        sample["chain_spec"] = self._chain_by_base_index.get(base_index)
        return sample

    @property
    def chain_count(self) -> int:
        return len(self._chain_by_base_index)

    def collate_fn(self, max_chains: int = 4):
        def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
            all_specs = [sample.pop("chain_spec", None) for sample in samples]
            specs = sorted(
                (spec for spec in all_specs if spec is not None),
                key=lambda spec: spec.key,
            )[:max_chains]
            batch = self.base.collate_fn(samples)
            # Keep collate lightweight. Large chain MRI volumes are materialized
            # only by the trainer when RRT is active (epoch >= start_epoch).
            batch["chain_specs"] = specs
            return batch

        return collate

    def materialize_chain_groups(
        self, specs: Sequence[ChainSpec]
    ) -> list[dict[str, Any]]:
        """Load selected chain MRI volumes lazily in the training process."""
        torch = __import__("torch")
        groups: list[dict[str, Any]] = []
        by_horizon: dict[int, list[ChainSpec]] = {}
        for spec in specs:
            by_horizon.setdefault(spec.horizon, []).append(spec)
        for horizon in sorted(by_horizon):
            horizon_specs = by_horizon[horizon]
            mri = torch.stack(
                [
                    torch.stack(
                        [self.base.mri_loader.load(mri_id) for mri_id in spec.mri_ids]
                    )
                    for spec in horizon_specs
                ]
            )
            groups.append(
                {
                    "horizon": horizon,
                    "keys": [spec.key for spec in horizon_specs],
                    "mri": mri,
                    "delta": torch.tensor(
                        [
                            [
                                right - left
                                for left, right in zip(
                                    spec.mri_days, spec.mri_days[1:]
                                )
                            ]
                            for spec in horizon_specs
                        ],
                        dtype=torch.float32,
                    ),
                    "drugs_text_steps": [
                        [spec.drugs_text[step] for spec in horizon_specs]
                        for step in range(horizon)
                    ],
                    "clinical_text": [spec.clinical_text for spec in horizon_specs],
                }
            )
        return groups


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-split", help="create the one immutable patient split")
    create.add_argument("--timeline-json", required=True)
    create.add_argument("--mri-data-dir", required=True)
    create.add_argument("--output", default="data/splits.json")
    create.add_argument("--seed", type=int, default=20260927)
    create.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    audit = sub.add_parser("audit", help="audit an existing split against current data")
    audit.add_argument("--split", default="data/splits.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "create-split":
        payload = create_fixed_split(
            args.timeline_json, args.mri_data_dir, args.output, args.seed, args.ratios
        )
        print(json.dumps({"split_sha256": payload["split_sha256"], "counts": payload["counts"]}, indent=2))
    else:
        payload = load_split(args.split)
        all_ids = [pid for ids in payload["splits"].values() for pid in ids]
        print(json.dumps({"split_sha256": payload.get("split_sha256"), "patients": len(all_ids), "counts": payload.get("counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
