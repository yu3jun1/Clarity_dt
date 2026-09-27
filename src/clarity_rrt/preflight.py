"""Fail-fast audit for code revision, paths, split integrity, and dependencies."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .data import eligible_patient_summary, load_split
from .train import load_config


def run_preflight(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    report: dict[str, Any] = {"ok": True, "checks": {}}

    def check(name: str, ok: bool, detail: Any) -> None:
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        report["ok"] = report["ok"] and bool(ok)

    upstream = Path(config["upstream_root"])
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        commit = str(error)
    check("upstream_commit", commit == config["upstream_commit"], commit)
    for label, path in (
        ("timeline_json", config["data"]["timeline_json"]),
        ("mri_data_dir", config["data"]["mri_data_dir"]),
        ("brainiac_ckpt", config["model"]["brainiac_ckpt"]),
        ("text_encoder", config["model"]["text_encoder_name"]),
        ("split_file", config["data"]["split_file"]),
    ):
        resolved = Path(path)
        check(label, resolved.exists(), str(resolved))
    dependencies = {}
    for module in ("torch", "numpy", "pandas", "yaml", "nibabel", "transformers", "peft", "monai", "sksurv"):
        try:
            loaded = importlib.import_module(module)
            dependencies[module] = getattr(loaded, "__version__", "installed")
        except Exception as error:  # import failures can include binary ABI problems
            dependencies[module] = f"ERROR: {type(error).__name__}: {error}"
    check(
        "dependencies",
        all(not str(value).startswith("ERROR") for value in dependencies.values()),
        dependencies,
    )
    if Path(config["data"]["split_file"]).is_file():
        split = load_split(config["data"]["split_file"])
        eligible = eligible_patient_summary(
            config["data"]["timeline_json"], config["data"]["mri_data_dir"]
        )
        split_ids = set().union(*(set(ids) for ids in split["splits"].values()))
        check(
            "split_cohort",
            split_ids == set(eligible),
            {
                "split_patients": len(split_ids),
                "eligible_patients": len(eligible),
                "split_sha256": split.get("split_sha256"),
            },
        )
    check(
        "cuda",
        dependencies.get("torch", "").startswith("ERROR") is False
        and __import__("torch").cuda.is_available(),
        __import__("torch").cuda.device_count() if not dependencies.get("torch", "").startswith("ERROR") else 0,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output", default="runs/preflight.json")
    args = parser.parse_args(argv)
    report = run_preflight(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
