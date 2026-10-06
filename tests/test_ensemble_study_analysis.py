from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from clarity_ensemble_study.analysis import analyze, summarize, write_report


def make_rows(errors, uncertainty, member_count=2):
    rows = []
    for error, value in zip(errors, uncertainty):
        for horizon in (1, 2, 3):
            rows.append({
                "horizon": horizon, "latent_mse": float(error ** 2),
                "cosine_similarity": 0.8, "error_l2": float(error),
                "latent_uncertainty": float(value),
                "survival_uncertainty": float(value / 10),
                "member_latent_mse": [float(error ** 2 + member) for member in range(member_count)],
                "member_cosine_similarity": [0.7 + member / 10 for member in range(member_count)],
            })
    return rows


def predictions(member_count, sample_count):
    values = np.arange(sample_count * 3 * 2, dtype=float).reshape(sample_count, 3, 1, 2)
    return np.stack([values * (member + 1) + member for member in range(member_count)])


def test_correlations_use_l2_error_not_mse():
    rows = make_rows(np.array([1.0, 2.0, 4.0, 8.0]), [1, 2, 4, 8])
    result = analyze(rows, predictions(2, 4))["H1"]
    assert result["latent_error_correlation"]["pearson"] == pytest.approx(1.0)
    assert result["latent_error_correlation"]["spearman"] == pytest.approx(1.0)
    assert result["survival_error_correlation"]["pearson"] == pytest.approx(1.0)
    mse_correlation = np.corrcoef([1, 2, 4, 8], [1, 4, 16, 64])[0, 1]
    assert result["latent_error_correlation"]["pearson"] != pytest.approx(mse_correlation)


def test_single_member_uncertainty_is_mathematically_undefined():
    rows = make_rows(np.array([1.0, 2.0, 3.0]), [0, 0, 0], member_count=1)
    result = analyze(rows, predictions(1, 3))["H3"]
    assert result["latent_error_correlation"] == {"pearson": None, "spearman": None}
    assert result["survival_error_correlation"] == {"pearson": None, "spearman": None}
    assert result["risk_coverage"] == []
    assert result["member_pairs"] == []
    assert result["averaging_gain_mse"] == 0.0


def test_coverage_uses_ceiling_and_stable_ties():
    rows = make_rows(np.array([1.0, 10.0, 2.0]), [0, 0, 1])
    result = analyze(rows, predictions(2, 3))["H2"]
    points = result["risk_coverage"]
    assert [point["coverage"] for point in points] == [1.0, 0.8, 0.6, 0.4]
    assert [point["samples"] for point in points] == [3, 3, 2, 2]
    assert points[2]["latent_mse"] == pytest.approx((1 + 100) / 2)
    assert points[3]["latent_mse"] == pytest.approx((1 + 100) / 2)
    tied_rows = make_rows(np.array([1.0, 10.0, 2.0, 3.0, 4.0]), [0, 0, 0, 1, 2])
    tied_point = analyze(tied_rows, predictions(2, 5))["H1"]["risk_coverage"][-1]
    assert tied_point["samples"] == 2
    assert tied_point["latent_mse"] == pytest.approx((1 + 100) / 2)


def test_member_diversity_uses_raw_predictions_not_member_errors():
    rows = make_rows(np.array([1.0, 2.0, 3.0]), [1, 2, 3])
    member_states = predictions(2, 3)
    member_states[1] = -member_states[0]
    result = analyze(rows, member_states)["H1"]
    assert result["member_pairs"] == [
        {"member_i": 0, "member_j": 1, "prediction_pearson": pytest.approx(-1.0)},
    ]
    assert result["members"][0]["latent_mse"] == pytest.approx(14 / 3)
    assert result["members"][1]["latent_mse"] == pytest.approx(14 / 3 + 1)
    assert result["averaging_gain_mse"] == pytest.approx(0.5)


def make_run(size, seed, value):
    errors = np.array([1.0, 2.0, 4.0])
    rows = make_rows(errors, [0, 0, 0] if size == 1 else errors, size)
    reliability = analyze(rows, predictions(size, len(errors)))
    return {
        "ensemble_size": size, "seed": seed,
        "recursive": {f"H{horizon}": {"latent_mse": value, "cosine_similarity": 0.8}
                      for horizon in (1, 2, 3)},
        "reliability": reliability,
    }, rows


def test_three_seed_stability_is_sample_standard_deviation():
    runs = [make_run(3, seed, value)[0] for seed, value in zip((42, 43, 44), (1.0, 2.0, 4.0))]
    result = summarize(runs)
    assert result["performance"]["H3"]["latent_mse"]["mean"] == pytest.approx(7 / 3)
    assert result["h3_stability_std"] == pytest.approx(np.std([1, 2, 4], ddof=1))
    assert result["h3_stability_std"] != pytest.approx(np.std([1, 2, 4], ddof=0))


def test_summary_averages_within_seed_correlations_without_pooling():
    runs = [make_run(3, seed, 1.0)[0] for seed in (42, 43, 44)]
    values = (0.2, 0.4, -0.3)
    for run, value in zip(runs, values):
        run["reliability"]["H1"]["latent_error_correlation"]["pearson"] = value
    result = summarize(runs)
    assert result["reliability"]["H1"]["latent_error_correlation"]["pearson"] == pytest.approx(sum(values) / 3)


def test_reports_write_only_the_requested_new_root(tmp_path):
    for size in (1, 3):
        for seed in (42, 43, 44):
            directory = tmp_path / "runs" / f"M{size}_seed{seed}"
            directory.mkdir(parents=True)
            run, rows = make_run(size, seed, seed / 100)
            (directory / "metrics.json").write_text(json.dumps(run))
            if size == 3:
                with (directory / "recursive_predictions.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("horizon", "latent_uncertainty", "error_l2"))
                    writer.writeheader()
                    writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    summary = write_report(tmp_path, "step1", (1, 3))
    destination = tmp_path / "reports" / "step1"
    saved = json.loads((destination / "summary.json").read_text())
    assert saved == summary
    assert saved["models"]["M1"]["reliability"]["H1"]["latent_error_correlation"]["pearson"] is None
    assert saved["correlation_aggregation"] == "mean of within-seed correlations; no cross-seed pooling"
    assert (destination / "report.md").is_file()
    assert (destination / "risk_coverage.png").is_file()
    assert all((destination / f"uncertainty_error_M3_seed{seed}.png").is_file() for seed in (42, 43, 44))
