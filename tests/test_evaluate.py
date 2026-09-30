import pytest

from clarity_rrt_v3.evaluate import (
    member_diagnostics,
    summarize_runs,
    uncertainty_metrics,
)


def test_uncertainty_reports_v3_secondary_metrics():
    rows = []
    for horizon in (1, 2, 3):
        for index in range(4):
            rows.append(
                {
                    "horizon": horizon,
                    "latent_disagreement": float(index),
                    "latent_mse": float(index + 1),
                    "survival_probability_std": float(index) / 10.0,
                    "primary_survival_window": int(index < 2),
                }
            )
    result = uncertainty_metrics(rows)
    assert result["H3"]["latent_disagreement_error_pearson"] == 1.0
    assert result["H1"]["latent_disagreement_mean"] == 1.5
    assert result["H2"]["survival_probability_disagreement_mean"] == pytest.approx(0.05)


def test_member_diagnostics_reports_each_member_and_horizon():
    rows = []
    for horizon in (1, 2, 3):
        rows.extend(
            [
                {
                    "horizon": horizon,
                    "member_latent_mse": [1.0, 3.0, 5.0],
                    "member_cosine_similarity": [0.9, 0.7, 0.5],
                },
                {
                    "horizon": horizon,
                    "member_latent_mse": [3.0, 5.0, 7.0],
                    "member_cosine_similarity": [0.7, 0.5, 0.3],
                },
            ]
        )

    result = member_diagnostics(rows)

    assert result["H1"][0] == {
        "member_index": 0,
        "latent_mse": 2.0,
        "cosine_similarity": pytest.approx(0.8),
    }
    assert result["H3"][2] == {
        "member_index": 2,
        "latent_mse": 6.0,
        "cosine_similarity": pytest.approx(0.4),
    }


def test_cross_seed_summary_keeps_secondary_and_representation_metrics():
    uncertainty = {
        f"H{horizon}": {
            "latent_disagreement_mean": 0.25,
            "latent_disagreement_error_pearson": 0.5,
            "survival_probability_disagreement_mean": 0.05,
        }
        for horizon in (1, 2, 3)
    }
    representation = {
        "observed_latent_variance": 0.4,
        "observed_transition_l2_mean": 0.3,
        "encoder_trainable_parameter_rms_delta": 0.02,
    }
    recursive = {
        f"H{horizon}": {
            "latent_mse": 1.0,
            "cosine_similarity": 0.8,
            "c_index": 0.7,
            "brier365": 0.2,
        }
        for horizon in (1, 2, 3)
    }
    members = {
        f"H{horizon}": [
            {
                "member_index": member,
                "latent_mse": float(member + horizon),
                "cosine_similarity": 0.9 - 0.1 * member,
            }
            for member in range(3)
        ]
        for horizon in (1, 2, 3)
    }
    summary = summarize_runs(
        [
            {
                "optimizer_steps": 2400,
                "warmup_steps": 240,
                "samples_seen": 35520,
                "effective_dataset_passes": 480.0,
                "checkpoint_optimizer_step": 1200,
                "recursive": recursive,
                "error_accumulation": 0.1,
                "representation_sanity": representation,
                "uncertainty": uncertainty,
                "member_diagnostics": members,
            }
        ]
    )
    assert summary["H2_latent_disagreement_error_pearson"]["mean"] == 0.5
    assert summary["H3_survival_probability_disagreement_mean"]["mean"] == 0.05
    assert summary["observed_latent_variance"]["mean"] == 0.4
    assert summary["optimizer_steps"]["mean"] == 2400
    assert summary["H3_member_2_latent_mse"]["mean"] == 5.0
