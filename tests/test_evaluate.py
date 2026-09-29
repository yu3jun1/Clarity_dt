import pytest

from clarity_rrt_v3.evaluate import summarize_runs, uncertainty_metrics


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
    summary = summarize_runs(
        [
            {
                "recursive": recursive,
                "error_accumulation": 0.1,
                "representation_sanity": representation,
                "uncertainty": uncertainty,
            }
        ]
    )
    assert summary["H2_latent_disagreement_error_pearson"]["mean"] == 0.5
    assert summary["H3_survival_probability_disagreement_mean"]["mean"] == 0.05
    assert summary["observed_latent_variance"]["mean"] == 0.4
