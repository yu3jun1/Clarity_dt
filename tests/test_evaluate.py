from clarity_rrt.evaluate import summarize_runs, uncertainty_metrics


def test_uncertainty_reports_correlation_and_selective_error():
    rows = []
    for horizon in (1, 2, 3):
        for index in range(4):
            rows.append(
                {
                    "horizon": horizon,
                    "uncertainty": float(index),
                    "latent_mse": float(index + 1),
                }
            )
    result = uncertainty_metrics(rows, [0.5, 1.0])
    assert result["H3"]["variance_error_pearson"] == 1.0
    assert result["H1"]["selective_rollout"]["0.5"]["latent_mse"] == 1.5
    assert result["H1"]["selective_rollout"]["1"]["latent_mse"] == 2.5


def test_cross_seed_summary_keeps_uncertainty_metrics():
    uncertainty = {
        f"H{horizon}": {
            "variance_error_pearson": 0.5,
            "selective_rollout": {"0.5": {"retained": 2, "latent_mse": 0.25}},
        }
        for horizon in (1, 2, 3)
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
                "uncertainty": uncertainty,
            }
        ]
    )
    assert summary["H2_uncertainty_error_pearson"]["mean"] == 0.5
    assert summary["H3_selective_0.5_latent_mse"]["mean"] == 0.25
