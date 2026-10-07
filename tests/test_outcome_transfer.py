import copy
import json

import numpy as np
import pytest
import torch
from sksurv.metrics import brier_score, concordance_index_censored
from sksurv.util import Surv

from clarity_outcome_transfer import analysis, prediction
from clarity_rrt_v3.evaluate import recursive_predictions as native_predictions
from clarity_rrt_v3.model import StagewiseDynamics


REFERENCE = (np.array([100., 250., 500., 700., 900.]), np.array([1, 0, 1, 0, 1], dtype=bool))


class Predictor(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, initial, condition, delta):
        return initial + self.increment


class ToyModel(torch.nn.Module):
    rollout = StagewiseDynamics.rollout
    survival = StagewiseDynamics.survival

    def __init__(self, members=3):
        super().__init__()
        self.ensemble_size = members
        self.predictors = torch.nn.ModuleList([Predictor(i + 1) for i in range(members)])
        self.mri_encoder = torch.nn.Identity()

    def encode_conditions(self, text, clinical):
        return torch.zeros(len(clinical), 3, 1)

    def survival_module(self, initial, post, condition):
        value = post.flatten(1).mean(dim=1, keepdim=True)
        return value.square(), value / 8


def batch():
    initial = torch.arange(8.).reshape(2, 2, 2)
    return {
        "mri": torch.stack([initial + 10 * step for step in range(4)], dim=1),
        "deltas": torch.ones(2, 3), "step_text": [["a", "b"]] * 3,
        "prefix_text": [["a", "b"]] * 3, "clinical_text": ["a", "b"],
        "patient": ["P1", "P2"], "timepoints": [["t0", "t1", "t2", "t3"]] * 2,
        "primary_survival_window": torch.tensor([1, 1]),
        "survival_time": torch.tensor([[200., 150., 100.], [800., 700., 600.]]),
        "event": torch.tensor([[1., 1., 1.], [0., 0., 0.]]),
    }


def concordance(risk, time, event):
    return concordance_index_censored(np.asarray(event, dtype=bool), np.asarray(time), np.asarray(risk))[0]


@pytest.mark.parametrize("members", [1, 3])
def test_native_prediction_metrics_unchanged_and_true_latent_never_enters_rollout(members):
    data = batch()
    rows, _ = prediction.recursive_predictions(ToyModel(members), [data], "cpu")
    native, _ = native_predictions(ToyModel(members), [data], "cpu")
    native = {(r["patient"], r["horizon"]): r for r in native}
    for row in rows:
        original = native[row["patient"], row["horizon"]]
        for key in ("latent_mse", "cosine_similarity", "risk", "survival365"):
            assert row[key] == pytest.approx(original[key])
    changed = copy.deepcopy(data)
    changed["mri"][:, 1:] += 100
    altered, _ = prediction.recursive_predictions(ToyModel(members), [changed], "cpu")
    for first, second in zip(rows, altered):
        assert first["risk"] == second["risk"]
        assert first["survival365"] == second["survival365"]
        assert first["true_survival365"] != second["true_survival365"]


def test_survival_member_averaging_is_distinct_from_mean_latent_control():
    rows, _ = prediction.recursive_predictions(ToyModel(), [batch()], "cpu")
    selected = next(r for r in rows if r["patient"] == "P1" and r["horizon"] == 3)
    values = torch.tensor([4.5, 7.5, 10.5])
    assert selected["survival365"] == pytest.approx(float((values / 8).sigmoid().mean()))
    assert selected["mean_latent_survival365"] == pytest.approx(float((values.mean() / 8).sigmoid()))
    assert selected["survival365"] != pytest.approx(selected["mean_latent_survival365"])
    assert selected["latent_disagreement"] == pytest.approx(24.)
    assert selected["survival_disagreement"] == pytest.approx(float((values / 8).sigmoid().var(unbiased=False)))


def test_ipcw_contributions_match_sksurv_and_censoring_boundary():
    times, events = np.array([120., 200., 365., 365., 600.]), np.array([1, 0, 1, 0, 0], dtype=bool)
    probabilities = np.array([.2, .9, .4, .1, .8])
    rows = [{"survival_time": t, "event": e, "survival365": p} for t, e, p in zip(times, events, probabilities)]
    known, label, squared, contribution = analysis.outcome_errors(REFERENCE, rows, "survival365")
    np.testing.assert_array_equal(known, [True, False, True, False, True])
    np.testing.assert_array_equal(label[known], [0., 0., 1.])
    np.testing.assert_allclose(squared[known], [.04, .16, .04])
    assert contribution[1] == contribution[3] == 0
    _, expected = brier_score(Surv.from_arrays(REFERENCE[1], REFERENCE[0]),
                             Surv.from_arrays(events, times), probabilities[:, None], [365.])
    assert contribution.mean() == pytest.approx(expected[0], abs=1e-12)


def test_patient_error_averages_windows_and_masks_early_censoring():
    rows, _ = prediction.recursive_predictions(ToyModel(), [batch()], "cpu")
    for row in rows:
        if row["patient"] == "P1":
            row["event"] = 0
        else:
            row["event"] = 1
    extra = [{**r, "start": "t1", "window_end": "t4", "primary_survival_window": 0,
              "latent_mse": 0.0} for r in rows if r["patient"] == "P1"]
    result = analysis.analyze_run(rows + extra, REFERENCE, concordance=lambda *args: .5)
    assert result["patient_count"] == 2
    assert result["known365_count"] == 1
    first = result["patients"][0]
    original = next(r for r in rows if r["patient"] == "P1" and r["horizon"] == 3)
    assert first["H3_latent_mse"] == original["latent_mse"] / 2
    assert first["predicted_latent_squared_error365"] is None
    assert first["predicted_latent_brier_contribution"] == 0
    assert result["disagreement_vs_survival_error"]["latent_disagreement"] == {
        "n": 1, "pearson": None, "spearman": None,
    }


def test_pairing_uses_patient_identity_not_row_order_and_rejects_unmatched_labels():
    rows, _ = prediction.recursive_predictions(ToyModel(), [batch()], "cpu")
    a = analysis.analyze_run(rows, REFERENCE, concordance)["patients"]
    c = copy.deepcopy(a[::-1])
    for row in c:
        row["H3_latent_mse"] -= 1
    paired = analysis.pair_patients(a, c, 42)
    assert [r["patient"] for r in paired] == ["P1", "P2"]
    assert all(r["C_minus_A_H3_latent_mse"] == -1 for r in paired)
    c[0]["end"] = "another_endpoint"
    with pytest.raises(AssertionError, match="Unmatched outcome"):
        analysis.pair_patients(a, c, 42)


def test_report_keeps_seed_matched_survival_differences_and_writes_artifacts(tmp_path):
    for variant, members in (("A", 1), ("C", 3)):
        for seed in (42, 43, 44):
            directory = tmp_path / "runs" / f"{variant}_seed{seed}"
            directory.mkdir(parents=True)
            rows, _ = prediction.recursive_predictions(ToyModel(members), [batch()], "cpu")
            transfer = analysis.analyze_run(rows, REFERENCE, concordance)
            metrics = {"recursive": {f"H{h}": {
                "latent_mse": float(np.mean([r["latent_mse"] for r in rows if r["horizon"] == h])),
                **transfer["H3"]["predicted_latent"],
            } for h in (1, 2, 3)}, "outcome_transfer": transfer}
            analysis.write_json(directory / "metrics.json", metrics)
            analysis.write_csv(directory / "recursive_predictions.csv", rows)
    report = analysis.write_report(tmp_path, [42, 43, 44])
    assert len(report["seed_matched_C_minus_A"]) == 3
    for path in ("summary.json", "report.md", "patient_paired_errors.csv", "C_disagreement_survival_error.png"):
        assert (tmp_path / "reports" / path).stat().st_size > 0
    saved = json.loads((tmp_path / "reports/summary.json").read_text())
    assert saved == report
    assert len((tmp_path / "reports/patient_paired_errors.csv").read_text().splitlines()) == 7


def test_window_pairing_rejects_mismatched_nonprimary_windows():
    rows, _ = prediction.recursive_predictions(ToyModel(), [batch()], "cpu")
    changed = copy.deepcopy(rows)
    changed[0]["window_end"] = "different"
    with pytest.raises(AssertionError):
        analysis.matched_windows(rows, changed)
