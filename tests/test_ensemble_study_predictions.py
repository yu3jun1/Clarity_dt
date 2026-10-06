import numpy as np
import pytest
import torch

from clarity_ensemble_study import prediction
from clarity_rrt_v3.model import StagewiseDynamics


class RecordingPredictor:
    def __init__(self, increment):
        self.increment = increment
        self.inputs = []

    def __call__(self, state, condition, delta):
        self.inputs.append(state.detach().clone())
        return state + self.increment


class ToyEnsemble:
    def __init__(self, members=2):
        self.predictors = tuple(RecordingPredictor(1 + 2 * index) for index in range(members))
        self.ensemble_size = members
        self.mri_encoder = lambda tensor: tensor
        self.rollout_calls = 0

    def eval(self):
        return self

    def encode_conditions(self, text_by_stage, clinical_text):
        return torch.zeros(len(clinical_text), len(text_by_stage), 1)

    def rollout(self, initial, conditions, deltas):
        self.rollout_calls += 1
        return StagewiseDynamics.rollout(self, initial, conditions, deltas)

    def teacher_forced(self, *args, **kwargs):
        raise AssertionError("Reliability evaluation must not use observed intermediate states")

    def survival(self, initial, member_states, conditions):
        members, samples, horizons = member_states.shape[:3]
        probability = torch.linspace(0.2, 0.8, members) if members > 1 else torch.tensor([0.5])
        logits = torch.logit(probability)[:, None, None].expand(members, samples, horizons)
        risks = torch.arange(members, dtype=torch.float32)[:, None, None].expand_as(logits)
        return risks, logits


def toy_batch(patient_ids=("P1", "P2"), offset=0):
    initial = torch.arange(1, 1 + 4 * len(patient_ids), dtype=torch.float32)
    initial = initial.reshape(len(patient_ids), 2, 2) + offset
    return {
        "mri": torch.stack([initial + 3 * stage for stage in range(4)], dim=1),
        "deltas": torch.ones(len(patient_ids), 3),
        "step_text": [["action"] * len(patient_ids) for _ in range(3)],
        "prefix_text": [["prefix"] * len(patient_ids) for _ in range(3)],
        "clinical_text": ["clinical"] * len(patient_ids),
        "patient": list(patient_ids),
        "timepoints": [["t0", "t1", "t2", "t3"] for _ in patient_ids],
        "primary_survival_window": torch.tensor([int(index == 0) for index in range(len(patient_ids))]),
        "survival_time": torch.tensor([[800.0, 700.0, 600.0] for _ in patient_ids]),
        "event": torch.ones(len(patient_ids), 3),
    }


def capture_analysis(monkeypatch):
    captured = {}

    def analyze(rows, member_states):
        captured["rows"] = rows
        captured["member_states"] = member_states
        return {"test_analysis": True}

    monkeypatch.setattr(prediction, "analyze", analyze)
    return captured


def test_latent_uncertainty_is_population_variance_l2_sum_not_coordinate_mean(monkeypatch):
    capture_analysis(monkeypatch)
    rows, diagnostics = prediction.recursive_predictions(
        ToyEnsemble(), [toy_batch()], torch.device("cpu")
    )

    assert len(rows) == 6
    for row in rows:
        horizon = row["horizon"]
        # Member increments 1 and 3 differ by 2h at horizon h. Each of the
        # four latent coordinates has population variance h**2.
        assert row["latent_uncertainty"] == pytest.approx(4.0 * horizon**2)
        assert row["error_l2"] == pytest.approx(2.0 * horizon)
        assert row["latent_mse"] == pytest.approx(float(horizon**2))
        assert row["member_latent_mse"] == pytest.approx([4.0 * horizon**2, 0.0])
    assert diagnostics["reliability"] == {"test_analysis": True}


def test_survival_uncertainty_is_population_variance_not_standard_deviation(monkeypatch):
    capture_analysis(monkeypatch)
    rows, _ = prediction.recursive_predictions(ToyEnsemble(), [toy_batch()], torch.device("cpu"))

    for row in rows:
        assert row["survival365"] == pytest.approx(0.5)
        assert row["survival_uncertainty"] == pytest.approx(0.09)
        assert row["risk"] == pytest.approx(0.5)


def test_each_member_rolls_out_from_its_own_state_without_teacher_forcing(monkeypatch):
    capture_analysis(monkeypatch)
    model = ToyEnsemble()
    batch = toy_batch()
    rows, _ = prediction.recursive_predictions(model, [batch], torch.device("cpu"))

    assert model.rollout_calls == 1
    for member in model.predictors:
        assert len(member.inputs) == 3
        for step, state in enumerate(member.inputs):
            torch.testing.assert_close(state, batch["mri"][:, 0] + step * member.increment)
    assert [(row["patient"], row["horizon"]) for row in rows] == [
        (patient, horizon) for patient in batch["patient"] for horizon in (1, 2, 3)
    ]
    for row in rows:
        assert row["start"] == "t0"
        assert row["end"] == f"t{row['horizon']}"
        assert row["window_end"] == "t3"
        assert row["primary_survival_window"] == int(row["patient"] == "P1")
        assert row["survival_time"] == 900.0 - 100.0 * row["horizon"]
        assert row["event"] == 1
        sample = batch["patient"].index(row["patient"])
        horizon = row["horizon"]
        initial = batch["mri"][sample, 0].flatten()
        target = batch["mri"][sample, horizon].flatten()
        mean_cosine = torch.nn.functional.cosine_similarity(
            initial + 2 * horizon, target, dim=0
        )
        assert row["cosine_similarity"] == pytest.approx(float(mean_cosine))
        member_cosines = [
            float(torch.nn.functional.cosine_similarity(initial + increment * horizon, target, dim=0))
            for increment in (1, 3)
        ]
        assert row["member_cosine_similarity"] == pytest.approx(member_cosines)


def test_analysis_receives_members_samples_horizons_and_latent_axes_across_batches(monkeypatch):
    captured = capture_analysis(monkeypatch)
    batches = [toy_batch(), toy_batch(("P3",), offset=20)]
    rows, _ = prediction.recursive_predictions(ToyEnsemble(), batches, torch.device("cpu"))

    states = captured["member_states"]
    assert isinstance(states, np.ndarray)
    assert states.shape == (2, 3, 3, 2, 2)
    assert captured["rows"] is rows
    initial = torch.cat([batch["mri"][:, 0] for batch in batches]).numpy()
    for member, increment in enumerate((1, 3)):
        for horizon in range(3):
            np.testing.assert_allclose(states[member, :, horizon], initial + increment * (horizon + 1))


def test_single_member_has_zero_latent_and_survival_uncertainty(monkeypatch):
    capture_analysis(monkeypatch)
    rows, _ = prediction.recursive_predictions(ToyEnsemble(members=1), [toy_batch()], torch.device("cpu"))

    assert all(row["latent_uncertainty"] == 0.0 for row in rows)
    assert all(row["survival_uncertainty"] == 0.0 for row in rows)
    assert all(len(row["member_latent_mse"]) == 1 for row in rows)


def test_single_member_end_to_end_reports_undefined_correlations():
    rows, diagnostics = prediction.recursive_predictions(
        ToyEnsemble(members=1), [toy_batch()], torch.device("cpu")
    )

    assert len(rows) == 6
    for horizon in ("H1", "H2", "H3"):
        reliability = diagnostics["reliability"][horizon]
        assert reliability["latent_error_correlation"] == {"pearson": None, "spearman": None}
        assert reliability["survival_error_correlation"] == {"pearson": None, "spearman": None}
        assert reliability["risk_coverage"] == []
