from __future__ import annotations

import torch
from torch import nn

from clarity_rrt.model import ClarityDynamicsEnsemble


class TinyTextEncoder(nn.Module):
    def __init__(self, dim: int = 2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, texts):
        lengths = torch.tensor([len(text) for text in texts], dtype=self.scale.dtype)
        return lengths[:, None] * self.scale[None]


class TinyPredictor(nn.Module):
    def __init__(
        self,
        latent_dim=3,
        drug_dim=4,
        time_dim=2,
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        ffn_dim=8,
        dropout=0.0,
    ):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, ffn_dim, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.calls = []

    def forward(self, state, condition, delta):
        self.calls.append(state)
        time_features = torch.stack((delta, delta), dim=-1)
        hidden = self.latent_proj(state)
        hidden = hidden + self.drug_proj(condition)[:, None]
        hidden = hidden + self.time_proj(time_features)[:, None]
        return self.output_proj(hidden)


class TinySurvival(nn.Module):
    def __init__(self):
        super().__init__()
        self.risk = nn.Linear(7, 1)
        self.survival = nn.Linear(7, 1)

    def forward(self, pre, post, condition):
        features = torch.cat((pre.mean(1), post.mean(1), condition[:, :1]), dim=-1)
        return self.risk(features), self.survival(features)


class TinyOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_predictor = TinyPredictor()
        self.shared_text_encoder = TinyTextEncoder()
        self.survival_module = TinySurvival()
        self.mri_encoder = None

    def forward(self, pre, drugs, delta, clinical=None):
        drug = self.shared_text_encoder(drugs)
        clinical_embedding = (
            self.shared_text_encoder(clinical)
            if clinical and any(clinical)
            else torch.zeros_like(drug)
        )
        condition = torch.cat((drug, clinical_embedding), dim=-1)
        post = self.latent_predictor(pre, condition, delta)
        risk, logit = self.survival_module(pre, post, condition)
        return post, risk, logit, condition


def test_single_member_is_exact_official_path():
    torch.manual_seed(3)
    official = TinyOfficial().eval()
    wrapped = ClarityDynamicsEnsemble(official, ensemble_size=1, seed=42).eval()
    pre = torch.randn(2, 5, 3)
    delta = torch.tensor([10.0, 20.0])
    expected = official(pre, ["a", "bb"], delta, ["x", "yy"])
    actual = wrapped.forward_pair(pre, ["a", "bb"], delta, ["x", "yy"])
    torch.testing.assert_close(actual["member_post"][0], expected[0])
    torch.testing.assert_close(actual["member_risk"][0], expected[1].squeeze(-1))
    torch.testing.assert_close(actual["member_logit"][0], expected[2].squeeze(-1))


def test_members_are_independent_and_differently_initialized():
    ensemble = ClarityDynamicsEnsemble(TinyOfficial(), ensemble_size=3, seed=42)
    ensemble.assert_independent_members()
    pointers = [next(member.parameters()).data_ptr() for member in ensemble.predictors]
    assert len(set(pointers)) == 3


def test_rollout_uses_own_prediction_and_backpropagates_through_time():
    ensemble = ClarityDynamicsEnsemble(TinyOfficial(), ensemble_size=1, seed=42)
    initial = torch.randn(2, 4, 3, requires_grad=True)
    conditions = torch.randn(2, 2, 4)
    deltas = torch.tensor([[10.0, 20.0], [11.0, 21.0]])
    states = ensemble.rollout_encoded(initial, conditions, deltas)
    predictor = ensemble.predictors[0]
    torch.testing.assert_close(predictor.calls[1], states[0, :, 0])
    states[:, :, 1].sum().backward()
    assert initial.grad is not None
    assert torch.isfinite(initial.grad).all()
    assert any(parameter.grad is not None for parameter in predictor.parameters())


def test_rollout_rejects_nonpositive_real_interval():
    ensemble = ClarityDynamicsEnsemble(TinyOfficial(), ensemble_size=1, seed=42)
    try:
        ensemble.rollout_encoded(
            torch.randn(1, 2, 3), torch.randn(1, 2, 4), torch.tensor([[1.0, 0.0]])
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("zero interval was accepted")


def test_survival_losses_reach_predictor_and_shared_head():
    ensemble = ClarityDynamicsEnsemble(TinyOfficial(), ensemble_size=2, seed=42)
    pre = torch.randn(3, 4, 3)
    output = ensemble.forward_pair(
        pre,
        ["a", "bb", "ccc"],
        torch.tensor([10.0, 20.0, 30.0]),
        ["x", "yy", "zzz"],
    )
    (output["member_risk"].sum() + output["member_logit"].sum()).backward()
    predictor_grads = [p.grad for p in ensemble.predictors[0].parameters() if p.grad is not None]
    survival_grads = [p.grad for p in ensemble.survival_module.parameters() if p.grad is not None]
    assert predictor_grads and survival_grads
    assert all(torch.isfinite(grad).all() for grad in predictor_grads + survival_grads)


def test_full_state_round_trip_reproduces_predictions():
    torch.manual_seed(12)
    source = ClarityDynamicsEnsemble(TinyOfficial(), ensemble_size=3, seed=42).eval()
    pre = torch.randn(2, 4, 3)
    delta = torch.tensor([12.0, 24.0])
    expected = source.forward_pair(pre, ["a", "bb"], delta, ["x", "yy"])
    restored = ClarityDynamicsEnsemble(TinyOfficial(), ensemble_size=3, seed=99).eval()
    restored.load_state_dict(source.state_dict())
    actual = restored.forward_pair(pre, ["a", "bb"], delta, ["x", "yy"])
    for key in ("member_post", "member_risk", "member_logit"):
        torch.testing.assert_close(actual[key], expected[key])
