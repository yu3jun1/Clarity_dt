from __future__ import annotations

import torch
from torch import nn

from clarity_rrt_v3.model import StagewiseDynamics
from clarity_rrt_v3.train import mean_horizon_l1


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(2))

    def forward(self, texts):
        lengths = torch.tensor([len(text) for text in texts], dtype=self.scale.dtype)
        return lengths[:, None] * self.scale


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
        hidden = self.latent_proj(state)
        hidden = hidden + self.drug_proj(condition)[:, None]
        time = torch.stack((delta, delta), dim=-1)
        hidden = hidden + self.time_proj(time)[:, None]
        return self.output_proj(hidden)


class TinySurvival(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(10, 2)

    def forward(self, initial, predicted, condition):
        features = torch.cat(
            (initial.mean(1), predicted.mean(1), condition), dim=-1
        )
        output = self.head(features)
        return output[:, :1], output[:, 1:]


class TinyClarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_predictor = TinyPredictor()
        self.shared_text_encoder = TinyTextEncoder()
        self.survival_module = TinySurvival()
        self.mri_encoder = nn.Identity()


def inputs():
    states = torch.randn(2, 4, 5, 3)
    conditions = torch.randn(2, 3, 4)
    deltas = torch.tensor([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    return states, conditions, deltas


def test_direct_path_uses_only_initial_state_and_full_plan():
    model = StagewiseDynamics(TinyClarity(), ensemble_size=1, seed=42)
    states, conditions, deltas = inputs()
    output = model.direct(states[:, 0], conditions[:, -1], deltas.sum(dim=1))
    assert output.shape == (1, 2, 5, 3)
    torch.testing.assert_close(model.predictors[0].calls[0], states[:, 0])


def test_recursive_loss_weights_all_three_horizons_equally():
    member_states = torch.zeros(1, 1, 3, 1, 1)
    targets = torch.tensor([[[[1.0]], [[2.0]], [[6.0]]]])
    loss = mean_horizon_l1(member_states, targets)
    torch.testing.assert_close(loss, torch.tensor(3.0))


def test_rollout_feeds_each_member_its_own_prediction():
    model = StagewiseDynamics(TinyClarity(), ensemble_size=2, seed=42)
    states, conditions, deltas = inputs()
    output = model.rollout(states[:, 0], conditions, deltas)
    assert output.shape == (2, 2, 3, 5, 3)
    for member, predictor in enumerate(model.predictors):
        torch.testing.assert_close(predictor.calls[1], output[member, :, 0])
        assert next(predictor.parameters()).data_ptr() != next(model.predictors[1 - member].parameters()).data_ptr()


def test_survival_head_is_shared_across_members_and_horizons():
    model = StagewiseDynamics(TinyClarity(), ensemble_size=3, seed=42)
    states, conditions, deltas = inputs()
    rollout = model.rollout(states[:, 0], conditions, deltas)
    risks, logits = model.survival(states[:, 0], rollout, conditions)
    assert risks.shape == (3, 2, 3)
    assert logits.shape == (3, 2, 3)
