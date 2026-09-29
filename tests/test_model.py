from __future__ import annotations

import json

import torch
from torch import nn

from clarity_rrt_v3.model import StagewiseDynamics
from clarity_rrt_v3.train import (
    CINDEX_CHECKPOINT_NAME,
    PRIMARY_CHECKPOINT_NAME,
    Trainer,
    counterfactual_loss,
    mean_horizon_l1,
)


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
        self.cf_calls = []

    def drug_swap_diversity_loss(
        self,
        predictor,
        pre_latent,
        condition_emb,
        time_delta,
        pred_latent,
        drug_categories,
        cos_margin,
    ):
        self.cf_calls.append(
            {
                "condition": condition_emb,
                "categories": drug_categories,
                "margin": cos_margin,
            }
        )
        return condition_emb.mean()



def interval_text(agent):
    actions = (
        {"chemotherapy": [{"agent": agent}]}
        if agent is not None
        else {}
    )
    return json.dumps({"intervals": [{"actions": actions}]})


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


def test_open_loop_counterfactual_loss_uses_full_plan_condition():
    model = StagewiseDynamics(TinyClarity(), ensemble_size=2, seed=42)
    initial = torch.zeros(4, 5, 3)
    member_states = torch.zeros(2, 4, 1, 5, 3)
    conditions = torch.full((4, 1, 4), 4.0)
    deltas = torch.ones(4, 1)
    texts = [
        [
            interval_text(agent)
            for agent in ("Temozolomide", "Avastin", "Lomustine", None)
        ]
    ]

    loss = counterfactual_loss(
        model,
        initial,
        member_states,
        conditions,
        deltas,
        texts,
        0.9,
    )

    torch.testing.assert_close(loss, torch.tensor(4.0))
    assert len(model.clarity.cf_calls) == 2
    assert all(
        call["categories"] == ["TMZ", "BEV", "OTHER", "no_treatment"]
        for call in model.clarity.cf_calls
    )


def test_recursive_counterfactual_loss_averages_stage_conditions():
    model = StagewiseDynamics(TinyClarity(), ensemble_size=2, seed=42)
    initial = torch.zeros(4, 5, 3)
    member_states = torch.zeros(2, 4, 3, 5, 3)
    conditions = torch.stack(
        [torch.full((4, 4), value) for value in (1.0, 2.0, 6.0)],
        dim=1,
    )
    deltas = torch.ones(4, 3)
    texts = [
        [
            interval_text(agent)
            for agent in ("Temozolomide", "Avastin", "Lomustine", None)
        ]
        for _ in range(3)
    ]

    loss = counterfactual_loss(
        model,
        initial,
        member_states,
        conditions,
        deltas,
        texts,
        0.9,
    )

    torch.testing.assert_close(loss, torch.tensor(3.0))
    assert len(model.clarity.cf_calls) == 6
    assert [call["categories"] for call in model.clarity.cf_calls[:3]] == [
        ["TMZ", "BEV", "OTHER", "no_treatment"] for _ in range(3)
    ]


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


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class CheckpointSelectionTrainer(Trainer):
    def __init__(self):
        self.config = {"training": {"epochs": 4, "warmup_epochs": 1}}
        self.scheduler = CountingScheduler()
        self.history = []
        self.best_val_loss = float("inf")
        self.best_c_index = -float("inf")
        self.saved = []

    def epoch(self, epoch, training):
        if training:
            return {"loss": 10.0, "c_index": 0.0}
        return {
            1: {"loss": 1.0, "c_index": 0.9},
            2: {"loss": 4.0, "c_index": 0.6},
            3: {"loss": 3.0, "c_index": 0.5},
            4: {"loss": 3.5, "c_index": 0.7},
        }[epoch]

    def write_history(self):
        pass

    def save(self, epoch, filename, selection_criterion, validation):
        self.saved.append((epoch, filename, selection_criterion, dict(validation)))


def test_checkpoint_selection_uses_val_loss_for_primary_and_cindex_for_secondary():
    trainer = CheckpointSelectionTrainer()
    trainer.fit()

    saved = [
        (epoch, filename, criterion)
        for epoch, filename, criterion, _ in trainer.saved
    ]
    assert saved == [
        (2, PRIMARY_CHECKPOINT_NAME, "validation_total_loss"),
        (2, CINDEX_CHECKPOINT_NAME, "validation_patient_level_c_index"),
        (3, PRIMARY_CHECKPOINT_NAME, "validation_total_loss"),
        (4, CINDEX_CHECKPOINT_NAME, "validation_patient_level_c_index"),
    ]
    assert trainer.best_val_loss == 3.0
    assert trainer.best_c_index == 0.7
    assert trainer.scheduler.steps == 4
