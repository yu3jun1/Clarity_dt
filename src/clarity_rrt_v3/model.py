"""CLARITY dynamics members with open-loop and recursive execution paths."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def fresh_predictor(template: nn.Module) -> nn.Module:
    layer = template.transformer.layers[0]
    return type(template)(
        latent_dim=template.latent_proj.in_features,
        drug_dim=template.drug_proj.in_features,
        time_dim=template.time_proj.in_features,
        hidden_dim=template.latent_proj.out_features,
        num_layers=len(template.transformer.layers),
        num_heads=layer.self_attn.num_heads,
        ffn_dim=layer.linear1.out_features,
        dropout=layer.dropout.p,
    )


class StagewiseDynamics(nn.Module):
    """Share CLARITY encoders/outcome and replicate only the transition model."""

    def __init__(self, clarity_model: nn.Module, ensemble_size: int, seed: int) -> None:
        super().__init__()
        self.clarity = clarity_model
        members = []
        for index in range(1, ensemble_size):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + index)
                members.append(fresh_predictor(clarity_model.latent_predictor))
        self.extra_predictors = nn.ModuleList(members)

    @property
    def predictors(self) -> tuple[nn.Module, ...]:
        return (self.clarity.latent_predictor, *tuple(self.extra_predictors))

    @property
    def ensemble_size(self) -> int:
        return len(self.predictors)

    @property
    def mri_encoder(self) -> nn.Module:
        return self.clarity.mri_encoder

    @property
    def text_encoder(self) -> nn.Module:
        return self.clarity.shared_text_encoder

    @property
    def survival_module(self) -> nn.Module:
        return self.clarity.survival_module

    def encode_conditions(
        self,
        text_by_stage: Sequence[Sequence[str]],
        clinical_text: Sequence[str],
    ) -> torch.Tensor:
        clinical = self.text_encoder(list(clinical_text))
        conditions = []
        for texts in text_by_stage:
            treatment = self.text_encoder(list(texts))
            conditions.append(torch.cat((treatment, clinical), dim=-1))
        return torch.stack(conditions, dim=1)

    def direct(
        self,
        initial: torch.Tensor,
        full_condition: torch.Tensor,
        total_delta: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [predictor(initial, full_condition, total_delta) for predictor in self.predictors]
        )

    def rollout(
        self,
        initial: torch.Tensor,
        step_conditions: torch.Tensor,
        step_deltas: torch.Tensor,
    ) -> torch.Tensor:
        members = []
        for predictor in self.predictors:
            state = initial
            trajectory = []
            for step in range(3):
                state = predictor(
                    state,
                    step_conditions[:, step],
                    step_deltas[:, step],
                )
                trajectory.append(state)
            members.append(torch.stack(trajectory, dim=1))
        return torch.stack(members)

    def survival(
        self,
        initial: torch.Tensor,
        member_states: torch.Tensor,
        conditions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        risks = []
        logits = []
        for member in range(self.ensemble_size):
            member_risks = []
            member_logits = []
            for horizon in range(member_states.shape[2]):
                risk, logit = self.survival_module(
                    initial,
                    member_states[member, :, horizon],
                    conditions[:, horizon],
                )
                member_risks.append(risk.squeeze(-1))
                member_logits.append(logit.squeeze(-1))
            risks.append(torch.stack(member_risks, dim=1))
            logits.append(torch.stack(member_logits, dim=1))
        return torch.stack(risks), torch.stack(logits)
