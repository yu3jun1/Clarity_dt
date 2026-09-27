"""Shared-encoder/shared-outcome dynamics ensemble for official CLARITY."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn


def _reset_module(module: nn.Module) -> None:
    """Reset leaf parameters without resetting descendants more than once."""
    children = list(module.children())
    if not children and hasattr(module, "reset_parameters"):
        module.reset_parameters()
    for child in children:
        _reset_module(child)


def _fresh_predictor(template: nn.Module) -> nn.Module:
    """Reconstruct the official LatentPredictor with its own init policy."""
    try:
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
    except (AttributeError, IndexError, TypeError):
        predictor = copy.deepcopy(template)
        _reset_module(predictor)
        return predictor


class ClarityDynamicsEnsemble(nn.Module):
    """Wrap an unmodified official model and replicate only its dynamics.

    Member zero is the official ``latent_predictor``. Additional predictors have
    the identical architecture but independently reset parameters. MRI/text
    encoders and the SurvivalModule remain shared.
    """

    def __init__(self, official_model: nn.Module, ensemble_size: int = 1, seed: int = 42):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be at least one")
        self.base = official_model
        self.ensemble_size = int(ensemble_size)
        text_model = getattr(official_model.shared_text_encoder, "model", None)
        if text_model is not None:
            # PEFT exposes its own wrapper config while Transformers checks the
            # nested backbone configs during gradient-checkpointed forwards.
            for module in text_model.modules():
                text_config = getattr(module, "config", None)
                if text_config is not None and hasattr(text_config, "use_cache"):
                    text_config.use_cache = False
        extras = []
        for member_index in range(1, ensemble_size):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 10_000 + member_index)
                predictor = _fresh_predictor(official_model.latent_predictor)
            extras.append(predictor)
        self.extra_predictors = nn.ModuleList(extras)

    @property
    def predictors(self) -> tuple[nn.Module, ...]:
        return (self.base.latent_predictor, *tuple(self.extra_predictors))

    @property
    def mri_encoder(self) -> nn.Module | None:
        return self.base.mri_encoder

    @property
    def shared_text_encoder(self) -> nn.Module:
        return self.base.shared_text_encoder

    @property
    def survival_module(self) -> nn.Module:
        return self.base.survival_module

    def encode_condition(
        self, drugs_text: Sequence[str], clinical_text: Sequence[str] | None = None
    ) -> torch.Tensor:
        drug_embedding = self.shared_text_encoder(list(drugs_text))
        if clinical_text and any(clinical_text):
            clinical_embedding = self.shared_text_encoder(list(clinical_text))
        else:
            clinical_embedding = torch.zeros_like(drug_embedding)
        return torch.cat([drug_embedding, clinical_embedding], dim=-1)

    def forward_encoded(
        self,
        pre_latent: torch.Tensor,
        condition: torch.Tensor,
        time_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        posts = []
        risks = []
        logits = []
        for predictor in self.predictors:
            post = predictor(pre_latent, condition, time_delta)
            risk, logit = self.survival_module(pre_latent, post, condition)
            posts.append(post)
            risks.append(risk.squeeze(-1))
            logits.append(logit.squeeze(-1))
        return {
            "member_post": torch.stack(posts),
            "member_risk": torch.stack(risks),
            "member_logit": torch.stack(logits),
            "condition": condition,
        }

    def forward_pair(
        self,
        pre_latent: torch.Tensor,
        drugs_text: Sequence[str],
        time_delta: torch.Tensor,
        clinical_text: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        # Preserve the exact upstream computation for the baseline/member zero.
        post0, risk0, logit0, condition = self.base(
            pre_latent, list(drugs_text), time_delta, list(clinical_text) if clinical_text else None
        )
        posts = [post0]
        risks = [risk0.squeeze(-1)]
        logits = [logit0.squeeze(-1)]
        for predictor in self.extra_predictors:
            post = predictor(pre_latent, condition, time_delta)
            risk, logit = self.survival_module(pre_latent, post, condition)
            posts.append(post)
            risks.append(risk.squeeze(-1))
            logits.append(logit.squeeze(-1))
        return {
            "member_post": torch.stack(posts),
            "member_risk": torch.stack(risks),
            "member_logit": torch.stack(logits),
            "condition": condition,
        }

    def rollout_encoded(
        self,
        initial_latent: torch.Tensor,
        step_conditions: torch.Tensor,
        step_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Roll each member through its own states.

        Args:
            initial_latent: ``[B,N,D]``
            step_conditions: ``[B,H,C]``
            step_deltas: ``[B,H]``, all strictly positive
        Returns:
            Member states after each transition, ``[M,B,H,N,D]``.
        """
        if step_conditions.ndim != 3 or step_deltas.ndim != 2:
            raise ValueError("step_conditions and step_deltas must be [B,H,C] and [B,H]")
        if step_conditions.shape[:2] != step_deltas.shape:
            raise ValueError("condition/delta batch and horizon dimensions do not match")
        if torch.any(step_deltas <= 0):
            raise ValueError("all real rollout intervals must be positive")
        all_members = []
        for predictor in self.predictors:
            state = initial_latent
            states = []
            for step in range(step_deltas.shape[1]):
                state = predictor(state, step_conditions[:, step], step_deltas[:, step])
                states.append(state)
            all_members.append(torch.stack(states, dim=1))
        return torch.stack(all_members)

    def encode_step_conditions(
        self,
        drugs_text_steps: Sequence[Sequence[str]],
        clinical_text: Sequence[str],
    ) -> torch.Tensor:
        conditions = [
            self.encode_condition(step_text, clinical_text) for step_text in drugs_text_steps
        ]
        return torch.stack(conditions, dim=1)

    def endpoint_predictions(
        self,
        initial_latent: torch.Tensor,
        member_terminal: torch.Tensor,
        endpoint_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        risks, logits = [], []
        for member_index in range(self.ensemble_size):
            risk, logit = self.survival_module(
                initial_latent, member_terminal[member_index], endpoint_condition
            )
            risks.append(risk.squeeze(-1))
            logits.append(logit.squeeze(-1))
        return torch.stack(risks), torch.stack(logits)

    def assert_independent_members(self) -> None:
        if self.ensemble_size == 1:
            return
        first_parameters = list(self.predictors[0].parameters())
        for index, predictor in enumerate(self.predictors[1:], start=1):
            parameters = list(predictor.parameters())
            if any(left.data_ptr() == right.data_ptr() for left, right in zip(first_parameters, parameters)):
                raise AssertionError(f"predictor member {index} shares parameter storage")
            if all(torch.equal(left, right) for left, right in zip(first_parameters, parameters)):
                raise AssertionError(f"predictor member {index} has identical initialization")
