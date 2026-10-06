"""Recursive predictions with the reliability study's L2 uncertainty units."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from clarity_rrt_v3.train import encode_mri_stages
from .analysis import analyze


@torch.no_grad()
def recursive_predictions(model, loader, device):
    model.eval()
    rows, member_batches = [], []
    for batch in loader:
        states = encode_mri_stages(model, batch["mri"].to(device), (0, 1, 2, 3))
        targets = states[:, 1:]
        step_conditions = model.encode_conditions(batch["step_text"], batch["clinical_text"])
        prefix_conditions = model.encode_conditions(batch["prefix_text"], batch["clinical_text"])
        members = model.rollout(states[:, 0], step_conditions, batch["deltas"].to(device))
        risks, logits = model.survival(states[:, 0], members, prefix_conditions)
        member_batches.append(members.float().cpu())
        mean_state = members.mean(dim=0)
        squared_error = (mean_state - targets).square()
        mse = squared_error.mean(dim=(-2, -1))
        error_l2 = squared_error.sum(dim=(-2, -1)).sqrt()
        latent_uncertainty = members.var(dim=0, unbiased=False).sum(dim=(-2, -1))
        expanded_targets = targets.unsqueeze(0).expand_as(members)
        member_mse = (members - expanded_targets).square().mean(dim=(-2, -1))
        member_cosine = F.cosine_similarity(members.flatten(3), expanded_targets.flatten(3), dim=-1)
        cosine = F.cosine_similarity(mean_state.flatten(2), targets.flatten(2), dim=-1)
        probabilities = logits.sigmoid()
        survival_uncertainty = probabilities.var(dim=0, unbiased=False)
        mean_probability, mean_risk = probabilities.mean(dim=0), risks.mean(dim=0)
        for sample, patient in enumerate(batch["patient"]):
            for horizon in range(3):
                rows.append({
                    "patient": patient,
                    "start": batch["timepoints"][sample][0],
                    "end": batch["timepoints"][sample][horizon + 1],
                    "window_end": batch["timepoints"][sample][-1],
                    "primary_survival_window": int(batch["primary_survival_window"][sample]),
                    "horizon": horizon + 1,
                    "latent_mse": float(mse[sample, horizon]),
                    "cosine_similarity": float(cosine[sample, horizon]),
                    "error_l2": float(error_l2[sample, horizon]),
                    "latent_uncertainty": float(latent_uncertainty[sample, horizon]),
                    "survival_uncertainty": float(survival_uncertainty[sample, horizon]),
                    "member_latent_mse": member_mse[:, sample, horizon].tolist(),
                    "member_cosine_similarity": member_cosine[:, sample, horizon].tolist(),
                    "risk": float(mean_risk[sample, horizon]),
                    "survival365": float(mean_probability[sample, horizon]),
                    "survival_time": float(batch["survival_time"][sample, horizon]),
                    "event": int(batch["event"][sample, horizon]),
                })
    members = torch.cat(member_batches, dim=1).numpy()
    return rows, {"reliability": analyze(rows, members)}
