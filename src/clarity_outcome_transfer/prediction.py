"""Recursive deployment and observed-latent controls in each model's own space."""

import torch
import torch.nn.functional as F

from clarity_rrt_v3.train import encode_mri_stages


@torch.no_grad()
def recursive_predictions(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        states = encode_mri_stages(model, batch["mri"].to(device), (0, 1, 2, 3))
        initial, targets = states[:, 0], states[:, 1:]
        steps = model.encode_conditions(batch["step_text"], batch["clinical_text"])
        prefixes = model.encode_conditions(batch["prefix_text"], batch["clinical_text"])
        members = model.rollout(initial, steps, batch["deltas"].to(device))
        risks, logits = model.survival(initial, members, prefixes)
        probabilities = logits.sigmoid()
        mean_state = members.mean(dim=0)
        mse = (mean_state - targets).square().mean(dim=(-2, -1))
        cosine = F.cosine_similarity(mean_state.flatten(2), targets.flatten(2), dim=-1)
        uz = members.var(dim=0, unbiased=False).sum(dim=(-2, -1))
        us = probabilities.var(dim=0, unbiased=False)
        mean_risk, mean_probability = risks.mean(dim=0), probabilities.mean(dim=0)
        for horizon in range(3):
            true_risk, true_logit = model.survival_module(
                initial, targets[:, horizon], prefixes[:, horizon],
            )
            fused_risk, fused_logit = model.survival_module(
                initial, mean_state[:, horizon], prefixes[:, horizon],
            )
            for sample, patient in enumerate(batch["patient"]):
                rows.append({
                    "patient": patient,
                    "start": batch["timepoints"][sample][0],
                    "end": batch["timepoints"][sample][horizon + 1],
                    "window_end": batch["timepoints"][sample][-1],
                    "primary_survival_window": int(batch["primary_survival_window"][sample]),
                    "horizon": horizon + 1,
                    "latent_mse": float(mse[sample, horizon]),
                    "cosine_similarity": float(cosine[sample, horizon]),
                    "latent_disagreement": float(uz[sample, horizon]),
                    "survival_disagreement": float(us[sample, horizon]),
                    "risk": float(mean_risk[sample, horizon]),
                    "survival365": float(mean_probability[sample, horizon]),
                    "true_risk": float(true_risk[sample].squeeze()),
                    "true_survival365": float(true_logit[sample].squeeze().sigmoid()),
                    "mean_latent_risk": float(fused_risk[sample].squeeze()),
                    "mean_latent_survival365": float(fused_logit[sample].squeeze().sigmoid()),
                    "survival_time": float(batch["survival_time"][sample, horizon]),
                    "event": int(batch["event"][sample, horizon]),
                })
    return rows, {}
