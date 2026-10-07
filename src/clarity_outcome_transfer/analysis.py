"""Censoring-aware outcome errors and seed/patient matched A versus C reports."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from clarity_rrt_v3.evaluate import brier365, statistics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def outcome_errors(reference, rows, probability_key):
    """Individual terms whose unnormalised mean is sksurv's IPCW Brier365."""
    train_time, train_event = reference
    censoring = CensoringDistributionEstimator().fit(Surv.from_arrays(train_event, train_time))
    times = np.asarray([row["survival_time"] for row in rows], dtype=float)
    events = np.asarray([bool(row["event"]) for row in rows])
    probabilities = np.asarray([row[probability_key] for row in rows], dtype=float)
    case, control = (times <= 365) & events, times > 365
    known = case | control
    g_time = censoring.predict_proba(times)
    g_365 = censoring.predict_proba(np.array([365.0]))
    # Follow the existing sksurv Brier convention for zero censoring survival.
    g_time[g_time == 0] = np.inf
    g_365[g_365 == 0] = np.inf
    contribution = probabilities ** 2 * case / g_time + (1 - probabilities) ** 2 * control / g_365[0]
    errors = (probabilities - control.astype(float)) ** 2
    return known, control.astype(float), errors, contribution


def correlations(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "pearson": None, "spearman": None}
    return {"n": len(x), "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])}


def analyze_run(rows, reference, concordance):
    primary = sorted(
        [row for row in rows if row["horizon"] == 3 and row["primary_survival_window"]],
        key=lambda row: row["patient"],
    )
    assert len(primary) == len({row["patient"] for row in primary})
    scores, errors = {}, {}
    for condition, prefix in (("predicted_latent", ""), ("true_latent", "true_"),
                              ("mean_predicted_latent", "mean_latent_")):
        known, label, squared, contribution = outcome_errors(reference, primary, prefix + "survival365")
        selected = [{**row, "risk": row[prefix + "risk"],
                     "survival365": row[prefix + "survival365"]} for row in primary]
        brier = brier365(*reference, selected)
        np.testing.assert_allclose(contribution.mean(), brier, rtol=1e-12, atol=1e-12)
        scores[condition] = {
            "c_index": float(concordance([r["risk"] for r in selected],
                                        [r["survival_time"] for r in selected],
                                        [r["event"] for r in selected])),
            "brier365": brier,
            "known365_mse": float(squared[known].mean()) if known.any() else None,
        }
        errors[condition] = (squared, contribution)
    patients = []
    for index, row in enumerate(primary):
        patient = {key: row[key] for key in ("patient", "start", "end", "window_end", "survival_time", "event",
                                            "latent_disagreement", "survival_disagreement")}
        patient.update(known365=bool(known[index]), label365=float(label[index]) if known[index] else None,
                       predicted_survival365=row["survival365"], true_survival365=row["true_survival365"],
                       mean_latent_survival365=row["mean_latent_survival365"],
                       transfer_abs_probability_error=abs(row["survival365"] - row["true_survival365"]),
                       primary_H3_latent_mse=row["latent_mse"])
        for horizon in (1, 2, 3):
            patient[f"H{horizon}_latent_mse"] = float(np.mean([
                r["latent_mse"] for r in rows if r["patient"] == row["patient"] and r["horizon"] == horizon
            ]))
        for condition in ("predicted_latent", "true_latent"):
            squared, contribution = errors[condition]
            patient[f"{condition}_squared_error365"] = float(squared[index]) if known[index] else None
            patient[f"{condition}_brier_contribution"] = float(contribution[index])
        patients.append(patient)
    return {
        "survival_time_origin": "H3 endpoint; 365 days after that endpoint",
        "patient_count": len(primary), "known365_count": int(known.sum()),
        "censored_by365_count": int((~known).sum()), "H3": scores,
        "predicted_minus_true": {metric: scores["predicted_latent"][metric] - scores["true_latent"][metric]
                                 for metric in ("c_index", "brier365")},
        "disagreement_vs_survival_error": {
            signal: correlations([r[signal] for r, include in zip(primary, known) if include],
                                 errors["predicted_latent"][0][known])
            for signal in ("latent_disagreement", "survival_disagreement")
        },
        "patients": patients,
    }


def pair_patients(a, c, seed):
    """One row per patient/seed, after averaging that patient's latent windows."""
    by_a = {row["patient"]: row for row in a}
    by_c = {row["patient"]: row for row in c}
    assert len(by_a) == len(a) and len(by_c) == len(c) and by_a.keys() == by_c.keys()
    paired = []
    for patient in sorted(by_a):
        first, second = by_a[patient], by_c[patient]
        shared = ("patient", "start", "end", "window_end", "survival_time", "event", "known365", "label365")
        assert all(first[key] == second[key] for key in shared), f"Unmatched outcome: {patient}"
        row = {"seed": seed, **{key: first[key] for key in shared}}
        fields = [f"H{h}_latent_mse" for h in (1, 2, 3)] + [
            "predicted_survival365", "true_survival365", "predicted_latent_squared_error365",
            "predicted_latent_brier_contribution", "true_latent_brier_contribution",
            "transfer_abs_probability_error",
        ]
        for key in fields:
            row[f"A_{key}"], row[f"C_{key}"] = first[key], second[key]
            row[f"C_minus_A_{key}"] = None if first[key] is None else second[key] - first[key]
        row["C_latent_disagreement"] = second["latent_disagreement"]
        row["C_survival_disagreement"] = second["survival_disagreement"]
        paired.append(row)
    return paired


def matched_windows(a_rows, c_rows):
    keys = ("patient", "start", "end", "window_end", "horizon")
    a = {tuple(r[k] for k in keys): r for r in a_rows}
    c = {tuple(r[k] for k in keys): r for r in c_rows}
    assert len(a) == len(a_rows) and len(c) == len(c_rows) and a.keys() == c.keys()
    assert all(all(a[key][field] == c[key][field] for field in
                   ("survival_time", "event", "primary_survival_window")) for key in a)


def write_report(root, seeds):
    root = Path(root)
    runs = {variant: [json.loads((root / "runs" / f"{variant}_seed{seed}" / "metrics.json").read_text())
                      for seed in seeds] for variant in ("A", "C")}
    paired, seed_differences = [], []
    for index, seed in enumerate(seeds):
        row_sets = []
        for variant in ("A", "C"):
            with (root / "runs" / f"{variant}_seed{seed}" / "recursive_predictions.csv").open(newline="") as handle:
                row_sets.append(list(csv.DictReader(handle)))
        matched_windows(*row_sets)
        a, c = (runs[v][index] for v in ("A", "C"))
        seed_pairs = pair_patients(a["outcome_transfer"]["patients"], c["outcome_transfer"]["patients"], seed)
        paired.extend(seed_pairs)
        seed_differences.append({
            "seed": seed,
            **{f"H{h}_latent_mse": c["recursive"][f"H{h}"]["latent_mse"] - a["recursive"][f"H{h}"]["latent_mse"]
               for h in (1, 2, 3)},
            **{f"H3_{metric}": c["recursive"]["H3"][metric] - a["recursive"]["H3"][metric]
               for metric in ("c_index", "brier365")},
            "patients_with_lower_C_H3_latent_mse": sum(r["C_minus_A_H3_latent_mse"] < 0 for r in seed_pairs),
            "patients_with_lower_C_brier_contribution": sum(r["C_minus_A_predicted_latent_brier_contribution"] < 0 for r in seed_pairs),
        })
    summary = {"seeds": seeds, "models": {}, "seed_matched_C_minus_A": seed_differences}
    for variant, group in runs.items():
        summary["models"][variant] = {
            "latent_mse": {f"H{h}": statistics([r["recursive"][f"H{h}"]["latent_mse"] for r in group]) for h in (1, 2, 3)},
            "H3": {condition: {metric: statistics([r["outcome_transfer"]["H3"][condition][metric] for r in group])
                               for metric in ("c_index", "brier365")}
                   for condition in ("true_latent", "predicted_latent", "mean_predicted_latent")},
            "predicted_minus_true": {metric: statistics([r["outcome_transfer"]["predicted_minus_true"][metric] for r in group])
                                     for metric in ("c_index", "brier365")},
        }
    summary["C_disagreement_vs_survival_error"] = [
        {"seed": seed, **run["outcome_transfer"]["disagreement_vs_survival_error"]}
        for seed, run in zip(seeds, runs["C"])
    ]
    report = root / "reports"
    report.mkdir()
    write_csv(report / "patient_paired_errors.csv", paired)
    write_json(report / "summary.json", summary)
    fmt = lambda s: f"{s['mean']:.4f} ± {s['std']:.4f}"
    lines = ["# A vs C: End-to-End Outcome Transfer", "",
             "Seed-matched 42/43/44; mean ± sample SD across seeds. Descriptive analysis only.", "",
             "## Latent MSE (all trajectory windows)", "", "|Model|H1|H2|H3|", "|---|---:|---:|---:|"]
    for variant in ("A", "C"):
        values = summary["models"][variant]["latent_mse"]
        lines.append(f"|{variant}|" + "|".join(fmt(values[f"H{h}"]) for h in (1, 2, 3)) + "|")
    lines += ["", "## H3 survival inputs", "",
              "True latent is a checkpoint-specific observed-state reference, not a guaranteed performance upper bound.",
              "A and C have separately trained encoders/heads: no cross-model latent swapping.",
              "C predicted = average of member risks/probabilities. C mean latent is an auxiliary fusion control.", "",
              "|Head/model|Input|C-index ↑|IPCW Brier365 ↓|", "|---|---|---:|---:|"]
    for variant, condition in (("A", "true_latent"), ("A", "predicted_latent"), ("C", "true_latent"),
                               ("C", "predicted_latent"), ("C", "mean_predicted_latent")):
        scores = summary["models"][variant]["H3"][condition]
        lines.append(f"|{variant}|{condition}|{fmt(scores['c_index'])}|{fmt(scores['brier365'])}|")
    lines += ["", "## Within-head predicted minus true", "", "|Model|Δ C-index|Δ Brier365|", "|---|---:|---:|"]
    for variant in ("A", "C"):
        delta = summary["models"][variant]["predicted_minus_true"]
        lines.append(f"|{variant}|{fmt(delta['c_index'])}|{fmt(delta['brier365'])}|")
    lines += ["", "## Seed-matched C minus A (native deployment)", "",
              "|Seed|Δ H1 MSE|Δ H2 MSE|Δ H3 MSE|Δ H3 C-index|Δ H3 Brier365|",
              "|---|---:|---:|---:|---:|---:|"]
    for row in seed_differences:
        lines.append(f"|{row['seed']}|" + "|".join(f"{row[key]:.4f}" for key in
                      ("H1_latent_mse", "H2_latent_mse", "H3_latent_mse", "H3_c_index", "H3_brier365")) + "|")
    lines += ["", "## Patient-paired errors", "",
              "[Full patient × seed paired errors](patient_paired_errors.csv). Each patient's latent MSE is averaged over their windows.",
              "Survival uses only the earliest eligible window. Negative C−A errors favor C; C-index is a cohort metric, not a patient error.",
              "Unknown 365-day outcomes have blank squared errors and zero IPCW contribution; zero is not evidence of correct prediction.", "",
              "## C disagreement versus survival prediction error", "",
              "Error = (predicted survival365 − observed alive365)² among known outcomes only, separately within each seed.",
              "Uz = sum of population latent variances; Us = population variance of member survival probabilities.", "",
              "|Seed|Known patients|Uz Pearson|Uz Spearman|Us Pearson|Us Spearman|", "|---|---:|---:|---:|---:|---:|"]
    for row in summary["C_disagreement_vs_survival_error"]:
        uz, us = row["latent_disagreement"], row["survival_disagreement"]
        values = [uz["pearson"], uz["spearman"], us["pearson"], us["spearman"]]
        lines.append(f"|{row['seed']}|{uz['n']}|" + "|".join("undefined" if v is None else f"{v:.4f}" for v in values) + "|")
    lines += ["", "![C disagreement versus survival error](C_disagreement_survival_error.png)", "",
              "## Interpretation limits", "",
              "365 days is measured from the H3 endpoint, using the existing endpoint survival labels and treatment-prefix conditions.",
              "This is retrospective conditional outcome transfer, not a new baseline-only survival forecasting protocol.",
              "IPCW uses only training patients at the same H3 landmark. Censored-by-365 patients are excluded from error correlations.",
              "A/C latent MSEs use their own fine-tuned encoder coordinates; within-head true-versus-predicted gaps diagnose transfer.",
              "A and C also have different trained heads; cross-model changes are end-to-end effects, not isolated dynamics causality.",
              "Only eight test patients and three training seeds; seeds are not independent patient cohorts. No significance claims."]
    (report / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(seeds), figsize=(4 * len(seeds), 3.5), squeeze=False)
    for axis, seed, run in zip(axes[0], seeds, runs["C"]):
        patients = [p for p in run["outcome_transfer"]["patients"] if p["known365"]]
        axis.scatter([p["latent_disagreement"] for p in patients],
                     [p["predicted_latent_squared_error365"] for p in patients])
        axis.set(title=f"C seed {seed}, n={len(patients)}", xlabel="H3 latent disagreement Uz", ylabel="365-day squared error")
    figure.tight_layout()
    figure.savefig(report / "C_disagreement_survival_error.png", dpi=160)
    plt.close(figure)
    return summary
