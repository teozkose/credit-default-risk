"""Stages: data -> features -> run (cross-validated models, evaluation, decision) -> report.

Usage: python -m credit.pipeline {data,features,run,report}
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from credit import decision, evaluate, figures
from credit.config import load_config, resolve
from credit.data import TARGET, fetch_and_cache, load_raw
from credit.features import FEATURES, build_table
from credit.models import (
    MODEL_ORDER,
    PriorBaseline,
    make_isotonic,
    make_logistic,
    make_scorecard,
    make_xgboost,
    tune_xgboost,
)
from credit.validate import validate_raw
from credit.woe import points_table, score

DECISION_MODEL = "xgboost_isotonic"
SEGMENTS = ["age_band", "education_label", "limit_band"]


def cross_validate(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Out-of-fold default probabilities for every model from stratified k-fold CV.

    Tuning and isotonic calibration happen inside each training fold, so no prediction ever comes from a
    model that saw that account."""
    X, y = table[FEATURES], table[TARGET].to_numpy()
    folds = StratifiedKFold(n_splits=cfg["validation"]["folds"], shuffle=True, random_state=cfg["validation"]["seed"])
    oof = {m: np.zeros(len(table)) for m in MODEL_ORDER}
    fold_auc: dict[str, list[float]] = {m: [] for m in MODEL_ORDER}
    tuning = []
    for k, (tr, te) in enumerate(folds.split(X, y)):
        t0 = time.time()
        Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
        params, inner_auc = tune_xgboost(Xtr, ytr, cfg)
        tuning.append({"fold": k + 1, "inner_auc": inner_auc, "params": params})
        fitted = {
            "baseline": PriorBaseline().fit(Xtr, ytr),
            "scorecard": make_scorecard(cfg).fit(Xtr, ytr),
            "logistic": make_logistic(cfg, FEATURES).fit(Xtr, ytr),
            "xgboost": make_xgboost(cfg, **params).fit(Xtr, ytr),
            "xgboost_isotonic": make_isotonic(cfg, params).fit(Xtr, ytr),
        }
        for name, model in fitted.items():
            oof[name][te] = model.predict_proba(Xte)[:, 1]
            fold_auc[name].append(evaluate.auc(y[te], oof[name][te]))
        print(f"fold {k + 1}: xgboost AUC {fold_auc['xgboost'][-1]:.3f} "
              f"scorecard {fold_auc['scorecard'][-1]:.3f} ({time.time() - t0:.0f}s)")
    info = {"fold_auc": {m: [float(a) for a in v] for m, v in fold_auc.items()}, "tuning": tuning}
    return oof, info


def run_experiment(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Everything after feature building. Returns JSON-ready metrics and in-memory artefacts for figures."""
    table = table.reset_index(drop=True)
    ev, dc, sc = cfg["evaluation"], cfg["decision"], cfg["scorecard"]
    seed, n_boot = cfg["validation"]["seed"], ev["n_boot"]
    y = table[TARGET].to_numpy()
    exposure = table["exposure"].to_numpy(dtype=float)

    oof, cv_info = cross_validate(table, cfg)
    scores = evaluate.metrics_with_ci(y, oof, n_boot, seed)
    draws = scores.pop("_draws")
    for name in MODEL_ORDER:
        scores[name]["fold_auc_sd"] = float(np.std(cv_info["fold_auc"][name], ddof=1))
        scores[name]["ece"] = (evaluate.expected_calibration_error(y, oof[name], ev["calibration_bins"])
                               if name != "baseline" else float(abs(oof[name].mean() - y.mean())))
    comparisons = {
        "xgboost_vs_scorecard_auc": evaluate.paired_difference(draws, "xgboost", "scorecard", "roc_auc"),
        "xgboost_vs_logistic_auc": evaluate.paired_difference(draws, "xgboost", "logistic", "roc_auc"),
        "isotonic_vs_raw_brier": evaluate.paired_difference(draws, "xgboost_isotonic", "xgboost", "brier"),
    }
    curves = {m: evaluate.calibration_curve(y, oof[m], ev["calibration_bins"])
              for m in ["scorecard", "xgboost", "xgboost_isotonic"]}

    # decision: threshold on the calibrated PD
    start, stop, step = dc["threshold_grid"]
    grid = np.round(np.arange(start, stop + step / 2, step), 4)
    pd_hat = oof[DECISION_MODEL]
    chosen = decision.choose_threshold(y, pd_hat, exposure, grid, dc["lgd"], dc["margin"])
    thr = float(chosen["best"]["threshold"])
    approve = pd_hat < thr
    saving = chosen["approve_all"]["per_1000"] - chosen["best"]["per_1000"]
    saving_ci = decision.saving_ci(y, pd_hat, exposure, thr, dc["lgd"], dc["margin"],
                                   evaluate.bootstrap_weights(len(y), n_boot, seed))
    sens, sweeps = [], {}
    for lgd in dc["lgd_sensitivity"]:
        c = decision.choose_threshold(y, pd_hat, exposure, grid, lgd, dc["margin"])
        sweeps[lgd] = c["sweep"]
        at_base = decision.policy_cost(y, pd_hat, exposure, thr, lgd, dc["margin"])
        sens.append({"lgd": lgd, "threshold": float(c["best"]["threshold"]), "analytic": c["analytic"],
                     "approval_rate": float(c["best"]["approval_rate"]),
                     "saving_per_1000": c["approve_all"]["per_1000"] - c["best"]["per_1000"],
                     "saving_share": float(c["saving_share"]),
                     "regret_of_base_threshold": at_base["per_1000"] - c["best"]["per_1000"]})
    scorecard_choice = decision.choose_threshold(y, oof["scorecard"], exposure, grid, dc["lgd"], dc["margin"])
    vs_scorecard = decision.cost_difference_ci(
        y, pd_hat, thr, oof["scorecard"], float(scorecard_choice["best"]["threshold"]), exposure, dc["lgd"],
        dc["margin"], evaluate.bootstrap_weights(len(y), n_boot, seed))

    table["pd"] = pd_hat
    segments = evaluate.segment_table(table, "pd", approve, SEGMENTS, min(n_boot, 300), seed)
    fairness = decision.fairness_table(y, approve, pd_hat, table["sex"].to_numpy())

    # final models on every account: the points table and the driver ranking
    card = make_scorecard(cfg).fit(table[FEATURES], y)
    points = points_table(card, sc["base_score"], sc["base_odds"], sc["pdo"])
    card_scores = score(card, table[FEATURES], sc["base_score"], sc["base_odds"], sc["pdo"])
    best_fold = max(cv_info["tuning"], key=lambda t: t["inner_auc"])
    xgb = make_xgboost(cfg, **best_fold["params"]).fit(table[FEATURES], y)
    gain = xgb.get_booster().get_score(importance_type="gain")
    total_gain = sum(gain.values())
    gain_rank = sorted(({"feature": f, "share": v / total_gain} for f, v in gain.items()), key=lambda r: -r["share"])

    metrics = {
        "headline": {
            "model": DECISION_MODEL, "roc_auc": scores[DECISION_MODEL]["roc_auc"],
            "roc_auc_ci": scores[DECISION_MODEL]["roc_auc_ci"],
            "threshold": thr, "approval_rate": float(chosen["best"]["approval_rate"]),
            "saving_per_1000": saving, "saving_ci": saving_ci, "saving_share": float(chosen["saving_share"]),
        },
        "cv": {"folds": cfg["validation"]["folds"], "rows": len(table), "default_rate": float(y.mean()),
               **cv_info},
        "models": {m: scores[m] for m in MODEL_ORDER},
        "comparisons": comparisons,
        "calibration": {m: c.round(4).to_dict(orient="list") for m, c in curves.items()},
        "decision": {
            "lgd": dc["lgd"], "margin": dc["margin"], "min_exposure": dc["min_exposure"],
            "best": chosen["best"], "approve_all": chosen["approve_all"], "analytic_threshold": chosen["analytic"],
            "scorecard_best": scorecard_choice["best"], "scorecard_saving_share": scorecard_choice["saving_share"],
            "xgboost_minus_scorecard_cost": vs_scorecard,
            "sensitivity": sens,
        },
        "segments": segments,
        "fairness": fairness,
        "scorecard": {
            "selected": card.selected, "iv": card.iv_table(), "points_rows": len(points),
            "score_range": [float(np.percentile(card_scores, 1)), float(np.percentile(card_scores, 99))],
            "score_median_default": float(np.median(card_scores[y == 1])),
            "score_median_good": float(np.median(card_scores[y == 0])),
        },
        "importance": {"xgboost_gain": gain_rank[:12]},
    }
    artefacts = {"oof": oof, "y": y, "table": table, "curves": curves, "sweeps": sweeps, "points": points,
                 "card_scores": card_scores, "threshold": thr}
    return metrics, artefacts


def stage_data(cfg: dict[str, Any]) -> None:
    raw = fetch_and_cache(cfg)
    summary = validate_raw(raw)
    print(f"data: {summary['rows']:,} accounts, default rate {summary['default_rate']:.3f}")


def stage_features(cfg: dict[str, Any]) -> None:
    raw = load_raw(resolve(cfg["data"]["raw"]))
    summary = validate_raw(raw)
    table, log = build_table(raw, cfg)
    table.to_parquet(resolve(cfg["data"]["table"]), index=False)
    results = resolve(cfg["paths"]["results"])
    results.mkdir(exist_ok=True)
    (results / "data_summary.json").write_text(json.dumps({**summary, "cleaning": log,
                                                           "modelling_rows": len(table)}, indent=2))
    print(f"features: {len(table):,} rows x {len(FEATURES)} model inputs")


def stage_run(cfg: dict[str, Any]) -> None:
    t0 = time.time()
    table = pd.read_parquet(resolve(cfg["data"]["table"]))
    results = resolve(cfg["paths"]["results"])
    metrics, art = run_experiment(table, cfg)
    metrics["data"] = json.loads((results / "data_summary.json").read_text())
    metrics["runtime_s"] = round(time.time() - t0, 1)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))
    art["points"].to_csv(results / "scorecard_points.csv", index=False)
    preds = art["table"][["account_id", TARGET]].copy()
    for m, p in art["oof"].items():
        preds[m] = p
    preds.to_parquet(resolve(cfg["data"]["predictions"]), index=False)
    figures.make_all(metrics, art, results)
    print(f"run: done in {metrics['runtime_s']:.0f}s")


def stage_report(cfg: dict[str, Any]) -> None:
    from credit.report import render_all

    render_all(cfg)


STAGES = {"data": stage_data, "features": stage_features, "run": stage_run, "report": stage_report}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in STAGES:
        raise SystemExit(f"usage: python -m credit.pipeline {{{','.join(STAGES)}}}")
    STAGES[args[0]](load_config())


if __name__ == "__main__":
    main()
