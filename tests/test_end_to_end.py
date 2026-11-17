"""Small end-to-end run on a synthetic sample generated inside the test."""

import json

from conftest import make_source

from credit.data import tidy
from credit.features import build_table
from credit.models import MODEL_ORDER
from credit.pipeline import run_experiment
from credit.validate import validate_raw


def test_pipeline_end_to_end(small_cfg):
    raw = tidy(make_source(n=1500, seed=3))
    summary = validate_raw(raw)
    table, log = build_table(raw, small_cfg)
    metrics, art = run_experiment(table, small_cfg)
    json.dumps(metrics)  # must be JSON-ready

    models = metrics["models"]
    assert set(models) == set(MODEL_ORDER)
    assert abs(models["baseline"]["roc_auc"] - 0.5) < 0.03  # fold rates differ only slightly
    assert models["scorecard"]["roc_auc"] > 0.65 and models["xgboost"]["roc_auc"] > 0.65
    h = metrics["headline"]
    assert h["roc_auc_ci"][0] <= h["roc_auc"] <= h["roc_auc_ci"][1]
    assert h["saving_per_1000"] >= 0
    assert len(metrics["decision"]["sensitivity"]) == 3
    assert {r["group"] for r in metrics["fairness"]} == {"male", "female"}
    assert len(art["oof"]["xgboost"]) == len(table)
    assert summary["rows"] == len(raw) and log
