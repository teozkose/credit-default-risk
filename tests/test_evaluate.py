import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit.evaluate import (
    SortedScores,
    all_metrics,
    bootstrap_weights,
    calibration_curve,
    metrics_with_ci,
    paired_difference,
)


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, n), 0, 1).round(2)  # rounding creates ties
    return y, p


def test_metrics_match_sklearn_with_ties():
    y, p = _data()
    m = all_metrics(y, p)
    assert np.isclose(m["roc_auc"], roc_auc_score(y, p))
    assert np.isclose(m["pr_auc"], average_precision_score(y, p))
    assert np.isclose(m["brier"], brier_score_loss(y, p))


def test_weights_equal_repeating_rows():
    y, p = _data(300, 1)
    w = next(bootstrap_weights(len(y), 1, seed=3))
    rep = np.repeat(np.arange(len(y)), w)
    weighted = SortedScores(y, p).metrics(w)
    explicit = all_metrics(y[rep], p[rep])
    for k in weighted:
        assert np.isclose(weighted[k], explicit[k])


def test_bootstrap_ci_brackets_point_and_pairs_models():
    y, p = _data(1500, 2)
    noisy = np.clip(p + np.random.default_rng(0).normal(0, 0.3, len(p)), 0, 1)
    out = metrics_with_ci(y, {"good": p, "noisy": noisy}, n_boot=100, seed=0)
    draws = out.pop("_draws")
    lo, hi = out["good"]["roc_auc_ci"]
    assert lo <= out["good"]["roc_auc"] <= hi
    diff = paired_difference(draws, "good", "noisy", "roc_auc")
    assert diff["ci"][0] > 0 and diff["share_a_better"] == 1.0


def test_calibration_curve_on_calibrated_scores():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    c = calibration_curve(y, p, 10)
    assert len(c) == 10 and c["n"].sum() == p.size
    assert np.max(np.abs(c["observed"] - c["predicted"])) < 0.03
