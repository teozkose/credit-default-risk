import numpy as np
import pandas as pd

from credit.evaluate import auc
from credit.woe import WoEScorecard, merge_small, points_table, score, woe_iv


def test_woe_is_positive_for_safer_bins_and_iv_nonnegative():
    w, iv = woe_iv(bad=np.array([10, 90]), good=np.array([90, 10]))
    assert w[0] > 0 > w[1]
    assert iv > 0


def test_merge_small_leaves_no_tiny_bins():
    x = np.r_[np.zeros(500), np.ones(5), np.full(500, 2.0)]
    edges = merge_small(x, np.array([0.0, 1.0]), min_count=50)
    counts = np.bincount(np.searchsorted(edges, x, side="left"))
    assert counts.min() >= 50


def test_scorecard_learns_signal_and_points_add_up_to_score():
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 4, n), "noise": rng.normal(size=n)})
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X["a"] + 0.5 * X["b"] - 1)))).astype(int).to_numpy()
    card = WoEScorecard(max_bins=6, min_bin_share=0.05, min_iv=0.02).fit(X, y)
    assert "a" in card.selected and "noise" not in card.selected
    assert auc(y, card.predict_proba(X)[:, 1]) > 0.7

    pts = points_table(card, 600, 50, 20)
    total = np.zeros(n)
    for f in card.selected:
        rows = pts[pts["feature"] == f]["points"].to_numpy()
        total += rows[card.bins[f].bin_index(X[f].to_numpy(dtype=float))]
    exact = score(card, X, 600, 50, 20)
    assert np.max(np.abs(total - exact)) <= len(card.selected) * 0.5 + 1e-6  # rounding only
    assert np.corrcoef(exact, card.predict_proba(X)[:, 1])[0, 1] < -0.99  # higher score = safer
