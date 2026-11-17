import numpy as np

from credit.decision import analytic_threshold, choose_threshold, fairness_table, policy_cost


def test_policy_cost_by_hand():
    y = np.array([1, 0, 1, 0])
    pd_hat = np.array([0.1, 0.2, 0.6, 0.7])
    exposure = np.array([100.0, 100.0, 100.0, 100.0])
    c = policy_cost(y, pd_hat, exposure, threshold=0.5, lgd=0.6, margin=0.2)
    # approved: rows 0 (default -> 60 lost) and 1 (good); declined: 2 (default) and 3 (good -> 20 margin lost)
    assert c["credit_loss"] == 60 and c["lost_margin"] == 20 and c["approval_rate"] == 0.5
    assert c["per_1000"] == 80 / 4 * 1000


def test_empirical_threshold_close_to_analytic_on_calibrated_pd():
    rng = np.random.default_rng(0)
    pd_hat = rng.beta(2, 6, 50000)
    y = (rng.uniform(size=pd_hat.size) < pd_hat).astype(int)
    exposure = rng.uniform(1000, 50000, pd_hat.size)
    out = choose_threshold(y, pd_hat, exposure, np.arange(0.02, 0.9, 0.01), lgd=0.6, margin=0.18)
    assert abs(out["best"]["threshold"] - analytic_threshold(0.6, 0.18)) <= 0.04
    assert out["best"]["total"] < out["approve_all"]["total"]


def test_fairness_rates_are_conditional_on_outcome():
    y = np.array([0, 0, 1, 1, 0, 1])
    approve = np.array([True, False, True, False, True, True])
    group = np.array(["a", "a", "a", "a", "b", "b"])
    rows = {r["group"]: r for r in fairness_table(y, approve, np.full(6, 0.3), group)}
    assert rows["a"]["rejected_good_rate"] == 0.5
    assert rows["a"]["approved_bad_rate"] == 0.5
    assert rows["b"]["approval_rate"] == 1.0
