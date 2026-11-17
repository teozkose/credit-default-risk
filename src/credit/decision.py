"""Cost-based approval threshold: expected credit loss on approved defaulters vs margin lost on rejected goods."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def policy_cost(y: np.ndarray, pd_hat: np.ndarray, exposure: np.ndarray, threshold: float,
                lgd: float, margin: float, w: np.ndarray | None = None) -> dict[str, float]:
    """Realised cost of approving every account whose PD is below the threshold.

    Approved defaulter: lose LGD x exposure. Rejected good account: forgo margin x exposure.
    Rejected defaulters and approved goods cost nothing relative to a perfect decision.
    """
    w = np.ones_like(exposure, dtype=float) if w is None else w
    approve = pd_hat < threshold
    loss = np.sum(w * approve * y * lgd * exposure)
    lost_margin = np.sum(w * ~approve * (1 - y) * margin * exposure)
    n = w.sum()
    return {"threshold": threshold, "approval_rate": float(np.sum(w * approve) / n),
            "credit_loss": float(loss), "lost_margin": float(lost_margin), "total": float(loss + lost_margin),
            "per_1000": float((loss + lost_margin) / n * 1000)}


def analytic_threshold(lgd: float, margin: float) -> float:
    """With calibrated PDs, approving is worth it while PD x LGD < (1 - PD) x margin,
    i.e. PD < margin / (LGD + margin). Exposure cancels out."""
    return margin / (lgd + margin)


def sweep(y: np.ndarray, pd_hat: np.ndarray, exposure: np.ndarray, grid: np.ndarray,
          lgd: float, margin: float) -> pd.DataFrame:
    """Policy cost at every threshold on the grid (threshold 1.0 = approve everyone)."""
    return pd.DataFrame([policy_cost(y, pd_hat, exposure, float(t), lgd, margin) for t in grid])


def choose_threshold(y: np.ndarray, pd_hat: np.ndarray, exposure: np.ndarray, grid: np.ndarray,
                     lgd: float, margin: float) -> dict[str, Any]:
    """Threshold with the lowest realised cost, compared with approving everyone."""
    s = sweep(y, pd_hat, exposure, grid, lgd, margin)
    best = s.loc[s["total"].idxmin()].to_dict()
    everyone = policy_cost(y, pd_hat, exposure, 1.01, lgd, margin)
    return {"best": best, "approve_all": everyone, "sweep": s,
            "saving_share": 1 - best["total"] / everyone["total"],
            "analytic": analytic_threshold(lgd, margin)}


def saving_ci(y: np.ndarray, pd_hat: np.ndarray, exposure: np.ndarray, threshold: float, lgd: float,
              margin: float, weights: Iterable[np.ndarray]) -> list[float]:
    """Bootstrap CI of the saving per 1,000 accounts against approving everyone, at a fixed threshold."""
    out = []
    for w in weights:
        a = policy_cost(y, pd_hat, exposure, threshold, lgd, margin, w)["per_1000"]
        b = policy_cost(y, pd_hat, exposure, 1.01, lgd, margin, w)["per_1000"]
        out.append(b - a)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def cost_difference_ci(y: np.ndarray, pd_a: np.ndarray, thr_a: float, pd_b: np.ndarray, thr_b: float,
                       exposure: np.ndarray, lgd: float, margin: float,
                       weights: Iterable[np.ndarray]) -> dict[str, Any]:
    """Paired bootstrap of cost per 1,000 accounts, policy A minus policy B (negative = A is cheaper)."""
    d = np.array([policy_cost(y, pd_a, exposure, thr_a, lgd, margin, w)["per_1000"]
                  - policy_cost(y, pd_b, exposure, thr_b, lgd, margin, w)["per_1000"] for w in weights])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "share_a_cheaper": float(np.mean(d < 0))}


def fairness_table(y: np.ndarray, approve: np.ndarray, pd_hat: np.ndarray, group: np.ndarray) -> list[dict]:
    """Approval and error rates by group. Rejected-good rate = share of non-defaulters turned down;
    approved-bad rate = share of defaulters let through."""
    rows = []
    for g in sorted(pd.unique(group)):
        m = group == g
        yy, aa = y[m], approve[m]
        rows.append({"group": str(g), "n": int(m.sum()), "default_rate": float(yy.mean()),
                     "mean_pd": float(pd_hat[m].mean()), "approval_rate": float(aa.mean()),
                     "rejected_good_rate": float((~aa & (yy == 0)).sum() / max((yy == 0).sum(), 1)),
                     "approved_bad_rate": float((aa & (yy == 1)).sum() / max((yy == 1).sum(), 1))})
    return rows
