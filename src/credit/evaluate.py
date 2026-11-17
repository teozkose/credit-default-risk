"""Ranking and probability metrics with fast weighted bootstrap CIs, calibration and segment tables.

Bootstrap resamples are expressed as integer weights on the original rows (how often each row was
drawn). Scores are sorted once, so each resample is a few vectorised sums instead of a re-sort.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd


class SortedScores:
    """Scores sorted high to low and grouped into tied blocks, reusable across bootstrap weights."""

    def __init__(self, y: np.ndarray, p: np.ndarray):
        order = np.argsort(-p, kind="mergesort")
        self.y = np.asarray(y, dtype=float)[order]
        self.p = np.asarray(p, dtype=float)[order]
        self.order = order
        change = np.r_[True, self.p[1:] != self.p[:-1]]
        self.starts = np.flatnonzero(change)

    def curve(self, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative weighted positives and negatives at each distinct threshold (high to low)."""
        w = np.ones_like(self.y) if w is None else np.asarray(w, dtype=float)[self.order]
        pos = np.add.reduceat(w * self.y, self.starts)
        neg = np.add.reduceat(w * (1 - self.y), self.starts)
        keep = (pos + neg) > 0  # a tied block that the resample never drew is not a threshold
        return np.cumsum(pos[keep]), np.cumsum(neg[keep])

    def metrics(self, w: np.ndarray | None = None) -> dict[str, float]:
        tp, fp = self.curve(w)
        P, N = tp[-1], fp[-1]
        if P == 0 or N == 0:
            return {"roc_auc": np.nan, "pr_auc": np.nan, "ks": np.nan, "brier": np.nan}
        tpr = np.r_[0.0, tp / P]
        fpr = np.r_[0.0, fp / N]
        roc_auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
        precision = tp / (tp + fp)
        pr_auc = float(np.sum((tpr[1:] - tpr[:-1]) * precision))  # average precision
        ks = float(np.max(tpr - fpr))
        ww = np.ones_like(self.y) if w is None else np.asarray(w, dtype=float)[self.order]
        brier = float(np.sum(ww * (self.p - self.y) ** 2) / ww.sum())
        return {"roc_auc": roc_auc, "pr_auc": pr_auc, "ks": ks, "brier": brier}


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC with ties counted as half."""
    return SortedScores(y, p).metrics()["roc_auc"]


def all_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """ROC-AUC, PR-AUC (average precision), KS and Brier score."""
    return SortedScores(y, p).metrics()


def bootstrap_weights(n: int, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Yield n_boot vectors of resample counts (each sums to n). The same seed gives the same resamples,
    so calling it again per model keeps model comparisons paired without holding a large matrix."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n)


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> list[float]:
    lo, hi = np.nanpercentile(draws, [50 * (1 - level), 50 * (1 + level)])
    return [float(lo), float(hi)]


def metrics_with_ci(y: np.ndarray, preds: dict[str, np.ndarray], n_boot: int, seed: int) -> dict[str, dict]:
    """Point estimate and 95% CI for every metric of every model, using the same resamples for all models
    so differences between models are paired."""
    out: dict[str, dict] = {}
    draws: dict[str, dict[str, np.ndarray]] = {}
    for name, p in preds.items():
        s = SortedScores(y, p)
        point = s.metrics()
        boots = [s.metrics(w) for w in bootstrap_weights(len(y), n_boot, seed)]
        draws[name] = {k: np.array([b[k] for b in boots]) for k in point}
        out[name] = {k: v for k, v in point.items()}
        out[name].update({f"{k}_ci": percentile_ci(draws[name][k]) for k in point})
    out["_draws"] = draws  # removed before writing JSON
    return out


def paired_difference(draws: dict[str, dict[str, np.ndarray]], a: str, b: str, metric: str) -> dict[str, Any]:
    """CI of metric(a) - metric(b) over the shared resamples and the share of resamples where a wins."""
    d = draws[a][metric] - draws[b][metric]
    return {"mean": float(np.nanmean(d)), "ci": percentile_ci(d), "share_a_better": float(np.mean(d > 0))}


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int) -> pd.DataFrame:
    """Mean predicted vs observed default rate in equal-count bins of the prediction."""
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"y": y, "p": p, "bin": q})
    g = df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return g.reset_index(drop=True)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    """Count-weighted mean |observed - predicted| over equal-count bins."""
    c = calibration_curve(y, p, bins)
    return float(np.sum(c["n"] * (c["observed"] - c["predicted"]).abs()) / c["n"].sum())


def group_ci(values: np.ndarray, fn: Callable[[np.ndarray], float], n_boot: int, seed: int) -> list[float]:
    """Bootstrap CI for a statistic of a 1-D array (used for per-segment rates)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    return percentile_ci(np.array([fn(values[rng.integers(0, n, n)]) for _ in range(n_boot)]))


def segment_table(frame: pd.DataFrame, pd_col: str, approve: np.ndarray, segments: list[str],
                  n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Per segment: size, default rate, mean PD, ROC-AUC with CI, approval rate at the chosen threshold."""
    rows = []
    for seg in segments:
        for level, g in frame.groupby(seg, sort=True, observed=True):
            idx = g.index.to_numpy()
            y, p = g["default"].to_numpy(), g[pd_col].to_numpy()
            s = SortedScores(y, p)
            boots = np.array([s.metrics(wi)["roc_auc"] for wi in bootstrap_weights(len(g), n_boot, seed)])
            rows.append({"segment": seg, "level": str(level), "n": len(g), "default_rate": float(y.mean()),
                         "mean_pd": float(p.mean()), "roc_auc": s.metrics()["roc_auc"],
                         "roc_auc_ci": percentile_ci(boots), "approval_rate": float(approve[idx].mean())})
    return rows
