"""Weight-of-evidence binning and a points-based logistic scorecard."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class FeatureBins:
    """Interior cut points for one feature and the WoE of each resulting bin."""

    name: str
    edges: np.ndarray            # interior cut points; bin i holds values in (edges[i-1], edges[i]]
    woe: np.ndarray
    iv: float
    counts: np.ndarray
    bad_rates: np.ndarray

    def bin_index(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, x, side="left")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.woe[self.bin_index(x)]

    def labels(self) -> list[str]:
        """Readable bin ranges, e.g. '(0.25, 0.5]'."""
        e = [-np.inf, *self.edges.tolist(), np.inf]
        return [f"({_fmt(lo)}, {_fmt(hi)}]" for lo, hi in zip(e[:-1], e[1:], strict=True)]


def _fmt(v: float) -> str:
    if np.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def woe_iv(bad: np.ndarray, good: np.ndarray, smooth: float = 0.5) -> tuple[np.ndarray, float]:
    """WoE = ln(share of goods / share of bads) per bin (positive = safer), and information value."""
    bad = bad.astype(float) + smooth
    good = good.astype(float) + smooth
    pb, pg = bad / bad.sum(), good / good.sum()
    w = np.log(pg / pb)
    return w, float(((pg - pb) * w).sum())


def candidate_edges(x: np.ndarray, max_bins: int) -> np.ndarray:
    """Interior cut points at quantiles; discrete features with few values get one bin per value."""
    uniq = np.unique(x)
    if len(uniq) <= max_bins:
        return uniq[:-1].astype(float)
    qs = np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1])
    return np.unique(qs)


def merge_small(x: np.ndarray, edges: np.ndarray, min_count: int) -> np.ndarray:
    """Drop cut points until every bin holds at least `min_count` rows."""
    edges = edges.copy()
    while len(edges):
        counts = np.bincount(np.searchsorted(edges, x, side="left"), minlength=len(edges) + 1)
        small = np.flatnonzero(counts < min_count)
        if not len(small):
            break
        i = int(small[0])
        # merge bin i with its smaller neighbour by removing the shared edge
        if i == 0:
            drop = 0
        elif i == len(counts) - 1:
            drop = i - 1
        else:
            drop = i - 1 if counts[i - 1] <= counts[i + 1] else i
        edges = np.delete(edges, drop)
    return edges


def fit_bins(x: np.ndarray, y: np.ndarray, name: str, max_bins: int, min_share: float) -> FeatureBins:
    """Quantile bins, small bins merged, then WoE and IV from the training labels."""
    edges = merge_small(x, candidate_edges(x, max_bins), max(1, int(min_share * len(x))))
    idx = np.searchsorted(edges, x, side="left")
    k = len(edges) + 1
    bad = np.bincount(idx, weights=y, minlength=k)
    total = np.bincount(idx, minlength=k)
    w, iv = woe_iv(bad, total - bad)
    return FeatureBins(name, edges, w, iv, total, bad / np.maximum(total, 1))


@dataclass
class WoEScorecard:
    """Bin every feature, keep those with IV above a floor, fit a logistic model on the WoE values."""

    max_bins: int = 6
    min_bin_share: float = 0.05
    min_iv: float = 0.02
    C: float = 1.0
    bins: dict[str, FeatureBins] = field(default_factory=dict)
    selected: list[str] = field(default_factory=list)
    model: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> WoEScorecard:
        y = np.asarray(y, dtype=float)
        self.bins = {c: fit_bins(X[c].to_numpy(dtype=float), y, c, self.max_bins, self.min_bin_share)
                     for c in X.columns}
        self.selected = [c for c, b in self.bins.items() if b.iv >= self.min_iv]
        if not self.selected:
            raise ValueError("no feature passed the IV floor")
        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(self.woe_frame(X), y)
        return self

    def woe_frame(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.bins[c].transform(X[c].to_numpy(dtype=float)) for c in self.selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit first"
        return self.model.predict_proba(self.woe_frame(X))

    def iv_table(self) -> list[dict[str, Any]]:
        rows = [{"feature": c, "iv": b.iv, "bins": len(b.woe), "selected": c in self.selected}
                for c, b in self.bins.items()]
        return sorted(rows, key=lambda r: -r["iv"])


def scaling(base_score: float, base_odds: float, pdo: float) -> tuple[float, float]:
    """Factor and offset so that score = offset + factor * ln(good:bad odds)."""
    factor = pdo / math.log(2)
    return factor, base_score - factor * math.log(base_odds)


def points_table(card: WoEScorecard, base_score: float, base_odds: float, pdo: float) -> pd.DataFrame:
    """Points per bin; an account's score is the sum of its bins' points (intercept spread evenly)."""
    assert card.model is not None, "fit first"
    factor, offset = scaling(base_score, base_odds, pdo)
    b0 = float(card.model.intercept_[0])
    coefs = card.model.coef_[0]
    n = len(card.selected)
    rows = []
    for c, beta in zip(card.selected, coefs, strict=True):
        fb = card.bins[c]
        for label, w, cnt, br in zip(fb.labels(), fb.woe, fb.counts, fb.bad_rates, strict=True):
            pts = (offset - factor * b0) / n - factor * beta * w
            rows.append({"feature": c, "bin": label, "accounts": int(cnt), "default_rate": float(br),
                         "woe": float(w), "points": round(float(pts))})
    return pd.DataFrame(rows)


def score(card: WoEScorecard, X: pd.DataFrame, base_score: float, base_odds: float, pdo: float) -> np.ndarray:
    """Exact (unrounded) score for each account: offset + factor * ln(odds of not defaulting)."""
    factor, offset = scaling(base_score, base_odds, pdo)
    p = np.clip(card.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return offset + factor * np.log((1 - p) / p)
