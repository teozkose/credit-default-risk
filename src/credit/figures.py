"""Figures for the README and report page. Each function writes one PNG under results/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from credit.evaluate import SortedScores  # noqa: E402

LABELS = {"baseline": "Baseline (default rate)", "scorecard": "WoE scorecard", "logistic": "Logistic regression",
          "xgboost": "XGBoost", "xgboost_isotonic": "XGBoost + isotonic"}
COLORS = {"baseline": "#a0aec0", "scorecard": "#2b6cb0", "logistic": "#38a169", "xgboost": "#dd6b20",
          "xgboost_isotonic": "#805ad5"}
DPI = 120


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def roc_pr(art: dict[str, Any], out: Path) -> None:
    """ROC and precision-recall curves for every model on out-of-fold predictions."""
    y = art["y"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name in ["baseline", "scorecard", "logistic", "xgboost"]:
        tp, fp = SortedScores(y, art["oof"][name]).curve()
        tpr, fpr = np.r_[0, tp / tp[-1]], np.r_[0, fp / fp[-1]]
        prec = np.r_[1, tp / (tp + fp)]
        a.plot(fpr, tpr, color=COLORS[name], label=LABELS[name])
        b.plot(tpr, prec, color=COLORS[name], label=LABELS[name])
    a.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8)
    a.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (5-fold out-of-fold)")
    b.axhline(y.mean(), ls=":", color="grey", lw=0.8)
    b.set(xlabel="Recall (defaulters caught)", ylabel="Precision", title="Precision-recall", ylim=(0, 1.02))
    a.legend(loc="lower right", fontsize=8)
    _save(fig, out / "roc_pr.png")


def calibration(art: dict[str, Any], out: Path) -> None:
    """Reliability diagram: mean predicted PD against observed default rate in deciles of the prediction."""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, c in art["curves"].items():
        ax.plot(c["predicted"], c["observed"], marker="o", ms=4, color=COLORS[name], label=LABELS[name])
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8, label="Perfect calibration")
    lim = max(max(float(max(c["observed"].max(), c["predicted"].max())) for c in art["curves"].values()) + 0.05, 0.2)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted default probability",
           ylabel="Observed default rate", title="Calibration by decile of predicted PD")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, out / "calibration.png")


def threshold_sweep(m: dict[str, Any], art: dict[str, Any], out: Path) -> None:
    """Expected cost per 1,000 accounts against the PD cut-off, for each LGD assumption."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for lgd, s in sorted(art["sweeps"].items()):
        line, = ax.plot(s["threshold"], s["per_1000"] / 1e6, label=f"LGD {lgd:.0%}")
        best = s.loc[s["total"].idxmin()]
        ax.scatter([best["threshold"]], [best["per_1000"] / 1e6], color=line.get_color(), zorder=3)
    base = m["decision"]["lgd"]
    ax.axvline(art["threshold"], ls="--", color="grey", lw=0.8)
    ax.text(art["threshold"], ax.get_ylim()[1] * 0.97, f"  chosen {art['threshold']:.2f} (LGD {base:.0%})",
            va="top", fontsize=8, color="grey")
    ax.set(xlabel="Decline when predicted PD is at or above", ylabel="Cost per 1,000 accounts (NT$ millions)",
           title="Credit loss + lost margin across cut-offs")
    ax.legend(fontsize=8)
    _save(fig, out / "threshold_sweep.png")


def segments(m: dict[str, Any], out: Path) -> None:
    """ROC-AUC with 95% CIs and approval rate by segment."""
    rows = m["segments"]  # already ordered by segment, then band
    labels = [f"{r['segment'].replace('_label', '').replace('_', ' ')}: {r['level']}" for r in rows]
    auc = np.array([r["roc_auc"] for r in rows])
    lo = auc - np.array([r["roc_auc_ci"][0] for r in rows])
    hi = np.array([r["roc_auc_ci"][1] for r in rows]) - auc
    ypos = np.arange(len(rows))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 0.36 * len(rows) + 1.4), sharey=True)
    a.errorbar(auc, ypos, xerr=[lo, hi], fmt="o", color="#2b6cb0", ms=4, capsize=2)
    a.axvline(m["models"]["xgboost_isotonic"]["roc_auc"], ls=":", color="grey")
    a.set(yticks=ypos, yticklabels=labels, xlabel="ROC-AUC (95% CI)", title="Ranking quality by segment")
    b.barh(ypos, [r["approval_rate"] for r in rows], color="#38a169", alpha=0.8, label="approved")
    b.scatter([r["default_rate"] for r in rows], ypos, color="#c53030", zorder=3, s=14, label="default rate")
    b.set(xlabel="Share of accounts", xlim=(0, 1), title="Approval rate at the chosen cut-off")
    b.legend(fontsize=8, loc="lower right")
    _save(fig, out / "segments.png")


def drivers(m: dict[str, Any], out: Path) -> None:
    """Information value of each scorecard input and XGBoost gain share."""
    iv = [r for r in m["scorecard"]["iv"]][:12][::-1]
    gain = m["importance"]["xgboost_gain"][:12][::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6))
    a.barh([r["feature"] for r in iv], [r["iv"] for r in iv],
           color=["#2b6cb0" if r["selected"] else "#cbd5e0" for r in iv])
    a.set(xlabel="Information value", title="Scorecard: information value (grey = dropped)")
    b.barh([r["feature"] for r in gain], [r["share"] for r in gain], color="#dd6b20")
    b.set(xlabel="Share of total gain", title="XGBoost: gain importance")
    _save(fig, out / "drivers.png")


def score_distribution(art: dict[str, Any], out: Path) -> None:
    """Scorecard points for accounts that defaulted vs did not."""
    s, y = art["card_scores"], art["y"]
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 45)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(s[y == 0], bins=bins, alpha=0.6, density=True, color="#38a169", label="paid next month")
    ax.hist(s[y == 1], bins=bins, alpha=0.6, density=True, color="#c53030", label="defaulted next month")
    ax.set(xlabel="Scorecard score (higher = safer)", ylabel="Density", title="Score distribution by outcome")
    ax.legend(fontsize=8)
    _save(fig, out / "score_distribution.png")


def make_all(m: dict[str, Any], art: dict[str, Any], out: Path) -> None:
    roc_pr(art, out)
    calibration(art, out)
    threshold_sweep(m, art, out)
    segments(m, out)
    drivers(m, out)
    score_distribution(art, out)
