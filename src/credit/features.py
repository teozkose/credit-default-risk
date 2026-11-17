"""Cleaning steps and account-level features: utilisation, payment ratios, delinquency streaks, bill trend."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit.data import BILL_COLS, MONTHS, PAID_COLS, STATUS_COLS, TARGET

EDUCATION_LABELS = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
SEX_LABELS = {1: "male", 2: "female"}

# Model inputs. SEX and MARRIAGE are never inputs (see config.features.excluded).
FEATURES: list[str] = [
    "limit", "age", "education",
    *STATUS_COLS,
    "util_1", "util_mean", "util_max",
    "pay_ratio_1", "pay_ratio_mean", "pay_ratio_min_3",
    "months_late", "max_late", "late_streak", "months_revolving", "months_no_payment",
    "bill_trend", "balance_change_3",
]
CATEGORICAL = ["education"]


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fold undocumented category codes and drop exact duplicate accounts; return the frame and a step log."""
    log: list[dict[str, Any]] = []
    out = df.copy()

    def step(name: str, before: int, after: int, changed: int | None = None) -> None:
        log.append({"step": name, "rows_before": before, "dropped": before - after, "rows_after": after,
                    "changed": changed if changed is not None else 0})

    n = len(out)
    mask = out["education"].isin(cfg["cleaning"]["education_other"])
    out.loc[mask, "education"] = 4
    step("Fold undocumented EDUCATION codes 0, 5, 6 into 4 (other)", n, n, int(mask.sum()))

    mask = out["marriage"].isin(cfg["cleaning"]["marriage_other"])
    out.loc[mask, "marriage"] = 3
    step("Fold undocumented MARRIAGE code 0 into 3 (other)", n, n, int(mask.sum()))

    cols = [c for c in out.columns if c != "account_id"]
    out = out.loc[~out[cols].duplicated()].reset_index(drop=True)
    step("Drop accounts identical on every field except the id", n, len(out))
    return out, log


def payment_ratios(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Share of the previous statement paid in each month.

    A payment in month k settles the bill issued in month k+1 (one month older). When nothing was
    owed the ratio is 1 (nothing to miss). Ratios are clipped to [0, cap].
    """
    ratios = {}
    for k in range(1, MONTHS):
        owed = df[f"bill_{k + 1}"].astype(float)
        paid = df[f"paid_{k}"].astype(float)
        r = np.where(owed > 0, paid / owed.where(owed > 0, 1.0), 1.0)
        ratios[f"pay_ratio_{k}"] = np.clip(r, 0.0, cap)
    return pd.DataFrame(ratios, index=df.index)


def late_streak(status: np.ndarray) -> np.ndarray:
    """Consecutive months, counting back from the most recent, with a status of 1+ months late."""
    late = status > 0
    # cumprod stops counting at the first month that was not late
    return np.cumprod(late, axis=1).sum(axis=1)


def bill_slope(bills: np.ndarray) -> np.ndarray:
    """Least-squares slope of the bill per month, oldest to newest (positive = balance growing)."""
    t = np.arange(bills.shape[1], dtype=float)
    t -= t.mean()
    chrono = bills[:, ::-1]  # column 0 becomes the oldest month
    centred = chrono - chrono.mean(axis=1, keepdims=True)
    return centred @ t / (t @ t)


def band_labels(edges: list[int], kind: str) -> list[str]:
    """Readable labels: age [20, 30) -> '20-29'; limit (0, 50000] -> 'NT$0-50k'."""
    if kind == "age":
        return [f"{lo}-{hi - 1}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return [f"NT${lo // 1000}k-{hi // 1000}k" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]


def build_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Account-level feature table plus the target, exposure and segment labels."""
    fc = cfg["features"]
    limit = df["limit"].astype(float)
    bills = df[BILL_COLS].to_numpy(dtype=float)
    status = df[STATUS_COLS].to_numpy()
    util = np.clip(bills / limit.to_numpy()[:, None], -1.0, fc["utilisation_cap"])
    ratios = payment_ratios(df, fc["payment_ratio_cap"])
    owed_prev = df[BILL_COLS[1:]].to_numpy() > 0
    unpaid = (df[PAID_COLS[:-1]].to_numpy() == 0) & owed_prev

    out = pd.DataFrame(index=df.index)
    out["account_id"] = df["account_id"]
    out["limit"] = df["limit"]
    out["age"] = df["age"]
    out["education"] = df["education"]
    for col in STATUS_COLS:
        out[col] = df[col]
    out["util_1"] = util[:, 0]
    out["util_mean"] = util.mean(axis=1)
    out["util_max"] = util.max(axis=1)
    out["pay_ratio_1"] = ratios["pay_ratio_1"]
    out["pay_ratio_mean"] = ratios.mean(axis=1)
    out["pay_ratio_min_3"] = ratios[["pay_ratio_1", "pay_ratio_2", "pay_ratio_3"]].min(axis=1)
    out["months_late"] = (status > 0).sum(axis=1)
    out["max_late"] = status.max(axis=1).clip(min=0)
    out["late_streak"] = late_streak(status)
    out["months_revolving"] = (status == 0).sum(axis=1)
    out["months_no_payment"] = unpaid.sum(axis=1)
    out["bill_trend"] = bill_slope(bills) / limit
    out["balance_change_3"] = (bills[:, 0] - bills[:, 2]) / limit

    # exposure for the decision step: current balance with a small floor
    out["exposure"] = np.maximum(df["bill_1"].astype(float), cfg["decision"]["min_exposure"])
    out["sex"] = df["sex"].map(SEX_LABELS)
    out["education_label"] = df["education"].map(EDUCATION_LABELS)
    out["age_band"] = pd.cut(df["age"], fc["age_bands"], right=False, labels=band_labels(fc["age_bands"], "age"))
    out["limit_band"] = pd.cut(df["limit"], fc["limit_bands"], right=True,
                               labels=band_labels(fc["limit_bands"], "limit"))
    out[TARGET] = df[TARGET]
    return out


def build_table(raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Clean then featurise; returns the modelling table and the cleaning log."""
    cleaned, log = clean(raw, cfg)
    return build_features(cleaned, cfg), log
