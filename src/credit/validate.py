"""Schema, range, null and duplicate checks on the raw table. Any hard failure stops the run."""

from __future__ import annotations

from typing import Any

import pandas as pd

from credit.data import BILL_COLS, PAID_COLS, RENAME, STATUS_COLS, TARGET

# Documented codes plus the undocumented ones that appear in the file (reported, not rejected).
ALLOWED: dict[str, set[int]] = {
    "sex": {1, 2},
    "education": {0, 1, 2, 3, 4, 5, 6},
    "marriage": {0, 1, 2, 3},
    TARGET: {0, 1},
}
DOCUMENTED: dict[str, set[int]] = {"education": {1, 2, 3, 4}, "marriage": {1, 2, 3}}
# Repayment status: -2 no balance to pay, -1 paid in full, 0 revolving (minimum paid), 1-9 months late.
STATUS_RANGE = (-2, 9)
RANGES: dict[str, tuple[float, float]] = {"limit": (1, 5_000_000), "age": (18, 100)}


class ValidationError(ValueError):
    """Raised with every failed check listed, so one run shows all problems."""


def validate_raw(df: pd.DataFrame) -> dict[str, Any]:
    """Check the tidy raw table and return a summary of what was found.

    Hard failures (missing columns, nulls, duplicate ids, values outside the allowed ranges) raise
    ValidationError. Undocumented category codes and negative bills are counted, not rejected.
    """
    problems: list[str] = []
    expected = set(RENAME.values())
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValidationError(f"missing columns: {missing}")

    nulls = df[list(expected)].isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{col}: {n} nulls")
    dup_ids = int(df["account_id"].duplicated().sum())
    if dup_ids:
        problems.append(f"account_id: {dup_ids} duplicate ids")

    for col, allowed in ALLOWED.items():
        bad = ~df[col].isin(allowed)
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} values outside {sorted(allowed)}")
    for col, (lo, hi) in RANGES.items():
        bad = ~df[col].between(lo, hi)
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} values outside [{lo}, {hi}]")
    for col in STATUS_COLS:
        bad = ~df[col].between(*STATUS_RANGE)
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} values outside {list(STATUS_RANGE)}")
    for col in PAID_COLS:
        if (df[col] < 0).any():
            problems.append(f"{col}: {int((df[col] < 0).sum())} negative payments")

    if problems:
        raise ValidationError("raw data failed validation:\n  " + "\n  ".join(problems))

    undocumented = {col: int((~df[col].isin(codes)).sum()) for col, codes in DOCUMENTED.items()}
    feature_cols = [c for c in df.columns if c != "account_id"]
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "default_rate": float(df[TARGET].mean()),
        "undocumented_codes": undocumented,
        "negative_bill_cells": int((df[BILL_COLS] < 0).sum().sum()),
        "duplicate_rows_ignoring_id": int(df[feature_cols].duplicated().sum()),
        "limit_range": [int(df["limit"].min()), int(df["limit"].max())],
        "age_range": [int(df["age"].min()), int(df["age"].max())],
    }
