"""Synthetic accounts in the source file's shape, so tests never touch the network."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from credit.config import load_config
from credit.data import RENAME, tidy


def make_source(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Source-format frame (original column names) where a latent risk drives lateness and default."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(0, 1, n)
    limit = rng.choice([20_000, 50_000, 100_000, 200_000, 360_000], n)
    data: dict[str, np.ndarray] = {
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": limit,
        "SEX": rng.choice([1, 2], n),
        "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.01, 0.35, 0.45, 0.15, 0.02, 0.01, 0.01]),
        "MARRIAGE": rng.choice([0, 1, 2, 3], n, p=[0.01, 0.45, 0.5, 0.04]),
        "AGE": rng.integers(21, 70, n),
    }
    status_names = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    for k, name in enumerate(status_names):
        late = np.clip(np.round(risk + rng.normal(0, 0.8, n) - 0.2 * k), -2, 8)
        data[name] = late.astype(int)
    for k in range(1, 7):
        util = np.clip(0.4 + 0.2 * risk + rng.normal(0, 0.2, n), -0.05, 1.2)
        data[f"BILL_AMT{k}"] = np.round(util * limit).astype(int)
    for k in range(1, 7):
        share = np.clip(0.3 - 0.15 * risk + rng.normal(0, 0.1, n), 0, 1.5)
        data[f"PAY_AMT{k}"] = np.round(share * np.maximum(data[f"BILL_AMT{min(k + 1, 6)}"], 0)).astype(int)
    p = 1 / (1 + np.exp(-(-1.4 + 1.3 * risk)))
    data["default payment next month"] = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame(data)[list(RENAME)]


@pytest.fixture
def raw() -> pd.DataFrame:
    """Tidy raw frame, as cached by `make data`."""
    return tidy(make_source())


@pytest.fixture
def cfg() -> dict:
    return copy.deepcopy(load_config())


@pytest.fixture
def small_cfg(cfg: dict) -> dict:
    """The project config shrunk for a fast end-to-end run."""
    cfg["validation"]["folds"] = 3
    cfg["models"]["xgboost"]["n_estimators"] = 60
    cfg["models"]["tuning"].update({"n_iter": 2, "folds": 2})
    cfg["models"]["calibration_folds"] = 2
    cfg["evaluation"]["n_boot"] = 40
    return cfg
