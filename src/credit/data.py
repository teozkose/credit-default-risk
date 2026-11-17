"""Download the UCI Default of Credit Card Clients file and load it with tidy column names."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from credit.config import resolve

TARGET = "default"
MONTHS = 6  # month 1 is the most recent statement (September 2005), month 6 the oldest (April 2005)

# The source labels the most recent repayment status PAY_0 but the bill and payment columns *_1;
# PAY_0 becomes pay_status_1 so every monthly column uses the same index.
RENAME: dict[str, str] = {
    "ID": "account_id", "LIMIT_BAL": "limit", "SEX": "sex", "EDUCATION": "education",
    "MARRIAGE": "marriage", "AGE": "age", "PAY_0": "pay_status_1",
    **{f"PAY_{k}": f"pay_status_{k}" for k in range(2, MONTHS + 1)},
    **{f"BILL_AMT{k}": f"bill_{k}" for k in range(1, MONTHS + 1)},
    **{f"PAY_AMT{k}": f"paid_{k}" for k in range(1, MONTHS + 1)},
    "default payment next month": TARGET,
}

STATUS_COLS = [f"pay_status_{k}" for k in range(1, MONTHS + 1)]
BILL_COLS = [f"bill_{k}" for k in range(1, MONTHS + 1)]
PAID_COLS = [f"paid_{k}" for k in range(1, MONTHS + 1)]


def download(cfg: dict[str, Any], force: bool = False) -> Path:
    """Fetch the zip once into data/ and return its path."""
    path = resolve(cfg["data"]["zip"])
    if path.exists() and path.stat().st_size >= cfg["data"]["min_zip_bytes"] and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(cfg["data"]["url"], timeout=120)
    resp.raise_for_status()
    if len(resp.content) < cfg["data"]["min_zip_bytes"]:
        raise RuntimeError(f"download too small ({len(resp.content)} bytes); the source may have moved")
    path.write_bytes(resp.content)
    return path


def read_source(zip_path: Path, xls_name: str) -> pd.DataFrame:
    """Read the Excel sheet inside the zip (row 0 is a second header line, so header=1)."""
    with zipfile.ZipFile(zip_path) as zf:
        payload = zf.read(xls_name)
    return pd.read_excel(io.BytesIO(payload), header=1, engine="xlrd")


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to snake_case and cast every column to int64."""
    missing = sorted(set(RENAME) - set(df.columns))
    if missing:
        raise ValueError(f"source file is missing columns: {missing}")
    return df[list(RENAME)].rename(columns=RENAME).astype("int64")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the cached tidy parquet written by `make data`."""
    return pd.read_parquet(path)


def fetch_and_cache(cfg: dict[str, Any]) -> pd.DataFrame:
    """Download, read, tidy and cache the raw table as parquet."""
    zip_path = download(cfg)
    df = tidy(read_source(zip_path, cfg["data"]["xls_name"]))
    out = resolve(cfg["data"]["raw"])
    df.to_parquet(out, index=False)
    return df
