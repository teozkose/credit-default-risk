import numpy as np
import pandas as pd

from credit.features import FEATURES, bill_slope, build_table, clean, late_streak, payment_ratios


def test_clean_folds_codes_and_drops_exact_duplicates(raw, cfg):
    dup = raw.iloc[[0]].copy()
    dup["account_id"] = 999_999
    cleaned, log = clean(pd.concat([raw, dup], ignore_index=True), cfg)
    assert set(cleaned["education"]) <= {1, 2, 3, 4}
    assert set(cleaned["marriage"]) <= {1, 2, 3}
    assert log[-1]["dropped"] >= 1
    assert len(cleaned) == log[-1]["rows_after"]


def test_payment_ratio_uses_previous_bill_and_handles_nothing_owed():
    df = pd.DataFrame({**{f"bill_{k}": [0, 0] for k in range(1, 7)}, **{f"paid_{k}": [0, 0] for k in range(1, 7)}})
    df.loc[0, "bill_2"], df.loc[0, "paid_1"] = 1000, 250   # paid a quarter of last month's bill
    df.loc[1, "bill_2"], df.loc[1, "paid_1"] = 100, 900    # overpaid: clipped at the cap
    r = payment_ratios(df, cap=2.0)
    assert r.loc[0, "pay_ratio_1"] == 0.25
    assert r.loc[1, "pay_ratio_1"] == 2.0
    assert r.loc[0, "pay_ratio_2"] == 1.0  # nothing owed counts as fully paid


def test_late_streak_counts_back_from_the_latest_month():
    status = np.array([[2, 1, 0, 3, 3, 3], [0, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1]])
    assert late_streak(status).tolist() == [2, 0, 6]


def test_bill_slope_sign_follows_time_order():
    growing = np.array([[600.0, 500, 400, 300, 200, 100]])  # bill_1 is the newest month
    assert bill_slope(growing)[0] == 100.0
    assert bill_slope(growing[:, ::-1])[0] == -100.0


def test_build_table_has_every_model_input_and_no_protected_attributes(raw, cfg):
    table, _ = build_table(raw, cfg)
    assert set(FEATURES) <= set(table.columns)
    assert not {"sex", "marriage"} & set(FEATURES)
    assert table[FEATURES].notna().all().all()
    assert (table["exposure"] >= cfg["decision"]["min_exposure"]).all()
