import pytest

from credit.validate import ValidationError, validate_raw


def test_clean_synthetic_passes_and_counts_undocumented_codes(raw):
    summary = validate_raw(raw)
    assert summary["rows"] == len(raw)
    assert summary["undocumented_codes"]["education"] == int(raw["education"].isin([0, 5, 6]).sum())
    assert summary["undocumented_codes"]["marriage"] == int((raw["marriage"] == 0).sum())
    assert 0 < summary["default_rate"] < 1


def test_nulls_duplicates_and_ranges_fail_loudly(raw):
    bad = raw.copy()
    bad = bad.astype({"age": "float64"})
    bad.loc[0, "age"] = None
    bad.loc[1, "account_id"] = bad.loc[2, "account_id"]
    bad.loc[3, "pay_status_2"] = 12
    bad.loc[4, "sex"] = 3
    with pytest.raises(ValidationError) as err:
        validate_raw(bad)
    msg = str(err.value)
    for expected in ["age: 1 nulls", "duplicate ids", "pay_status_2", "sex"]:
        assert expected in msg


def test_missing_column_fails(raw):
    with pytest.raises(ValidationError, match="missing columns"):
        validate_raw(raw.drop(columns=["bill_3"]))
