import numpy as np
import pandas as pd
import pytest
from qlib.data.base import Feature

from rdagent.scenarios.qlib.experiment.limit_operators import OpeningLimit


def load_limits(monkeypatch, fields, start, end):
    reads = []

    def load(feature, instrument, left, right, freq):
        reads.append((str(feature), left, right))
        return pd.Series(fields[str(feature)], dtype=np.float32).loc[left:right]

    monkeypatch.setattr(Feature, "load", load)
    result = [OpeningLimit(Feature("limit_rate"), direction)._load_internal("SH600000", start, end, "day")
              for direction in (1, -1)]
    return result, reads


@pytest.mark.parametrize("opening,buy_blocked,sell_blocked", [
    (1000, False, False), (1096, True, False), (904, False, True),
    (1095, False, False), (905, False, False),
    (1095.01, True, False), (904.99, False, True),
    (np.nan, True, True), (0, True, True), (-1, True, True), (np.inf, True, True),
])
@pytest.mark.parametrize("current_close", [900, 1000, 1100])
def test_direction_and_existing_strict_boundaries(monkeypatch, opening, buy_blocked, sell_blocked, current_close):
    result, reads = load_limits(monkeypatch, {
        "$open": {1: opening}, "$close": {0: 1000, 1: current_close}, "$limit_rate": {1: 0.095},
    }, 1, 1)
    assert [bool(series.iloc[0]) for series in result] == [buy_blocked, sell_blocked]
    assert all(right < 1 for field, left, right in reads if field == "$close")


def test_resumption_queries_prior_close_across_long_suspension(monkeypatch):
    result, reads = load_limits(monkeypatch, {
        "$open": {150: 11, 151: 10}, "$close": {0: 10, 150: 11}, "$limit_rate": {150: .095, 151: .095},
    }, 150, 151)
    assert result[0].tolist() == [True, False]
    assert result[1].tolist() == [False, False]
    assert any(left == 0 for field, left, right in reads if field == "$close")
    assert all(right < 151 for field, left, right in reads if field == "$close")


@pytest.mark.parametrize("previous", [None, 0, -1, np.inf, np.nan])
def test_missing_valid_reference_blocks_both_directions(monkeypatch, previous):
    result, _ = load_limits(monkeypatch, {
        "$open": {1: 10}, "$close": {} if previous is None else {0: previous}, "$limit_rate": {1: .095},
    }, 1, 1)
    assert all(bool(series.iloc[0]) for series in result)


@pytest.mark.parametrize("threshold", [np.nan, 0, -1, np.inf])
def test_invalid_threshold_blocks_both_directions(monkeypatch, threshold):
    result, _ = load_limits(monkeypatch, {
        "$open": {1: 10}, "$close": {0: 10}, "$limit_rate": {1: threshold},
    }, 1, 1)
    assert all(bool(series.iloc[0]) for series in result)


def test_first_calendar_day_has_no_prior_reference(monkeypatch):
    result, reads = load_limits(monkeypatch, {
        "$open": {0: 10}, "$close": {0: 10}, "$limit_rate": {0: .095},
    }, 0, 0)
    assert all(bool(series.iloc[0]) for series in result)
    assert not any(field == "$close" for field, _, _ in reads)


def test_in_window_suspension_and_price_scaling(monkeypatch):
    for scale in (1, 3):
        result, _ = load_limits(monkeypatch, {
            "$open": {1: np.nan, 2: 10 * scale, 3: 9 * scale},
            "$close": {0: 10 * scale, 1: np.nan, 2: 10 * scale},
            "$limit_rate": {1: .095, 2: .095, 3: .095},
        }, 1, 3)
        assert result[0].tolist() == [True, False, False]
        assert result[1].tolist() == [True, False, True]


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="direction"):
        OpeningLimit(Feature("limit_rate"), 0)


def test_query_partition_and_future_closes_do_not_change_current_limit(monkeypatch):
    fields = {
        "$open": {1: 11, 2: np.nan, 3: np.nan, 4: 10, 5: 9, 6: 10},
        "$close": {0: 10, 1: 11, 2: np.nan, 3: np.nan, 4: 10, 5: 9, 6: 10},
        "$limit_rate": {day: .095 for day in range(1, 7)},
    }
    baseline, _ = load_limits(monkeypatch, fields, 1, 6)
    for day in range(1, 7):
        single, _ = load_limits(monkeypatch, fields, day, day)
        changed = {**fields, "$close": {**fields["$close"], **{d: 1000 for d in range(day, 7)}}}
        changed_result, _ = load_limits(monkeypatch, changed, 1, 6)
        for direction in (0, 1):
            assert single[direction].loc[day] == baseline[direction].loc[day]
            pd.testing.assert_series_equal(changed_result[direction].loc[:day], baseline[direction].loc[:day])
