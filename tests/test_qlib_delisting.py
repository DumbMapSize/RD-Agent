from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qlib.backtest.decision import Order
from qlib.backtest.position import Position
from qlib.backtest.utils import CommonInfrastructure, LevelInfrastructure
from qlib.contrib.strategy import TopkDropoutStrategy

from rdagent.scenarios.qlib.experiment import delisting


def test_listing_lifetimes_are_separate_and_index_exit_is_not_delisting(tmp_path, monkeypatch):
    table = tmp_path / "stock_delistings.csv"
    table.write_text("instrument,list_date,delist_date\n"
                     "SH600001,2000-01-01,2005-02-01\n"
                     "SH600001,2008-01-01,2010-01-01\n")
    monkeypatch.setattr(delisting, "D", SimpleNamespace(instruments=lambda market: market, list_instruments=lambda *a, **kw: {
        "SH600001": [(pd.Timestamp("2015-01-01"), pd.Timestamp("2026-11-01"))],
        "SH600002": [(pd.Timestamp("2001-01-01"), pd.Timestamp("2006-01-01"))],
    }))
    assert delisting._inactive_intervals(table) == [
        ("SH600001", date(2005, 2, 1), date(2008, 1, 1)),
        ("SH600001", date(2010, 1, 1), date(2015, 1, 1)),
    ]


@pytest.mark.parametrize("rows", [
    "",
    "SH600001,2005-01-01,2000-01-01\n",
    "SH600001,2000-01-01,2005-01-01\nSH600001,2000-01-01,2005-01-02\n",
])
def test_invalid_tables_fail_before_any_orders(tmp_path, rows):
    table = tmp_path / "stock_delistings.csv"
    table.write_text("instrument,list_date,delist_date\n" + rows)
    with pytest.raises(ValueError):
        delisting._inactive_intervals(table)


def _orders(cls, scores, held, *, retired=None, only_tradable=False, as_frame=False):
    day = pd.Timestamp("2025-09-05")
    signal = pd.concat({day: scores}, names=["datetime"])
    calendar = SimpleNamespace(get_trade_step=lambda: 0, get_step_time=lambda *a, **kw: (day, day),
                               get_freq=lambda: "day")
    exchange = SimpleNamespace(is_stock_tradable=lambda **kw: True, check_order=lambda order: True,
                               deal_order=lambda order, position: (order.amount, 0, 1),
                               get_deal_price=lambda **kw: 1, get_factor=lambda **kw: 1,
                               round_amount_by_trade_unit=lambda amount, factor: amount)

    pos = Position(cash=100, position_dict={c: {"amount": 1, "price": 1, "count_day": 2} for c in held})
    strategy = cls(signal=signal.to_frame("score") if as_frame else signal, topk=20, n_drop=2,
                   only_tradable=only_tradable, level_infra=LevelInfrastructure(trade_calendar=calendar),
                   common_infra=CommonInfrastructure(trade_account=SimpleNamespace(current_position=pos),
                                                     trade_exchange=exchange))
    strategy._delisting_intervals = [] if retired is None else [(retired, day.date(), date.max)]
    return [(o.stock_id, o.direction, o.amount) for o in strategy.generate_trade_decision().get_decision()]


@pytest.mark.parametrize("seed,retired_score", [(1, -100), (6, -100), (6, 200)])
@pytest.mark.parametrize("only_tradable,as_frame", [(False, False), (True, True)])
def test_retired_candidate_preserves_native_order_and_refills(seed, retired_score, only_tradable, as_frame):
    codes = pd.Index([f"SH{600000+i}" for i in range(50)], name="instrument")
    held, retired = codes[:20], codes[20]
    scores = pd.Series(np.random.default_rng(seed).integers(0, 5, len(codes)), index=codes)
    scores.loc[held] = 100
    scores.iloc[:2] = -10
    scores.loc[retired] = retired_score
    actual = _orders(delisting.HistoricalTopkDropoutStrategy, scores, held, retired=retired,
                     only_tradable=only_tradable, as_frame=as_frame)
    ranked = scores.loc[~scores.index.isin(held)].sort_values(ascending=False)
    expected_buys = ranked.drop(index=retired).index[:2].tolist()
    assert [c for c, side, _ in actual if side == Order.SELL] == held[:2].tolist()
    assert [c for c, side, _ in actual if side == Order.BUY] == expected_buys
    if retired_score < 0:
        assert actual == _orders(TopkDropoutStrategy, scores, held, only_tradable=only_tradable, as_frame=as_frame)
    else:
        assert ranked.index[0] == retired


@pytest.mark.parametrize("as_frame,with_retirement", [(False, False), (True, True)])
def test_invalid_scores_are_not_used_to_fill_buy_slots(as_frame, with_retirement):
    codes = pd.Index([f"SH{600000+i}" for i in range(8)], name="instrument")
    scores = pd.Series([1., np.inf, 3., -np.inf, 3., np.nan, 2., 4.], index=codes)
    retired = codes[-1] if with_retirement else None
    finite = scores[np.isfinite(scores)]
    native = _orders(TopkDropoutStrategy, finite, [])
    expected_buys = [c for c, side, _ in native if c != retired and side == Order.BUY]
    actual = _orders(delisting.HistoricalTopkDropoutStrategy, scores, [], retired=retired, as_frame=as_frame)
    assert actual == [(c, Order.BUY, 95. / len(expected_buys)) for c in expected_buys]


def test_invalid_held_score_is_a_missing_score_not_a_position_removed_from_the_book():
    codes = pd.Index([f"SH{600000+i}" for i in range(26)], name="instrument")
    held = codes[:20]
    scores = pd.Series(1., index=codes)
    scores.loc[held[:3]] = [np.nan, -np.inf, np.inf]
    scores.loc[codes[20:]] = 2.
    expected = _orders(TopkDropoutStrategy, scores[np.isfinite(scores)], held)
    actual = _orders(delisting.HistoricalTopkDropoutStrategy, scores, held)
    assert actual == expected
    sells = {c for c, side, _ in actual if side == Order.SELL}
    assert len(sells) == 2 and sells < set(held[:3])
