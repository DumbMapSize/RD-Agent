"""Cash settlement of formally delisted A-shares in historical daily simulations."""

import csv
import logging
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from qlib.backtest.signal import Signal
from qlib.config import C
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D

logger = logging.getLogger(__name__)


def _inactive_intervals(path: Path):
    listings = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            code = row["instrument"]
            start, end = date.fromisoformat(row["list_date"]), date.fromisoformat(row["delist_date"])
            if not re.fullmatch(r"SH6\d{5}|SZ[03]\d{5}", code) or start >= end or start in listings[code]:
                raise ValueError(f"Invalid or duplicate listing in {path}: {row}")
            listings[code][start] = end
    if not listings:
        raise ValueError(f"Empty delisting table: {path}")
    # A subsequent listing of the same code is a distinct security lifetime.
    # Never infer a delisting from the end of a QLib instrument interval.
    spans = D.list_instruments(D.instruments("all"), start_time="1900-01-01", freq="day")
    intervals = []
    for code, periods in listings.items():
        starts = set(periods) | {s.date() for s, _ in spans.get(code, ())}
        for end in periods.values():
            resume = min((s for s in starts if s > end), default=date.max)
            intervals.append((code, end, resume))
    return intervals


def load_delisting_intervals():
    return _inactive_intervals(C.dpm.get_data_uri("day") / "metadata" / "stock_delistings.csv")


def inactive_instruments(day: date, intervals) -> set[str]:
    return {code for code, end, resume in intervals if end <= day < resume}


class _EligibleScores(pd.Series):
    """QLib takes candidates from sort_values().index; filter that ranking, not its input."""

    # Preserve exclusions when QLib selects non-holdings or the production adapter drops NaNs.
    _metadata = ["_excluded_codes"]

    @property
    def _constructor(self):
        return _EligibleScores

    def sort_values(self, *args, **kwargs):
        ranked = pd.Series(self, copy=False).sort_values(*args, **kwargs)
        return ranked[~ranked.index.isin(self._excluded_codes)]


def eligible_scores(scores, inactive):
    """Preserve native TopK tie ordering while excluding inactive candidates."""
    if scores is None:
        return None
    excluded = scores.index.intersection(inactive)
    if excluded.empty:
        return scores
    # Native TopK uses the first column of a DataFrame signal.
    view = _EligibleScores(scores.iloc[:, 0] if isinstance(scores, pd.DataFrame) else scores, copy=False)
    view._excluded_codes = excluded
    return view


class _ListedSignal(Signal):
    def __init__(self, source, inactive):
        self.source, self.inactive = source, inactive

    def get_signal(self, start_time, end_time):
        scores = self.source.get_signal(start_time=start_time, end_time=end_time)
        return eligible_scores(scores, self.inactive)


class DelistingSettlementMixin:
    """Settle at the last book mark before ranking; leave normal trade execution alone."""

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._delisting_intervals = load_delisting_intervals()

    def generate_trade_decision(self, execute_result=None):
        day = self.trade_calendar.get_step_time()[0].date()
        inactive = inactive_instruments(day, self._delisting_intervals)
        position = self.trade_position
        position.delisting_settlements = []
        for code in position.get_stock_list():
            if code not in inactive:
                continue
            amount, price = float(position.get_stock_amount(code)), float(position.get_stock_price(code))
            value = amount * price
            if not (math.isfinite(value) and price > 0 and amount > 0):
                raise ValueError(f"Cannot settle {code} on {day}: amount={amount}, price={price}")
            # Use QLib's matching adjusted units exactly once. No order, fee or turnover.
            position.position["cash"] += value
            del position.position[code]
            event = dict(date=day.isoformat(), instrument=code, amount=amount, price=price, value=value)
            position.delisting_settlements.append(event)
            logger.info("Delisting cash settlement: %s", event)
        signal = self.signal
        if inactive:
            self.signal = _ListedSignal(signal, inactive)
        try:
            return super().generate_trade_decision(execute_result)
        finally:
            self.signal = signal


class _FiniteSignal(Signal):
    def __init__(self, source):
        self.source = source

    def get_signal(self, start_time, end_time):
        scores = self.source.get_signal(start_time=start_time, end_time=end_time)
        if scores is None:
            return None
        if isinstance(scores, pd.DataFrame):
            scores = scores.iloc[:, 0]
        return scores.replace([math.inf, -math.inf], math.nan).dropna()


class HistoricalTopkDropoutStrategy(DelistingSettlementMixin, TopkDropoutStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signal = _FiniteSignal(self.signal)
