from __future__ import annotations

import numpy as np
import pandas as pd
from qlib.data.base import ExpressionOps, Feature


class OpeningLimit(ExpressionOps):
    """Directional opening-price limit using only prior valid closes."""

    def __init__(self, threshold, direction):
        if direction not in (1, -1):
            raise ValueError("OpeningLimit direction must be 1 (buy) or -1 (sell)")
        self.threshold = threshold
        self.direction = direction

    def __str__(self):
        return f"OpeningLimit({self.threshold},{self.direction})"

    def get_longest_back_rolling(self):
        return np.inf

    def get_extended_window_size(self):
        # Suspension gaps are resolved inside _load_internal, without future data.
        left, right = self.threshold.get_extended_window_size()
        return max(1, left), right

    def _load_internal(self, instrument, start_index, end_index, freq):
        index = pd.RangeIndex(start_index, end_index + 1)
        opening = Feature("open").load(instrument, start_index, end_index, freq).reindex(index).astype(float)
        threshold = self.threshold.load(instrument, start_index, end_index, freq).reindex(index)
        close = Feature("close")
        history = pd.Series(dtype=float)
        if end_index > 0:
            history = close.load(instrument, max(0, start_index - 1), end_index - 1, freq)
        history = history.reindex(pd.RangeIndex(start_index - 1, end_index)).astype(float)
        history = history.where(np.isfinite(history) & (history > 0))

        # Seed the first reference across a suspension that began before this query.
        cursor, span = start_index - 2, 32
        while pd.isna(history.iloc[0]) and cursor >= 0:
            left = max(0, cursor - span + 1)
            earlier = close.load(instrument, left, cursor, freq)
            earlier = earlier[np.isfinite(earlier) & (earlier > 0)]
            if not earlier.empty:
                history.iloc[0] = earlier.iloc[-1]
                break
            cursor, span = left - 1, span * 2

        previous = pd.Series(history.ffill().to_numpy(), index=index)
        valid = np.isfinite(opening) & (opening > 0) & previous.notna()
        valid &= np.isfinite(threshold) & (threshold > 0)
        # Match the existing float32 change field and strict threshold comparisons.
        change = ((opening - previous) / previous).astype(np.float32)
        limited = change > threshold if self.direction == 1 else change < -threshold
        return ~valid | limited
