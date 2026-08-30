from __future__ import annotations

import copy
from typing import Any

from qlib.workflow.record_temp import PortAnaRecord


def normalize_limit_threshold(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert safe-YAML expression pairs to QLib's tuple-only API."""
    if config is None:
        return None
    normalized = copy.deepcopy(config)
    backtest = normalized.get("backtest")
    exchange_kwargs = backtest.get("exchange_kwargs") if isinstance(backtest, dict) else None
    if not isinstance(exchange_kwargs, dict):
        return normalized

    value = exchange_kwargs.get("limit_threshold")
    if not isinstance(value, list):
        return normalized
    if len(value) != 2 or not all(isinstance(item, str) and item for item in value):
        raise ValueError("limit_threshold must contain exactly two non-empty expressions")
    exchange_kwargs["limit_threshold"] = tuple(value)
    return normalized


class LimitExpressionPortAnaRecord(PortAnaRecord):
    """PortAnaRecord accepting a safe-YAML limit expression pair."""

    def __init__(self, recorder, config=None, **kwargs):
        super().__init__(recorder=recorder, config=normalize_limit_threshold(config), **kwargs)
