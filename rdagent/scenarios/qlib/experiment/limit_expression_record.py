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
    if not isinstance(value, (list, tuple)):
        return normalized
    if len(value) != 2 or not all(isinstance(item, str) and item for item in value):
        raise ValueError("limit_threshold must contain exactly two non-empty expressions")
    exchange_kwargs["limit_threshold"] = tuple(value)
    if any("OpeningLimit(" in item for item in value):
        from qlib.config import C
        from qlib.data.ops import Operators

        from rdagent.scenarios.qlib.experiment.limit_operators import OpeningLimit

        # QLib workers reconstruct their operator registry from this configuration.
        operator = {"class": "OpeningLimit", "module_path": OpeningLimit.__module__}
        custom_ops = C.get("custom_ops", [])
        if operator not in custom_ops:
            C.custom_ops = [*custom_ops, operator]
        if getattr(Operators, "OpeningLimit", None) is not OpeningLimit:
            Operators.register([OpeningLimit])
    return normalized


class LimitExpressionPortAnaRecord(PortAnaRecord):
    """PortAnaRecord accepting a safe-YAML limit expression pair."""

    def __init__(self, recorder, config=None, **kwargs):
        super().__init__(recorder=recorder, config=normalize_limit_threshold(config), **kwargs)
