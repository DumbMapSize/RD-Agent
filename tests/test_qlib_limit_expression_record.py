from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML
from qlib.workflow.cli import render_template

from rdagent.scenarios.qlib.experiment.limit_expression_record import normalize_limit_threshold
from rdagent.scenarios.qlib.experiment.model_training import build_model_run_env


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = (
    "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
    "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
    "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
    "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
    "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
)
EXPRESSIONS = (
    "$change > $limit_rate",
    "$change < 0 - $limit_rate",
)


def test_normalizer_converts_expression_list_without_mutating_source() -> None:
    source = {"backtest": {"exchange_kwargs": {"limit_threshold": list(EXPRESSIONS)}}}

    normalized = normalize_limit_threshold(source)

    assert normalized["backtest"]["exchange_kwargs"]["limit_threshold"] == EXPRESSIONS
    assert source["backtest"]["exchange_kwargs"]["limit_threshold"] == list(EXPRESSIONS)


def test_normalizer_keeps_legacy_float_compatible() -> None:
    source = {"backtest": {"exchange_kwargs": {"limit_threshold": 0.095}}}

    assert normalize_limit_threshold(source) == source


@pytest.mark.parametrize("value", [[], ["buy"], ["buy", ""], ["buy", "sell", "extra"]])
def test_normalizer_rejects_invalid_expression_lists(value: list[str]) -> None:
    source = {"backtest": {"exchange_kwargs": {"limit_threshold": value}}}

    with pytest.raises(ValueError, match="exactly two"):
        normalize_limit_threshold(source)


@pytest.mark.parametrize("relative_path", TEMPLATES)
def test_templates_use_uniform_per_instrument_limit_expression(monkeypatch, relative_path: str) -> None:
    for key, value in build_model_run_env({}, "Tabular", num_features=158).items():
        monkeypatch.setenv(key, value)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(render_template(str(ROOT / relative_path)))
    port_config = config["port_analysis_config"]
    exchange_kwargs = port_config["backtest"]["exchange_kwargs"]
    record = next(item for item in config["task"]["record"] if item["class"] == "LimitExpressionPortAnaRecord")

    assert exchange_kwargs["limit_threshold"] == list(EXPRESSIONS)
    assert record["module_path"] == "rdagent.scenarios.qlib.experiment.limit_expression_record"
