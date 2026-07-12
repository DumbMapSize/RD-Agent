from types import SimpleNamespace

import pytest

from rdagent.scenarios.qlib.proposal.bandit import extract_metrics_from_experiment


def test_extract_metrics_reads_qlib_annualized_return_key():
    experiment = SimpleNamespace(
        result={
            "IC": 0.04,
            "ICIR": 0.30,
            "Rank IC": 0.03,
            "Rank ICIR": 0.25,
            "1day.excess_return_with_cost.annualized_return": 0.12,
            "1day.excess_return_with_cost.information_ratio": 1.50,
            "1day.excess_return_with_cost.max_drawdown": -0.06,
        }
    )

    metrics = extract_metrics_from_experiment(experiment)

    assert metrics.arr == pytest.approx(0.12)
    assert metrics.sharpe == pytest.approx(2.0)
