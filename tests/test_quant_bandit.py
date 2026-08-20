from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rdagent.scenarios.qlib.proposal.bandit import (
    EnvController,
    LinearThompsonTwoArm,
    Metrics,
    extract_metrics_from_experiment,
)
from rdagent.scenarios.qlib.proposal.quant_proposal import (
    _current_strategy_metrics,
    _record_bandit_experiment,
)


def _result(*, arr=0.12, ir=1.5, mdd=-0.06):
    return {
        "IC": 0.04,
        "ICIR": 0.30,
        "Rank IC": 0.03,
        "Rank ICIR": 0.25,
        "1day.excess_return_with_cost.annualized_return": arr,
        "1day.excess_return_with_cost.information_ratio": ir,
        "1day.excess_return_with_cost.max_drawdown": mdd,
    }


def test_extract_metrics_reads_qlib_annualized_return_key():
    experiment = SimpleNamespace(result=_result())

    metrics = extract_metrics_from_experiment(experiment)

    assert metrics is not None
    assert metrics.arr == pytest.approx(0.12)
    assert metrics.sharpe == pytest.approx(2.0)


def test_extract_metrics_rejects_incomplete_or_non_finite_results():
    incomplete = _result()
    incomplete.pop("Rank ICIR")
    non_finite = _result(arr=float("nan"))

    assert extract_metrics_from_experiment(SimpleNamespace(result=incomplete)) is None
    assert extract_metrics_from_experiment(SimpleNamespace(result=non_finite)) is None
    assert extract_metrics_from_experiment(SimpleNamespace(result=None)) is None


def test_drawdown_score_is_better_for_smaller_drawdown_regardless_of_sign():
    smaller = Metrics(mdd=-0.05)
    larger = Metrics(mdd=-0.20)
    positive_convention = Metrics(mdd=0.05)

    assert smaller.as_vector()[6] > larger.as_vector()[6]
    assert smaller.as_vector()[6] == positive_convention.as_vector()[6]


def test_linear_thompson_update_matches_closed_form_posterior():
    model = LinearThompsonTwoArm(dim=2, prior_var=2.0, noise_var=0.5)
    observations = (
        (np.array([1.0, 2.0]), 0.4),
        (np.array([-0.5, 1.5]), -0.2),
    )

    for context, reward in observations:
        model.update("factor", context, reward)

    expected_precision = np.eye(2) / 2.0
    expected_natural_mean = np.zeros(2)
    for context, reward in observations:
        expected_precision += np.outer(context, context) / 0.5
        expected_natural_mean += context * reward / 0.5

    np.testing.assert_allclose(model.precision["factor"], expected_precision)
    np.testing.assert_allclose(
        model.mean["factor"],
        np.linalg.solve(expected_precision, expected_natural_mean),
    )


def test_controller_records_net_annualized_return_improvement():
    controller = EnvController()
    controller.bandit.update = Mock()
    context = Metrics(ic=0.10, arr=0.10, ir=2.0, mdd=-0.10, sharpe=1.0)
    outcome = Metrics(ic=-0.10, arr=0.16, ir=-2.0, mdd=-0.50, sharpe=-1.0)

    controller.record_improvement(context, outcome, "factor")

    arm, recorded_context, reward = controller.bandit.update.call_args.args
    assert arm == "factor"
    np.testing.assert_array_equal(recorded_context, context.as_vector())
    assert reward == pytest.approx(0.06)


def test_model_history_recording_uses_global_incumbent_instead_of_previous_model():
    baseline = SimpleNamespace(result=_result(arr=0.08))
    accepted_model = SimpleNamespace(
        result=_result(arr=0.10),
        based_experiments=[],
        hypothesis=SimpleNamespace(action="model"),
    )
    accepted_factor = SimpleNamespace(
        result=_result(arr=0.16),
        based_experiments=[baseline],
        hypothesis=SimpleNamespace(action="factor"),
    )
    candidate_model = SimpleNamespace(
        result=_result(arr=0.12),
        based_experiments=[accepted_model],
        hypothesis=SimpleNamespace(action="model"),
    )
    history = [
        (accepted_model, SimpleNamespace(decision=True, exception=None)),
        (accepted_factor, SimpleNamespace(decision=True, exception=None)),
        (candidate_model, SimpleNamespace(decision=False, exception=None)),
    ]
    controller = Mock(spec=EnvController)

    assert _record_bandit_experiment(controller, history, 2) is True

    context, outcome, arm = controller.record_improvement.call_args.args
    assert context.arr == pytest.approx(0.16)
    assert outcome.arr == pytest.approx(0.12)
    assert arm == "model"


def test_history_recording_uses_incumbent_and_skips_execution_failures():
    baseline = SimpleNamespace(result=_result(arr=0.10))
    accepted_factor = SimpleNamespace(
        result=_result(arr=0.16),
        based_experiments=[baseline],
        hypothesis=SimpleNamespace(action="factor"),
    )
    rejected_model = SimpleNamespace(
        result=_result(arr=0.08),
        based_experiments=[accepted_factor],
        hypothesis=SimpleNamespace(action="model"),
    )
    failed_factor = SimpleNamespace(
        result=_result(arr=0.30),
        based_experiments=[accepted_factor],
        hypothesis=SimpleNamespace(action="factor"),
    )
    history = [
        (accepted_factor, SimpleNamespace(decision=True, exception=None)),
        (rejected_model, SimpleNamespace(decision=False, exception=None)),
        (failed_factor, SimpleNamespace(decision=False, exception=RuntimeError("failed"))),
    ]
    controller = EnvController()

    assert _record_bandit_experiment(controller, history, 0) is True
    assert _record_bandit_experiment(controller, history, 1) is True
    factor_precision = controller.bandit.precision["factor"].copy()
    model_precision = controller.bandit.precision["model"].copy()
    assert _record_bandit_experiment(controller, history, 2) is False

    np.testing.assert_array_equal(controller.bandit.precision["factor"], factor_precision)
    np.testing.assert_array_equal(controller.bandit.precision["model"], model_precision)
    current = _current_strategy_metrics(history)
    assert current is not None
    assert current.arr == pytest.approx(0.16)


def test_current_strategy_falls_back_to_initial_baseline_before_first_acceptance():
    baseline = SimpleNamespace(result=_result(arr=0.10))
    rejected = SimpleNamespace(
        result=_result(arr=0.08),
        based_experiments=[baseline],
        hypothesis=SimpleNamespace(action="factor"),
    )
    history = [(rejected, SimpleNamespace(decision=False, exception=None))]

    current = _current_strategy_metrics(history)

    assert current is not None
    assert current.arr == pytest.approx(0.10)
