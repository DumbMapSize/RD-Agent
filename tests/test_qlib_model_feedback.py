import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from rdagent.scenarios.qlib.developer import feedback as feedback_module
from rdagent.scenarios.qlib.developer.feedback import (
    IMPORTANT_METRICS,
    QlibModelExperiment2Feedback,
    process_results,
)


def _response(*, decision: bool, observation: str) -> str:
    return json.dumps(
        {
            "Observations": observation,
            "Feedback for Hypothesis": "evaluation",
            "New Hypothesis": "next",
            "Reasoning": "reason",
            "Decision": decision,
        }
    )


def _full_result(*, scale: float) -> pd.Series:
    return pd.Series(
        {
            "IC": 0.03 * scale,
            "ICIR": 0.24 * scale,
            "Rank IC": 0.02 * scale,
            "Rank ICIR": 0.15 * scale,
            "1day.excess_return_without_cost.annualized_return": 0.22 * scale,
            "1day.excess_return_without_cost.information_ratio": 1.30 * scale,
            "1day.excess_return_without_cost.max_drawdown": -0.16 * scale,
            "1day.excess_return_with_cost.annualized_return": 0.17 * scale,
            "1day.excess_return_with_cost.information_ratio": 1.05 * scale,
            "1day.excess_return_with_cost.max_drawdown": -0.19 * scale,
        }
    )


def _model_experiment(*, result: pd.Series) -> SimpleNamespace:
    return SimpleNamespace(
        hypothesis=SimpleNamespace(hypothesis="test hypothesis", reason="test reason"),
        sub_tasks=[SimpleNamespace(get_task_information=lambda: "model task")],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "model code"})],
        result=result,
        stdout="training log",
    )


def test_model_feedback_calls_llm_once_and_uses_that_response(monkeypatch) -> None:
    backend = Mock()
    rendered_contexts = {}
    backend.build_messages_and_create_chat_completion.side_effect = [
        _response(decision=True, observation="first"),
        _response(decision=False, observation="second"),
    ]
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)

    def template(uri):
        def render(**context):
            rendered_contexts[uri] = context
            return "rendered prompt"

        return SimpleNamespace(r=render)

    monkeypatch.setattr(feedback_module, "T", template)

    result = pd.Series({metric: 0.1 for metric in IMPORTANT_METRICS})
    exp = SimpleNamespace(
        hypothesis=SimpleNamespace(hypothesis="test hypothesis", reason="test reason"),
        sub_tasks=[SimpleNamespace(get_task_information=lambda: "task")],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "code"})],
        result=result,
        stdout="training log",
    )
    sota_hypothesis = SimpleNamespace(hypothesis="SOTA hypothesis")
    sota_experiment = SimpleNamespace(
        sub_tasks=[SimpleNamespace(get_task_information=lambda: "SOTA task")],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "SOTA code"})],
        result=pd.Series({metric: 0.05 for metric in IMPORTANT_METRICS}),
    )
    trace = SimpleNamespace(
        get_sota_hypothesis_and_experiment=lambda: (sota_hypothesis, sota_experiment)
    )
    scenario = SimpleNamespace(get_scenario_all_desc=lambda **_kwargs: "scenario")

    generated = QlibModelExperiment2Feedback(scenario).generate_feedback(exp, trace)

    assert backend.build_messages_and_create_chat_completion.call_count == 1
    assert generated.decision is True
    assert generated.observations == "first"
    evidence = rendered_contexts["scenarios.qlib.prompts:model_feedback_generation.user"]["combined_result"]
    assert "IC: Current=0.100000, SOTA=0.050000, Delta=+0.050000" in evidence


def test_model_feedback_renders_full_evidence_with_sota(monkeypatch) -> None:
    backend = Mock()
    backend.build_messages_and_create_chat_completion.return_value = _response(
        decision=False,
        observation="observed",
    )
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)
    monkeypatch.setattr("rdagent.utils.agent.tpl.logger.log_object", lambda *_args, **_kwargs: None)

    class QuantScenario:
        def get_scenario_all_desc(self, **_kwargs):
            return "scenario"

    monkeypatch.setattr(feedback_module, "QlibQuantScenario", QuantScenario)
    exp = _model_experiment(result=_full_result(scale=1.0))
    sota_experiment = _model_experiment(result=_full_result(scale=0.5))
    trace = SimpleNamespace(
        get_sota_hypothesis_and_experiment=lambda: (sota_experiment.hypothesis, sota_experiment)
    )

    generated = QlibModelExperiment2Feedback(QuantScenario()).generate_feedback(exp, trace)

    call = backend.build_messages_and_create_chat_completion.call_args.kwargs
    user_prompt = call["user_prompt"]
    system_prompt = call["system_prompt"]
    assert generated.decision is False
    assert "Predictive metrics" in user_prompt
    assert "Portfolio metrics" in user_prompt
    assert "ICIR: Current=0.240000, SOTA=0.120000, Delta=+0.120000" in user_prompt
    assert "1day.excess_return_with_cost.information_ratio" in user_prompt
    assert "Annualized return cost drag" in user_prompt
    assert "Do not claim that turnover" in system_prompt


def test_model_feedback_renders_current_metrics_without_sota(monkeypatch) -> None:
    backend = Mock()
    backend.build_messages_and_create_chat_completion.return_value = _response(
        decision=True,
        observation="observed",
    )
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)
    monkeypatch.setattr("rdagent.utils.agent.tpl.logger.log_object", lambda *_args, **_kwargs: None)
    exp = _model_experiment(result=_full_result(scale=1.0))
    trace = SimpleNamespace(get_sota_hypothesis_and_experiment=lambda: (None, None))
    scenario = SimpleNamespace(get_scenario_all_desc=lambda **_kwargs: "scenario")

    QlibModelExperiment2Feedback(scenario).generate_feedback(exp, trace)

    user_prompt = backend.build_messages_and_create_chat_completion.call_args.kwargs["user_prompt"]
    assert "This is the first round" in user_prompt
    assert "IC: Current=0.030000, SOTA=unavailable, Delta=unavailable" in user_prompt
    assert "Annualized return cost drag: Current=0.050000, SOTA=unavailable" in user_prompt


def test_process_results_includes_predictive_portfolio_and_derived_evidence() -> None:
    current = _full_result(scale=1.0)
    sota = _full_result(scale=0.5)

    evidence = process_results(current, sota)

    assert "Predictive metrics" in evidence
    assert "Portfolio metrics" in evidence
    assert "IC: Current=0.030000, SOTA=0.015000, Delta=+0.015000" in evidence
    assert "1day.excess_return_with_cost.information_ratio" in evidence
    assert "Annualized return cost drag: Current=0.050000, SOTA=0.025000, Delta=+0.025000" in evidence


def test_process_results_skips_unavailable_metrics_without_failing() -> None:
    evidence = process_results(
        pd.Series({"IC": 0.03}),
        pd.Series({"IC": 0.02}),
    )

    assert "IC: Current=0.030000, SOTA=0.020000, Delta=+0.010000" in evidence
    assert "Rank IC" not in evidence
    assert "cost drag" not in evidence
