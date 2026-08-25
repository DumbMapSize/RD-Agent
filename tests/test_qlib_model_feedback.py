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
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment


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


def _model_experiment(*, result: pd.Series) -> QlibModelExperiment:
    experiment = QlibModelExperiment.__new__(QlibModelExperiment)
    experiment.hypothesis = SimpleNamespace(hypothesis="test hypothesis", reason="test reason", action="model")
    experiment.sub_tasks = [SimpleNamespace(get_task_information=lambda: "model task")]
    experiment.sub_workspace_list = [SimpleNamespace(file_dict={"model.py": "model code"})]
    experiment.running_info = SimpleNamespace(result=result)
    experiment.stdout = "training log"
    return experiment


def _factor_experiment(*, result: pd.Series | None, based_experiments=()) -> QlibFactorExperiment:
    experiment = QlibFactorExperiment.__new__(QlibFactorExperiment)
    experiment.hypothesis = SimpleNamespace(hypothesis="factor hypothesis", action="factor")
    experiment.sub_tasks = []
    experiment.sub_workspace_list = []
    experiment.based_experiments = list(based_experiments)
    experiment.running_info = SimpleNamespace(result=result)
    experiment.experiment_workspace = SimpleNamespace(file_dict={})
    experiment.stdout = ""
    return experiment


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
    sota_experiment = _model_experiment(result=pd.Series({metric: 0.05 for metric in IMPORTANT_METRICS}))
    sota_experiment.hypothesis = SimpleNamespace(hypothesis="SOTA hypothesis", action="model")
    sota_experiment.sub_tasks = [SimpleNamespace(get_task_information=lambda: "SOTA task")]
    sota_experiment.sub_workspace_list = [SimpleNamespace(file_dict={"model.py": "SOTA code"})]
    trace = SimpleNamespace(
        hist=[(sota_experiment, SimpleNamespace(decision=True))],
        get_sota_hypothesis_and_experiment=lambda: (sota_experiment.hypothesis, sota_experiment),
    )
    scenario = SimpleNamespace(get_scenario_all_desc=lambda **_kwargs: "scenario")

    generated = QlibModelExperiment2Feedback(scenario).generate_feedback(exp, trace)

    assert backend.build_messages_and_create_chat_completion.call_count == 1
    assert generated.decision is True
    assert generated.observations == "first"
    context = rendered_contexts["scenarios.qlib.prompts:model_feedback_generation.user"]
    evidence = context["combined_result"]
    assert "IC: Current=0.100000, SOTA=0.050000, Delta=+0.050000" in evidence
    assert context["has_sota"] is True
    assert context["sota_model_hypothesis"] is sota_experiment.hypothesis
    assert context["sota_model_task"] == "SOTA task"
    assert context["sota_model_code"] == "SOTA code"


def test_model_feedback_reads_lightgbm_when_no_model_experiment_is_accepted(monkeypatch) -> None:
    backend = Mock()
    rendered_contexts = {}
    backend.build_messages_and_create_chat_completion.return_value = _response(
        decision=False,
        observation="observed",
    )
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)

    def template(uri):
        def render(**context):
            rendered_contexts[uri] = context
            return "rendered prompt"

        return SimpleNamespace(r=render)

    monkeypatch.setattr(feedback_module, "T", template)
    accepted_factor = SimpleNamespace(
        hypothesis=SimpleNamespace(hypothesis="accepted factor", action="factor"),
        sub_tasks=[SimpleNamespace(get_task_information=lambda: "factor task")],
        sub_workspace_list=[SimpleNamespace(file_dict={"factor.py": "factor code"})],
        experiment_workspace=SimpleNamespace(
            file_dict={
                "conf_combined_factors.yaml": """
task:
  model:
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
      learning_rate: 0.2
      num_threads: 16
"""
            }
        ),
        result=_full_result(scale=0.5),
    )
    rejected_model = _model_experiment(result=_full_result(scale=0.4))
    rejected_model.hypothesis = SimpleNamespace(hypothesis="rejected model", action="model")
    trace = SimpleNamespace(
        hist=[
            (rejected_model, SimpleNamespace(decision=False)),
            (accepted_factor, SimpleNamespace(decision=True)),
        ],
        get_sota_hypothesis_and_experiment=lambda: (accepted_factor.hypothesis, accepted_factor),
    )
    scenario = SimpleNamespace(get_scenario_all_desc=lambda **_kwargs: "scenario")

    QlibModelExperiment2Feedback(scenario).generate_feedback(
        _model_experiment(result=_full_result(scale=1.0)),
        trace,
    )

    context = rendered_contexts["scenarios.qlib.prompts:model_feedback_generation.user"]
    assert context["has_sota"] is True
    assert context["sota_model_hypothesis"] is None
    assert context["sota_model_task"] is None
    assert "class: LGBModel" in context["sota_model_code"]
    assert "module_path: qlib.contrib.model.gbdt" in context["sota_model_code"]
    assert "num_threads: 16" in context["sota_model_code"]


def test_model_feedback_keeps_global_sota_metrics_but_reads_last_accepted_model(monkeypatch) -> None:
    backend = Mock()
    rendered_contexts = {}
    backend.build_messages_and_create_chat_completion.return_value = _response(
        decision=False,
        observation="observed",
    )
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)

    def template(uri):
        def render(**context):
            rendered_contexts[uri] = context
            return "rendered prompt"

        return SimpleNamespace(r=render)

    monkeypatch.setattr(feedback_module, "T", template)
    accepted_model = _model_experiment(result=_full_result(scale=0.4))
    accepted_model.hypothesis = SimpleNamespace(hypothesis="accepted model", action="model")
    accepted_model.sub_tasks = [SimpleNamespace(get_task_information=lambda: "accepted model task")]
    accepted_model.sub_workspace_list = [SimpleNamespace(file_dict={"model.py": "accepted model code"})]
    accepted_factor = SimpleNamespace(
        hypothesis=SimpleNamespace(hypothesis="accepted factor", action="factor"),
        result=_full_result(scale=0.5),
    )
    trace = SimpleNamespace(
        hist=[
            (accepted_model, SimpleNamespace(decision=True)),
            (accepted_factor, SimpleNamespace(decision=True)),
        ],
        get_sota_hypothesis_and_experiment=lambda: (accepted_factor.hypothesis, accepted_factor),
    )
    scenario = SimpleNamespace(get_scenario_all_desc=lambda **_kwargs: "scenario")

    QlibModelExperiment2Feedback(scenario).generate_feedback(
        _model_experiment(result=_full_result(scale=1.0)),
        trace,
    )

    context = rendered_contexts["scenarios.qlib.prompts:model_feedback_generation.user"]
    assert context["sota_model_hypothesis"] is accepted_model.hypothesis
    assert context["sota_model_task"] == "accepted model task"
    assert context["sota_model_code"] == "accepted model code"
    assert "IC: Current=0.030000, SOTA=0.015000, Delta=+0.015000" in context["combined_result"]


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
        hist=[(sota_experiment, SimpleNamespace(decision=True))],
        get_sota_hypothesis_and_experiment=lambda: (sota_experiment.hypothesis, sota_experiment),
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
    assert "Training Log: training log" in user_prompt
    assert "Do not claim that turnover" in system_prompt


def test_model_feedback_uses_initial_factor_baseline_without_global_sota(monkeypatch) -> None:
    backend = Mock()
    backend.build_messages_and_create_chat_completion.return_value = _response(
        decision=False,
        observation="observed",
    )
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)
    monkeypatch.setattr("rdagent.utils.agent.tpl.logger.log_object", lambda *_args, **_kwargs: None)
    exp = _model_experiment(result=_full_result(scale=1.0))
    baseline = _factor_experiment(result=_full_result(scale=0.5))
    baseline.experiment_workspace.file_dict = {
        "conf_baseline.yaml": """
task:
  model:
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
      learning_rate: 0.2
      num_threads: 16
"""
    }
    rejected_factor = _factor_experiment(result=_full_result(scale=0.4), based_experiments=[baseline])
    trace = SimpleNamespace(
        hist=[(rejected_factor, SimpleNamespace(decision=False))],
        get_sota_hypothesis_and_experiment=lambda: (None, None),
    )
    scenario = SimpleNamespace(get_scenario_all_desc=lambda **_kwargs: "scenario")

    QlibModelExperiment2Feedback(scenario).generate_feedback(exp, trace)

    user_prompt = backend.build_messages_and_create_chat_completion.call_args.kwargs["user_prompt"]
    assert "class: LGBModel" in user_prompt
    assert "num_threads: 16" in user_prompt
    assert "IC: Current=0.030000, SOTA=0.015000, Delta=+0.015000" in user_prompt
    assert "Annualized return cost drag: Current=0.050000, SOTA=0.025000, Delta=+0.025000" in user_prompt
    assert "ICIR is greater than 0" not in user_prompt


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
