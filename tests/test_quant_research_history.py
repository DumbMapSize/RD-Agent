from types import SimpleNamespace

import pytest

from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.coder.model_coder.model import ModelTask
from rdagent.core.proposal import Trace
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesis2Experiment
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesis2Experiment
from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen, QuantTrace
from rdagent.utils.agent.tpl import T


class DummyHypothesis:
    def __init__(self, action, body, reason):
        self.action = action
        self.hypothesis = body
        self.reason = reason

    def __str__(self):
        return f"Chosen Action: {self.action}\nHypothesis: {self.hypothesis}\nReason: {self.reason}"


class DummyQuantScenario(QlibQuantScenario):
    def __init__(self):
        pass

    def get_scenario_all_desc(self, *args, **kwargs):
        return "scenario"


class DummyScenario:
    def get_scenario_all_desc(self, *args, **kwargs):
        return "scenario"


def make_feedback(decision=False):
    return SimpleNamespace(
        observations="observation-marker",
        hypothesis_evaluation="evaluation-marker",
        decision=decision,
        new_hypothesis="next-marker",
        reason="feedback-reason-marker",
    )


def test_factor_history_keeps_reason_and_formula_without_repeated_hypothesis():
    task = FactorTask(
        factor_name="factor-marker",
        factor_description="description-marker",
        factor_formulation="formula-marker",
        variables={"price-marker": "long-variable-explanation-marker"},
    )
    experiment = SimpleNamespace(
        hypothesis=DummyHypothesis("factor", "factor-hypothesis-marker", "factor-reason-marker"),
        sub_tasks=[task],
        result=None,
    )
    trace = Trace(DummyQuantScenario())
    trace.hist = [(experiment, make_feedback())]

    rendered = T("scenarios.qlib.prompts:quant_hypothesis_and_feedback").r(trace=trace)

    for marker in (
        "factor-reason-marker",
        "factor-marker",
        "description-marker",
        "formula-marker",
        "price-marker",
        "observation-marker",
        "evaluation-marker",
    ):
        assert marker in rendered
    assert "factor-hypothesis-marker" not in rendered
    assert "long-variable-explanation-marker" not in rendered


def test_model_history_uses_actual_task_and_reason_instead_of_repeated_hypothesis():
    task = ModelTask(
        name="model-marker",
        description="description-marker",
        architecture="architecture-marker",
        hyperparameters={"hidden-marker": 16},
        training_hyperparameters={"learning-rate-marker": 0.001},
        model_type="Tabular-marker",
    )
    experiment = SimpleNamespace(
        hypothesis=DummyHypothesis("model", "repeated-model-hypothesis-marker", "model-reason-marker"),
        sub_tasks=[task],
        result=None,
    )
    trace = Trace(DummyQuantScenario())
    trace.hist = [(experiment, make_feedback(True))]

    rendered = T("scenarios.qlib.prompts:quant_hypothesis_and_feedback").r(trace=trace)

    assert "repeated-model-hypothesis-marker" not in rendered
    for marker in (
        "model-reason-marker",
        "model-marker",
        "description-marker",
        "architecture-marker",
        "hidden-marker",
        "learning-rate-marker",
        "Tabular-marker",
        "observation-marker",
        "evaluation-marker",
    ):
        assert marker in rendered


@pytest.mark.parametrize("action", ["factor", "model"])
@pytest.mark.parametrize("tasks", [[], [None]])
def test_history_without_task_keeps_full_hypothesis(action, tasks):
    experiment = SimpleNamespace(
        hypothesis=DummyHypothesis(action, "failed-hypothesis-marker", "reason-marker"),
        sub_tasks=tasks,
        result=None,
    )
    trace = Trace(DummyQuantScenario())
    trace.hist = [(experiment, make_feedback())]

    rendered = T("scenarios.qlib.prompts:quant_hypothesis_and_feedback").r(trace=trace)

    assert "failed-hypothesis-marker" in rendered


def test_quant_experiment_generation_uses_research_history():
    scenario = DummyQuantScenario()
    factor_task = FactorTask("factor-marker", "description-marker", "formula-marker")
    factor_exp = SimpleNamespace(
        hypothesis=DummyHypothesis("factor", "factor-hypothesis", "factor-reason"),
        sub_tasks=[factor_task],
        result=None,
        stdout="factor-training-log",
    )
    factor_trace = Trace(scenario)
    factor_trace.hist = [(factor_exp, make_feedback())]
    factor_context, _ = QlibFactorHypothesis2Experiment().prepare_context(
        DummyHypothesis("factor", "new-factor", "new-reason"), factor_trace
    )

    model_task = ModelTask(
        name="model-marker",
        description="description-marker",
        architecture="architecture-marker",
        hyperparameters={},
        training_hyperparameters={},
        model_type="Tabular",
    )
    model_exp = SimpleNamespace(
        hypothesis=DummyHypothesis("model", "repeated-model-hypothesis-marker", "model-reason"),
        sub_tasks=[model_task],
        result=None,
        stdout="model-training-log",
    )
    model_trace = Trace(scenario)
    model_trace.hist = [(model_exp, make_feedback(True))]
    model_context, _ = QlibModelHypothesis2Experiment().prepare_context(
        DummyHypothesis("model", "new-model", "new-reason"), model_trace
    )

    assert "factor-marker" in factor_context["hypothesis_and_feedback"]
    assert "repeated-model-hypothesis-marker" not in model_context["hypothesis_and_feedback"]
    assert "architecture-marker" not in model_context["hypothesis_and_feedback"]
    assert "architecture-marker" in model_context["last_hypothesis_and_feedback"]
    assert "architecture-marker" not in model_context["SOTA_hypothesis_and_feedback"]
    assert "sole executable source" in model_context["experiment_output_format"]
    assert "n_epochs" in model_context["experiment_output_format"]
    assert "optimizer" in model_context["experiment_output_format"]
    assert "time_series_lookback" in model_context["experiment_output_format"]
    assert "weight_decay" in model_context["experiment_output_format"]


def test_quant_hypothesis_generation_uses_research_history(monkeypatch):
    monkeypatch.setattr(QUANT_PROP_SETTING, "action_selection", "random")
    monkeypatch.setattr("rdagent.scenarios.qlib.proposal.quant_proposal.random.choice", lambda _: "model")
    scenario = DummyQuantScenario()
    task = ModelTask(
        name="model-marker",
        description="description-marker",
        architecture="architecture-marker",
        hyperparameters={},
        training_hyperparameters={},
        model_type="Tabular",
    )
    experiment = SimpleNamespace(
        hypothesis=DummyHypothesis("model", "repeated-model-hypothesis-marker", "model-reason"),
        sub_tasks=[task],
        result=None,
        stdout="model-training-log",
    )
    trace = QuantTrace(scenario)
    trace.hist = [(experiment, make_feedback(True))]

    context, _ = QlibQuantHypothesisGen(scenario).prepare_context(trace)

    assert "repeated-model-hypothesis-marker" not in context["hypothesis_and_feedback"]
    assert "architecture-marker" not in context["hypothesis_and_feedback"]
    assert "architecture-marker" in context["last_hypothesis_and_feedback"]
    assert "architecture-marker" not in context["SOTA_hypothesis_and_feedback"]
    assert "repeated-model-hypothesis-marker" in context["last_hypothesis_and_feedback"]
    assert "supported root fields" in context["hypothesis_specification"]
    assert "`adamw`" in context["hypothesis_specification"]
    assert "do not propose or claim any other training-loop behavior" in context["hypothesis_specification"].lower()
    assert "executes one candidate implementation" in context["hypothesis_specification"]
    assert "not as current-round requirements or reasons to reject the candidate" in context["hypothesis_specification"]
    assert "Without matched controls, do not claim causal attribution" in context["hypothesis_specification"]


def test_quant_hypothesis_generation_keeps_action_split(monkeypatch):
    monkeypatch.setattr(QUANT_PROP_SETTING, "action_selection", "random")
    monkeypatch.setattr("rdagent.scenarios.qlib.proposal.quant_proposal.random.choice", lambda _: "factor")
    scenario = DummyQuantScenario()

    def make_experiment(action, marker):
        task = (
            FactorTask(marker, f"{marker}-description", f"{marker}-formula")
            if action == "factor"
            else ModelTask(
                name=marker,
                description=f"{marker}-description",
                architecture=f"{marker}-architecture",
                hyperparameters={},
                training_hyperparameters={},
                model_type="Tabular",
            )
        )
        return SimpleNamespace(
            hypothesis=DummyHypothesis(action, f"{marker}-hypothesis", f"{marker}-reason"),
            sub_tasks=[task],
            result=None,
            stdout=f"{marker}-training-log",
        )

    trace = QuantTrace(scenario)
    trace.hist = [
        (make_experiment("factor", "factor-history-marker"), make_feedback(True)),
        (make_experiment("model", "old-accepted-model-marker"), make_feedback(True)),
        (make_experiment("model", "new-rejected-model-marker"), make_feedback(False)),
    ]

    context, _ = QlibQuantHypothesisGen(scenario).prepare_context(trace)
    factor_history = context["hypothesis_and_feedback"]

    assert "factor-history-marker" not in factor_history
    assert "factor-history-marker" in context["last_hypothesis_and_feedback"]
    assert "old-accepted-model-marker" in factor_history
    assert "new-rejected-model-marker" not in factor_history

    monkeypatch.setattr("rdagent.scenarios.qlib.proposal.quant_proposal.random.choice", lambda _: "model")
    context, _ = QlibQuantHypothesisGen(scenario).prepare_context(trace)
    model_history = context["hypothesis_and_feedback"]

    assert "factor-history-marker" in model_history
    assert "old-accepted-model-marker" in model_history
    assert "new-rejected-model-marker" not in model_history
    assert "new-rejected-model-marker" in context["last_hypothesis_and_feedback"]


def test_non_quant_experiment_generation_keeps_original_history():
    scenario = DummyScenario()
    task = ModelTask(
        name="model-marker",
        description="description-marker",
        architecture="architecture-marker",
        hyperparameters={},
        training_hyperparameters={},
        model_type="Tabular",
    )
    experiment = SimpleNamespace(
        hypothesis=DummyHypothesis("model", "original-hypothesis-marker", "model-reason"),
        sub_tasks=[task],
        result=None,
        stdout="model-training-log",
    )
    trace = Trace(scenario)
    trace.hist = [(experiment, make_feedback(True))]

    context, _ = QlibModelHypothesis2Experiment().prepare_context(
        DummyHypothesis("model", "new-model", "new-reason"), trace
    )

    assert "original-hypothesis-marker" in context["hypothesis_and_feedback"]


def test_latest_factor_details_are_compact_only_when_requested():
    task = FactorTask(
        "factor-marker",
        "description-marker",
        "formula-marker",
        variables={"price-marker": "long-variable-explanation-marker"},
    )
    experiment = SimpleNamespace(
        hypothesis=DummyHypothesis("factor", "hypothesis-marker", "reason-marker"),
        sub_tasks=[task],
        result=None,
        stdout="training-log-marker",
    )
    feedback = make_feedback()
    template = T("scenarios.qlib.prompts:last_hypothesis_and_feedback")

    compact = template.r(experiment=experiment, feedback=feedback, compact_factor_tasks=True)
    original = template.r(experiment=experiment, feedback=feedback)

    assert "long-variable-explanation-marker" not in compact
    assert "long-variable-explanation-marker" in original
    for marker in (
        "factor-marker",
        "description-marker",
        "formula-marker",
        "price-marker",
        "hypothesis-marker",
        "reason-marker",
        "training-log-marker",
        "observation-marker",
        "evaluation-marker",
        "next-marker",
        "feedback-reason-marker",
    ):
        assert marker in compact


@pytest.mark.parametrize("action_selection", ["random", "llm"])
def test_factor_context_excludes_only_the_separately_rendered_experiment(monkeypatch, action_selection):
    monkeypatch.setattr(QUANT_PROP_SETTING, "action_selection", action_selection)
    monkeypatch.setattr("rdagent.scenarios.qlib.proposal.quant_proposal.random.choice", lambda _: "factor")
    captured = []

    class FakeBackend:
        def build_messages_and_create_chat_completion(self, user_prompt, *args, **kwargs):
            captured.append(user_prompt)
            return '{"action": "factor"}'

    monkeypatch.setattr("rdagent.scenarios.qlib.proposal.quant_proposal.APIBackend", FakeBackend)

    def make_experiment():
        return SimpleNamespace(
            hypothesis=DummyHypothesis("factor", "hypothesis-marker", "reason-marker"),
            sub_tasks=[
                FactorTask(
                    "factor-marker",
                    "description-marker",
                    "formula-marker",
                    variables={"price-marker": "long-variable-explanation-marker"},
                )
            ],
            result=None,
            stdout="training-log-marker",
        )

    older, latest = make_experiment(), make_experiment()
    assert older is not latest
    old_feedback, latest_feedback = make_feedback(), make_feedback()
    old_feedback.observations = "older-result-marker"
    latest_feedback.observations = "latest-result-marker"
    trace = QuantTrace(DummyQuantScenario())
    trace.hist = [(older, old_feedback), (latest, latest_feedback)]
    original_nodes = list(trace.hist)

    context, _ = QlibQuantHypothesisGen(trace.scen).prepare_context(trace)

    history = context["hypothesis_and_feedback"]
    details = context["last_hypothesis_and_feedback"]
    assert "older-result-marker" in history
    assert "latest-result-marker" not in history
    assert "latest-result-marker" in details
    assert "long-variable-explanation-marker" not in history + details
    assert "hypothesis-marker" not in history
    assert "hypothesis-marker" in details
    assert trace.hist == original_nodes
    assert trace.hist[0][0] is older and trace.hist[1][0] is latest
    assert latest.sub_tasks[0].variables["price-marker"] == "long-variable-explanation-marker"
    if captured:
        assert captured[0].count("latest-result-marker") == 1
        assert captured[0].count("older-result-marker") == 1
        assert "long-variable-explanation-marker" not in captured[0]


def test_history_keeps_all_tasks_and_partial_failure_feedback():
    experiment = SimpleNamespace(
        hypothesis=DummyHypothesis("factor", "hypothesis-marker", "reason-marker"),
        sub_tasks=[
            FactorTask("successful-factor", "first-description", "first-formula", factor_implementation=True),
            FactorTask("failed-factor", "second-description", "second-formula", factor_implementation=False),
        ],
        result=None,
    )
    feedback = make_feedback()
    feedback.observations = "successful-factor evaluated; failed-factor timed out"
    trace = Trace(DummyQuantScenario())
    trace.hist = [(experiment, feedback)]

    rendered = T("scenarios.qlib.prompts:quant_hypothesis_and_feedback").r(trace=trace)

    for value in ("first-formula", "second-formula", "reason-marker", feedback.observations):
        assert value in rendered
    assert experiment.sub_tasks[0].factor_implementation is True
    assert experiment.sub_tasks[1].factor_implementation is False
