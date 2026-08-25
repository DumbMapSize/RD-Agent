from types import SimpleNamespace

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


def test_factor_history_keeps_hypothesis_and_formula_but_not_variable_prose():
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
        "factor-hypothesis-marker",
        "factor-reason-marker",
        "factor-marker",
        "description-marker",
        "formula-marker",
        "price-marker",
        "observation-marker",
        "evaluation-marker",
    ):
        assert marker in rendered
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


def test_model_history_without_task_keeps_full_hypothesis():
    experiment = SimpleNamespace(
        hypothesis=DummyHypothesis("model", "failed-model-hypothesis-marker", "model-reason-marker"),
        sub_tasks=[],
        result=None,
    )
    trace = Trace(DummyQuantScenario())
    trace.hist = [(experiment, make_feedback())]

    rendered = T("scenarios.qlib.prompts:quant_hypothesis_and_feedback").r(trace=trace)

    assert "failed-model-hypothesis-marker" in rendered


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
    assert "architecture-marker" in model_context["hypothesis_and_feedback"]
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
    assert "architecture-marker" in context["hypothesis_and_feedback"]
    assert "repeated-model-hypothesis-marker" in context["last_hypothesis_and_feedback"]
    assert "supported root fields" in context["hypothesis_specification"]
    assert "`adamw`" in context["hypothesis_specification"]
    assert "do not propose or claim any other training-loop behavior" in context["hypothesis_specification"].lower()


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

    assert "factor-history-marker" in factor_history
    assert "old-accepted-model-marker" in factor_history
    assert "new-rejected-model-marker" not in factor_history

    monkeypatch.setattr("rdagent.scenarios.qlib.proposal.quant_proposal.random.choice", lambda _: "model")
    context, _ = QlibQuantHypothesisGen(scenario).prepare_context(trace)
    model_history = context["hypothesis_and_feedback"]

    assert "factor-history-marker" in model_history
    assert "old-accepted-model-marker" in model_history
    assert "new-rejected-model-marker" in model_history


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
