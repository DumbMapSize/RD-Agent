import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rdagent.components.coder.model_coder import evaluators
from rdagent.components.coder.model_coder import eva_utils
from rdagent.components.coder.model_coder.evaluators import ModelCoSTEEREvaluator
from rdagent.components.coder.model_coder.eva_utils import ModelFinalEvaluator, _parse_final_decision
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        (" false ", False),
        ("False", False),
    ],
)
def test_parse_final_decision(value: object, expected: bool) -> None:
    assert _parse_final_decision(value) is expected


@pytest.mark.parametrize("value", [None, 0, 1, [], {}, "yes", "", "not false"])
def test_parse_final_decision_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="Invalid final_decision"):
        _parse_final_decision(value)


def test_model_final_evaluator_treats_false_string_as_false(monkeypatch) -> None:
    backend = Mock()
    backend.chat_token_limit = 1000
    backend.build_messages_and_calculate_token.return_value = 1
    backend.build_messages_and_create_chat_completion.return_value = json.dumps(
        {"final_feedback": "implementation is invalid", "final_decision": "false"}
    )
    monkeypatch.setattr(eva_utils, "APIBackend", lambda: backend)
    monkeypatch.setattr(eva_utils, "T", lambda _uri: SimpleNamespace(r=lambda **_kwargs: "prompt"))

    task = ModelTask(
        name="test_model",
        description="test model",
        architecture="test architecture",
        hyperparameters={},
        training_hyperparameters={},
        model_type="Tabular",
    )
    workspace = ModelFBWorkspace(target_task=task)

    feedback, decision = ModelFinalEvaluator(scen=None).evaluate(
        target_task=task,
        implementation=workspace,
        gt_implementation=None,
        model_execution_feedback="success",
        model_shape_feedback="shape ok",
        model_value_feedback="value unavailable",
        model_code_feedback="code does not match task",
    )

    assert feedback == "implementation is invalid"
    assert decision is False


def test_model_costeer_evaluator_uses_declared_time_series_lookback(monkeypatch) -> None:
    task = ModelTask(
        name="test_time_series_model",
        description="test model",
        architecture="fixed 20-step model",
        hyperparameters={},
        training_hyperparameters={"time_series_lookback": 20},
        model_type="TimeSeries",
    )
    workspace = ModelFBWorkspace(target_task=task)
    execute = Mock(return_value=("execution succeeded", np.zeros((8, 1))))
    monkeypatch.setattr(workspace, "execute", execute)
    monkeypatch.setattr(evaluators, "shape_evaluator", lambda *_args: ("shape ok", True))
    monkeypatch.setattr(evaluators, "value_evaluator", lambda *_args: ("value ok", True))
    monkeypatch.setattr(evaluators.ModelCodeEvaluator, "evaluate", lambda *_args, **_kwargs: ("code ok", True))
    monkeypatch.setattr(
        evaluators.ModelFinalEvaluator,
        "evaluate",
        lambda *_args, **_kwargs: ("implementation accepted", True),
    )

    feedback = ModelCoSTEEREvaluator(scen=SimpleNamespace(model_output_channel=1)).evaluate(
        target_task=task,
        implementation=workspace,
        gt_implementation=None,
    )

    assert execute.call_args.kwargs["num_timesteps"] == 20
    assert feedback.final_decision is True
