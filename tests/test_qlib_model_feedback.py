import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from rdagent.scenarios.qlib.developer import feedback as feedback_module
from rdagent.scenarios.qlib.developer.feedback import IMPORTANT_METRICS, QlibModelExperiment2Feedback


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


def test_model_feedback_calls_llm_once_and_uses_that_response(monkeypatch) -> None:
    backend = Mock()
    backend.build_messages_and_create_chat_completion.side_effect = [
        _response(decision=True, observation="first"),
        _response(decision=False, observation="second"),
    ]
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)
    monkeypatch.setattr(
        feedback_module,
        "T",
        lambda _uri: SimpleNamespace(r=lambda **_context: "rendered prompt"),
    )

    result = pd.Series({metric: 0.1 for metric in IMPORTANT_METRICS})
    exp = SimpleNamespace(
        hypothesis=SimpleNamespace(hypothesis="test hypothesis", reason="test reason"),
        sub_tasks=[SimpleNamespace(get_task_information=lambda: "task")],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "code"})],
        result=result,
        stdout="training log",
    )
    trace = SimpleNamespace(get_sota_hypothesis_and_experiment=lambda: (None, None))
    scenario = SimpleNamespace(get_scenario_all_desc=lambda **_kwargs: "scenario")

    generated = QlibModelExperiment2Feedback(scenario).generate_feedback(exp, trace)

    assert backend.build_messages_and_create_chat_completion.call_count == 1
    assert generated.decision is True
    assert generated.observations == "first"
