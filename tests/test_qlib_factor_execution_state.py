import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.scenarios.qlib.developer import feedback as feedback_module
from rdagent.scenarios.qlib.developer import utils as utils_module
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
from rdagent.scenarios.qlib.developer.feedback import QlibFactorExperiment2Feedback
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesis2Experiment


@dataclass
class FactorWorkspace:
    frame: pd.DataFrame | None
    message: str
    workspace_path: Path
    file_dict: dict = field(default_factory=lambda: {"factor.py": "# test implementation"})
    calls: int = 0

    def execute(self, mode):
        assert mode == "All"
        self.calls += 1
        return self.message, self.frame


@pytest.fixture(autouse=True)
def isolated_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "workspaces")
    monkeypatch.setattr(feedback_module.logger, "log_object", Mock())
    monkeypatch.setattr(
        utils_module, "multiprocessing_wrapper", lambda jobs, n: [func(*args) for func, args in jobs]
    )


def make_exp(tmp_path, names, failures=()):
    exp = QlibFactorExperiment(
        sub_tasks=[FactorTask(name, name, "x", variables={}, factor_implementation=True) for name in names],
        hypothesis=SimpleNamespace(hypothesis="test hypothesis", concise_justification="test", action="factor"),
    )
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06"]), ["SH600000", "SH600001"]],
        names=["datetime", "instrument"],
    )
    exp.sub_workspace_list = [
        FactorWorkspace(
            None if name in failures else pd.DataFrame({name: [1.0, 3.0, 2.0, 4.0]}, index=index),
            f"timeout:{name}" if name in failures else "execution succeeded",
            tmp_path / name,
        )
        for name in names
    ]
    exp.prop_dev_feedback = CoSTEERMultiFeedback(
        [CoSTEERSingleFeedback("debug passed", "valid", "ok", True) for _ in names]
    )
    return exp


@pytest.mark.parametrize("legacy_feedback", [False, True])
def test_partial_execution_aligns_multiple_failures_without_mutating_coding_objects(tmp_path, legacy_feedback):
    exp = make_exp(tmp_path, ["ok0", "bad1", "debug_bad", "ok3", "bad4", "ok5"], ["bad1", "bad4"])
    if legacy_feedback:
        exp.prop_dev_feedback = CoSTEERMultiFeedback(
            [FactorSingleFeedback(execution_feedback="debug passed", final_decision=True) for _ in exp.sub_tasks]
        )
    exp.sub_tasks[2].factor_implementation = False
    exp.prop_dev_feedback[2].final_decision = False
    original_tasks = exp.sub_tasks
    original_feedback = exp.prop_dev_feedback

    result = process_factor_data(exp, update_candidate_feedback=True)

    assert result.columns.tolist() == ["ok0", "ok3", "ok5"]
    assert [task.factor_implementation for task in exp.sub_tasks] == [True, False, False, True, False, True]
    assert [bool(fb) for fb in exp.prop_dev_feedback] == [True, False, False, True, False, True]
    assert [ws.calls for ws in exp.sub_workspace_list] == [1, 1, 0, 1, 1, 1]
    for index in [1, 4]:
        assert exp.sub_tasks[index] is not original_tasks[index]
        assert exp.prop_dev_feedback[index] is not original_feedback[index]
        assert f"timeout:{exp.sub_tasks[index].factor_name}" in exp.prop_dev_feedback[index].execution
        assert original_tasks[index].factor_implementation is True
        assert original_feedback[index].final_decision is True
        assert original_feedback[index].execution == "debug passed"


def test_reading_based_experiments_does_not_modify_their_state(tmp_path):
    exp = make_exp(tmp_path, ["old0", "old1"])
    original_tasks, original_feedback = exp.sub_tasks, exp.prop_dev_feedback
    result = process_factor_data([exp])
    assert result.columns.tolist() == ["old0", "old1"]
    assert exp.sub_tasks is original_tasks
    assert exp.prop_dev_feedback is original_feedback


@pytest.mark.parametrize("count", [1, 2, 5])
def test_all_failed_candidates_raise_with_each_failure_and_skip_training(tmp_path, count):
    names = [f"bad{i}" for i in range(count)]
    exp = make_exp(tmp_path, names, names)
    baseline = make_exp(tmp_path, [])
    baseline.result = pd.Series({"IC": 0.01})
    exp.based_experiments = [baseline]
    exp.experiment_workspace.execute = Mock()

    with pytest.raises(FactorEmptyError) as raised:
        QlibFactorRunner(SimpleNamespace()).develop(exp)

    exp.experiment_workspace.execute.assert_not_called()
    assert not any(task.factor_implementation for task in exp.sub_tasks)
    assert not any(exp.prop_dev_feedback)
    for name in names:
        assert f"timeout:{name}" in str(raised.value)


@pytest.mark.parametrize("invalid", ["missing_datetime", "intraday"])
def test_existing_invalid_output_checks_also_clear_candidate_state(tmp_path, invalid):
    exp = make_exp(tmp_path, ["invalid", "valid"])
    ws = exp.sub_workspace_list[0]
    if invalid == "missing_datetime":
        ws.frame.index = ws.frame.index.set_names(["date", "instrument"])
    else:
        ws.frame.index = pd.MultiIndex.from_product(
            [pd.date_range("2026-01-05 09:30", periods=2, freq="min"), ["SH600000", "SH600001"]],
            names=["datetime", "instrument"],
        )
    result = process_factor_data(exp, update_candidate_feedback=True)
    assert result.columns.tolist() == ["valid"]
    assert exp.sub_tasks[0].factor_implementation is False
    assert not exp.prop_dev_feedback[0]
    assert exp.prop_dev_feedback[0].execution != "debug passed"


def test_all_success_preserves_values_order_and_sparse_rows(tmp_path):
    exp = make_exp(tmp_path, ["zeta", "alpha", "middle"])
    exp.sub_workspace_list[1].frame = exp.sub_workspace_list[1].frame.iloc[1:]
    expected = pd.concat([ws.frame for ws in exp.sub_workspace_list], axis=1)
    result = process_factor_data(exp, update_candidate_feedback=True)
    pd.testing.assert_frame_equal(result, expected)
    assert all(task.factor_implementation for task in exp.sub_tasks)
    assert all(exp.prop_dev_feedback)


def test_failed_factor_can_be_proposed_again_after_partial_acceptance(tmp_path):
    exp = make_exp(tmp_path, ["bad", "ok"], ["bad"])
    process_factor_data(exp, update_candidate_feedback=True)
    proposed = QlibFactorHypothesis2Experiment().convert_response(
        json.dumps(
            {name: {"description": name, "formulation": "x", "variables": {}} for name in ["bad", "ok", "new"]}
        ),
        hypothesis=SimpleNamespace(),
        trace=SimpleNamespace(hist=[(exp, True)]),
    )
    assert [task.factor_name for task in proposed.sub_tasks] == ["bad", "new"]


def test_partial_result_reaches_feedback_and_survives_reload(tmp_path, monkeypatch):
    exp = make_exp(tmp_path, ["bad0", "ok1", "bad2", "ok3"], ["bad0", "bad2"])
    baseline = make_exp(tmp_path, [])
    baseline.result = pd.Series({"IC": 0.01})
    exp.based_experiments = [baseline]
    trained_columns = []

    def training_boundary(**kwargs):
        frame = pd.read_parquet(exp.experiment_workspace.workspace_path / "combined_factors_df.parquet")
        trained_columns.extend(frame.columns.get_level_values(1))
        return pd.Series({"IC": 0.02}), "test training boundary"

    exp.experiment_workspace.execute = training_boundary
    QlibFactorRunner(SimpleNamespace()).develop(exp)
    assert trained_columns == ["ok1", "ok3"]
    del exp.experiment_workspace.execute

    backend = Mock()
    backend.build_messages_and_create_chat_completion.return_value = json.dumps({"Replace Best Result": "yes"})
    monkeypatch.setattr(feedback_module, "APIBackend", lambda: backend)
    feedback = QlibFactorExperiment2Feedback(
        SimpleNamespace(get_scenario_all_desc=lambda: "test scenario")
    ).generate_feedback(exp, SimpleNamespace())
    prompt = backend.build_messages_and_create_chat_completion.call_args.kwargs["user_prompt"]
    assert feedback.decision is True
    for name in ["bad0", "bad2"]:
        section = prompt.split(f"- {name}:", 1)[1].split("- Factor Implementation:", 1)[1]
        assert section.lstrip().startswith("False")
        assert f"timeout:{name}" in section
    for name in ["ok1", "ok3"]:
        section = prompt.split(f"- {name}:", 1)[1].split("- Factor Implementation:", 1)[1]
        assert section.lstrip().startswith("True")

    restored = pickle.loads(pickle.dumps(exp))
    assert process_factor_data([restored]).columns.tolist() == ["ok1", "ok3"]
    assert [ws.calls for ws in restored.sub_workspace_list] == [1, 2, 1, 2]


def test_partial_failure_and_correlation_dedup_keep_states_aligned(tmp_path, monkeypatch):
    baseline = make_exp(tmp_path, [])
    accepted = make_exp(tmp_path, ["old"])
    baseline.result = accepted.result = pd.Series({"IC": 0.01})
    exp = make_exp(tmp_path, ["bad", "duplicate", "novel"], ["bad"])
    exp.based_experiments = [baseline, accepted]
    runner = QlibFactorRunner(SimpleNamespace())
    monkeypatch.setattr(runner, "deduplicate_new_factors", lambda old, new: new[["novel"]])
    exp.experiment_workspace.execute = Mock(return_value=(pd.Series({"IC": 0.02}), "ok"))
    runner.develop(exp)
    frame = pd.read_parquet(exp.experiment_workspace.workspace_path / "combined_factors_df.parquet")
    assert frame.columns.get_level_values(1).tolist() == ["old", "novel"]
    assert [task.factor_name for task in exp.sub_tasks] == ["bad", "novel"]
    assert [bool(fb) for fb in exp.prop_dev_feedback] == [False, True]
    assert [ws.workspace_path.name for ws in exp.sub_workspace_list] == ["bad", "novel"]


def test_real_factor_process_failures_reach_the_training_input_and_states(tmp_path, monkeypatch):
    monkeypatch.setattr(RD_AGENT_SETTINGS, "cache_with_pickle", False)
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "data_folder", str(tmp_path / "source"))
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "python_bin", sys.executable)
    exp = make_exp(tmp_path, ["bad0", "ok1", "bad2", "ok3"])
    baseline = make_exp(tmp_path, [])
    baseline.result = pd.Series({"IC": 0.01})
    exp.based_experiments = [baseline]
    exp.sub_workspace_list = []
    for task in exp.sub_tasks:
        workspace = FactorFBWorkspace(target_task=task)
        if task.factor_name.startswith("bad"):
            code = f"raise RuntimeError('full-sample failure: {task.factor_name}')\n"
        else:
            code = (
                "import pandas as pd\n"
                "index = pd.MultiIndex.from_product([pd.to_datetime(['2026-01-05', '2026-01-06']), "
                "['SH600000', 'SH600001']], names=['datetime', 'instrument'])\n"
                f"pd.DataFrame({{{task.factor_name!r}: [1.0, 3.0, 2.0, 4.0]}}, index=index)"
                ".to_hdf('result.h5', key='data')\n"
            )
        workspace.inject_files(**{"factor.py": code})
        exp.sub_workspace_list.append(workspace)
    exp.experiment_workspace.execute = Mock(return_value=(pd.Series({"IC": 0.02}), "mock training"))

    QlibFactorRunner(SimpleNamespace()).develop(exp)

    frame = pd.read_parquet(exp.experiment_workspace.workspace_path / "combined_factors_df.parquet")
    assert frame.columns.get_level_values(1).tolist() == ["ok1", "ok3"]
    assert [task.factor_implementation for task in exp.sub_tasks] == [False, True, False, True]
    assert [bool(fb) for fb in exp.prop_dev_feedback] == [False, True, False, True]
    for index in [0, 2]:
        assert f"full-sample failure: bad{index}" in exp.prop_dev_feedback[index].execution
