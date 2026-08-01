from types import SimpleNamespace

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesis2Experiment


def test_factor_proposal_removes_names_already_present_in_accepted_experiments(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "workspaces")
    accepted = QlibFactorExperiment(
        sub_tasks=[FactorTask("existing", "existing", "existing", variables={})]
    )
    trace = SimpleNamespace(hist=[(accepted, SimpleNamespace(decision=True))])
    response = """{
        "existing": {"description": "duplicate", "formulation": "x", "variables": {}},
        "novel": {"description": "new", "formulation": "y", "variables": {}}
    }"""

    experiment = QlibFactorHypothesis2Experiment().convert_response(
        response,
        hypothesis=SimpleNamespace(),
        trace=trace,
    )

    assert [task.factor_name for task in experiment.sub_tasks] == ["novel"]
    assert len(experiment.sub_workspace_list) == 1
