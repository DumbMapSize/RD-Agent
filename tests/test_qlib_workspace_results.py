from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from rdagent.scenarios.qlib.experiment import workspace
from rdagent.utils.env import EnvResult


@pytest.mark.parametrize("outcome", ["qrun_failed", "no_run", "failed_run", "missing_report", "two_runs", "success"])
def test_only_this_invocations_completed_result_is_returned(tmp_path, monkeypatch, outcome):
    store = tmp_path / "mlruns"
    client = MlflowClient(tracking_uri=store.as_uri())
    exp = client.create_experiment("workflow")
    metrics_path, report_path = tmp_path / "qlib_res.csv", tmp_path / "ret.pkl"

    def record(execution_id, score, *, report=True, status="FINISHED"):
        run_id = client.create_run(exp).info.run_id
        client.log_param(run_id, "_QLIB_RDAGENT_EXECUTION_ID", execution_id)
        client.log_metric(run_id, "IC", score)
        if report:
            path = store / exp / run_id / "artifacts/portfolio_analysis"
            path.mkdir(parents=True)
            pd.DataFrame({"return": [score]}).to_pickle(path / "report_normal_1day.pkl")
        client.set_terminated(run_id, status=status)

    record("previous", 0.11)
    pd.Series({"IC": 0.11}).to_csv(metrics_path)
    pd.DataFrame({"return": [0.11]}).to_pickle(report_path)

    def qrun(*, local_path, entry, env):
        assert local_path == str(tmp_path) and entry == "qrun conf.yaml"
        if outcome == "qrun_failed":
            return EnvResult("original error", 23, 0.1)
        if outcome != "no_run":
            record(env["_QLIB_RDAGENT_EXECUTION_ID"], 0.22, report=outcome != "missing_report",
                   status="FAILED" if outcome == "failed_run" else "FINISHED")
        if outcome == "two_runs":
            record(env["_QLIB_RDAGENT_EXECUTION_ID"], 0.33)
        return EnvResult("qrun finished", 0, 0.1)

    env = Mock()
    env.run.side_effect = qrun
    monkeypatch.setattr(workspace, "MODEL_COSTEER_SETTINGS", SimpleNamespace(env_type="conda"))
    monkeypatch.setattr(workspace, "QlibCondaEnv", lambda conf: env)
    monkeypatch.setattr(workspace, "logger", Mock())
    ws = object.__new__(workspace.QlibFBWorkspace)
    ws.workspace_path = tmp_path
    result, log = ws.execute()
    env.run.assert_called_once()
    if outcome == "success":
        assert result.to_dict() == {"IC": 0.22}
        pd.testing.assert_series_equal(result, pd.read_csv(metrics_path, index_col=0).iloc[:, 0], check_exact=True)
        assert pd.read_pickle(report_path)["return"].tolist() == [0.22]
    else:
        assert result is None and not metrics_path.exists() and not report_path.exists()
        if outcome == "qrun_failed":
            assert "code 23" in log and "original error" in log
        else:
            assert "extraction failed" in log and "qrun finished" in log
