from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.scenarios.qlib.developer import factor_runner as factor_runner_module
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
from rdagent.scenarios.qlib.developer.model_runner import QlibModelRunner
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


def _factor_frame(name: str, dates: list[str], values: list[float]) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates), ["SH600000"] * len(dates)],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({name: values}, index=index)


def test_factor_runner_preserves_sota_rows_when_new_factor_is_sparse(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "workspaces")
    baseline = QlibFactorExperiment(sub_tasks=[])
    accepted = QlibFactorExperiment(sub_tasks=[])
    candidate = QlibFactorExperiment(sub_tasks=[], based_experiments=[baseline, accepted])
    baseline.result = pd.Series({"IC": 0.01})
    accepted.result = pd.Series({"IC": 0.02})
    candidate.base_features = {}

    sota = _factor_frame("sota", ["2026-01-05", "2026-01-06", "2026-01-07"], [1.0, 2.0, 3.0])
    new = _factor_frame("new", ["2026-01-06", "2026-01-07"], [4.0, 5.0])

    monkeypatch.setattr(RD_AGENT_SETTINGS, "cache_with_pickle", False)
    monkeypatch.setattr(
        factor_runner_module,
        "process_factor_data",
        lambda value: new if value is candidate else sota,
    )
    runner = QlibFactorRunner(SimpleNamespace())
    monkeypatch.setattr(runner, "deduplicate_new_factors", lambda _sota, retained: retained)
    candidate.experiment_workspace.execute = Mock(return_value=(pd.Series({"IC": 0.03}), "ok"))

    runner.develop(candidate)

    combined = pd.read_parquet(candidate.experiment_workspace.workspace_path / "combined_factors_df.parquet")
    assert combined.index.equals(sota.index)
    assert combined[("feature", "sota")].tolist() == [1.0, 2.0, 3.0]
    assert pd.isna(combined.loc[(pd.Timestamp("2026-01-05"), "SH600000"), ("feature", "new")])


def test_qlib_runners_do_not_cache_whole_experiment_results() -> None:
    assert not hasattr(QlibFactorRunner.develop, "__wrapped__")
    assert not hasattr(QlibModelRunner.develop, "__wrapped__")
