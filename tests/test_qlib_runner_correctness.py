from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.scenarios.qlib.developer import factor_runner as factor_runner_module
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
from rdagent.scenarios.qlib.developer.model_runner import QlibModelRunner
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.components.coder.model_coder.model import ModelTask


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


def test_model_runner_forwards_supported_training_hyperparameters() -> None:
    training_hyperparameters = {
        "n_epochs": 7,
        "lr": 0.003,
        "early_stop": 3,
        "batch_size": 128,
        "weight_decay": 0.002,
    }
    experiment_workspace = SimpleNamespace(
        inject_files=Mock(),
        execute=Mock(return_value=(pd.Series({"IC": 0.03}), "training complete")),
    )
    experiment = SimpleNamespace(
        based_experiments=[],
        sub_tasks=[
            SimpleNamespace(
                name="test model",
                model_type="Tabular",
                training_hyperparameters=training_hyperparameters,
            )
        ],
        sub_workspace_list=[SimpleNamespace(file_dict={"model.py": "model code"})],
        experiment_workspace=experiment_workspace,
        result=None,
        stdout="",
    )

    QlibModelRunner(SimpleNamespace()).develop(experiment)

    assert experiment_workspace.inject_files.call_args_list[0].kwargs == {"model.py": "model code"}
    assert set(experiment_workspace.inject_files.call_args_list[1].kwargs) == {"rdagent_general_ptnn.py"}
    experiment_workspace.execute.assert_called_once_with(
        qlib_config_name="conf_baseline_factors_model.yaml",
        run_env={
            "PYTHONPATH": "./",
            "n_epochs": "7",
            "lr": "0.003",
            "early_stop": "3",
            "batch_size": "128",
            "weight_decay": "0.002",
            "optimizer": "adam",
            "optimizer_momentum": "0.0",
            "loss": "mse",
            "huber_delta": "1.0",
            "loss_temperature": "1.0",
            "tail_fraction": "0.2",
            "tail_top_weight": "2.0",
            "tail_bottom_weight": "1.0",
            "ordinal_num_bins": "5",
            "sam_enabled": "false",
            "sam_rho": "0.05",
            "sam_adaptive": "false",
            "batch_mode": "sample",
            "train_shuffle": "true",
            "train_drop_last": "true",
            "checkpoint_metric": "loss",
            "checkpoint_topk": "20",
            "gradient_clip_mode": "value",
            "gradient_clip_threshold": "3.0",
            "scheduler": "plateau",
            "scheduler_factor": "0.5",
            "scheduler_patience": "5",
            "scheduler_min_lr": "1e-06",
            "scheduler_threshold": "1e-05",
            "dataset_cls": "DatasetH",
        },
    )
    assert experiment.result.loc["IC"] == 0.03
    assert experiment.stdout == "training complete"


def test_factor_runner_keeps_only_partially_deduplicated_factor_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "workspaces")
    baseline = QlibFactorExperiment(sub_tasks=[])
    accepted = QlibFactorExperiment(sub_tasks=[])
    candidate = QlibFactorExperiment(
        sub_tasks=[
            FactorTask("duplicate", "duplicate", "duplicate", variables={}),
            FactorTask("novel", "novel", "novel", variables={}),
        ],
        based_experiments=[baseline, accepted],
    )
    baseline.result = pd.Series({"IC": 0.01})
    accepted.result = pd.Series({"IC": 0.02})
    candidate.base_features = {}
    candidate.sub_workspace_list = [SimpleNamespace(name="duplicate"), SimpleNamespace(name="novel")]
    candidate.prop_dev_feedback = CoSTEERMultiFeedback(
        [SimpleNamespace(name="duplicate"), SimpleNamespace(name="novel")]
    )

    sota = _factor_frame("sota", ["2026-01-05", "2026-01-06"], [1.0, 2.0])
    new = pd.concat(
        [
            _factor_frame("duplicate", ["2026-01-05", "2026-01-06"], [1.0, 2.0]),
            _factor_frame("novel", ["2026-01-05", "2026-01-06"], [2.0, 1.0]),
        ],
        axis=1,
    )
    monkeypatch.setattr(
        factor_runner_module,
        "process_factor_data",
        lambda value: new if value is candidate else sota,
    )
    runner = QlibFactorRunner(SimpleNamespace())
    monkeypatch.setattr(runner, "deduplicate_new_factors", lambda _sota, factors: factors[["novel"]])
    candidate.experiment_workspace.execute = Mock(return_value=(pd.Series({"IC": 0.03}), "ok"))

    runner.develop(candidate)

    assert [task.factor_name for task in candidate.sub_tasks] == ["novel"]
    assert [workspace.name for workspace in candidate.sub_workspace_list] == ["novel"]
    assert [feedback.name for feedback in candidate.prop_dev_feedback] == ["novel"]


def test_factor_runner_reuses_sota_model_training_config_and_adapter(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "workspaces")
    accepted_factor = QlibFactorExperiment(sub_tasks=[])
    accepted_factor.result = pd.Series({"IC": 0.02})
    model_task = ModelTask(
        name="time series model",
        description="test",
        architecture="test",
        hyperparameters={},
        training_hyperparameters={
            "optimizer": {"name": "adamw"},
            "loss": {
                "name": "tail_listnet",
                "temperature": 0.8,
                "tail_fraction": 0.15,
                "top_weight": 3.0,
                "bottom_weight": 1.5,
            },
            "sam": {"enabled": True, "rho": 0.04, "adaptive": True},
            "data_loader": {"batch_mode": "date", "shuffle": False, "drop_last": False},
            "checkpoint": {"metric": "rank_ic", "topk": 20},
            "gradient_clip": {"mode": "norm", "threshold": 0.6},
            "scheduler": {"name": "none"},
            "time_series_lookback": 31,
        },
        model_type="TimeSeries",
    )
    accepted_model = QlibModelExperiment(sub_tasks=[model_task])
    accepted_model.result = pd.Series({"IC": 0.03})
    accepted_model.sub_workspace_list = [SimpleNamespace(file_dict={"model.py": "model code"})]
    candidate = QlibFactorExperiment(
        sub_tasks=[FactorTask("new", "new", "new", variables={})],
        based_experiments=[accepted_factor, accepted_model],
    )
    candidate.base_features = {}

    sota = _factor_frame("sota", ["2026-01-05", "2026-01-06"], [1.0, 2.0])
    new = _factor_frame("new", ["2026-01-05", "2026-01-06"], [2.0, 1.0])
    monkeypatch.setattr(
        factor_runner_module,
        "process_factor_data",
        lambda value: new if value is candidate else sota,
    )
    runner = QlibFactorRunner(SimpleNamespace())
    monkeypatch.setattr(runner, "deduplicate_new_factors", lambda _sota, factors: factors)
    candidate.experiment_workspace.inject_files = Mock(wraps=candidate.experiment_workspace.inject_files)
    candidate.experiment_workspace.execute = Mock(return_value=(pd.Series({"IC": 0.04}), "ok"))

    runner.develop(candidate)

    injected_names = [set(call.kwargs) for call in candidate.experiment_workspace.inject_files.call_args_list]
    assert {"model.py"} in injected_names
    assert {"rdagent_general_ptnn.py"} in injected_names
    run_env = candidate.experiment_workspace.execute.call_args.kwargs["run_env"]
    assert run_env["optimizer"] == "adamw"
    assert run_env["loss"] == "tail_listnet"
    assert run_env["loss_temperature"] == "0.8"
    assert run_env["tail_fraction"] == "0.15"
    assert run_env["sam_enabled"] == "true"
    assert run_env["sam_adaptive"] == "true"
    assert run_env["batch_mode"] == "date"
    assert run_env["train_shuffle"] == "false"
    assert run_env["train_drop_last"] == "false"
    assert run_env["checkpoint_metric"] == "rank_ic"
    assert run_env["gradient_clip_mode"] == "norm"
    assert run_env["scheduler"] == "none"
    assert run_env["step_len"] == run_env["num_timesteps"] == "31"
