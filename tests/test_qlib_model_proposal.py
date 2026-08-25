import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.scenarios.qlib.experiment.model_training import normalize_generated_training_hyperparameters
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesis2Experiment


def _base_training_config() -> dict:
    return {
        "n_epochs": 6,
        "lr": 0.0002,
        "early_stop": 2,
        "batch_size": 256,
        "weight_decay": 0.001,
        "optimizer": {"name": "adam"},
        "loss": {"name": "mse"},
        "sam": {"enabled": False},
        "data_loader": {"batch_mode": "sample", "shuffle": True, "drop_last": True},
        "checkpoint": {"metric": "loss"},
        "gradient_clip": {"mode": "value", "threshold": 3.0},
        "scheduler": {"name": "none"},
    }


def test_model_experiment_prompt_has_one_valid_and_unambiguous_config_contract() -> None:
    prompts = yaml.safe_load(
        (Path(__file__).parents[1] / "rdagent/scenarios/qlib/prompts.yaml").read_text()
    )
    output_format = prompts["model_experiment_output_format"]

    assert "The current target hypothesis is authoritative" in output_format
    assert "`training_hyperparameters` is the sole executable source" in output_format
    assert "Do not duplicate training settings in `hyperparameters`" in output_format
    assert '"training_hyperparameters" {' not in output_format
    json.loads(output_format[output_format.index("{") :])


def test_generated_training_contract_preserves_all_active_values() -> None:
    config = _base_training_config()
    config.update(
        {
            "optimizer": {"name": "sgd", "momentum": 0.8},
            "loss": {
                "name": "tail_listnet",
                "temperature": 0.2,
                "tail_fraction": 0.15,
                "top_weight": 0.5,
                "bottom_weight": 0.5,
            },
            "sam": {"enabled": True, "rho": 0.04, "adaptive": True},
            "data_loader": {"batch_mode": "date", "shuffle": False, "drop_last": False},
            "checkpoint": {"metric": "topk_precision", "topk": 50},
            "gradient_clip": {"mode": "norm", "threshold": 0.2},
            "scheduler": {
                "name": "plateau",
                "factor": 0.4,
                "patience": 2,
                "min_lr": 1e-6,
                "threshold": 1e-5,
            },
            "time_series_lookback": 20,
        }
    )

    normalized = normalize_generated_training_hyperparameters(config, "TimeSeries")

    assert normalized["optimizer"] == {"name": "sgd", "momentum": 0.8}
    assert normalized["loss"] == {
        "name": "tail_listnet",
        "huber_delta": 1.0,
        "temperature": 0.2,
        "tail_fraction": 0.15,
        "top_weight": 0.5,
        "bottom_weight": 0.5,
    }
    assert normalized["sam"] == {"enabled": True, "rho": 0.04, "adaptive": True}
    assert normalized["checkpoint"] == {"metric": "topk_precision", "topk": 50}
    assert normalized["time_series_lookback"] == 20


@pytest.mark.parametrize(
    ("path", "value", "missing"),
    [
        (("loss",), {"name": "huber"}, "huber_delta"),
        (("loss",), {"name": "pairwise"}, "temperature"),
        (("loss",), {"name": "listnet"}, "temperature"),
        (("loss",), {"name": "tail_listnet"}, "bottom_weight"),
        (("loss",), {"name": "ordinal"}, "num_bins"),
        (("optimizer",), {"name": "sgd"}, "momentum"),
        (("sam",), {"enabled": True}, "adaptive"),
        (("checkpoint",), {"metric": "topk_precision"}, "topk"),
        (("gradient_clip",), {"mode": "norm"}, "threshold"),
        (("scheduler",), {"name": "plateau"}, "factor"),
    ],
)
def test_generated_training_contract_rejects_missing_active_fields(path, value, missing) -> None:
    config = _base_training_config()
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    if value.get("name") in {"pairwise", "listnet", "tail_listnet"}:
        config["data_loader"] = {"batch_mode": "date", "shuffle": False, "drop_last": False}
    if value.get("metric") in {"ic", "rank_ic", "icir", "topk_precision"}:
        config["data_loader"] = {"batch_mode": "date", "shuffle": False, "drop_last": False}

    with pytest.raises(ValueError, match=missing):
        normalize_generated_training_hyperparameters(config, "Tabular")


def test_generated_training_contract_rejects_missing_root_field() -> None:
    config = _base_training_config()
    del config["lr"]

    with pytest.raises(ValueError, match="explicitly provide: lr"):
        normalize_generated_training_hyperparameters(config, "Tabular")


def test_generated_training_contract_requires_explicit_time_series_lookback() -> None:
    with pytest.raises(ValueError, match="time_series_lookback"):
        normalize_generated_training_hyperparameters(_base_training_config(), "TimeSeries")


def test_loop_94_style_response_cannot_silently_default_tail_loss_parameters() -> None:
    config = _base_training_config()
    config.update(
        {
            "loss": {"name": "tail_listnet", "huber_delta": 1.0},
            "data_loader": {"batch_mode": "date", "shuffle": False, "drop_last": False},
            "checkpoint": {"metric": "topk_precision", "topk": 50},
            "time_series_lookback": 20,
        }
    )

    with pytest.raises(ValueError, match="loss.*bottom_weight"):
        normalize_generated_training_hyperparameters(config, "TimeSeries")


def test_model_response_conversion_keeps_hypothesis_training_values(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RD_AGENT_SETTINGS, "workspace_path", tmp_path / "workspaces")
    training = _base_training_config()
    training.update(
        {
            "loss": {
                "name": "tail_listnet",
                "temperature": 0.2,
                "tail_fraction": 0.15,
                "top_weight": 0.5,
                "bottom_weight": 0.5,
            },
            "data_loader": {"batch_mode": "date", "shuffle": False, "drop_last": False},
            "checkpoint": {"metric": "topk_precision", "topk": 50},
            "time_series_lookback": 20,
        }
    )
    response = {
        "candidate": {
            "description": "test model",
            "formulation": "y = f(x)",
            "architecture": "one small time-series network",
            "variables": {"y": "prediction"},
            "hyperparameters": {"hidden_size": 8},
            "training_hyperparameters": deepcopy(training),
            "model_type": "TimeSeries",
        }
    }

    experiment = QlibModelHypothesis2Experiment().convert_response(
        json.dumps(response),
        hypothesis=SimpleNamespace(),
        trace=SimpleNamespace(hist=[]),
    )

    task = experiment.sub_tasks[0]
    assert task.hyperparameters == {"hidden_size": 8}
    assert task.training_hyperparameters["loss"]["temperature"] == 0.2
    assert task.training_hyperparameters["loss"]["tail_fraction"] == 0.15
    assert task.training_hyperparameters["loss"]["top_weight"] == 0.5
    assert task.training_hyperparameters["loss"]["bottom_weight"] == 0.5
