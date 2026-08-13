import math
import sys
from pathlib import Path

import pytest
import torch
import yaml
from qlib.workflow.cli import render_template
from qlib.contrib.model.pytorch_general_nn import GeneralPTNN as QlibGeneralPTNN
from qlib.utils import init_instance_by_config
from torch import nn, optim

from rdagent.scenarios.qlib.experiment.model_training import (
    build_model_run_env,
    normalize_model_type,
    normalize_training_hyperparameters,
)
from rdagent.scenarios.qlib.experiment.rdagent_general_ptnn import GeneralPTNN


class TabularModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.projection = nn.Linear(num_features, 1)

    def forward(self, features):
        return self.projection(features)


class TimeSeriesModel(nn.Module):
    def __init__(self, num_features: int, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.projection = nn.Linear(num_features, 1)

    def forward(self, features):
        assert features.shape[1] == self.num_timesteps
        return self.projection(features[:, -1, :])


class RecurrentTimeSeriesModel(nn.Module):
    def __init__(self, num_features: int, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.encoder = nn.LSTM(num_features, 5, batch_first=True)
        self.projection = nn.Linear(5, 1)

    def forward(self, features):
        encoded, _ = self.encoder(features)
        return self.projection(encoded[:, -1, :])


class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, num_features: int, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.input_projection = nn.Linear(num_features, 4)
        layer = nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_projection = nn.Linear(4, 1)

    def forward(self, features):
        encoded = self.encoder(self.input_projection(features))
        return self.output_projection(encoded[:, -1, :])


class ConvolutionalTimeSeriesModel(nn.Module):
    def __init__(self, num_features: int, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.encoder = nn.Conv1d(num_features, 4, kernel_size=3, padding=1)
        self.projection = nn.Linear(4, 1)

    def forward(self, features):
        encoded = self.encoder(features.transpose(1, 2)).mean(dim=-1)
        return self.projection(encoded)


def _trainer(model_cls=TabularModel, **kwargs) -> GeneralPTNN:
    model_kwargs = {"num_features": 3}
    if model_cls is not TabularModel:
        model_kwargs["num_timesteps"] = 4
    return GeneralPTNN(
        pt_model_uri=f"{__name__}.{model_cls.__name__}",
        pt_model_kwargs=model_kwargs,
        n_jobs=0,
        GPU=-1,
        **kwargs,
    )


def test_training_hyperparameters_default_to_official_general_ptnn_behavior() -> None:
    config = normalize_training_hyperparameters({}, "Tabular")

    assert config == {
        "n_epochs": 100,
        "lr": 2e-4,
        "early_stop": 10,
        "batch_size": 256,
        "weight_decay": 1e-4,
        "optimizer": {"name": "adam", "momentum": 0.0},
        "loss": {"name": "mse", "huber_delta": 1.0},
        "gradient_clip": {"mode": "value", "threshold": 3.0},
        "scheduler": {"name": "plateau", "factor": 0.5, "patience": 5, "min_lr": 1e-6, "threshold": 1e-5},
    }


def test_training_hyperparameters_normalize_common_llm_aliases() -> None:
    config = normalize_training_hyperparameters(
        {
            "optimizer": "torch.optim.SGD",
            "optimizer_momentum": "0.9",
            "loss": "torch.nn.HuberLoss",
            "huber_delta": "0.75",
            "gradient_clip_norm": "1.5",
            "scheduler": "ReduceLROnPlateau",
            "lookback": "32",
        },
        "time-series",
    )

    assert normalize_model_type("TS") == "TimeSeries"
    assert config["optimizer"] == {"name": "sgd", "momentum": 0.9}
    assert config["loss"] == {"name": "huber", "huber_delta": 0.75}
    assert config["gradient_clip"] == {"mode": "norm", "threshold": 1.5}
    assert config["scheduler"]["name"] == "plateau"
    assert config["time_series_lookback"] == 32


@pytest.mark.parametrize("gradient_clip", ["none", {"mode": "none"}, {"mode": "none", "threshold": 0.0}])
def test_training_hyperparameters_allow_zero_threshold_when_clipping_is_disabled(gradient_clip) -> None:
    config = normalize_training_hyperparameters({"gradient_clip": gradient_clip}, "Tabular")

    assert config["gradient_clip"] == {"mode": "none", "threshold": 0.0}


@pytest.mark.parametrize(
    "scheduler",
    [
        "none",
        {"name": "none"},
        {"name": "none", "factor": 1.0, "patience": 0, "min_lr": 8e-5, "threshold": 0.0},
    ],
)
def test_training_hyperparameters_ignore_inactive_scheduler_options(scheduler) -> None:
    config = normalize_training_hyperparameters({"scheduler": scheduler}, "Tabular")

    assert config["scheduler"] == {
        "name": "none",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6,
        "threshold": 1e-5,
    }


def test_training_hyperparameters_still_validate_active_scheduler_options() -> None:
    with pytest.raises(ValueError, match="scheduler.factor must be < 1.0"):
        normalize_training_hyperparameters(
            {"scheduler": {"name": "plateau", "factor": 1.0}},
            "Tabular",
        )


@pytest.mark.parametrize(
    ("field", "value", "section", "normalized_field", "expected"),
    [
        ("optimizer_momentum", 0.0, "optimizer", "momentum", 0.0),
        ("huber_delta", 0.75, "loss", "huber_delta", 0.75),
        ("scheduler_factor", 0.25, "scheduler", "factor", 0.25),
        ("scheduler_patience", 2, "scheduler", "patience", 2),
        ("scheduler_min_lr", 1e-7, "scheduler", "min_lr", 1e-7),
        ("scheduler_threshold", 1e-4, "scheduler", "threshold", 1e-4),
    ],
)
def test_flat_compatibility_field_does_not_conflict_with_implicit_defaults(
    field, value, section, normalized_field, expected
) -> None:
    config = normalize_training_hyperparameters({field: value}, "Tabular")

    assert config[section][normalized_field] == expected


def test_explicit_nested_and_flat_training_fields_still_conflict() -> None:
    with pytest.raises(ValueError, match="not both"):
        normalize_training_hyperparameters(
            {
                "scheduler": {"name": "plateau", "patience": 3},
                "scheduler_patience": 2,
            },
            "Tabular",
        )


@pytest.mark.parametrize(
    ("config", "model_type", "message"),
    [
        ({"training_adjustment": "custom loop"}, "Tabular", "Unsupported"),
        ({"optimizer": {"name": "adam", "momentum": 0.9}}, "Tabular", "only valid for SGD"),
        ({"gradient_clip": {"mode": "norm", "threshold": 0}}, "Tabular", "must be >"),
        ({"time_series_lookback": 20}, "Tabular", "only valid for TimeSeries"),
        ({"scheduler": {"name": "cosine"}}, "Tabular", "Unsupported"),
        ({"loss": "torch.nn.SmoothL1Loss"}, "Tabular", "Unsupported"),
    ],
)
def test_training_hyperparameters_reject_unsupported_or_contradictory_values(config, model_type, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_training_hyperparameters(config, model_type)


def test_time_series_run_env_uses_one_lookback_for_dataset_and_model() -> None:
    env = build_model_run_env(
        {"time_series_lookback": 40, "optimizer": {"name": "adamw"}},
        "TimeSeries",
        num_features=167,
    )

    assert env["dataset_cls"] == "TSDatasetH"
    assert env["step_len"] == env["num_timesteps"] == "40"
    assert env["num_features"] == "167"
    assert env["optimizer"] == "adamw"


@pytest.mark.parametrize(
    "relative_path",
    [
        "model_template/conf_baseline_factors_model.yaml",
        "model_template/conf_sota_factors_model.yaml",
        "factor_template/conf_combined_factors_sota_model.yaml",
    ],
)
def test_all_general_ptnn_qrun_templates_render_effective_training_config(monkeypatch, relative_path) -> None:
    env = build_model_run_env(
        {
            "optimizer": {"name": "sgd", "momentum": 0.7},
            "loss": {"name": "huber", "huber_delta": 0.4},
            "gradient_clip": {"mode": "norm", "threshold": 0.8},
            "scheduler": {"name": "none"},
            "time_series_lookback": 19,
        },
        "TimeSeries",
        num_features=163,
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    template = Path(__file__).parents[1] / "rdagent/scenarios/qlib/experiment" / relative_path

    rendered = yaml.safe_load(render_template(str(template)))

    model = rendered["task"]["model"]
    kwargs = model["kwargs"]
    assert model["module_path"] == "rdagent_general_ptnn"
    assert kwargs["optimizer"] == "sgd"
    assert kwargs["optimizer_momentum"] == 0.7
    assert kwargs["loss"] == "huber"
    assert kwargs["huber_delta"] == 0.4
    assert kwargs["gradient_clip_mode"] == "norm"
    assert kwargs["scheduler"] == "none"
    assert rendered["task"]["dataset"]["kwargs"]["step_len"] == 19


def test_rendered_qrun_model_imports_injected_adapter_and_generated_model(monkeypatch, tmp_path) -> None:
    adapter_source = Path(__file__).parents[1] / (
        "rdagent/scenarios/qlib/experiment/rdagent_general_ptnn.py"
    )
    (tmp_path / "rdagent_general_ptnn.py").write_text(adapter_source.read_text())
    (tmp_path / "model.py").write_text(
        "import torch\n"
        "class GeneratedModel(torch.nn.Module):\n"
        "    def __init__(self, num_features):\n"
        "        super().__init__()\n"
        "        self.projection = torch.nn.Linear(num_features, 1)\n"
        "    def forward(self, features):\n"
        "        return self.projection(features)\n"
        "model_cls = GeneratedModel\n"
    )
    env = build_model_run_env(
        {
            "optimizer": {"name": "adamw"},
            "loss": {"name": "huber", "huber_delta": 0.5},
            "gradient_clip": {"mode": "norm", "threshold": 0.9},
            "scheduler": {"name": "none"},
        },
        "Tabular",
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.syspath_prepend(str(tmp_path))
    template = Path(__file__).parents[1] / (
        "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml"
    )
    model_config = yaml.safe_load(render_template(str(template)))["task"]["model"]
    sys.modules.pop("model", None)
    sys.modules.pop("rdagent_general_ptnn", None)

    trainer = init_instance_by_config(model_config)

    assert type(trainer).__module__ == "rdagent_general_ptnn"
    assert type(trainer.dnn_model).__name__ == "GeneratedModel"
    assert isinstance(trainer.train_optimizer, optim.AdamW)
    assert trainer.loss == "huber"
    assert trainer.gradient_clip_mode == "norm"


@pytest.mark.parametrize(
    ("name", "expected_type", "momentum"),
    [("adam", optim.Adam, 0.0), ("adamw", optim.AdamW, 0.0), ("sgd", optim.SGD, 0.8)],
)
def test_adapter_constructs_supported_optimizer_and_scheduler_for_final_optimizer(
    name, expected_type, momentum
) -> None:
    trainer = _trainer(optimizer=name, optimizer_momentum=momentum)

    assert isinstance(trainer.train_optimizer, expected_type)
    assert trainer.lr_scheduler.optimizer is trainer.train_optimizer
    if name == "sgd":
        assert trainer.train_optimizer.param_groups[0]["momentum"] == 0.8


@pytest.mark.parametrize(("loss", "expected"), [("mae", 2.5), ("huber", 1.75)])
def test_adapter_pointwise_losses_honor_sample_weights(loss, expected) -> None:
    trainer = _trainer(loss=loss, huber_delta=1.0)
    prediction = torch.tensor([[1.0], [3.0]])
    label = torch.tensor([2.0, 1.0])
    weight = torch.tensor([1.0, 2.0])

    assert trainer.loss_fn(prediction, label, weight).item() == pytest.approx(expected)


def test_adapter_norm_clipping_constrains_gradient_norm() -> None:
    trainer = _trainer(gradient_clip_mode="norm", gradient_clip_threshold=0.25)
    for parameter in trainer.dnn_model.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)

    trainer._clip_gradients()

    total_norm = math.sqrt(sum(float(parameter.grad.square().sum()) for parameter in trainer.dnn_model.parameters()))
    assert total_norm <= 0.250001


def test_adapter_accepts_zero_gradient_clip_threshold_when_clipping_is_disabled() -> None:
    trainer = _trainer(gradient_clip_mode="none", gradient_clip_threshold=0.0)

    assert trainer.gradient_clip_mode == "none"
    assert trainer.gradient_clip_threshold == 0.0


def test_adapter_trains_tabular_and_time_series_model_shapes() -> None:
    tabular = _trainer(scheduler="none", gradient_clip_mode="none")
    tabular_data = torch.randn(8, 4)
    tabular.train_epoch([(tabular_data, torch.ones(8))])

    time_series = _trainer(TimeSeriesModel, loss="mae", optimizer="adamw")
    time_series_data = torch.randn(8, 4, 4)
    time_series.train_epoch([(time_series_data, torch.ones(8))])

    assert type(tabular.lr_scheduler).__name__ == "_NoOpScheduler"
    assert time_series.dnn_model.num_timesteps == 4


@pytest.mark.parametrize(
    "model_cls",
    [RecurrentTimeSeriesModel, TransformerTimeSeriesModel, ConvolutionalTimeSeriesModel],
)
def test_adapter_trains_common_llm_generated_time_series_architectures(model_cls) -> None:
    trainer = _trainer(model_cls, optimizer="adamw", loss="huber", gradient_clip_mode="norm")
    data = torch.randn(8, 4, 4)
    before = [parameter.detach().clone() for parameter in trainer.dnn_model.parameters()]

    trainer.train_epoch([(data, torch.ones(8))])

    after = list(trainer.dnn_model.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert all(torch.isfinite(parameter).all() for parameter in after)


def test_adapter_defaults_match_official_general_ptnn_training_step() -> None:
    common = {
        "pt_model_uri": f"{__name__}.TabularModel",
        "pt_model_kwargs": {"num_features": 3},
        "n_jobs": 0,
        "GPU": -1,
        "lr": 0.002,
        "weight_decay": 0.001,
    }
    torch.manual_seed(2026)
    official = QlibGeneralPTNN(**common)
    torch.manual_seed(2026)
    adapter = GeneralPTNN(**common)
    data = torch.tensor(
        [
            [0.2, -0.3, 0.5, 0.1],
            [0.7, 0.4, -0.2, -0.3],
            [-0.1, 0.8, 0.6, 0.2],
            [0.3, -0.5, 0.9, -0.4],
        ]
    )
    batch = [(data, torch.ones(len(data)))]

    official.train_epoch(batch)
    adapter.train_epoch(batch)

    for official_parameter, adapter_parameter in zip(
        official.dnn_model.parameters(), adapter.dnn_model.parameters(), strict=True
    ):
        assert torch.equal(official_parameter, adapter_parameter)
    assert official.lr_scheduler.state_dict() == adapter.lr_scheduler.state_dict()
