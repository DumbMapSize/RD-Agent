import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from qlib.contrib.model.pytorch_general_nn import GeneralPTNN as QlibGeneralPTNN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.utils import init_instance_by_config
from qlib.workflow.cli import render_template
from torch import nn, optim

from rdagent.scenarios.qlib.experiment.model_training import (
    build_model_run_env,
    normalize_model_type,
    normalize_training_hyperparameters,
)
from rdagent.scenarios.qlib.experiment.rdagent_general_ptnn import GeneralPTNN
from rdagent.scenarios.qlib.experiment.workspace import _extract_training_log_summary


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


class BatchNormTabularModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.normalization = nn.BatchNorm1d(num_features)
        self.projection = nn.Linear(num_features, 1)

    def forward(self, features):
        return self.projection(self.normalization(features))


class CountingEvalModel(TabularModel):
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.forward_calls = 0

    def forward(self, features):
        self.forward_calls += 1
        return super().forward(features)


class SchemaAwareTabularModel(nn.Module):
    required_feature_names = ("KMID",)

    def __init__(self, num_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, features):
        kmid_index = self.feature_index["KMID"]
        return features[:, kmid_index : kmid_index + 1] * self.scale


class SchemaAwareTimeSeriesModel(nn.Module):
    required_feature_names = ("KMID",)

    def __init__(self, num_features: int, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, features):
        assert features.shape[1] == self.num_timesteps
        kmid_index = self.feature_index["KMID"]
        return features[:, -1, kmid_index : kmid_index + 1] * self.scale


class RecordingReweighter(Reweighter):
    def __init__(self):
        self.sample_counts = []

    def reweight(self, data):
        self.sample_counts.append(len(data))
        return np.ones(len(data))


def _trainer(model_cls=TabularModel, *, num_features=3, **kwargs) -> GeneralPTNN:
    model_kwargs = {"num_features": num_features}
    if "TimeSeries" in model_cls.__name__:
        model_kwargs["num_timesteps"] = 4
    return GeneralPTNN(
        pt_model_uri=f"{__name__}.{model_cls.__name__}",
        pt_model_kwargs=model_kwargs,
        n_jobs=0,
        GPU=-1,
        **kwargs,
    )


class FrameDataset:
    def __init__(self) -> None:
        columns = pd.MultiIndex.from_tuples(
            [("feature", "f0"), ("feature", "f1"), ("feature", "f2"), ("label", "LABEL0")]
        )
        self.frames = {}
        for offset, segment in enumerate(("train", "valid", "test")):
            dates = pd.date_range(f"202{offset + 2}-01-03", periods=4, freq="D")
            index = pd.MultiIndex.from_product(
                [dates, ["SH600000", "SH600001", "SH600002", "SH600003"]],
                names=["datetime", "instrument"],
            )
            first_feature = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), len(dates))
            self.frames[segment] = pd.DataFrame(
                np.column_stack(
                    [
                        first_feature,
                        np.sin(np.arange(len(index))),
                        np.cos(np.arange(len(index))),
                        first_feature + 0.05 * np.sin(np.arange(len(index))),
                    ]
                ),
                index=index,
                columns=columns,
            )

    def prepare(self, segment, col_set=None, **_kwargs):
        frame = self.frames[segment]
        if col_set == "label":
            return frame[[column for column in frame.columns if column[0] == "label"]]
        return frame


class NamedFrameDataset(FrameDataset):
    def __init__(self, feature_names, *, test_feature_names=None) -> None:
        super().__init__()
        feature_names = tuple(feature_names)
        test_feature_names = tuple(test_feature_names or feature_names)
        for segment, frame in self.frames.items():
            names = test_feature_names if segment == "test" else feature_names
            feature_values = [frame.iloc[:, position % 3].to_numpy() for position in range(len(names))]
            label = frame.iloc[:, -1].to_numpy()
            columns = pd.MultiIndex.from_tuples(
                [("feature", name) for name in names] + [("label", "LABEL0")]
            )
            self.frames[segment] = pd.DataFrame(
                np.column_stack([*feature_values, label]),
                index=frame.index,
                columns=columns,
            )


class PreparedTimeSeriesData:
    def __init__(self) -> None:
        self.index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2026-01-05"), "SH600000"),
                (pd.Timestamp("2026-01-06"), "SH600001"),
                (pd.Timestamp("2026-01-05"), "SH600002"),
                (pd.Timestamp("2026-01-06"), "SH600003"),
            ],
            names=["datetime", "instrument"],
        )
        self.values = torch.randn(4, 4, 4)
        self.fillna_type = None

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def get_index(self):
        return self.index

    def config(self, *, fillna_type):
        self.fillna_type = fillna_type


class FeatureNameHandler:
    def __init__(self, feature_names) -> None:
        self.feature_names = tuple(feature_names)

    def get_cols(self, *, col_set, data_key):
        assert col_set == "feature"
        assert data_key in {DataHandlerLP.DK_L, DataHandlerLP.DK_I}
        return list(self.feature_names)


class NamedTimeSeriesDataset:
    def __init__(self, feature_names) -> None:
        self.handler = FeatureNameHandler(feature_names)
        self.prepared = {segment: PreparedTimeSeriesData() for segment in ("train", "valid", "test")}

    def prepare(self, segment, **_kwargs):
        return self.prepared[segment]


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
        "sam": {"enabled": False, "rho": 0.05, "adaptive": False},
        "data_loader": {"batch_mode": "sample", "shuffle": True, "drop_last": True},
        "checkpoint": {"metric": "loss", "topk": 20},
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


def test_training_hyperparameters_normalize_first_batch_training_extensions() -> None:
    config = normalize_training_hyperparameters(
        {
            "optimizer": {"name": "adamw"},
            "sam": {"enabled": True, "rho": 0.08, "adaptive": True},
            "loss": {
                "name": "tail_listnet",
                "temperature": 0.7,
                "tail_fraction": 0.15,
                "top_weight": 3.0,
                "bottom_weight": 0.5,
            },
            "data_loader": {"batch_mode": "date", "shuffle": False, "drop_last": False},
            "checkpoint": {"metric": "rank_ic", "topk": 20},
        },
        "Tabular",
    )

    assert config["sam"] == {"enabled": True, "rho": 0.08, "adaptive": True}
    assert config["loss"] == {
        "name": "tail_listnet",
        "huber_delta": 1.0,
        "temperature": 0.7,
        "tail_fraction": 0.15,
        "top_weight": 3.0,
        "bottom_weight": 0.5,
    }
    assert config["data_loader"] == {"batch_mode": "date", "shuffle": False, "drop_last": False}
    assert config["checkpoint"] == {"metric": "rank_ic", "topk": 20}
    env = build_model_run_env(config, "Tabular")
    assert env["sam_enabled"] == "true"
    assert env["loss"] == "tail_listnet"
    assert env["batch_mode"] == "date"
    assert env["train_shuffle"] == "false"
    assert env["checkpoint_metric"] == "rank_ic"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"loss": {"name": "pairwise"}}, "require.*date"),
        ({"checkpoint": {"metric": "rank_ic"}}, "require date"),
        (
            {
                "loss": {"name": "tail_listnet"},
                "data_loader": {"batch_mode": "date", "drop_last": True},
            },
            "drop_last must be false",
        ),
        ({"loss": {"name": "ordinal", "num_bins": 1}}, ">= 2"),
        ({"sam": {"enabled": True, "rho": 0.0}}, "must be >"),
    ],
)
def test_training_hyperparameters_reject_invalid_first_batch_combinations(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_training_hyperparameters(config, "Tabular")


@pytest.mark.parametrize(
    ("loss", "message"),
    [
        ({"name": "mse", "temperature": 0.5}, "temperature.*only valid"),
        ({"name": "listnet", "tail_fraction": 0.2}, "tail_fraction.*only valid"),
        ({"name": "pairwise", "top_weight": 2.0}, "top_weight.*only valid"),
        ({"name": "listnet", "bottom_weight": 1.0}, "bottom_weight.*only valid"),
        ({"name": "tail_listnet", "num_bins": 5}, "num_bins.*only valid"),
        ({"name": "ordinal", "temperature": 0.5}, "temperature.*only valid"),
    ],
)
def test_training_hyperparameters_reject_loss_options_that_do_not_apply(loss, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_training_hyperparameters({"loss": loss}, "Tabular")


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
            "loss": {
                "name": "tail_listnet",
                "temperature": 0.4,
                "tail_fraction": 0.15,
                "top_weight": 3.0,
                "bottom_weight": 1.0,
            },
            "sam": {"enabled": True, "rho": 0.06, "adaptive": True},
            "data_loader": {"batch_mode": "date", "shuffle": False, "drop_last": False},
            "checkpoint": {"metric": "rank_ic", "topk": 15},
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
    assert kwargs["loss"] == "tail_listnet"
    assert kwargs["loss_temperature"] == 0.4
    assert kwargs["tail_fraction"] == 0.15
    assert kwargs["tail_top_weight"] == 3.0
    assert kwargs["sam_enabled"] is True
    assert kwargs["sam_adaptive"] is True
    assert kwargs["batch_mode"] == "date"
    assert kwargs["train_shuffle"] is False
    assert kwargs["train_drop_last"] is False
    assert kwargs["checkpoint_metric"] == "rank_ic"
    assert kwargs["checkpoint_topk"] == 15
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


def test_date_loader_keeps_each_cross_section_in_one_batch() -> None:
    trainer = _trainer(
        loss="listnet",
        batch_mode="date",
        train_shuffle=False,
        train_drop_last=False,
        scheduler="none",
    )
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-05"), "SH600000"),
            (pd.Timestamp("2026-01-06"), "SH600001"),
            (pd.Timestamp("2026-01-05"), "SH600002"),
            (pd.Timestamp("2026-01-06"), "SH600003"),
            (pd.Timestamp("2026-01-06"), "SH600004"),
        ],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.2],
            [2.0, 0.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, 0.1],
            [2.0, 0.0, 0.0, 0.1],
            [2.0, 0.0, 0.0, 0.2],
        ],
        index=index,
    )

    loader, ordered_index = trainer.build_data_loader(frame, np.ones(len(frame)), train=False)
    batches = list(loader)

    assert [len(data) for data, _weight in batches] == [2, 3]
    assert [set(data[:, 0].tolist()) for data, _weight in batches] == [{1.0}, {2.0}]
    assert ordered_index.tolist() == [index[0], index[2], index[1], index[3], index[4]]


def test_date_loader_groups_time_series_samples_without_changing_windows() -> None:
    trainer = _trainer(
        TimeSeriesModel,
        loss="pairwise",
        batch_mode="date",
        train_shuffle=False,
        train_drop_last=False,
        scheduler="none",
    )
    prepared = PreparedTimeSeriesData()

    loader, ordered_index = trainer.build_data_loader(prepared, np.ones(len(prepared)), train=False)
    batches = list(loader)

    assert [data.shape for data, _weight in batches] == [(2, 4, 4), (2, 4, 4)]
    assert prepared.fillna_type == "ffill+bfill"
    assert ordered_index.tolist() == [prepared.index[0], prepared.index[2], prepared.index[1], prepared.index[3]]


@pytest.mark.parametrize("loss", ["pairwise", "listnet", "tail_listnet"])
def test_ranking_losses_prefer_correct_cross_sectional_order(loss) -> None:
    trainer = _trainer(
        loss=loss,
        batch_mode="date",
        train_drop_last=False,
        scheduler="none",
    )
    label = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    aligned = label.view(-1, 1)
    reversed_order = -aligned

    assert trainer.loss_fn(aligned, label) < trainer.loss_fn(reversed_order, label)


@pytest.mark.parametrize(("optimizer", "momentum"), [("adam", 0.0), ("adamw", 0.0), ("sgd", 0.8)])
def test_sam_executes_two_forward_backward_passes_and_updates_base_optimizer(optimizer, momentum) -> None:
    trainer = _trainer(
        optimizer=optimizer,
        optimizer_momentum=momentum,
        sam_enabled=True,
        sam_rho=0.05,
        scheduler="none",
        gradient_clip_mode="none",
    )
    forward_calls = []
    hook = trainer.dnn_model.register_forward_hook(lambda *_args: forward_calls.append(1))
    data = torch.randn(8, 4)
    before = [parameter.detach().clone() for parameter in trainer.dnn_model.parameters()]

    trainer.train_epoch([(data, torch.ones(len(data)))])
    hook.remove()

    assert len(forward_calls) == 2
    assert trainer.lr_scheduler.state_dict() == {}
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.dnn_model.parameters(), strict=True)
    )
    assert all(torch.isfinite(parameter).all() for parameter in trainer.dnn_model.parameters())


def test_sam_second_pass_does_not_update_batch_norm_running_statistics() -> None:
    data = torch.randn(8, 4)
    torch.manual_seed(2048)
    ordinary = _trainer(BatchNormTabularModel, scheduler="none", gradient_clip_mode="none")
    torch.manual_seed(2048)
    sam = _trainer(
        BatchNormTabularModel,
        sam_enabled=True,
        scheduler="none",
        gradient_clip_mode="none",
    )

    ordinary.train_epoch([(data, torch.ones(len(data)))])
    sam.train_epoch([(data, torch.ones(len(data)))])

    assert torch.equal(
        ordinary.dnn_model.normalization.running_mean,
        sam.dnn_model.normalization.running_mean,
    )
    assert torch.equal(
        ordinary.dnn_model.normalization.num_batches_tracked,
        sam.dnn_model.normalization.num_batches_tracked,
    )
    assert sam.dnn_model.normalization.momentum == ordinary.dnn_model.normalization.momentum


def test_ordinal_loss_uses_training_fold_boundaries_and_keeps_scalar_prediction() -> None:
    trainer = _trainer(
        loss="ordinal",
        ordinal_num_bins=4,
        scheduler="none",
        gradient_clip_mode="none",
    )
    training_labels = np.arange(8, dtype=float)
    training_frame = pd.DataFrame(
        np.column_stack([np.zeros((8, 3)), training_labels]),
        columns=pd.MultiIndex.from_tuples(
            [("feature", "f0"), ("feature", "f1"), ("feature", "f2"), ("label", "LABEL0")]
        ),
    )
    trainer.prepare_training_data(training_frame)
    data = torch.tensor(training_frame.to_numpy(), dtype=torch.float32)

    trainer.train_epoch([(data, torch.ones(len(data)))])
    prediction = trainer.dnn_model(data[:, :-1])

    assert trainer.ordinal_boundaries.cpu().numpy() == pytest.approx(np.quantile(training_labels, [0.25, 0.5, 0.75]))
    assert prediction.shape == (len(data), 1)
    assert torch.all((prediction >= 0.0) & (prediction <= 1.0))
    cutpoints = trainer.dnn_model.cutpoints().detach()
    assert torch.all(cutpoints[1:] > cutpoints[:-1])


def test_date_checkpoint_metric_uses_daily_rank_ic() -> None:
    trainer = _trainer(
        loss="listnet",
        batch_mode="date",
        train_shuffle=False,
        train_drop_last=False,
        checkpoint_metric="rank_ic",
        scheduler="none",
    )
    with torch.no_grad():
        trainer.dnn_model.projection.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        trainer.dnn_model.projection.bias.zero_()
    dates = [pd.Timestamp("2026-01-05")] * 3 + [pd.Timestamp("2026-01-06")] * 3
    index = pd.MultiIndex.from_arrays(
        [dates, [f"SH60000{i}" for i in range(6)]],
        names=["datetime", "instrument"],
    )
    first_feature = np.array([-1.0, 0.0, 1.0, -2.0, 0.0, 2.0])
    frame = pd.DataFrame(
        np.column_stack([first_feature, np.zeros((6, 2)), first_feature]),
        index=index,
    )
    loader, _ = trainer.build_data_loader(frame, np.ones(len(frame)), train=False)

    _loss, rank_ic = trainer._evaluate_loader(loader)

    assert rank_ic == pytest.approx(1.0)


def test_date_pointwise_training_accepts_numpy_float64_weights() -> None:
    trainer = _trainer(
        loss="mse",
        batch_mode="date",
        train_drop_last=False,
        checkpoint_metric="topk_precision",
        checkpoint_topk=2,
    )
    frame = FrameDataset().frames["train"]
    loader, _ = trainer.build_data_loader(frame, np.ones(len(frame)), train=True)
    before = trainer.dnn_model.projection.weight.detach().clone()

    trainer.train_epoch(loader)

    assert trainer.dnn_model.projection.weight.dtype == torch.float32
    assert not torch.equal(before, trainer.dnn_model.projection.weight)


def test_date_pointwise_sam_accepts_numpy_float64_weights_on_both_passes() -> None:
    trainer = _trainer(
        loss="mse",
        sam_enabled=True,
        batch_mode="date",
        train_drop_last=False,
    )
    frame = FrameDataset().frames["train"]
    loader, _ = trainer.build_data_loader(frame, np.ones(len(frame)), train=True)
    forward_calls = 0

    def count_forward(_module, _args, _output):
        nonlocal forward_calls
        forward_calls += 1

    handle = trainer.dnn_model.register_forward_hook(count_forward)
    before = trainer.dnn_model.projection.weight.detach().clone()
    try:
        trainer.train_epoch(loader)
    finally:
        handle.remove()

    assert forward_calls == 2 * len(loader)
    assert all(torch.isfinite(parameter).all() for parameter in trainer.dnn_model.parameters())
    assert not torch.equal(before, trainer.dnn_model.projection.weight)


def test_sample_evaluation_weights_partial_batch_by_finite_sample_count() -> None:
    trainer = _trainer(batch_size=4, train_drop_last=False, scheduler="none")
    with torch.no_grad():
        for parameter in trainer.dnn_model.parameters():
            parameter.zero_()
    frame = pd.DataFrame(np.column_stack([np.zeros((5, 3)), [0.0, 0.0, 0.0, 0.0, 10.0]]))
    loader, _ = trainer.build_data_loader(frame, np.ones(len(frame)), train=False)

    loss, score = trainer._evaluate_loader(loader)

    assert loss == pytest.approx(20.0)
    assert score == pytest.approx(20.0)


def test_date_evaluation_keeps_equal_date_weighting() -> None:
    trainer = _trainer(
        batch_mode="date",
        train_drop_last=False,
        checkpoint_metric="loss",
        scheduler="none",
    )
    with torch.no_grad():
        for parameter in trainer.dnn_model.parameters():
            parameter.zero_()
    index = pd.MultiIndex.from_arrays(
        [
            [pd.Timestamp("2026-01-05")] * 4 + [pd.Timestamp("2026-01-06")],
            [f"SH60000{i}" for i in range(5)],
        ],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        np.column_stack([np.zeros((5, 3)), [0.0, 0.0, 0.0, 0.0, 10.0]]),
        index=index,
    )
    loader, _ = trainer.build_data_loader(frame, np.ones(len(frame)), train=False)

    loss, score = trainer._evaluate_loader(loader)

    assert loss == pytest.approx(50.0)
    assert score == pytest.approx(50.0)


def test_default_sample_fit_keeps_partial_validation_batch(tmp_path) -> None:
    dataset = FrameDataset()
    dataset.frames["train"] = pd.concat([dataset.frames["train"]] * 3)
    trainer = _trainer(batch_size=32, n_epochs=1, early_stop=1, scheduler="none")
    evals_result = {}
    save_path = tmp_path / "sample-checkpoint.pt"

    trainer.fit(dataset, evals_result=evals_result, save_path=save_path)

    assert np.isfinite(evals_result["valid"][0])
    assert trainer.fitted is True
    assert save_path.exists()


def test_default_sample_fit_rejects_empty_training_loader(tmp_path) -> None:
    trainer = _trainer(batch_size=32, n_epochs=1, early_stop=1, scheduler="none")
    save_path = tmp_path / "empty-training-checkpoint.pt"

    with pytest.raises(ValueError, match="data loader produced no batches"):
        trainer.fit(FrameDataset(), save_path=save_path)

    assert trainer.fitted is False
    assert not save_path.exists()


def test_unified_fit_preserves_qlib_reweighter_support(tmp_path) -> None:
    trainer = _trainer(batch_size=8, n_epochs=1, early_stop=1, scheduler="none")
    reweighter = RecordingReweighter()

    trainer.fit(FrameDataset(), save_path=tmp_path / "reweighted.pt", reweighter=reweighter)

    assert reweighter.sample_counts == [16, 16]


@pytest.mark.parametrize(
    ("loss", "trainer_kwargs"),
    [
        ("mse", {}),
        ("listnet", {"batch_mode": "date", "train_drop_last": False}),
        ("ordinal", {"ordinal_num_bins": 3}),
    ],
)
def test_adapter_rejects_nonfinite_predictions_for_valid_labels(loss, trainer_kwargs) -> None:
    trainer = _trainer(loss=loss, **trainer_kwargs)
    if loss == "ordinal":
        trainer.ordinal_boundaries = torch.tensor([-0.5, 0.5])
    prediction = torch.tensor([[0.1], [float("inf")], [0.3]], requires_grad=True)
    label = torch.tensor([-1.0, 0.0, 1.0])

    with pytest.raises(FloatingPointError, match="non-finite prediction"):
        trainer.loss_fn(prediction, label)


def test_adapter_rejects_nonfinite_weights_for_valid_labels() -> None:
    trainer = _trainer(loss="mse")
    prediction = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)
    label = torch.tensor([-1.0, 0.0, 1.0])
    weight = torch.tensor([1.0, float("nan"), 1.0])

    with pytest.raises(FloatingPointError, match="non-finite sample weight"):
        trainer.loss_fn(prediction, label, weight)


def test_adapter_ignores_prediction_at_missing_label() -> None:
    trainer = _trainer(loss="mse")
    prediction = torch.tensor([[0.1], [float("inf")], [0.3]], requires_grad=True)
    label = torch.tensor([-1.0, float("nan"), 1.0])

    assert torch.isfinite(trainer.loss_fn(prediction, label))


def test_adapter_rejects_nonfinite_loss_from_finite_inputs() -> None:
    trainer = _trainer(loss="mse")
    prediction = torch.tensor([[3e38]], requires_grad=True)
    label = torch.tensor([-3e38])

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        trainer.loss_fn(prediction, label)


def test_custom_fit_rejects_run_without_finite_checkpoint_metric(monkeypatch, tmp_path) -> None:
    trainer = _trainer(
        loss="mse",
        batch_mode="date",
        train_drop_last=False,
        checkpoint_metric="rank_ic",
        n_epochs=2,
        early_stop=1,
        scheduler="none",
    )
    dataset = FrameDataset()
    dataset.frames["valid"].loc[:, ("label", "LABEL0")] = 1.0
    save_path = tmp_path / "invalid-checkpoint.pt"
    messages = []

    def capture(message, *args):
        messages.append(message % args if args else message)

    monkeypatch.setattr(trainer.logger, "info", capture)

    with pytest.raises(ValueError, match="No finite validation rank_ic checkpoint"):
        trainer.fit(dataset, save_path=save_path)

    assert trainer.fitted is False
    assert not save_path.exists()
    assert any("valid_rank_ic=unavailable" in message for message in messages)


@pytest.mark.parametrize(
    ("metric", "label", "direction"),
    [
        ("loss", "loss", "minimize"),
        ("ic", "ic", "maximize"),
        ("rank_ic", "rank_ic", "maximize"),
        ("icir", "icir", "maximize"),
        ("topk_precision", "topk_precision@17", "maximize"),
    ],
)
def test_checkpoint_log_metadata_names_metric_and_direction(metric, label, direction) -> None:
    kwargs = {"checkpoint_metric": metric, "checkpoint_topk": 17}
    if metric != "loss":
        kwargs.update({"batch_mode": "date", "train_drop_last": False})
    trainer = _trainer(**kwargs)

    assert trainer._checkpoint_label() == label
    assert trainer._checkpoint_direction() == direction


@pytest.mark.parametrize("loss", ["mse", "ordinal"])
def test_evaluation_reuses_one_forward_for_loss_and_checkpoint(loss) -> None:
    trainer = _trainer(
        CountingEvalModel,
        loss=loss,
        ordinal_num_bins=4,
        batch_mode="date",
        train_shuffle=False,
        train_drop_last=False,
        checkpoint_metric="rank_ic",
        scheduler="none",
    )
    dataset = FrameDataset()
    if loss == "ordinal":
        trainer.prepare_training_data(dataset.frames["train"])
        counted_model = trainer.dnn_model.base_model
    else:
        counted_model = trainer.dnn_model
    loader, _ = trainer.build_data_loader(dataset.frames["valid"], np.ones(len(dataset.frames["valid"])), train=False)
    counted_model.forward_calls = 0

    trainer._evaluate_loader(loader)

    assert counted_model.forward_calls == len(loader)


def test_custom_fit_logs_loss_checkpoint_context_and_early_stop(monkeypatch, tmp_path) -> None:
    trainer = _trainer(
        loss="mse",
        batch_mode="date",
        train_shuffle=False,
        train_drop_last=False,
        checkpoint_metric="topk_precision",
        checkpoint_topk=20,
        n_epochs=3,
        early_stop=1,
        scheduler="none",
    )
    messages = []

    def capture(message, *args):
        messages.append(message % args if args else message)

    monkeypatch.setattr(trainer.logger, "info", capture)
    monkeypatch.setattr(trainer, "build_data_loader", lambda data, weights, train: ([object()], None))
    monkeypatch.setattr(trainer, "train_epoch", lambda _loader: None)
    evaluations = iter([(0.9, 0.10), (1.0, 0.20), (0.8, 0.11), (1.1, 0.19)])
    monkeypatch.setattr(trainer, "_evaluate_loader", lambda _loader: next(evaluations))

    trainer.fit(FrameDataset(), save_path=tmp_path / "checkpoint.pt")

    assert messages == [
        "RD-Agent training context (authoritative): optimizer=adam; loss=mse; batch_mode=date; "
        "checkpoint=topk_precision@20; direction=maximize; scheduler=none; scheduler_monitor=none; "
        "epochs=3; early_stop_patience=1",
        "training...",
        "Epoch0: train_loss=0.900000; valid_loss=1.000000; "
        "train_topk_precision@20=0.100000; valid_topk_precision@20=0.200000; lr=0.001",
        "Epoch1: train_loss=0.800000; valid_loss=1.100000; "
        "train_topk_precision@20=0.110000; valid_topk_precision@20=0.190000; lr=0.001",
        "early stop: checkpoint=topk_precision@20; patience=1; epoch=1",
        "best checkpoint: metric=topk_precision@20; direction=maximize; value=0.200000; epoch=0",
    ]
    assert _extract_training_log_summary("\n".join(messages)) == "\n".join(
        message for message in messages if message != "training..."
    )


def test_non_loss_checkpoint_plateau_scheduler_monitors_validation_loss(monkeypatch, tmp_path) -> None:
    trainer = _trainer(
        loss="mse",
        batch_mode="date",
        train_drop_last=False,
        checkpoint_metric="rank_ic",
        n_epochs=1,
        early_stop=1,
        scheduler="plateau",
    )
    scheduler_values = []
    monkeypatch.setattr(trainer.logger, "info", lambda *_args: None)
    monkeypatch.setattr(trainer, "build_data_loader", lambda data, weights, train: ([object()], None))
    monkeypatch.setattr(trainer, "train_epoch", lambda _loader: None)
    evaluations = iter([(0.8, 0.10), (1.7, 0.25)])
    monkeypatch.setattr(trainer, "_evaluate_loader", lambda _loader: next(evaluations))
    monkeypatch.setattr(trainer.lr_scheduler, "step", scheduler_values.append)

    trainer.fit(FrameDataset(), save_path=tmp_path / "checkpoint.pt")

    assert scheduler_values == [1.7]


def test_model_prompt_explains_that_date_batches_ignore_batch_size() -> None:
    prompts = yaml.safe_load(
        (Path(__file__).parents[1] / "rdagent/scenarios/qlib/prompts.yaml").read_text()
    )

    specification = prompts["model_hypothesis_specification"]
    assert "each batch is one complete trading-date cross-section" in specification
    assert "`batch_size` has no effect" in specification


def test_date_ranking_fit_and_predict_complete_with_scalar_signal(tmp_path) -> None:
    trainer = _trainer(
        loss="tail_listnet",
        batch_mode="date",
        train_shuffle=False,
        train_drop_last=False,
        checkpoint_metric="rank_ic",
        n_epochs=2,
        early_stop=2,
        scheduler="none",
    )
    dataset = FrameDataset()
    evals_result = {}

    trainer.fit(dataset, evals_result=evals_result, save_path=tmp_path / "ranking.pt")
    prediction = trainer.predict(dataset)

    assert len(evals_result["train"]) == len(evals_result["valid"]) == 2
    assert len(prediction) == len(dataset.frames["test"])
    assert prediction.index.equals(dataset.frames["test"].index)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize(
    ("feature_names", "expected_kmid_index"),
    [
        (("ZETA", "KMID", "OMEGA"), 1),
        (("A_NEW_FACTOR", "ZETA", "KMID", "OMEGA"), 2),
    ],
)
def test_feature_schema_tracks_the_current_dataset_order(
    tmp_path,
    feature_names,
    expected_kmid_index,
) -> None:
    trainer = _trainer(
        SchemaAwareTabularModel,
        num_features=len(feature_names),
        batch_size=8,
        n_epochs=1,
        early_stop=1,
        scheduler="none",
    )
    dataset = NamedFrameDataset(feature_names)

    trainer.fit(dataset, save_path=tmp_path / "schema.pt")
    prediction = trainer.predict(dataset)

    assert trainer.feature_names == feature_names
    assert trainer.dnn_model.feature_index["KMID"] == expected_kmid_index
    assert np.isfinite(prediction).all()


def test_feature_schema_rejects_a_missing_required_feature(tmp_path) -> None:
    trainer = _trainer(
        SchemaAwareTabularModel,
        batch_size=8,
        n_epochs=1,
        early_stop=1,
        scheduler="none",
    )

    with pytest.raises(ValueError, match="requires unavailable features: KMID"):
        trainer.fit(
            NamedFrameDataset(("ALPHA", "BETA", "GAMMA")),
            save_path=tmp_path / "missing.pt",
        )


def test_feature_schema_rejects_training_inference_order_drift(tmp_path) -> None:
    trainer = _trainer(
        SchemaAwareTabularModel,
        batch_size=8,
        n_epochs=1,
        early_stop=1,
        scheduler="none",
    )
    dataset = NamedFrameDataset(
        ("KMID", "ALPHA", "BETA"),
        test_feature_names=("ALPHA", "KMID", "BETA"),
    )
    trainer.fit(dataset, save_path=tmp_path / "drift.pt")

    with pytest.raises(ValueError, match="changed between model training and inference"):
        trainer.predict(dataset)


def test_feature_schema_reaches_the_base_model_for_ordinal_training(tmp_path) -> None:
    trainer = _trainer(
        SchemaAwareTabularModel,
        loss="ordinal",
        ordinal_num_bins=4,
        batch_size=8,
        n_epochs=1,
        early_stop=1,
        scheduler="none",
    )
    dataset = NamedFrameDataset(("ALPHA", "KMID", "BETA"))

    trainer.fit(dataset, save_path=tmp_path / "ordinal-schema.pt")

    assert trainer.dnn_model.base_model.feature_index["KMID"] == 1


def test_feature_schema_supports_time_series_handler_columns(tmp_path) -> None:
    trainer = _trainer(
        SchemaAwareTimeSeriesModel,
        batch_size=2,
        n_epochs=1,
        early_stop=1,
        scheduler="none",
    )
    dataset = NamedTimeSeriesDataset(("ALPHA", "KMID", "BETA"))

    trainer.fit(dataset, save_path=tmp_path / "time-series-schema.pt")

    assert trainer.feature_names == ("ALPHA", "KMID", "BETA")
    assert trainer.dnn_model.feature_index["KMID"] == 1


def test_model_execution_template_binds_required_feature_schema(tmp_path, monkeypatch) -> None:
    model_module = ModuleType("model")
    model_module.model_cls = SchemaAwareTabularModel
    monkeypatch.setitem(sys.modules, "model", model_module)
    monkeypatch.chdir(tmp_path)
    namespace = {
        "MODEL_TYPE": "Tabular",
        "BATCH_SIZE": 4,
        "NUM_FEATURES": 3,
        "NUM_TIMESTEPS": 4,
        "NUM_EDGES": 8,
        "INPUT_VALUE": 1.0,
        "PARAM_INIT_VALUE": 0.5,
    }
    template_path = (
        Path(__file__).parents[1]
        / "rdagent/components/coder/model_coder/model_execute_template_v1.txt"
    )

    exec(compile(template_path.read_text(), str(template_path), "exec"), namespace)

    assert namespace["m"].feature_index["KMID"] == 0
    assert namespace["execution_model_output"].shape == (4, 1)
    assert np.isfinite(namespace["execution_model_output"]).all()


def test_ordinal_fit_and_predict_complete_with_training_fold_bins(tmp_path) -> None:
    trainer = _trainer(
        loss="ordinal",
        ordinal_num_bins=4,
        batch_size=8,
        n_epochs=2,
        early_stop=2,
        scheduler="none",
    )
    dataset = FrameDataset()

    trainer.fit(dataset, save_path=tmp_path / "ordinal.pt")
    prediction = trainer.predict(dataset)

    assert trainer.ordinal_boundaries is not None
    assert len(prediction) == len(dataset.frames["test"])
    assert prediction.index.equals(dataset.frames["test"].index)
    assert prediction.between(0.0, 1.0).all()


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
