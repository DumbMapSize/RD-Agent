from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from jinja2 import Template
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from qlib.utils import init_instance_by_config

from rdagent.scenarios.qlib.experiment.model_training import build_model_run_env


TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "rdagent/scenarios/qlib/experiment"
TEMPLATES = (
    "factor_template/conf_baseline.yaml",
    "factor_template/conf_combined_factors.yaml",
    "factor_template/conf_combined_factors_sota_model.yaml",
    "model_template/conf_baseline_factors_model.yaml",
    "model_template/conf_sota_factors_model.yaml",
)


@pytest.mark.parametrize("relative_path", TEMPLATES)
def test_research_templates_preserve_values_and_use_the_same_column_order(relative_path, monkeypatch, tmp_path):
    path = TEMPLATE_ROOT / relative_path
    env = build_model_run_env({}, "Tabular", num_features=159)
    config = yaml.safe_load(Template(path.read_text()).render(**env))
    segments = config["task"]["dataset"]["kwargs"]["segments"]
    dates = pd.to_datetime([segments["train"][0], segments["train"][1], segments["test"][0]])
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001", "SH600002", "SH600003"]],
        names=["datetime", "instrument"],
    )
    names = Alpha158DL.get_feature_config()[1]
    columns = pd.MultiIndex.from_tuples([*(("feature", name) for name in names), ("label", "LABEL0")])
    raw = pd.DataFrame(
        np.random.default_rng(42).normal(size=(len(index), len(columns))).astype("float32"),
        index=index,
        columns=columns,
    )
    raw.iloc[0, 0] = np.nan
    raw.iloc[-1, -1] = np.nan

    def load_group(self, instruments, exprs, names, start_time=None, end_time=None, gp_name=None):
        if gp_name == "label":
            assert exprs[0].replace(" ", "") == "Ref($close,-2)/Ref($open,-1)-1"
            return raw[gp_name].set_axis(names, axis=1).copy()
        return raw[gp_name].loc[:, names].copy()

    monkeypatch.setattr(QlibDataLoader, "load_group_df", load_group)
    monkeypatch.chdir(path.parent)
    handler_config = config["task"]["dataset"]["kwargs"]["handler"]
    loader = handler_config["kwargs"].get("data_loader", {})
    extra_columns = pd.MultiIndex.from_tuples([("feature", "AAA_test_factor")])
    factor_path = tmp_path / "combined_factors_df.parquet"
    pd.DataFrame(np.float32(0), index=index, columns=extra_columns).to_parquet(factor_path)
    for child in loader.get("kwargs", {}).get("dataloader_l", []):
        if child["class"].endswith("StaticDataLoader"):
            child["kwargs"]["config"] = str(factor_path)

    handler = init_instance_by_config(handler_config)
    assert handler._data.columns.tolist() == sorted(handler._data.columns.tolist())

    reference = Alpha158(
        instruments="csi300",
        start_time=segments["train"][0],
        end_time=segments["test"][1],
        fit_start_time=segments["train"][0],
        fit_end_time=segments["train"][1],
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
        label=[["Ref($close, -2) / Ref($open, -1) - 1"], ["LABEL0"]],
    )
    for data_key in (DataHandlerLP.DK_R, DataHandlerLP.DK_I, DataHandlerLP.DK_L):
        actual = handler.fetch(data_key=data_key, col_set=DataHandlerLP.CS_RAW).drop(
            columns=extra_columns.tolist(), errors="ignore"
        )
        expected = reference.fetch(data_key=data_key, col_set=DataHandlerLP.CS_RAW).sort_index(axis=1)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
