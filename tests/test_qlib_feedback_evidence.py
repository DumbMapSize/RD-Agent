import json
import pickle
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.qlib.developer import feedback as module


def _experiment(path, *, topk=2):
    artifacts = path / "mlruns/experiment/run/artifacts"
    (artifacts / "portfolio_analysis").mkdir(parents=True)
    dates = pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01"])
    report = pd.DataFrame({"return": [.02, .03, -.01], "bench": [.01, .01, 0], "cost": .001}, index=dates)
    report.to_pickle(path / "ret.pkl")
    report.to_pickle(artifacts / "portfolio_analysis/report_normal_1day.pkl")
    index = pd.MultiIndex.from_product([dates, list("abcd")], names=["datetime", "instrument"])
    pd.DataFrame({"score": [4., 3., 2., 1.] * 3}, index=index).to_pickle(artifacts / "pred.pkl")
    pd.DataFrame({"label": [.4, .2, -.2, -.4] * 3}, index=index).to_pickle(artifacts / "label.pkl")
    task = {"record": [{"kwargs": {"config": {"strategy": {
        "class": "TopkDropoutStrategy", "kwargs": {"topk": topk},
    }}}}]}
    pd.to_pickle(task, artifacts / "task")
    return SimpleNamespace(experiment_workspace=SimpleNamespace(workspace_path=path)), artifacts


def test_artifact_evidence_uses_configured_topk_and_actual_quarterly_returns(tmp_path):
    exp, _ = _experiment(tmp_path)
    result = module._experiment_evidence(exp)
    assert result["portfolio"]["quarterly_net_excess"] == {
        "2026Q1": {"valid_days": 2, "sum": pytest.approx(.028)},
        "2026Q2": {"valid_days": 1, "sum": pytest.approx(-.011)},
    }
    assert result["prediction"]["finite_fraction"] == 1
    assert result["prediction"]["daily_unique_min"] == 4
    assert result["prediction"]["largest_tie_fraction_median"] == .25
    assert result["topk_labels"]["topk"] == 2
    assert result["topk_labels"]["mean_excess_label"] == pytest.approx(.3)
    assert result["topk_labels"]["mean_label_coverage"] == 1
    assert result["topk_labels"]["quarterly_excess_label"]["2026Q2"]["mean"] == pytest.approx(.3)
    json.dumps(result, allow_nan=False)


def test_boundary_ties_are_fractional_and_row_order_independent(tmp_path):
    _, artifacts = _experiment(tmp_path)
    scores = pd.read_pickle(artifacts / "pred.pkl").iloc[:, 0]
    labels = pd.read_pickle(artifacts / "label.pkl").iloc[:, 0]
    scores.iloc[:] = [3., 2., 2., 1.] * 3
    result = module._topk_label_diagnostic(scores, labels, 2)
    np.testing.assert_allclose(result.excess_label, .2)
    pd.testing.assert_frame_equal(result, module._topk_label_diagnostic(scores.iloc[::-1], labels, 2))


def test_missing_labels_are_excluded_after_score_only_selection(tmp_path):
    exp, artifacts = _experiment(tmp_path)
    labels = pd.read_pickle(artifacts / "label.pkl")
    labels.loc[(slice(None), "a"), :] = np.nan
    labels.to_pickle(artifacts / "label.pkl")
    evidence = module._experiment_evidence(exp)["topk_labels"]
    assert evidence["mean_label_coverage"] == .5
    assert evidence["mean_excess_label"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("value", [0.0, 1e-15])
def test_constant_predictions_are_reported_without_an_artificial_topk_advantage(tmp_path, value):
    exp, artifacts = _experiment(tmp_path)
    scores = pd.read_pickle(artifacts / "pred.pkl")
    scores.iloc[:] = value
    scores.to_pickle(artifacts / "pred.pkl")
    evidence = module._experiment_evidence(exp)
    assert evidence["prediction"]["daily_unique_min"] == 1
    assert evidence["prediction"]["largest_tie_fraction_median"] == 1
    assert evidence["topk_labels"]["mean_excess_label"] == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("kind", ["all_scores_nan", "all_labels_nan", "insufficient_scores"])
def test_nonfinite_and_insufficient_samples_remain_unavailable_not_zero(tmp_path, kind):
    exp, artifacts = _experiment(tmp_path, topk=3)
    file = artifacts / ("label.pkl" if kind == "all_labels_nan" else "pred.pkl")
    frame = pd.read_pickle(file)
    frame.iloc[:] = np.nan
    if kind == "insufficient_scores":
        frame.iloc[:, 0] = [np.inf, np.nan, 2., 1.] * 3
    frame.to_pickle(file)
    evidence = module._experiment_evidence(exp)
    assert evidence["topk_labels"]["mean_excess_label"] is None
    assert evidence["topk_labels"]["quarterly_excess_label"]["2026Q1"]["valid_days"] == 0
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize("fault", ["missing", "ambiguous", "wrong_report", "bad_pickle"])
def test_missing_or_wrong_recorder_does_not_attach_unrelated_prediction_evidence(tmp_path, fault):
    exp, artifacts = _experiment(tmp_path)
    if fault == "missing":
        shutil.rmtree(artifacts)
    elif fault == "ambiguous":
        shutil.copytree(artifacts, tmp_path / "mlruns/experiment/second/artifacts")
    elif fault == "wrong_report":
        report = pd.read_pickle(artifacts / "portfolio_analysis/report_normal_1day.pkl")
        report["return"] += 1
        report.to_pickle(artifacts / "portfolio_analysis/report_normal_1day.pkl")
    else:
        (artifacts / "pred.pkl").write_bytes(b"not a pickle")
    evidence = module._experiment_evidence(exp)
    assert isinstance(evidence["portfolio"], dict)
    assert evidence["prediction"] == "unavailable"
    assert evidence["topk_labels"] == "unavailable"
    assert evidence["unavailable_reason"]


def test_misaligned_prediction_label_rows_do_not_produce_tail_evidence(tmp_path):
    exp, artifacts = _experiment(tmp_path)
    labels = pd.read_pickle(artifacts / "label.pkl").iloc[1:]
    labels.to_pickle(artifacts / "label.pkl")
    evidence = module._experiment_evidence(exp)
    assert isinstance(evidence["prediction"], dict)
    assert evidence["topk_labels"] == "unavailable"
    assert "indices differ" in evidence["unavailable_reason"]


@pytest.mark.parametrize("topk", [0, 1.5, "2", True])
def test_unsupported_topk_does_not_fall_back_to_a_hardcoded_value(tmp_path, topk):
    exp, _ = _experiment(tmp_path, topk=topk)
    evidence = module._experiment_evidence(exp)
    assert evidence["topk_labels"] == "unavailable"


def test_evidence_reads_current_artifacts_without_mutating_experiment(tmp_path):
    exp, artifacts = _experiment(tmp_path)
    before = pickle.dumps(exp)
    first = module._experiment_evidence(exp)
    scores = pd.read_pickle(artifacts / "pred.pkl")
    (-scores).to_pickle(artifacts / "pred.pkl")
    second = module._experiment_evidence(exp)
    assert first["topk_labels"]["mean_excess_label"] == pytest.approx(.3)
    assert second["topk_labels"]["mean_excess_label"] == pytest.approx(-.3)
    assert pickle.dumps(exp) == before


def test_comparison_reports_sample_mismatch_and_missing_history(tmp_path):
    current, _ = _experiment(tmp_path / "current")
    sota, artifacts = _experiment(tmp_path / "sota", topk=3)
    labels = pd.read_pickle(artifacts / "label.pkl")
    labels += 1
    labels.to_pickle(artifacts / "label.pkl")
    comparison = json.loads(module._comparison_evidence(current, sota).splitlines()[-1])
    assert comparison["Aligned_samples"] == {
        "benchmark_and_dates": True, "prediction_index": True, "labels": False, "topk": False,
    }
    missing = json.loads(module._comparison_evidence(current, None).splitlines()[-1])
    assert all(value is None for value in missing["Aligned_samples"].values())


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("action", ["factor", "model"])
def test_evidence_only_enters_fin_quant_and_does_not_add_llm_calls(monkeypatch, quant, action):
    class QuantScenario:
        def get_scenario_all_desc(self, **kwargs):
            return "scenario"

    monkeypatch.setattr(module, "QlibQuantScenario", QuantScenario)
    scenario = QuantScenario() if quant else SimpleNamespace(get_scenario_all_desc=lambda **kwargs: "scenario")
    evidence = Mock(return_value=" evidence")
    monkeypatch.setattr(module, "_comparison_evidence", evidence)
    contexts = {}

    def template(uri):
        def render(**kwargs):
            contexts[uri] = kwargs
            return "rendered"
        return SimpleNamespace(r=render)

    monkeypatch.setattr(module, "T", template)
    backend = Mock()
    backend.build_messages_and_create_chat_completion.return_value = json.dumps({"Decision": False, "Replace Best Result": "no"})
    monkeypatch.setattr(module, "APIBackend", lambda: backend)
    baseline = SimpleNamespace(result=pd.Series({module.NET_ANNUALIZED_RETURN: .2}), experiment_workspace=SimpleNamespace(file_dict={}))
    exp = SimpleNamespace(
        hypothesis=SimpleNamespace(hypothesis="hypothesis"), sub_tasks=[], based_experiments=[baseline],
        result=pd.Series({module.NET_ANNUALIZED_RETURN: .3}),
    )
    trace = SimpleNamespace(hist=[], get_sota_hypothesis_and_experiment=lambda: (True, baseline))
    cls = module.QlibFactorExperiment2Feedback if action == "factor" else module.QlibModelExperiment2Feedback
    cls(scenario).generate_feedback(exp, trace)
    assert evidence.call_count == int(quant)
    assert backend.build_messages_and_create_chat_completion.call_count == 1
    route = "factor_feedback_generation.system" if action == "factor" else "model_feedback_generation.user"
    policy = contexts[f"scenarios.qlib.prompts:{route}"].get("promotion_policy")
    assert bool(policy) is quant
