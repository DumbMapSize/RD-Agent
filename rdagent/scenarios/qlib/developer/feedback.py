import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent

IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]

PREDICTIVE_METRICS = (
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
)
PORTFOLIO_METRICS = (
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.information_ratio",
    "1day.excess_return_without_cost.max_drawdown",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_with_cost.max_drawdown",
)
GROSS_ANNUALIZED_RETURN = "1day.excess_return_without_cost.annualized_return"
NET_ANNUALIZED_RETURN = "1day.excess_return_with_cost.annualized_return"


def _default_model_implementation(experiment: Experiment) -> str | None:
    workspace_files = experiment.experiment_workspace.file_dict
    for config_name in ("conf_combined_factors.yaml", "conf_baseline.yaml"):
        if config_text := workspace_files.get(config_name):
            config = yaml.safe_load(config_text)
            model = config.get("task", {}).get("model") if isinstance(config, dict) else None
            if model:
                return "QLib built-in model configuration:\n" + yaml.safe_dump(
                    model,
                    sort_keys=False,
                ).strip()

    return None


def _last_accepted_model_experiment(trace: Trace) -> QlibModelExperiment | None:
    for experiment, feedback in reversed(trace.hist):
        if feedback.decision and isinstance(experiment, QlibModelExperiment):
            return experiment

    return None


def _initial_factor_baseline_experiment(trace: Trace) -> QlibFactorExperiment | None:
    for experiment, _ in trace.hist:
        if not isinstance(experiment, QlibFactorExperiment) or not experiment.based_experiments:
            continue
        baseline_experiment = experiment.based_experiments[0]
        if (
            isinstance(baseline_experiment, QlibFactorExperiment)
            and not baseline_experiment.sub_tasks
            and baseline_experiment.result is not None
        ):
            return baseline_experiment

    return None


def _as_metric_series(result: Any) -> pd.Series:
    if result is None:
        return pd.Series(dtype=float)
    if isinstance(result, pd.Series):
        return result
    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 1:
            return result.iloc[:, 0]
        if result.shape[0] == 1:
            return result.iloc[0]
        return pd.Series(dtype=float)
    try:
        return pd.Series(result)
    except (TypeError, ValueError):
        return pd.Series(dtype=float)


def _metric_value(result: pd.Series, metric: str) -> float | None:
    if metric not in result.index:
        return None
    try:
        value = float(result.loc[metric])
    except (TypeError, ValueError):
        return None
    return value if pd.notna(value) else None


def _format_value(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "unavailable"
    if abs(value) < 0.5e-6:
        value = 0.0
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _format_metric(metric: str, current: pd.Series, sota: pd.Series) -> str | None:
    current_value = _metric_value(current, metric)
    sota_value = _metric_value(sota, metric)
    if current_value is None and sota_value is None:
        return None
    delta = current_value - sota_value if current_value is not None and sota_value is not None else None
    return (
        f"{metric}: Current={_format_value(current_value)}, "
        f"SOTA={_format_value(sota_value)}, Delta={_format_value(delta, signed=True)}"
    )


def process_results(current_result: Any, sota_result: Any) -> str:
    current = _as_metric_series(current_result)
    sota = _as_metric_series(sota_result)
    sections = []
    for title, metrics in (
        ("Predictive metrics", PREDICTIVE_METRICS),
        ("Portfolio metrics", PORTFOLIO_METRICS),
    ):
        lines = [line for metric in metrics if (line := _format_metric(metric, current, sota)) is not None]
        if lines:
            sections.append(f"{title} (Delta = Current - SOTA):\n" + "\n".join(f"- {line}" for line in lines))

    current_gross = _metric_value(current, GROSS_ANNUALIZED_RETURN)
    current_net = _metric_value(current, NET_ANNUALIZED_RETURN)
    sota_gross = _metric_value(sota, GROSS_ANNUALIZED_RETURN)
    sota_net = _metric_value(sota, NET_ANNUALIZED_RETURN)
    current_drag = current_gross - current_net if current_gross is not None and current_net is not None else None
    sota_drag = sota_gross - sota_net if sota_gross is not None and sota_net is not None else None
    if current_drag is not None or sota_drag is not None:
        drag_delta = current_drag - sota_drag if current_drag is not None and sota_drag is not None else None
        sections.append(
            "Derived metric (gross annualized return - net annualized return; does not identify its cause):\n"
            f"- Annualized return cost drag: Current={_format_value(current_drag)}, "
            f"SOTA={_format_value(sota_drag)}, Delta={_format_value(drag_delta, signed=True)}"
        )

    return "\n".join(sections) if sections else "No comparable backtest metrics are available."


def _sample_digest(values: pd.Series | pd.Index) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(values).to_numpy().tobytes()).hexdigest()


def _quarterly_summary(values: pd.Series, *, aggregate: str) -> dict:
    return {
        str(quarter): {
            "valid_days": int(group.notna().sum()),
            aggregate: float(getattr(group, aggregate)()) if group.notna().any() else None,
        }
        for quarter, group in values.groupby(values.index.to_period("Q"))
    }


def _topk_label_diagnostic(scores: pd.Series, labels: pd.Series, topk: int) -> pd.DataFrame:
    scores = scores.replace([np.inf, -np.inf], np.nan)
    labels = labels.reindex(scores.index).replace([np.inf, -np.inf], np.nan)
    finite = scores.dropna()
    ranks = finite.groupby(level="datetime").rank(ascending=False, method="min")
    ties = finite.groupby([finite.index.get_level_values("datetime"), finite]).transform("size")
    # Fractional boundary ties avoid choosing stocks by index order or future labels.
    weights = ((topk - ranks + 1) / ties).clip(0, 1)
    selected_labels = labels.reindex(weights.index)
    covered_weight = weights.where(selected_labels.notna(), 0).groupby(level="datetime").sum()
    top_mean = (weights * selected_labels).groupby(level="datetime").sum(min_count=1) / covered_weight
    excess = top_mean - labels.groupby(level="datetime").mean()
    enough_scores = finite.groupby(level="datetime").size() >= topk
    return pd.DataFrame({
        "excess_label": excess.where(enough_scores.reindex(excess.index, fill_value=False)),
        "label_coverage": covered_weight / topk,
    })


def _experiment_evidence(exp: Experiment | None) -> dict:
    if exp is None:
        return {"unavailable": "No comparison experiment"}
    workspace = getattr(exp, "experiment_workspace", None)
    path = getattr(workspace, "workspace_path", None)
    evidence = {"portfolio": "unavailable", "prediction": "unavailable", "topk_labels": "unavailable"}
    samples = {}
    try:
        if path is None:
            return evidence
        path = Path(path)
        report = pd.read_pickle(path / "ret.pkl").sort_index()
        if not isinstance(report.index, pd.DatetimeIndex) or not report.index.is_unique:
            raise ValueError("Backtest dates must be unique")
        daily = report["return"] - report["bench"] - report["cost"]
        if report.empty or not np.isfinite(report[["return", "bench", "cost"]].to_numpy()).all():
            raise ValueError("Backtest daily returns are missing or nonfinite")
        evidence["portfolio"] = {
            "dates": [str(report.index.min().date()), str(report.index.max().date())],
            "quarterly_net_excess": _quarterly_summary(daily, aggregate="sum"),
        }
        samples["benchmark_and_dates"] = _sample_digest(report["bench"])
        reports = list((path / "mlruns").glob("*/*/artifacts/portfolio_analysis/report_normal_1day.pkl"))
        matches = [file for file in reports if pd.read_pickle(file).sort_index().equals(report)]
        if len(matches) != 1:
            raise ValueError("No unique recorder matches ret.pkl")
        artifacts = matches[0].parent.parent
        scores = pd.read_pickle(artifacts / "pred.pkl").iloc[:, 0].sort_index()
        if not scores.index.is_unique or "datetime" not in scores.index.names:
            raise ValueError("Prediction index must contain unique stock/date rows")
        scores = scores.replace([np.inf, -np.inf], np.nan)
        unique = scores.groupby(level="datetime").nunique()
        largest_tie = scores.groupby(level="datetime").apply(
            lambda group: group.value_counts(normalize=True).max()
        )
        evidence["prediction"] = {
            "rows": len(scores), "finite_fraction": float(scores.notna().mean()),
            "daily_unique_min": int(unique.min()), "daily_unique_median": float(unique.median()),
            "largest_tie_fraction_median": float(largest_tie.median()) if largest_tie.notna().any() else None,
        }
        samples["prediction_index"] = _sample_digest(scores.index)
        labels = pd.read_pickle(artifacts / "label.pkl").iloc[:, 0].sort_index()
        if not labels.index.is_unique or not labels.index.equals(scores.index):
            raise ValueError("Prediction and label indices differ")
        samples["labels"] = _sample_digest(labels)
        task = pd.read_pickle(artifacts / "task")
        strategies = [record.get("kwargs", {}).get("config", {}).get("strategy", {}) for record in task["record"]]
        topks = [s.get("kwargs", {}).get("topk") for s in strategies if s.get("class") == "TopkDropoutStrategy"]
        if len(topks) != 1 or type(topks[0]) is not int or topks[0] <= 0:
            raise ValueError("No supported TopkDropoutStrategy topk")
        topk = topks[0]
        tail = _topk_label_diagnostic(scores, labels, topk)
        evidence["topk_labels"] = {
            "topk": topk,
            "mean_excess_label": float(tail.excess_label.mean()) if tail.excess_label.notna().any() else None,
            "mean_label_coverage": float(tail.label_coverage.mean()) if tail.label_coverage.notna().any() else None,
            "quarterly_excess_label": _quarterly_summary(tail.excess_label, aggregate="mean"),
        }
        samples["topk"] = topk
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, EOFError, pickle.UnpicklingError, ImportError) as exc:
        evidence["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    evidence["_samples"] = samples
    return evidence


def _comparison_evidence(current: Experiment, sota: Experiment | None) -> str:
    left, right = _experiment_evidence(current), _experiment_evidence(sota)
    left_samples, right_samples = left.get("_samples", {}), right.get("_samples", {})
    aligned = {
        key: left_samples[key] == right_samples[key] if key in left_samples and key in right_samples else None
        for key in ("benchmark_and_dates", "prediction_index", "labels", "topk")
    }
    payload = {
        "Current": {k: v for k, v in left.items() if k != "_samples"},
        "SOTA": {k: v for k, v in right.items() if k != "_samples"},
        "Aligned_samples": aligned,
    }
    return (
        "\nArtifact diagnostics (null = unavailable; unequal samples are not controlled comparisons). "
        "Quarterly portfolio sums are sum(return - benchmark - cost), not compounded returns. "
        "Top-k labels use score-only selection, fractional boundary ties, and available labels after selection; "
        "excess is relative to the daily universe mean, not a holdings/cost backtest.\n"
        + json.dumps(payload, allow_nan=False)
    )


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        for index, task in enumerate(tasks_factors):
            task["execution_feedback"] = None
            if task["factor_implementation"] == "False" and exp.prop_dev_feedback is not None:
                feedback = exp.prop_dev_feedback[index]
                if feedback is not None:
                    task["execution_feedback"] = feedback.execution
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)
        promotion_policy = None
        if isinstance(self.scen, QlibQuantScenario):
            combined_result += _comparison_evidence(exp, exp.based_experiments[-1])
            promotion_policy = T("scenarios.qlib.prompts:quant_promotion_policy").r()

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor"), promotion_policy=promotion_policy,
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class QlibModelExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibModelExperiment): The experiment to generate feedback for.
            hypothesis (QlibModelHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            HypothesisFeedback: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        comparison_experiment = (
            SOTA_experiment if SOTA_hypothesis else _initial_factor_baseline_experiment(trace)
        )
        SOTA_model_experiment = _last_accepted_model_experiment(trace) if SOTA_hypothesis else None
        if SOTA_model_experiment is not None:
            SOTA_model_hypothesis = SOTA_model_experiment.hypothesis
            SOTA_model_task = SOTA_model_experiment.sub_tasks[0].get_task_information()
            SOTA_model_code = SOTA_model_experiment.sub_workspace_list[0].file_dict.get("model.py")
        else:
            SOTA_model_hypothesis = None
            SOTA_model_task = None
            SOTA_model_code = (
                _default_model_implementation(comparison_experiment) if comparison_experiment is not None else None
            )
        combined_result = process_results(
            exp.result,
            comparison_experiment.result if comparison_experiment is not None else None,
        ) if exp.result is not None else "execution failed"
        promotion_policy = None
        if isinstance(self.scen, QlibQuantScenario):
            combined_result += _comparison_evidence(exp, comparison_experiment)
            promotion_policy = T("scenarios.qlib.prompts:quant_promotion_policy").r()
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            has_sota=comparison_experiment is not None,
            sota_model_hypothesis=SOTA_model_hypothesis,
            sota_model_task=SOTA_model_task,
            sota_model_code=SOTA_model_code,
            hypothesis=hypothesis,
            exp=exp,
            combined_result=combined_result,
            promotion_policy=promotion_policy,
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response)
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
