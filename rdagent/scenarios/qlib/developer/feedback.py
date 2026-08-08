import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
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
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor")
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
        combined_result = process_results(
            exp.result,
            SOTA_experiment.result if SOTA_hypothesis else None,
        ) if exp.result is not None else "execution failed"
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            combined_result=combined_result,
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
