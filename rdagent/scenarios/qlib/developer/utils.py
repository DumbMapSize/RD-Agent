from copy import copy
from typing import List

import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


def process_factor_data(
    exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment,
    *,
    update_candidate_feedback: bool = False,
) -> pd.DataFrame:
    """
    Process and combine factor data from experiment implementations.

    Args:
        exp_or_list: The experiments containing factor data.
        update_candidate_feedback: Record full-sample failures on the current candidate only.

    Returns:
        pd.DataFrame: Combined successful factor data, preserving missing values.
    """
    if isinstance(exp_or_list, QlibFactorExperiment):
        exp_or_list = [exp_or_list]
    factor_dfs = []
    error_messages = []

    # Collect all exp's dataframes
    for exp in exp_or_list:
        if isinstance(exp, QlibFactorExperiment):
            if len(exp.sub_tasks) > 0:
                # if it has no sub_tasks, the experiment is results from template project.
                # otherwise, it is developed with designed task. So it should have feedback.
                assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
                if update_candidate_feedback:
                    # Coding knowledge may share these objects with the candidate.
                    exp.sub_tasks = list(exp.sub_tasks)
                    exp.prop_dev_feedback = copy(exp.prop_dev_feedback)
                    exp.prop_dev_feedback.feedback_list = list(exp.prop_dev_feedback.feedback_list)
                eligible = [
                    (index, implementation)
                    for index, (implementation, fb) in enumerate(zip(exp.sub_workspace_list, exp.prop_dev_feedback))
                    if implementation and fb
                ]
                # Iterate over sub-implementations and execute them to get each factor data
                message_and_df_list = multiprocessing_wrapper(
                    [(implementation.execute, ("All",)) for _, implementation in eligible],
                    n=RD_AGENT_SETTINGS.multi_proc_n,
                )
                for (index, _), (message, df) in zip(eligible, message_and_df_list, strict=True):
                    task = exp.sub_tasks[index]
                    # Check if factor generation was successful
                    if df is not None and "datetime" in df.index.names:
                        time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                        if pd.Timedelta(minutes=1) not in time_diff:
                            factor_dfs.append(df)
                            logger.info(f"Factor data from {task.factor_name} is successfully generated.")
                            continue
                        reason = f"Output contains one-minute intervals. {message}"
                    else:
                        reason = message if df is None else f"Output lacks a datetime index. {message}"
                    failure = f"Full-sample execution failed for {task.factor_name}: {reason}"
                    error_messages.append(failure)
                    logger.warning(failure)
                    if update_candidate_feedback:
                        task = copy(task)
                        task.factor_implementation = False
                        exp.sub_tasks[index] = task
                        feedback = copy(exp.prop_dev_feedback[index])
                        feedback.final_decision = False
                        feedback.execution = failure
                        exp.prop_dev_feedback.feedback_list[index] = feedback

    # Combine all successful factor data
    if factor_dfs:
        return pd.concat(factor_dfs, axis=1)
    else:
        error_message = "\n".join(error_messages)
        raise FactorEmptyError(
            f"No valid factor data found to merge (in process_factor_data) because of {error_message}."
        )
