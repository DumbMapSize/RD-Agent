import pandas as pd

from rdagent.components.runner import CachedRunner
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import ModelEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.model_training import (
    build_model_run_env,
    inject_model_training_adapter,
    normalize_model_type,
)


class QlibModelRunner(CachedRunner[QlibModelExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - Pytorch `model.py`
    - results in `mlflow`

    https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py
    - pt_model_uri:  hard-code `model.py:Net` in the config
    - let LLM modify model.py
    """

    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        exist_sota_factor_exp = False
        if exp.based_experiments:
            SOTA_factor = None
            # Filter and retain only QlibFactorExperiment instances
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 0:
                logger.info(f"SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            if SOTA_factor is not None and not SOTA_factor.empty:
                exist_sota_factor_exp = True
                combined_factors = SOTA_factor
                combined_factors = combined_factors.sort_index()
                combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
                new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
                combined_factors.columns = new_columns
                num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len(combined_factors.columns))

                target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

                # Save the combined factors to the workspace
                combined_factors.to_parquet(target_path, engine="pyarrow")

        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})
        inject_model_training_adapter(exp.experiment_workspace)

        task = exp.sub_tasks[0]
        model_type = normalize_model_type(task.model_type)
        env_to_use = build_model_run_env(
            task.training_hyperparameters,
            model_type,
            num_features=num_features if exist_sota_factor_exp else None,
        )

        logger.info(f"start to run {exp.sub_tasks[0].name} model")
        if model_type == "TimeSeries":
            if exist_sota_factor_exp:
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_sota_factors_model.yaml", run_env=env_to_use
                )
            else:
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline_factors_model.yaml", run_env=env_to_use
                )
        else:
            if exist_sota_factor_exp:
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_sota_factors_model.yaml", run_env=env_to_use
                )
            else:
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline_factors_model.yaml", run_env=env_to_use
                )

        exp.result = result
        exp.stdout = stdout

        if result is None:
            logger.error(f"Failed to run {exp.sub_tasks[0].name}, because {stdout}")
            raise ModelEmptyError(f"Failed to run {exp.sub_tasks[0].name} model, because {stdout}")

        return exp
