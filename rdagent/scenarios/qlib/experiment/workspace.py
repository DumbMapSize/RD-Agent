import re
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QTDockerEnv

_TRAINING_METRIC_VALUE_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TRAINING_LOG_SUMMARY_RE = re.compile(
    rf"^(?:Epoch\d+: train {_TRAINING_METRIC_VALUE_PATTERN}, valid {_TRAINING_METRIC_VALUE_PATTERN}"
    rf"|best score: {_TRAINING_METRIC_VALUE_PATTERN} @ \d+ epoch)$"
)
_QLIB_GENERAL_PTNN_LOG_PREFIX_RE = re.compile(
    r"^\[\d+:[^\]]+\]\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\) "
    r"INFO - qlib\.GeneralPTNN - \[(?:pytorch_general_nn|rdagent_general_ptnn)\.py:\d+\] - (?P<message>.*)$"
)


def _extract_training_log_summary(execute_qlib_log: str) -> str:
    matches = []
    for raw_line in execute_qlib_log.splitlines():
        line = raw_line.strip()
        qlib_match = _QLIB_GENERAL_PTNN_LOG_PREFIX_RE.fullmatch(line)
        candidate = qlib_match.group("message").strip() if qlib_match else line
        if _TRAINING_LOG_SUMMARY_RE.fullmatch(candidate):
            matches.append(candidate)
    return "\n".join(matches)


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs) -> str:
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            logger.error(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "Unknown environment type"
        qtde.prepare()

        # Run the Qlib backtest
        execute_qlib_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        execute_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )

        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            return None, execute_qlib_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # Here, we ensure that the qlib experiment has run successfully before extracting information from execute_qlib_log using regex; otherwise, we keep the original experiment stdout.
            execute_qlib_log = _extract_training_log_summary(execute_qlib_log)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log
