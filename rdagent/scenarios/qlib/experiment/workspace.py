import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from mlflow.tracking import MlflowClient

from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QTDockerEnv

_TRAINING_METRIC_VALUE_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CHECKPOINT_METRIC_PATTERN = r"(?:loss|ic|rank_ic|icir|topk_precision@\d+)"
_TRAINING_SCORE_PATTERN = rf"(?:{_TRAINING_METRIC_VALUE_PATTERN}|unavailable)"
_TRAINING_LOG_SUMMARY_PATTERNS = (
    re.compile(
        rf"^RD-Agent training context \(authoritative\): optimizer=(?:adam|adamw|sgd); "
        rf"loss=(?:mse|mae|huber|pairwise|listnet|tail_listnet|ordinal); batch_mode=(?:sample|date); "
        rf"checkpoint={_CHECKPOINT_METRIC_PATTERN}; direction=(?:minimize|maximize); "
        rf"scheduler=(?:none|plateau); scheduler_monitor=(?:none|valid_loss); epochs=\d+; "
        rf"early_stop_patience=\d+$"
    ),
    re.compile(
        rf"^Epoch\d+: train_loss={_TRAINING_METRIC_VALUE_PATTERN}; "
        rf"valid_loss={_TRAINING_METRIC_VALUE_PATTERN}; lr={_TRAINING_METRIC_VALUE_PATTERN}$"
    ),
    re.compile(
        rf"^Epoch\d+: train_loss={_TRAINING_METRIC_VALUE_PATTERN}; "
        rf"valid_loss={_TRAINING_METRIC_VALUE_PATTERN}; "
        rf"train_(?P<metric>{_CHECKPOINT_METRIC_PATTERN})={_TRAINING_SCORE_PATTERN}; "
        rf"valid_(?P=metric)={_TRAINING_SCORE_PATTERN}; lr={_TRAINING_METRIC_VALUE_PATTERN}$"
    ),
    re.compile(rf"^early stop: checkpoint={_CHECKPOINT_METRIC_PATTERN}; patience=\d+; epoch=\d+$"),
    re.compile(
        rf"^best checkpoint: metric={_CHECKPOINT_METRIC_PATTERN}; direction=(?:minimize|maximize); "
        rf"value={_TRAINING_METRIC_VALUE_PATTERN}; epoch=\d+$"
    ),
    # Keep QLib's original loss-only output readable for runs that use the base fit loop.
    re.compile(
        rf"^Epoch\d+: train {_TRAINING_METRIC_VALUE_PATTERN}, valid {_TRAINING_METRIC_VALUE_PATTERN}$"
    ),
    re.compile(rf"^best score: {_TRAINING_METRIC_VALUE_PATTERN} @ \d+ epoch$"),
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
        if any(pattern.fullmatch(candidate) for pattern in _TRAINING_LOG_SUMMARY_PATTERNS):
            matches.append(candidate)
    return "\n".join(matches)


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(
        self, qlib_config_name: str = "conf.yaml", run_env: dict | None = None, *args, **kwargs
    ) -> tuple[pd.Series | None, str]:
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            logger.error(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "Unknown environment type"
        qtde.prepare()

        metrics_path = self.workspace_path / "qlib_res.csv"
        report_path = self.workspace_path / "ret.pkl"
        metrics_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        # QLib records _QLIB_* environment variables as parameters on the actual run.
        execution_id = uuid4().hex
        result = qtde.run(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env={**(run_env or {}), "_QLIB_RDAGENT_EXECUTION_ID": execution_id},
        )
        logger.log_object(result.stdout, tag="Qlib_execute_log")
        if result.exit_code != 0:
            return None, f"qrun exited with code {result.exit_code}:\n{result.stdout}"

        store = self.workspace_path.resolve() / "mlruns"
        try:
            client = MlflowClient(tracking_uri=store.as_uri())
            runs = client.search_runs(
                [exp.experiment_id for exp in client.search_experiments()],
                filter_string=f"params._QLIB_RDAGENT_EXECUTION_ID = '{execution_id}'",
                max_results=2,
            )
            if len(runs) != 1 or runs[0].info.status != "FINISHED":
                raise ValueError("Expected one completed QLib recorder for this invocation")
            run = runs[0]
            metrics = pd.Series(run.data.metrics)
            if metrics.empty:
                raise ValueError(f"Recorder {run.info.run_id} contains no metrics")
            # Read the mounted workspace path, including when qrun ran inside Docker.
            artifacts = store / run.info.experiment_id / run.info.run_id / "artifacts"
            report = pd.read_pickle(artifacts / "portfolio_analysis/report_normal_1day.pkl")
            report.to_pickle(report_path)
            metrics.to_csv(metrics_path)
            metrics = pd.read_csv(metrics_path, index_col=0).iloc[:, 0]
        except Exception as exc:
            report_path.unlink(missing_ok=True)
            metrics_path.unlink(missing_ok=True)
            return None, f"QLib result extraction failed: {exc}\n{result.stdout}"

        logger.log_object(report, tag="Quantitative Backtesting Chart")
        return metrics, _extract_training_log_summary(result.stdout)
