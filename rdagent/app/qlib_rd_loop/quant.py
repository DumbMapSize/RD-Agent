"""
Quant (Factor & Model) workflow with session control
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import fire

from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.proposal.quant_proposal import QuantTrace


def _clean_external_knowledge_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_external_knowledge_file(knowledge_file: str) -> dict[str, str]:
    path = Path(knowledge_file)
    content = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return {"general": content.strip(), "factor": "", "model": ""}

    if raw is None:
        return {"general": "", "factor": "", "model": ""}
    if isinstance(raw, str):
        return {"general": raw.strip(), "factor": "", "model": ""}
    if not isinstance(raw, dict):
        return {"general": json.dumps(raw, ensure_ascii=False, indent=2), "factor": "", "model": ""}

    return {
        key: _clean_external_knowledge_value(raw.get(key, ""))
        for key in ("general", "factor", "model")
    }


class QuantRDLoop(RDLoop):
    skip_loop_error = (
        FactorEmptyError,
        ModelEmptyError,
    )

    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")

        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.quant_hypothesis_gen)(scen)
        logger.log_object(self.hypothesis_gen, tag="quant hypothesis generator")

        self.factor_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.factor_hypothesis2experiment
        )()
        logger.log_object(self.factor_hypothesis2experiment, tag="factor hypothesis2experiment")
        self.model_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.model_hypothesis2experiment
        )()
        logger.log_object(self.model_hypothesis2experiment, tag="model hypothesis2experiment")

        self.factor_coder: Developer = import_class(PROP_SETTING.factor_coder)(scen)
        logger.log_object(self.factor_coder, tag="factor coder")
        self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
        logger.log_object(self.model_coder, tag="model coder")

        self.factor_runner: Developer = import_class(PROP_SETTING.factor_runner)(scen)
        logger.log_object(self.factor_runner, tag="factor runner")
        self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
        logger.log_object(self.model_runner, tag="model runner")

        self.factor_summarizer: Experiment2Feedback = import_class(PROP_SETTING.factor_summarizer)(scen)
        logger.log_object(self.factor_summarizer, tag="factor summarizer")
        self.model_summarizer: Experiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)
        logger.log_object(self.model_summarizer, tag="model summarizer")

        self.plan: dict[str, Any] = {}
        self.trace = QuantTrace(scen=scen)
        super(RDLoop, self).__init__()

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                hypo = self._propose()
                assert hypo.action in ["factor", "model"]
                if hypo.action == "factor":
                    exp = self.factor_hypothesis2experiment.convert(hypo, self.trace)
                else:
                    exp = self.model_hypothesis2experiment.convert(hypo, self.trace)
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                return {"propose": hypo, "exp_gen": exp}
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        if prev_out["direct_exp_gen"]["propose"].action == "factor":
            exp = self.factor_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        elif prev_out["direct_exp_gen"]["propose"].action == "model":
            exp = self.model_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        if prev_out["direct_exp_gen"]["propose"].action == "factor":
            exp = self.factor_runner.develop(prev_out["coding"])
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
        elif prev_out["direct_exp_gen"]["propose"].action == "model":
            exp = self.model_runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
            logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["direct_exp_gen"]["exp_gen"], feedback))
        else:
            if prev_out["direct_exp_gen"]["propose"].action == "factor":
                feedback = self.factor_summarizer.generate_feedback(prev_out["running"], self.trace)
            elif prev_out["direct_exp_gen"]["propose"].action == "model":
                feedback = self.model_summarizer.generate_feedback(prev_out["running"], self.trace)
            logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["running"], feedback))


def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
    checkout_path: str | None = None,
    replace_timer: bool = True,
    knowledge_file: str | None = None,
):
    """
    Auto R&D Evolving loop for fintech factors.
    You can continue running session by
    .. code-block:: python
        dotenv run -- python rdagent/app/qlib_rd_loop/quant.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional paramter
    """
    if checkout_path is not None:
        checkout = Path(checkout_path)

    if path is None:
        quant_loop = QuantRDLoop(QUANT_PROP_SETTING)
    else:
        quant_loop = QuantRDLoop.load(path, checkout=checkout, replace_timer=replace_timer)

    if not hasattr(quant_loop, "plan") or quant_loop.plan is None:
        quant_loop.plan = {}
    if knowledge_file is not None:
        quant_loop.plan["external_knowledge"] = _load_external_knowledge_file(knowledge_file)
    else:
        quant_loop.plan.pop("external_knowledge", None)

    asyncio.run(quant_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
