from abc import abstractmethod
from typing import Tuple

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import (
    ExperimentPlan,
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


def _target_to_external_knowledge_key(targets: str) -> str | None:
    target = targets.strip().lower()
    if target in {"factor", "factors"}:
        return "factor"
    if target in {"model", "models", "model tuning"}:
        return "model"
    return None


def _normalise_external_knowledge(plan: ExperimentPlan | None) -> dict[str, str]:
    if not plan:
        return {}
    raw = plan.get("external_knowledge")
    if raw is None and plan.get("user_instruction"):
        raw = {"general": plan["user_instruction"]}
    if isinstance(raw, str):
        return {"general": raw}
    if not isinstance(raw, dict):
        return {}
    return {
        key: str(raw.get(key, "")).strip()
        for key in ("general", "factor", "model")
        if str(raw.get(key, "")).strip()
    }


def _select_external_knowledge(plan: ExperimentPlan | None, targets: str) -> list[str]:
    external = _normalise_external_knowledge(plan)
    action_key = _target_to_external_knowledge_key(targets)
    selected = []
    if external.get("general"):
        selected.append(external["general"])
    if action_key and external.get(action_key):
        selected.append(external[action_key])
    return selected


def _compose_rag_with_external_knowledge(base_rag: str | None, plan: ExperimentPlan | None, targets: str) -> str | None:
    selected = _select_external_knowledge(plan, targets)
    if not selected:
        return base_rag

    external_block = (
        "External research knowledge is provided as the current candidate under evaluation. "
        "For the current action, if an item marked with KNOWLEDGE_ID is present, "
        "design this hypothesis around that item and copy that exact ID into external_knowledge_ref. "
        "Prioritize observed experimental feedback when deciding how to adapt the candidate, "
        "but do not omit the ID when the candidate is used.\n\n"
        + "\n\n".join(selected)
    )
    if base_rag:
        return f"{base_rag}\n\n{external_block}"
    return external_block


def _get_sota_hypothesis_and_feedback(context: dict) -> str:
    return context.get("sota_hypothesis_and_feedback") or context.get("SOTA_hypothesis_and_feedback") or ""


class LLMHypothesisGen(HypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...

    def gen(
        self,
        trace: Trace,
        plan: ExperimentPlan | None = None,
    ) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)
        if _select_external_knowledge(plan, self.targets):
            context_dict["hypothesis_output_format"] = context_dict.get(
                "hypothesis_output_format_with_external_knowledge",
                context_dict["hypothesis_output_format"],
            )
        context_dict["RAG"] = _compose_rag_with_external_knowledge(context_dict.get("RAG"), plan, self.targets)

        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(
            targets=self.targets,
            scenario=(
                self.scen.get_scenario_all_desc(filtered_tag=self.targets)
                if self.targets in ["factor", "model"]
                else self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
            ),
            hypothesis_output_format=context_dict["hypothesis_output_format"],
            hypothesis_specification=context_dict["hypothesis_specification"],
        )
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(
            targets=self.targets,
            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
            last_hypothesis_and_feedback=(
                context_dict["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context_dict else ""
            ),
            sota_hypothesis_and_feedback=_get_sota_hypothesis_and_feedback(context_dict),
            RAG=context_dict["RAG"],
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, str]
        )

        hypothesis = self.convert_response(resp)

        return hypothesis


class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"


class ModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "model tuning"


class FactorAndModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "feature engineering and model building"


class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> Experiment: ...

    @wait_retry(retry_n=5)
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = T(".prompts:hypothesis2experiment.system_prompt").r(
            targets=self.targets,
            scenario=trace.scen.get_scenario_all_desc(filtered_tag=self.targets),
            experiment_output_format=context["experiment_output_format"],
        )
        user_prompt = T(".prompts:hypothesis2experiment.user_prompt").r(
            targets=self.targets,
            target_hypothesis=context["target_hypothesis"],
            hypothesis_and_feedback=(
                context["hypothesis_and_feedback"] if "hypothesis_and_feedback" in context else ""
            ),
            last_hypothesis_and_feedback=(
                context["last_hypothesis_and_feedback"] if "last_hypothesis_and_feedback" in context else ""
            ),
            sota_hypothesis_and_feedback=_get_sota_hypothesis_and_feedback(context),
            target_list=context["target_list"],
            RAG=context["RAG"],
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag, json_target_type=dict[str, dict[str, str | dict]]
        )

        return self.convert_response(resp, hypothesis, trace)


class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"


class ModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "model tuning"


class FactorAndModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "feature engineering and model building"
