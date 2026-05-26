import json
import inspect

from rdagent.app.qlib_rd_loop import quant as quant_module
from rdagent.app.qlib_rd_loop.quant import _load_external_knowledge_file
from rdagent.components.proposal import (
    LLMHypothesisGen,
    _compose_rag_with_external_knowledge,
    _get_sota_hypothesis_and_feedback,
)
from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen


def test_compose_rag_uses_only_current_action_knowledge():
    plan = {
        "external_knowledge": {
            "general": "General market regime note.",
            "factor": "Factor-only paper seed.",
            "model": "Model-only architecture note.",
        }
    }

    rag = _compose_rag_with_external_knowledge("Base guidance.", plan, "factor")

    assert "Base guidance." in rag
    assert "General market regime note." in rag
    assert "Factor-only paper seed." in rag
    assert "Model-only architecture note." not in rag
    assert "Prioritize observed experimental feedback" in rag


def test_sota_feedback_context_accepts_legacy_uppercase_key():
    assert _get_sota_hypothesis_and_feedback({"SOTA_hypothesis_and_feedback": "legacy sota"}) == "legacy sota"
    assert _get_sota_hypothesis_and_feedback({"sota_hypothesis_and_feedback": "canonical sota"}) == "canonical sota"
    assert _get_sota_hypothesis_and_feedback({"SOTA_hypothesis_and_feedback": None}) == ""


def test_load_external_knowledge_file_accepts_json_mapping(tmp_path):
    knowledge_file = tmp_path / "knowledge.json"
    knowledge_file.write_text(
        json.dumps({"general": "general", "factor": "factor", "model": "model"}),
        encoding="utf-8",
    )

    context = _load_external_knowledge_file(str(knowledge_file))

    assert context == {"general": "general", "factor": "factor", "model": "model"}


def test_load_external_knowledge_file_treats_json_null_as_empty(tmp_path):
    knowledge_file = tmp_path / "knowledge.json"
    knowledge_file.write_text(
        json.dumps({"general": None, "factor": "factor", "model": None}),
        encoding="utf-8",
    )

    context = _load_external_knowledge_file(str(knowledge_file))

    assert context == {"general": "", "factor": "factor", "model": ""}


def test_quant_main_clears_stale_external_knowledge_on_resume_without_file(monkeypatch, tmp_path):
    class DummyLoop:
        def __init__(self):
            self.plan = {"external_knowledge": {"factor": "stale seed"}}

        async def run(self, step_n=None, loop_n=None, all_duration=None):
            self.run_args = {"step_n": step_n, "loop_n": loop_n, "all_duration": all_duration}

    loop = DummyLoop()
    monkeypatch.setattr(quant_module.QuantRDLoop, "load", lambda *args, **kwargs: loop)

    quant_module.main(
        path=str(tmp_path / "session"),
        checkout_path=str(tmp_path / "checkout"),
        step_n=0,
        loop_n=0,
        all_duration=None,
        knowledge_file=None,
    )

    assert "external_knowledge" not in loop.plan
    assert loop.run_args == {"step_n": 0, "loop_n": 0, "all_duration": None}


def test_llm_hypothesis_gen_keeps_string_response_schema():
    source = inspect.getsource(LLMHypothesisGen.gen)

    assert "json_target_type=dict[str, str]" in source
    assert "dict[str, object]" not in source


def test_quant_hypothesis_preserves_external_knowledge_ref_string():
    gen = QlibQuantHypothesisGen(object())

    hypothesis = gen.convert_response(json.dumps({
        "action": "factor",
        "hypothesis": "Use the cited paper seed.",
        "reason": "The seed is relevant to the current factor search.",
        "concise_reason": "seed",
        "concise_observation": "obs",
        "concise_justification": "just",
        "concise_knowledge": "knowledge",
        "external_knowledge_ref": "factor_seed:paperseed",
    }))

    assert hypothesis.external_knowledge_refs == ["factor_seed:paperseed"]


def test_quant_hypothesis_defaults_missing_external_refs_to_empty_list():
    gen = QlibQuantHypothesisGen(object())

    hypothesis = gen.convert_response(json.dumps({
        "action": "model",
        "hypothesis": "No external knowledge was used.",
        "reason": "Trace feedback was enough.",
        "concise_reason": "trace",
        "concise_observation": "obs",
        "concise_justification": "just",
        "concise_knowledge": "knowledge",
    }))

    assert hypothesis.external_knowledge_refs == []


def test_quant_hypothesis_ignores_non_string_external_knowledge_ref():
    gen = QlibQuantHypothesisGen(object())

    for raw_ref in (None, ["factor_seed:paperseed"]):
        hypothesis = gen.convert_response(json.dumps({
            "action": "factor",
            "hypothesis": "Use trace feedback.",
            "reason": "No valid external ref was returned.",
            "concise_reason": "trace",
            "concise_observation": "obs",
            "concise_justification": "just",
            "concise_knowledge": "knowledge",
            "external_knowledge_ref": raw_ref,
        }))

        assert hypothesis.external_knowledge_refs == []
