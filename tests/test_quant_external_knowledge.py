import json

from rdagent.app.qlib_rd_loop.quant import _load_external_knowledge_file
from rdagent.components.proposal import _compose_rag_with_external_knowledge
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


def test_load_external_knowledge_file_accepts_json_mapping(tmp_path):
    knowledge_file = tmp_path / "knowledge.json"
    knowledge_file.write_text(
        json.dumps({"general": "general", "factor": "factor", "model": "model"}),
        encoding="utf-8",
    )

    context = _load_external_knowledge_file(str(knowledge_file))

    assert context == {"general": "general", "factor": "factor", "model": "model"}


def test_quant_hypothesis_preserves_external_knowledge_refs():
    gen = QlibQuantHypothesisGen(object())

    hypothesis = gen.convert_response(json.dumps({
        "action": "factor",
        "hypothesis": "Use the cited paper seed.",
        "reason": "The seed is relevant to the current factor search.",
        "concise_reason": "seed",
        "concise_observation": "obs",
        "concise_justification": "just",
        "concise_knowledge": "knowledge",
        "external_knowledge_refs": ["factor_seed:paperseed", " "],
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
