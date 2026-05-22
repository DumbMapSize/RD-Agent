import json

from rdagent.app.qlib_rd_loop.quant import _load_external_knowledge_file
from rdagent.components.proposal import _compose_rag_with_external_knowledge


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
    assert "prioritize observed experimental feedback" in rag


def test_load_external_knowledge_file_accepts_json_mapping(tmp_path):
    knowledge_file = tmp_path / "knowledge.json"
    knowledge_file.write_text(
        json.dumps({"general": "general", "factor": "factor", "model": "model"}),
        encoding="utf-8",
    )

    context = _load_external_knowledge_file(str(knowledge_file))

    assert context == {"general": "general", "factor": "factor", "model": "model"}
