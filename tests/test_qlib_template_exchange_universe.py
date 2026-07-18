from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = (
    "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
    "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
    "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
    "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
    "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
)


def _exchange_kwargs(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "exchange_kwargs:")
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    block: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
            break
        block.append(line.strip())
    return block


@pytest.mark.parametrize("relative_path", TEMPLATES)
def test_qlib_template_keeps_research_pool_without_restricting_exchange(relative_path: str) -> None:
    text = (ROOT / relative_path).read_text()

    assert "instruments: *market" in text
    assert not any(line.startswith("codes:") for line in _exchange_kwargs(text))
