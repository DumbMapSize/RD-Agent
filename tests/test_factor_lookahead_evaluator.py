import json
from types import SimpleNamespace
from types import SimpleNamespace

import pandas as pd
import pytest

from rdagent.components.coder.CoSTEER import evaluators as costeer_evaluators_module
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.factor_coder import eva_utils as eva_utils_module
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.eva_utils import FactorLookaheadAuditResult, FactorLookaheadEvaluator
from rdagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
from rdagent.components.coder.factor_coder.factor import FactorTask


class FakeScenario:
    def get_scenario_all_desc(self, *args, **kwargs):
        return "Daily China A-share factor research."


class FakeBackend:
    response: str = "{}"
    last_system_prompt: str | None = None
    last_user_prompt: str | None = None

    def build_messages_and_create_chat_completion(self, user_prompt, system_prompt, **kwargs):
        type(self).last_user_prompt = user_prompt
        type(self).last_system_prompt = system_prompt
        return type(self).response


class FakeImplementation:
    def __init__(self, code: str, df: pd.DataFrame):
        self.all_codes = code
        self.df = df
        self.execute_count = 0

    def execute(self):
        self.execute_count += 1
        return "Execution succeeded without error.", self.df


class StaticEvaluator:
    def __init__(self, feedback, decision):
        self.feedback = feedback
        self.decision = decision

    def evaluate(self, **kwargs):
        return self.feedback, self.decision


class StaticLookaheadEvaluator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def evaluate(self, **kwargs):
        self.calls += 1
        return self.result


class QueuedLookaheadEvaluator:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def evaluate(self, **kwargs):
        self.calls += 1
        decision = self.results.pop(0)
        if decision == "pass":
            return FactorLookaheadAuditResult(
                decision="pass",
                confidence="high",
                evidence="Queued audit pass.",
                risky_code_patterns=[],
                suggested_fix="",
            )
        return FactorLookaheadAuditResult(
            decision=decision,
            confidence="high",
            evidence=f"Queued audit {decision}.",
            risky_code_patterns=[f"queued_{decision}"],
            suggested_fix="Revise the factor implementation.",
        )


# These labels are queued audit outcomes for plumbing tests only.
# Semantic lookahead recognition is covered by the real Codex integration test.
LOOKAHEAD_PLUMBING_CASES = [
    pytest.param(
        "past_20d_momentum",
        "factor = close / close.groupby(instrument).shift(20) - 1",
        "pass",
        id="past-momentum",
    ),
    pytest.param("past_5d_reversal", "factor = -close.groupby(instrument).pct_change(5)", "pass", id="past-reversal"),
    pytest.param(
        "rolling_volatility",
        "factor = ret.groupby(instrument).rolling(20).std().reset_index(level=0, drop=True)",
        "pass",
        id="rolling-vol",
    ),
    pytest.param(
        "rolling_volume_zscore",
        "factor = (volume - volume.groupby(instrument).rolling(20).mean().reset_index(level=0, drop=True))",
        "pass",
        id="rolling-volume-zscore",
    ),
    pytest.param(
        "amihud_illiq",
        "factor = (ret.abs() / amount).groupby(instrument).rolling(20).mean()",
        "pass",
        id="amihud",
    ),
    pytest.param("lagged_turnover", "factor = turnover.groupby(instrument).shift(1)", "pass", id="lagged-turnover"),
    pytest.param("past_vwap_gap", "factor = close / vwap.groupby(instrument).shift(1) - 1", "pass", id="past-vwap-gap"),
    pytest.param(
        "same_day_cross_section_rank",
        "factor = raw.groupby(datetime).rank(pct=True)",
        "pass",
        id="same-day-rank",
    ),
    pytest.param(
        "same_day_cross_section_zscore",
        "factor = raw.groupby(datetime).transform(lambda x: (x - x.mean()) / x.std())",
        "pass",
        id="same-day-zscore",
    ),
    pytest.param(
        "same_day_demean",
        "factor = raw - raw.groupby(datetime).transform('mean')",
        "pass",
        id="same-day-demean",
    ),
    pytest.param(
        "historical_ewm",
        "factor = ret.groupby(instrument).ewm(span=20, adjust=False).mean().reset_index(level=0, drop=True)",
        "pass",
        id="historical-ewm",
    ),
    pytest.param(
        "historical_expanding",
        "factor = ret.groupby(instrument).expanding(min_periods=60).mean().reset_index(level=0, drop=True)",
        "pass",
        id="historical-expanding",
    ),
    pytest.param("future_shift_close", "factor = close.groupby(instrument).shift(-1)", "fail", id="future-shift-close"),
    pytest.param(
        "future_return",
        "future_return = close.shift(-5) / close - 1; factor = future_return",
        "fail",
        id="future-return",
    ),
    pytest.param("qlib_ref_future", "factor = Ref($close, -2) / $close - 1", "fail", id="qlib-ref-future"),
    pytest.param(
        "negative_pct_change",
        "factor = close.groupby(instrument).pct_change(-5)",
        "fail",
        id="negative-pct-change",
    ),
    pytest.param(
        "centered_rolling",
        "factor = close.groupby(instrument).rolling(5, center=True).mean()",
        "fail",
        id="centered-rolling",
    ),
    pytest.param("backward_fill_future", "factor = raw.groupby(instrument).bfill()", "fail", id="bfill"),
    pytest.param(
        "forward_asof_join",
        "factor = pd.merge_asof(price, events, on='datetime', direction='forward')",
        "fail",
        id="forward-asof",
    ),
    pytest.param("global_scaler", "factor = global_scaler.fit_transform(raw[['close']])", "fail", id="global-scaler"),
    pytest.param(
        "sklearn_fit_transform",
        "factor = scaler.fit_transform(raw[['close', 'volume']])",
        "fail",
        id="fit-transform",
    ),
    pytest.param(
        "all_sample_quantile",
        "factor = raw / raw.quantile(0.9)  # all_sample",
        "fail",
        id="all-sample-quantile",
    ),
    pytest.param("future_label_feature", "factor = feature_df['future_label']", "fail", id="future-label"),
    pytest.param("tomorrow_close", "factor = df['tomorrow_close'] / df['close'] - 1", "fail", id="tomorrow-close"),
    pytest.param("next_close", "factor = df['next_close'].rank()", "fail", id="next-close"),
    pytest.param("rank_future", "factor = rank_future(close.shift(-1))", "fail", id="future-rank"),
    pytest.param(
        "external_csv",
        "fund = pd.read_csv('fundamental.csv'); factor = fund['roe']",
        "uncertain",
        id="external-csv",
    ),
    pytest.param("opaque_loader", "factor = external_loader('vendor_alpha_42')", "uncertain", id="opaque-loader"),
    pytest.param(
        "custom_calendar_align",
        "factor = custom_calendar_align(announcements, prices)",
        "uncertain",
        id="custom-calendar-align",
    ),
    pytest.param(
        "announcement_date_missing",
        "factor = fundamental_panel['earnings_surprise']  # no announcement_date handling",
        "uncertain",
        id="announcement-missing",
    ),
    pytest.param("vendor_factor", "factor = vendor_factor('sentiment_alpha')", "uncertain", id="vendor-factor"),
]


def _task():
    return FactorTask(
        factor_name="future_close",
        factor_description="Potentially leaked close based factor.",
        factor_formulation="future_close = close.shift(-1)",
        variables={"close": "daily close price"},
    )


def _task_for_case(name: str):
    return FactorTask(
        factor_name=name,
        factor_description=f"Corpus test factor {name}.",
        factor_formulation=name,
        variables={"close": "daily close price", "volume": "daily volume"},
    )


def _df(rows: int = 4):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=rows, freq="D"), ["SH600000"]],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"factor": range(rows)}, index=idx)


def _mock_backend(monkeypatch, response: dict | str):
    FakeBackend.response = json.dumps(response) if isinstance(response, dict) else response
    FakeBackend.last_system_prompt = None
    FakeBackend.last_user_prompt = None
    monkeypatch.setattr(eva_utils_module, "APIBackend", lambda *args, **kwargs: FakeBackend())


def _make_factor_evaluator(monkeypatch, lookahead_decisions=None):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_audit_enabled", True, raising=False)
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_uncertain_policy", "fail", raising=False)
    evaluator = FactorEvaluatorForCoder(scen=FakeScenario())
    evaluator.value_evaluator = StaticEvaluator("Value evaluation passed.", True)
    evaluator.lookahead_evaluator = QueuedLookaheadEvaluator(lookahead_decisions or [])
    return evaluator


def test_factor_lookahead_audit_is_enabled_by_default():
    assert FACTOR_COSTEER_SETTINGS.lookahead_audit_enabled is True


def test_factor_lookahead_evaluator_fails_future_shift(monkeypatch):
    _mock_backend(
        monkeypatch,
        {
            "lookahead_decision": "fail",
            "confidence": "high",
            "evidence": "The code uses shift(-1), which reads future rows.",
            "risky_code_patterns": ["shift(-1)"],
            "suggested_fix": "Use only current and historical observations within each instrument.",
        },
    )

    result = FactorLookaheadEvaluator(scen=FakeScenario()).evaluate(
        target_task=_task(),
        code="df['factor'] = df.groupby('instrument')['$close'].shift(-1)",
        execution_feedback="Execution succeeded.",
        value_feedback="Value checks passed.",
        gen_df=_df(),
    )

    assert result.decision == "fail"
    assert result.final_decision is False
    assert "shift(-1)" in result.to_code_feedback()
    assert "Use only current and historical" in result.to_code_feedback()


def test_factor_lookahead_evaluator_passes_historical_window(monkeypatch):
    _mock_backend(
        monkeypatch,
        {
            "lookahead_decision": "pass",
            "confidence": "medium",
            "evidence": "The code uses rolling windows over historical observations.",
            "risky_code_patterns": [],
            "suggested_fix": "",
        },
    )

    result = FactorLookaheadEvaluator(scen=FakeScenario()).evaluate(
        target_task=_task(),
        code="df['factor'] = df.groupby('instrument')['$close'].rolling(20).mean().reset_index(level=0, drop=True)",
        execution_feedback="Execution succeeded.",
        value_feedback="Value checks passed.",
        gen_df=_df(),
    )

    assert result.decision == "pass"
    assert result.final_decision is True


def test_factor_lookahead_evaluator_treats_invalid_json_as_failed_uncertain(monkeypatch):
    _mock_backend(monkeypatch, "not json")
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_uncertain_policy", "fail", raising=False)

    result = FactorLookaheadEvaluator(scen=FakeScenario()).evaluate(
        target_task=_task(),
        code="df = complex_external_merge(df)",
        execution_feedback="Execution succeeded.",
        value_feedback="Value checks passed.",
        gen_df=_df(),
    )

    assert result.decision == "uncertain"
    assert result.final_decision is False
    assert "Failed to parse lookahead audit response" in result.evidence


def test_factor_lookahead_prompt_contains_time_semantics_without_full_dataframe(monkeypatch):
    _mock_backend(
        monkeypatch,
        {
            "lookahead_decision": "pass",
            "confidence": "high",
            "evidence": "No future access found.",
            "risky_code_patterns": [],
            "suggested_fix": "",
        },
    )

    FactorLookaheadEvaluator(scen=FakeScenario()).evaluate(
        target_task=_task(),
        code="df['factor'] = df.groupby('instrument')['$close'].shift(1)",
        execution_feedback="Execution succeeded.",
        value_feedback="Value checks passed.",
        gen_df=_df(rows=20),
    )

    assert "factor at date t may only use data known at date t or earlier" in FakeBackend.last_system_prompt
    assert "future returns in label configuration are not factor leakage" in FakeBackend.last_system_prompt
    assert "training labels for strictly historical rows" in FakeBackend.last_system_prompt
    assert "directly or indirectly depends on future returns or labels for date t or later" in FakeBackend.last_system_prompt
    assert "same-date cross-sectional rank" in FakeBackend.last_system_prompt
    assert "DataFrame info" in FakeBackend.last_user_prompt
    assert "2024-01-20" not in FakeBackend.last_user_prompt


def test_factor_evaluator_lookahead_failure_overrides_value_pass(monkeypatch):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_audit_enabled", True, raising=False)
    impl = FakeImplementation(
        code="df['factor'] = df.groupby('instrument')['$close'].shift(-1)",
        df=_df(),
    )
    evaluator = FactorEvaluatorForCoder(scen=FakeScenario())
    evaluator.value_evaluator = StaticEvaluator("Value evaluation passed.", True)
    lookahead = StaticLookaheadEvaluator(
        FactorLookaheadAuditResult(
            decision="fail",
            confidence="high",
            evidence="The code uses shift(-1), which reads future rows.",
            risky_code_patterns=["shift(-1)"],
            suggested_fix="Use lagged data only.",
        )
    )
    evaluator.lookahead_evaluator = lookahead

    feedback = evaluator.evaluate(target_task=_task(), implementation=impl)

    assert feedback.final_decision is False
    assert "Lookahead audit failed" in feedback.code_feedback
    assert "shift(-1)" in feedback.code_feedback
    assert "Lookahead audit" in feedback.value_feedback
    assert lookahead.calls == 1
    assert impl.execute_count == 1


def test_factor_evaluator_reuses_success_knowledge_without_lookahead(monkeypatch):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_audit_enabled", True, raising=False)
    task = _task()
    cached_feedback = SimpleNamespace(final_decision=True)
    queried_knowledge = SimpleNamespace(
        success_task_to_knowledge_dict={
            task.get_task_information(): SimpleNamespace(feedback=cached_feedback),
        },
        failed_task_info_set=set(),
    )
    impl = FakeImplementation(code="df['factor'] = df['$close']", df=_df())
    evaluator = FactorEvaluatorForCoder(scen=FakeScenario())
    evaluator.lookahead_evaluator = StaticLookaheadEvaluator(
        FactorLookaheadAuditResult(
            decision="fail",
            confidence="high",
            evidence="Should not be called for cached success knowledge.",
            risky_code_patterns=[],
            suggested_fix="",
        )
    )

    feedback = evaluator.evaluate(
        target_task=task,
        implementation=impl,
        queried_knowledge=queried_knowledge,
    )

    assert feedback is cached_feedback
    assert evaluator.lookahead_evaluator.calls == 0
    assert impl.execute_count == 0


def test_factor_evaluator_skips_lookahead_when_value_check_rejects(monkeypatch):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_audit_enabled", True, raising=False)
    impl = FakeImplementation(code="df['factor'] = df['$close']", df=_df())
    evaluator = FactorEvaluatorForCoder(scen=FakeScenario())
    evaluator.value_evaluator = StaticEvaluator("Value evaluation failed.", False)
    evaluator.code_evaluator = StaticEvaluator("Code evaluation completed.", False)
    evaluator.lookahead_evaluator = StaticLookaheadEvaluator(
        FactorLookaheadAuditResult(
            decision="pass",
            confidence="high",
            evidence="Should not be called for an already rejected implementation.",
            risky_code_patterns=[],
            suggested_fix="",
        )
    )

    feedback = evaluator.evaluate(target_task=_task(), implementation=impl)

    assert feedback.final_decision is False
    assert evaluator.lookahead_evaluator.calls == 0
    assert impl.execute_count == 1


def test_factor_evaluator_reuses_success_knowledge_without_lookahead(monkeypatch):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_audit_enabled", True, raising=False)
    task = _task()
    cached_feedback = SimpleNamespace(final_decision=True)
    queried_knowledge = SimpleNamespace(
        success_task_to_knowledge_dict={
            task.get_task_information(): SimpleNamespace(feedback=cached_feedback),
        },
        failed_task_info_set=set(),
    )
    impl = FakeImplementation(code="df['factor'] = df['$close']", df=_df())
    evaluator = FactorEvaluatorForCoder(scen=FakeScenario())
    evaluator.lookahead_evaluator = StaticLookaheadEvaluator(
        FactorLookaheadAuditResult(
            decision="fail",
            confidence="high",
            evidence="Should not be called for cached success knowledge.",
            risky_code_patterns=[],
            suggested_fix="",
        )
    )

    feedback = evaluator.evaluate(
        target_task=task,
        implementation=impl,
        queried_knowledge=queried_knowledge,
    )

    assert feedback is cached_feedback
    assert evaluator.lookahead_evaluator.calls == 0
    assert impl.execute_count == 0


def test_factor_evaluator_skips_lookahead_when_value_check_rejects(monkeypatch):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_audit_enabled", True, raising=False)
    impl = FakeImplementation(code="df['factor'] = df['$close']", df=_df())
    evaluator = FactorEvaluatorForCoder(scen=FakeScenario())
    evaluator.value_evaluator = StaticEvaluator("Value evaluation failed.", False)
    evaluator.code_evaluator = StaticEvaluator("Code evaluation completed.", False)
    evaluator.lookahead_evaluator = StaticLookaheadEvaluator(
        FactorLookaheadAuditResult(
            decision="pass",
            confidence="high",
            evidence="Should not be called for an already rejected implementation.",
            risky_code_patterns=[],
            suggested_fix="",
        )
    )

    feedback = evaluator.evaluate(target_task=_task(), implementation=impl)

    assert feedback.final_decision is False
    assert evaluator.lookahead_evaluator.calls == 0
    assert impl.execute_count == 1


def test_factor_evaluator_recreates_lookahead_evaluator_for_resumed_session(monkeypatch):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_audit_enabled", True, raising=False)
    impl = FakeImplementation(code="df['factor'] = df['$close']", df=_df())
    evaluator = FactorEvaluatorForCoder(scen=FakeScenario())
    evaluator.value_evaluator = StaticEvaluator("Value evaluation passed.", True)
    delattr(evaluator, "lookahead_evaluator")
    restored_lookahead = StaticLookaheadEvaluator(
        FactorLookaheadAuditResult(
            decision="pass",
            confidence="high",
            evidence="No lookahead bias found.",
            risky_code_patterns=[],
            suggested_fix="",
        )
    )
    monkeypatch.setattr(
        "rdagent.components.coder.factor_coder.evaluators.FactorLookaheadEvaluator",
        lambda scen: restored_lookahead,
    )

    feedback = evaluator.evaluate(target_task=_task(), implementation=impl)

    assert feedback.final_decision is True
    assert evaluator.lookahead_evaluator is restored_lookahead
    assert restored_lookahead.calls == 1


def test_factor_evaluator_batch_applies_supplied_lookahead_results_to_many_factor_implementations(monkeypatch):
    cases = LOOKAHEAD_PLUMBING_CASES[:6] + LOOKAHEAD_PLUMBING_CASES[12:18] + LOOKAHEAD_PLUMBING_CASES[-3:]
    evaluator = _make_factor_evaluator(monkeypatch, [case.values[2] for case in cases])

    feedback_by_name = {}
    for case in cases:
        name, code, expected_decision = case.values
        feedback_by_name[name] = evaluator.evaluate(
            target_task=_task_for_case(name),
            implementation=FakeImplementation(code=code, df=_df()),
        )
        assert feedback_by_name[name].final_decision is (expected_decision == "pass")

    failed_feedback = feedback_by_name["future_shift_close"]
    assert "Lookahead audit failed" in failed_feedback.code_feedback
    assert "Queued audit fail" in failed_feedback.code_feedback
    assert "Lookahead audit" in str(failed_feedback)
    assert evaluator.lookahead_evaluator.calls == len(cases)


def test_costeer_multi_evaluator_merges_lookahead_results_for_multi_factor_tasks(monkeypatch):
    selected_cases = [
        LOOKAHEAD_PLUMBING_CASES[0],
        LOOKAHEAD_PLUMBING_CASES[12],
        LOOKAHEAD_PLUMBING_CASES[-1],
    ]
    evaluator = _make_factor_evaluator(monkeypatch, [case.values[2] for case in selected_cases])
    monkeypatch.setattr(
        costeer_evaluators_module,
        "multiprocessing_wrapper",
        lambda jobs, n: [func(*args) for func, args in jobs],
    )
    evo = EvolvingItem(sub_tasks=[_task_for_case(case.values[0]) for case in selected_cases])
    evo.sub_workspace_list = [
        FakeImplementation(code=case.values[1], df=_df())
        for case in selected_cases
    ]
    evo.sub_gt_implementations = None

    feedback = CoSTEERMultiEvaluator(evaluator, scen=FakeScenario()).evaluate(evo)

    decisions = [item.final_decision for item in feedback.feedback_list]
    assert decisions == [True, False, False]
    assert evo.sub_tasks[0].factor_implementation is True
    assert evo.sub_tasks[1].factor_implementation is False
    assert evo.sub_tasks[2].factor_implementation is False
    assert "Lookahead audit failed" in feedback.feedback_list[1].code_feedback


def test_factor_evaluator_skips_lookahead_when_disabled(monkeypatch):
    monkeypatch.setattr(FACTOR_COSTEER_SETTINGS, "lookahead_audit_enabled", False, raising=False)
    impl = FakeImplementation(code="df['factor'] = df['$close']", df=_df())
    evaluator = FactorEvaluatorForCoder(scen=FakeScenario())
    evaluator.value_evaluator = StaticEvaluator("Value evaluation passed.", True)
    evaluator.lookahead_evaluator = StaticLookaheadEvaluator(
        FactorLookaheadAuditResult(
            decision="fail",
            confidence="high",
            evidence="Should not be used.",
            risky_code_patterns=[],
            suggested_fix="",
        )
    )

    feedback = evaluator.evaluate(target_task=_task(), implementation=impl)

    assert feedback.final_decision is True
    assert evaluator.lookahead_evaluator.calls == 0
    assert impl.execute_count == 1
