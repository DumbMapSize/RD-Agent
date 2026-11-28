import pickle
from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.qlib.experiment import feature_processors
from rdagent.scenarios.qlib.experiment.feature_processors import CSRiskNeutralizeLabel


LABEL = ("label", "LABEL0")
RISK_COLUMNS = [("feature", name) for name in ("STD60", "STD20", "BETA60")]


def _frame(date, labels, risks=None, instrument_start=0):
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp(date)], [f"SH600{i:03d}" for i in range(instrument_start, instrument_start + len(labels))]],
        names=["datetime", "instrument"],
    )
    data = {LABEL: labels}
    if risks is not None:
        data.update(zip(RISK_COLUMNS, risks.T, strict=True))
    return pd.DataFrame(data, index=index)


def test_winsorize_scales_mad_to_estimate_gaussian_standard_deviation():
    values = np.array([-100.0, 1.0, 2.0, 3.0, 100.0])

    actual = feature_processors._winsorize(values, mad_clip=3.0)

    np.testing.assert_allclose(actual, [2.0 - 4.4478, 1.0, 2.0, 3.0, 2.0 + 4.4478])


@pytest.mark.parametrize("spacing", [0.0, 1e-13])
def test_winsorize_preserves_cross_section_with_zero_or_near_zero_mad(spacing):
    values = np.array([-2.0, 0.0, spacing, 2 * spacing, 3.0])

    np.testing.assert_array_equal(feature_processors._winsorize(values, mad_clip=3.0), values)


def test_risk_neutralizer_round_trips_and_preserves_features_and_row_order():
    exposures = np.array(list(product([-1, 1], repeat=4)), dtype=np.float32)
    risks, residual = exposures[:, :3], exposures[:, 3]
    labels = 0.5 + risks @ [0.5, -1.0, 1.0] + residual
    frame = _frame("2025-01-02", labels, risks).sample(frac=1, random_state=42)
    original = frame.copy()
    processor = pickle.loads(pickle.dumps(CSRiskNeutralizeLabel(liquidity_expr=None)))

    actual = processor(frame)

    assert processor.is_for_infer() is False
    pd.testing.assert_index_equal(actual.index, original.index)
    pd.testing.assert_index_equal(actual.columns, original.columns)
    pd.testing.assert_frame_equal(actual["feature"], original["feature"], check_exact=True)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    np.testing.assert_allclose(actual["feature"].to_numpy().T @ actual[LABEL].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(actual[LABEL].reindex(original.sort_index().index), residual, atol=1e-12)


def test_risk_neutralizer_limits_outlier_influence_on_other_labels():
    risk = np.tile([-1.0, -1.0, 1.0, 1.0], 128)
    residual = np.tile([-1.0, 1.0, -1.0, 1.0], 128)
    risks = np.column_stack([risk, risk, np.full(len(risk), np.nan)])
    frame = _frame("2025-01-02", 2 * risk + residual, risks)
    contaminated = frame.copy()
    contaminated.loc[frame.index[0], LABEL] = 1e8
    processor = CSRiskNeutralizeLabel(liquidity_expr=None)

    clean = processor(frame)[LABEL]
    actual = processor(contaminated)[LABEL]

    np.testing.assert_allclose(actual.iloc[1:], clean.iloc[1:], atol=0.08)
    assert 2.5 < actual.iloc[0] <= 3.0


def test_risk_neutralizer_clips_and_handles_sparse_or_constant_labels():
    with_outlier = _frame("2025-01-02", [*([-1.0] * 26), *([0.0] * 45), *([1.0] * 28), 1e8])
    small = _frame("2025-01-03", [-1.0, 1.0, np.inf], np.ones((3, 3)))
    constant = _frame("2025-01-06", [0.1] * 3)
    sparse = _frame("2025-01-07", [np.nan, 1.0])
    frame = pd.concat([with_outlier, small, constant, sparse])

    labels = CSRiskNeutralizeLabel(liquidity_expr=None)(frame)[LABEL]

    assert labels.dropna().abs().max() == 3.0
    np.testing.assert_array_equal(labels.loc["2025-01-03"], [-1.0, 1.0, np.nan])
    np.testing.assert_array_equal(labels.loc["2025-01-06"], [0.0, 0.0, 0.0])
    assert labels.loc["2025-01-07"].isna().all()


def test_risk_neutralizer_uses_missing_exposure_for_empty_liquidity(monkeypatch):
    exposures = np.array(list(product([-1.0, 1.0], repeat=4)))
    risks, residual = exposures[:, :3], exposures[:, 3]
    frame = _frame("2025-01-02", 2 * risks[:, 0] + residual, risks)
    empty_liquidity = pd.DataFrame(index=frame.index[:0].swaplevel())
    monkeypatch.setattr(feature_processors, "D", SimpleNamespace(features=lambda *args, **kwargs: empty_liquidity))
    processor = CSRiskNeutralizeLabel()

    actual = processor._risk_values(frame)

    np.testing.assert_array_equal(actual, np.column_stack((risks, np.full(len(frame), np.nan))))
    np.testing.assert_allclose(processor(frame)[LABEL], residual, atol=1e-12)


@pytest.mark.parametrize("swap_input_levels", [False, True])
def test_risk_neutralizer_aligns_liquidity_and_isolates_trading_dates(monkeypatch, swap_input_levels):
    liquidity, residual = np.array(list(product([-1.0, 1.0], repeat=2)) * 4).T
    early = _frame("2025-01-02", 2 * liquidity + residual)
    later = _frame("2025-01-06", -3 * liquidity - residual, instrument_start=8)
    frame = pd.concat([early, later]).sample(frac=1, random_state=3)
    if swap_input_levels:
        frame = frame.swaplevel()

    processor = CSRiskNeutralizeLabel()
    expr = "Log(Mean($close * $volume + 1e-6, 20) + 1e-6)"
    source = pd.DataFrame({expr: np.tile(liquidity, 2)}, index=early.index.append(later.index))
    source = source.sample(frac=1, random_state=4)
    if not swap_input_levels:
        source = source.swaplevel()
    calls = []

    def features(instruments, fields, *, start_time, end_time):
        calls.append((set(instruments), fields, start_time, end_time))
        return source

    monkeypatch.setattr(feature_processors, "D", SimpleNamespace(features=features))
    transformed = processor(frame)
    expected = pd.Series(np.concatenate([residual, -residual]), index=early.index.append(later.index), name=LABEL)
    if swap_input_levels:
        expected = expected.swaplevel()
    pd.testing.assert_series_equal(transformed[LABEL], expected.reindex(frame.index), atol=1e-12)
    assert calls == [
        (set(frame.index.get_level_values("instrument")), [expr], pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-06"))
    ]

    # Changing later labels and liquidity must leave the earlier cross-section identical.
    later_rows = frame.index.get_level_values("datetime") == pd.Timestamp("2025-01-06")
    source.loc[source.index.get_level_values("datetime") == pd.Timestamp("2025-01-06"), expr] = np.nan
    frame.loc[later_rows, LABEL] = np.arange(len(later)) ** 2
    changed = processor(frame)[LABEL]
    pd.testing.assert_series_equal(changed.loc[~later_rows], transformed[LABEL].loc[~later_rows], check_exact=True)
    assert not np.allclose(changed.loc[later_rows], transformed[LABEL].loc[later_rows])
