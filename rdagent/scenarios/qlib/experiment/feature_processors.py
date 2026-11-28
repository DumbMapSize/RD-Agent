import numpy as np
import pandas as pd
from qlib.data import D
from qlib.data.dataset.processor import Processor, get_group_columns


class Float64Features(Processor):
    """Keep feature preprocessing precision independent of appended factor dtypes."""

    def __call__(self, df):
        columns = get_group_columns(df, "feature")
        df[columns] = df[columns].astype("float64", copy=False)
        return df


def _winsorize(values: np.ndarray, mad_clip: float) -> np.ndarray:
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    if mad <= 1e-12:
        return values
    limit = mad_clip * mad
    return np.clip(values, median - limit, median + limit)


def _winsorized_zscore(values: np.ndarray, mad_clip: float, z_clip: float) -> np.ndarray:
    winsorized = _winsorize(values, mad_clip)
    # Center relative to an observed value first so constant inputs stay exactly zero.
    centered = winsorized - winsorized[0]
    centered -= centered.mean()
    scale = np.sqrt(np.mean(centered**2))
    if scale == 0:
        return np.zeros_like(values)
    return np.clip(centered / scale, -z_clip, z_clip)


class CSRiskNeutralizeLabel(Processor):
    """Residualize and normalize learning labels within each trading date."""

    def __init__(
        self,
        fields_group: str = "label",
        risk_features: tuple[str, ...] | list[str] = ("STD60", "STD20", "BETA60"),
        liquidity_expr: str | None = "Log(Mean($close * $volume + 1e-6, 20) + 1e-6)",
        risk_clip: float = 3.0,
        ols_label_clip: float = 5.0,
        clip_std: float = 3.0,
        min_cross_section: int = 15,
    ) -> None:
        super().__init__()
        self.fields_group = fields_group
        self.risk_features = tuple(risk_features)
        self.liquidity_expr = liquidity_expr
        self.risk_clip = risk_clip
        self.ols_label_clip = ols_label_clip
        self.clip_std = clip_std
        self.min_cross_section = min_cross_section

    def is_for_infer(self) -> bool:
        return False

    def readonly(self) -> bool:
        return True

    def _risk_values(self, df: pd.DataFrame) -> np.ndarray:
        columns = [("feature", name) for name in self.risk_features if ("feature", name) in df.columns]
        risk_values = df[columns].to_numpy(dtype=float, na_value=np.nan)
        if self.liquidity_expr is None:
            return risk_values

        dates = df.index.get_level_values("datetime")
        instruments = df.index.get_level_values("instrument").unique().tolist()
        liquidity = D.features(instruments, [self.liquidity_expr], start_time=dates.min(), end_time=dates.max())
        if liquidity.empty:
            return np.column_stack((risk_values, np.full(len(df), np.nan)))
        if liquidity.index.names != df.index.names:
            liquidity = liquidity.reorder_levels(df.index.names)
        aligned = liquidity[self.liquidity_expr].reindex(df.index).to_numpy(dtype=float, na_value=np.nan)
        return np.column_stack((risk_values, aligned))

    def _normalize(self, y: np.ndarray, risks: np.ndarray) -> np.ndarray:
        if len(y) < self.min_cross_section or risks.shape[1] == 0:
            return _winsorized_zscore(y, self.clip_std, self.clip_std)

        design = np.zeros((len(y), risks.shape[1] + 1))
        design[:, 0] = 1.0
        for column, values in enumerate(risks.T, start=1):
            finite = np.isfinite(values)
            if finite.any():
                design[finite, column] = _winsorized_zscore(values[finite], 3.0, self.risk_clip)

        beta = np.linalg.lstsq(design, _winsorize(y, self.ols_label_clip), rcond=None)[0]
        residual = y - design[:, 1:] @ beta[1:]
        # Avoid amplifying floating-point noise when the exposures explain the entire cross-section.
        tolerance = np.finfo(float).eps * len(y) * np.max(np.abs(y))
        if np.ptp(residual) <= tolerance:
            return np.zeros_like(y)
        return _winsorized_zscore(residual, self.clip_std, self.clip_std)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        columns = get_group_columns(df, self.fields_group)
        labels = df[columns].to_numpy(dtype=float, na_value=np.nan)
        risk_values = self._risk_values(df)
        normalized = np.full_like(labels, np.nan)
        for rows in df.groupby(level="datetime", sort=False).indices.values():
            for column in range(labels.shape[1]):
                finite_rows = rows[np.isfinite(labels[rows, column])]
                if len(finite_rows) >= 2:
                    normalized[finite_rows, column] = self._normalize(labels[finite_rows, column], risk_values[finite_rows])

        # Replace only the label columns and let the unchanged feature blocks be shared.
        result = df.copy(deep=False)
        result[columns] = normalized
        return result
