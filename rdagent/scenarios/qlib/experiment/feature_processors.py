from qlib.data.dataset.processor import Processor, get_group_columns


class Float64Features(Processor):
    """Keep feature preprocessing precision independent of appended factor dtypes."""

    def __call__(self, df):
        columns = get_group_columns(df, "feature")
        df[columns] = df[columns].astype("float64", copy=False)
        return df
