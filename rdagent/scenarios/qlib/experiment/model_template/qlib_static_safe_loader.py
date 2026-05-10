import pandas as pd

from qlib.data.dataset.loader import DataLoader, StaticDataLoader
from qlib.utils import init_instance_by_config


class StaticSafeNestedDataLoader(DataLoader):
    """Nested loader that does not pass market aliases to static parquet data."""

    def __init__(self, dataloader_l: list[dict], join: str = "left") -> None:
        super().__init__()
        self.data_loader_l = [
            dl if isinstance(dl, DataLoader) else init_instance_by_config(dl)
            for dl in dataloader_l
        ]
        self.join = join

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        df_full = None
        for dl in self.data_loader_l:
            load_instruments = None if isinstance(dl, StaticDataLoader) else instruments
            df_current = dl.load(load_instruments, start_time, end_time)
            if df_full is None:
                df_full = df_current
            else:
                current_columns = df_current.columns.tolist()
                full_columns = df_full.columns.tolist()
                columns_to_drop = [col for col in current_columns if col in full_columns]
                df_full.drop(columns=columns_to_drop, inplace=True)
                df_full = pd.merge(df_full, df_current, left_index=True, right_index=True, how=self.join)
        return df_full.sort_index(axis=1)
