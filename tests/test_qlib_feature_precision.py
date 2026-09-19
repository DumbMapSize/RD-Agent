import pickle

import numpy as np
import pandas as pd

from rdagent.scenarios.qlib.experiment.feature_processors import Float64Features


def test_float64_features_preserves_values_missingness_labels_and_order():
    columns = pd.MultiIndex.from_tuples([
        ("feature", "Z"), ("label", "LABEL0"), ("feature", "A"), ("feature", "M")
    ])
    frame = pd.DataFrame({
        columns[0]: np.array([1.0000001, np.nan, 2.25], dtype="float32"),
        columns[1]: np.array([0.1, np.nan, -0.2], dtype="float32"),
        columns[2]: np.array([1.123456789012345, np.nan, 1e40], dtype="float64"),
        columns[3]: np.array([1, 2, 3], dtype="int64"),
    }, index=pd.date_range("2025-01-02", periods=3))
    expected = frame.astype({column: "float64" for column in columns if column[0] == "feature"})
    processor = pickle.loads(pickle.dumps(Float64Features()))
    processor.fit(frame)
    actual = processor(frame.copy())
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(processor(actual.copy()), expected, check_exact=True)
    assert actual["label"].dtypes.eq("float32").all()
