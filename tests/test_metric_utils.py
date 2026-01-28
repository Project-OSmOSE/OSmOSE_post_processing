import pytest
from pandas import DataFrame

from post_processing.utils.metrics_utils import detection_perf


def test_detection_perf(sample_df: DataFrame) -> None:
    try:
        detection_perf(df=sample_df[sample_df["annotator"].isin(["ann1", "ann4"])], ref=("ann1", "lbl1"))
    except ValueError:
        pytest.fail("test_detection_perf raised ValueError unexpectedly.")


def test_detection_perf_one_annotator(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="Two annotators needed"):
        detection_perf(df=sample_df[sample_df["annotator"] == "ann1"], ref=("ann1", "lbl1"))
