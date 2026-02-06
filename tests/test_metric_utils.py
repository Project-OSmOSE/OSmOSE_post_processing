from contextlib import nullcontext
from typing import ContextManager

import pytest
from _pytest.monkeypatch import MonkeyPatch
from numpy import array, ndarray
from pandas import DataFrame

from post_processing.utils.metrics_utils import detection_perf


@pytest.mark.parametrize(
    ("filter_annotator", "filter_annotation", "ref", "expected"),
    [
        pytest.param(
            ["ann1", "ann4"],
            None,
            ("ann1", "lbl1"),
            nullcontext(),
            id="no_timestamps_provided",
        ),
        pytest.param(
            ["ann1"],
            None,
            ("ann1", "lbl1"),
            pytest.raises(ValueError, match="Two annotators needed"),
            id="one_annotator_provided",
        ),
        pytest.param(
            ["ann1", "ann6"],
            ["lbl1"],
            ("ann1", "lbl6"),
            pytest.raises(ValueError, match="No detection found for ann1/lbl6"),
            id="empty_ref_df",
        ),
    ],
)
def test_detection_perf(
    sample_df: DataFrame,
    filter_annotator: list[str, str],
    filter_annotation: list[str, str],
    ref: tuple[str, str],
    expected: ContextManager[Exception],
) -> None:
    filtered_df = sample_df
    if filter_annotator:
        filtered_df = filtered_df[filtered_df["annotator"].isin(filter_annotator)]
    if filter_annotation:
        filtered_df = filtered_df[filtered_df["annotation"].isin(filter_annotation)]

    with expected:
        detection_perf(df=filtered_df, ref=ref)


def test_detection_perf_error_in_metric_computation(
    sample_df: DataFrame,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that ValueError is raised when there are errors in metric computation."""

    def mock_map_datetimes_to_vector(df: DataFrame, timestamps: list[int]) -> ndarray:
        return array(len(df) * [3], dtype=int)

    monkeypatch.setattr(
        "post_processing.utils.metrics_utils._map_datetimes_to_vector",
        mock_map_datetimes_to_vector,
    )

    filtered_df = sample_df[
        (sample_df["annotator"].isin(["ann1", "ann2"]))
        & (sample_df["annotation"] == "lbl1")
    ]
    with pytest.raises(ValueError, match=r"[0-9]+ errors in metric computation."):
        detection_perf(
            df=filtered_df,
            ref=("ann1", "lbl1"),
        )


def test_detection_perf_impossible_metric_computation(
    sample_df: DataFrame,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that ValueError is raised when there are errors in metric computation."""

    def mock_map_datetimes_to_vector(df: DataFrame, timestamps: list[int]) -> ndarray:
        return array(len(df) * [0], dtype=int)

    monkeypatch.setattr(
        "post_processing.utils.metrics_utils._map_datetimes_to_vector",
        mock_map_datetimes_to_vector,
    )

    filtered_df = sample_df[
        (sample_df["annotator"].isin(["ann1", "ann2"]))
        & (sample_df["annotation"] == "lbl1")
    ]
    with pytest.raises(ValueError, match=r"Precision/Recall computation impossible."):
        detection_perf(
            df=filtered_df,
            ref=("ann1", "lbl1"),
        )
