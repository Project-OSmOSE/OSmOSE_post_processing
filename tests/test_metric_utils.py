from contextlib import nullcontext
from typing import ContextManager

import pytest
from _pytest.monkeypatch import MonkeyPatch
from numpy import array
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
    filtered_df = sample_df[sample_df["type"] == "WEAK"]
    if filter_annotator:
        filtered_df = filtered_df[filtered_df["annotator"].isin(filter_annotator)]
    if filter_annotation:
        filtered_df = filtered_df[filtered_df["annotation"].isin(filter_annotation)]

    with expected:
        detection_perf(df=filtered_df, ref=ref)


def test_detection_perf_confusion_matrix_errors(
    monkeypatch: MonkeyPatch,
    sample_df: DataFrame,
) -> None:
    def fake_get_count(*args, **kwargs) -> DataFrame:
        return DataFrame({
            "lbl2-ann1": array([1, 1, 1, 0, 666, 1]),
            "lbl2-ann2": array([1, 0, 2, 1, 0, 1234]),
        })

    monkeypatch.setattr("post_processing.utils.metrics_utils.get_count", fake_get_count)

    filtered_df = sample_df[
        (sample_df["annotation"] == "lbl2")
        & (sample_df["annotator"].isin(["ann1", "ann2"]))
        & (sample_df["type"] == "WEAK")
    ]

    with pytest.raises(ValueError, match="3 errors in metric computation"):
        detection_perf(
            filtered_df,
            ref=("ann1", "lbl2"),
        )


def test_detection_perf_confusion_matrix_no_data(
    monkeypatch: MonkeyPatch,
    sample_df: DataFrame,
) -> None:
    def fake_get_count(*args, **kwargs) -> DataFrame:
        return DataFrame({
            "lbl2-ann1": array([0] * 10),
            "lbl2-ann2": array([0] * 10),
        })

    monkeypatch.setattr("post_processing.utils.metrics_utils.get_count", fake_get_count)

    filtered_df = sample_df[
        (sample_df["annotation"] == "lbl2")
        & (sample_df["annotator"].isin(["ann1", "ann2"]))
        & (sample_df["type"] == "WEAK")
    ]

    with pytest.raises(ValueError, match="Precision/Recall computation impossible"):
        detection_perf(
            filtered_df,
            ref=("ann1", "lbl2"),
        )
