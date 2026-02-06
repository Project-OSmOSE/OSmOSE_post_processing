"""Plot functions used for DataAplose objects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series, Timedelta, Timestamp, date_range

from post_processing.utils.filtering_utils import intersection_or_union

if TYPE_CHECKING:
    from post_processing.dataclass.recording_period import RecordingPeriod


def detection_perf(
    df: DataFrame,
    timestamps: list[Timestamp] | None = None,
    *,
    ref: tuple[str, str],
) -> tuple[float, float, float]:
    """Compute the performance metrics for detection.

    Performances are computed with a reference annotator/label pair
    in comparison to a second annotator/label pair.

    Parameters
    ----------
    df: DataFrame
        APLOSE formatted detection/annotation DataFrame
    ref: tuple[str, str]
        Tuple of annotator/detector pairs.
    timestamps: list[Timestamp]
        A list of Timestamps to base the computation on.

    Returns
    -------
    precision: float
    recall: float
    f_score: float

    """
    annotators = df["annotator"].unique().tolist()
    if len(annotators) != 2:  # noqa: PLR2004
        msg = f"Two annotators needed, DataFrame contains {len(annotators)} annotators"
        raise ValueError(msg)

    selected_annotator1, selected_label1 = ref

    if not timestamps:
        timestamps = [
            ts.timestamp()
            for ts in date_range(
                start=df["start_datetime"].min(),
                end=df["start_datetime"].max(),
                freq=str(df["end_time"].max()) + "s",
            )
        ]
    else:
        timestamps = [ts.timestamp() for ts in timestamps]

    # df1 - REFERENCE
    selected_annotations1 = df[
        (df["annotator"] == selected_annotator1) & (df["annotation"] == selected_label1)
    ]
    if selected_annotations1.empty:
        msg = f"No detection found for {selected_annotator1}/{selected_label1}"
        raise ValueError(msg)

    vec1 = _map_datetimes_to_vector(df=selected_annotations1, timestamps=timestamps)

    # df2
    labels = df["annotation"].unique().tolist()
    selected_annotator2 = next(ant for ant in annotators if ant != selected_annotator1)
    selected_label2 = (
        next(lbl for lbl in labels if lbl != selected_label1)
        if len(labels) == 2  # noqa: PLR2004
        else selected_label1
    )
    selected_annotations2 = df[
        (df["annotator"] == selected_annotator2) & (df["annotation"] == selected_label2)
    ]

    vec2 = _map_datetimes_to_vector(selected_annotations2, timestamps)

    # metrics computation
    confusion_matrix = {
        "true_pos": int(np.sum((vec1 == 1) & (vec2 == 1))),
        "false_pos": int(np.sum((vec1 == 0) & (vec2 == 1))),
        "false_neg": int(np.sum((vec1 == 1) & (vec2 == 0))),
        "true_neg": int(np.sum((vec1 == 0) & (vec2 == 0))),
        "error": int(np.sum((vec1 != 0) & (vec1 != 1) | (vec2 != 0) & (vec2 != 1))),
    }

    if confusion_matrix["error"] != 0:
        msg = f"{confusion_matrix['error']} errors in metric computation."
        raise ValueError(msg)

    if (
        confusion_matrix["true_pos"] + confusion_matrix["false_pos"] == 0
        or confusion_matrix["false_neg"] + confusion_matrix["true_pos"] == 0
    ):
        msg = "Precision/Recall computation impossible."
        raise ValueError(msg)

    _log_detection_results(
        selection1=(selected_annotator1, selected_label1),
        selection2=(selected_annotator2, selected_label2),
        matrix=confusion_matrix,
        df=df,
    )

    return (
        _get_precision(confusion_matrix),
        _get_recall(confusion_matrix),
        _get_f_score(confusion_matrix),
    )


def _get_precision(confusion_matrix: dict) -> float:
    """Compute precision."""
    tp = confusion_matrix["true_pos"]
    fp = confusion_matrix["false_pos"]
    return tp / (tp + fp)


def _get_recall(confusion_matrix: dict) -> float:
    """Compute recall."""
    tp = confusion_matrix["true_pos"]
    fn = confusion_matrix["false_neg"]
    return tp / (tp + fn)


def _get_f_score(confusion_matrix: dict) -> float:
    """Compute F-score."""
    precision = _get_precision(confusion_matrix)
    recall = _get_recall(confusion_matrix)
    return 2 * (precision * recall) / (precision + recall)


def _log_detection_results(
    selection1: tuple[str, str],
    selection2: tuple[str, str],
    matrix: dict,
    df: DataFrame,
) -> None:
    """Log detection performance results."""
    annotator1, label1 = selection1
    annotator2, label2 = selection2
    precision = _get_precision(matrix)
    recall = _get_recall(matrix)
    f_score = _get_f_score(matrix)

    msg_result = (
        f"{' Detection results ':#^50}\n"
        f"{'Config 1:':<10}{f'{annotator1}/{label1}':>40}\n"
        f"{'Config 2:':<10}{f'{annotator2}/{label2}':>40}\n\n"
        f"{'True positive:':<25}{matrix['true_pos']:>25}\n"
        f"{'True negative:':<25}{matrix['true_neg']:>25}\n"
        f"{'False positive:':<25}{matrix['false_pos']:>25}\n"
        f"{'False negative:':<25}{matrix['false_neg']:>25}\n\n"
        f"{'Precision:':<25}{precision:>25.2f}\n"
        f"{'Recall:':<25}{recall:>25.2f}\n"
        f"{'F-score:':<25}{f_score:>25.2f}\n\n"
        f"{'Union:':<25}{len(intersection_or_union(df, 'union')):>25.0f}\n"
        f"{'Intersection:':<25}{len(intersection_or_union(df, 'intersection')):>25.0f}\n"
    )
    logging.info(msg_result)


def _map_datetimes_to_vector(df: DataFrame, timestamps: list[int]) -> ndarray:
    """Map datetime ranges to a binary vector indicating overlap with timestamp bins.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame.
    timestamps : list of int
        List of UNIX timestamps representing bin start times.

    Returns
    -------
    ndarray
        Binary array (0/1) where 1 indicates overlap with a bin.

    """
    starts = df["start_datetime"].astype("int64") // 10**9
    ends = df["end_datetime"].astype("int64") // 10**9
    timebin = int(df["end_time"].iloc[0])  # duration in seconds

    timestamps = np.array(timestamps)
    ts_start = timestamps
    ts_end = timestamps + timebin

    vec = np.zeros(len(timestamps), dtype=int)

    for start, end in zip(starts, ends, strict=False):
        overlap = (ts_start < end) & (ts_end > start)
        vec[overlap] = 1

    return vec


def normalize_counts_by_effort(
    counts: DataFrame,
    effort: RecordingPeriod,
    time_bin: Timedelta,
) -> DataFrame:
    """Normalize detection counts given the observation effort."""
    timebin_origin = effort.timebin_origin
    effort_series = effort.counts
    effort_intervals = effort_series.index
    effort_series.index = [interval.left for interval in effort_series.index]
    for col in counts.columns:
        effort_ratio = effort_series * (timebin_origin / time_bin)
        effort_ratio = Series(
            np.where((effort_ratio > 0) & (effort_ratio < 1), 1.0, effort_ratio),
            index=effort_series.index,
            name=effort_series.name,
        )
        counts[f"{col}"] = (counts[col] / effort_ratio.reindex(counts[col].index)).clip(
            upper=1
        )
        effort_series.index = effort_intervals
    return counts
