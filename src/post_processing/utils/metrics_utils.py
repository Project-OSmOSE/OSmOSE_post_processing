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
    """Compute performances metrics for detection.

    Performances are computed with a reference annotator in
    comparison with a second annotator/detector.
    Precision and recall are computed in regard
    with a reference annotator/label pair.

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
    datetime_begin = df["start_datetime"].min()
    datetime_end = df["start_datetime"].max()
    df_freq = str(df["end_time"].max()) + "s"
    labels = df["annotation"].unique().tolist()
    annotators = df["annotator"].unique().tolist()

    num_annotators = 2
    if len(annotators) != num_annotators:
        msg = f"Two annotators needed, DataFrame contains {len(annotators)} annotators"
        raise ValueError(msg)

    if not timestamps:
        timestamps = [
            ts.timestamp()
            for ts in date_range(
                start=datetime_begin,
                end=datetime_end,
                freq=df_freq,
            )
        ]
    else:
        timestamps = [ts.timestamp() for ts in timestamps]

    # df1 - REFERENCE
    selected_annotator1 = ref[0]
    selected_label1 = ref[1]
    selected_annotations1 = df[
        (df["annotator"] == selected_annotator1) & (df["annotation"] == selected_label1)
    ]
    vec1 = _map_datetimes_to_vector(df=selected_annotations1, timestamps=timestamps)

    # df2
    selected_annotator2 = (
        next(ant for ant in annotators if ant != selected_annotator1)
        if len(annotators) == 2  # noqa: PLR2004
        else selected_annotator1
    )
    selected_label2 = (
        next(lbl for lbl in labels if lbl != selected_label1)
        if len(labels) == 2  # noqa: PLR2004
        else selected_label1
    )
    selected_annotations2 = df[
        (df["annotator"] == selected_annotator2) & (df["annotation"] == selected_label2)
    ]
    vec2 = _map_datetimes_to_vector(selected_annotations2, timestamps)

    # Metrics computation
    true_pos = int(np.sum((vec1 == 1) & (vec2 == 1)))
    false_pos = int(np.sum((vec1 == 0) & (vec2 == 1)))
    false_neg = int(np.sum((vec1 == 1) & (vec2 == 0)))
    true_neg = int(np.sum((vec1 == 0) & (vec2 == 0)))
    error = int(np.sum((vec1 != 0) & (vec1 != 1) | (vec2 != 0) & (vec2 != 1)))

    if error != 0:
        msg = f"Error : {error}"
        raise ValueError(msg)

    msg_result = f"{' Detection results ':#^50}\n"
    msg_result += f"{'Config 1:':<10}{f'{selected_annotator1}/{selected_label1}':>40}\n"
    msg_result += (
        f"{'Config 2:':<10}{f'{selected_annotator2}/{selected_label2}':>40}\n\n"
    )
    msg_result += f"{'True positive:':<25}{true_pos:>25}\n"
    msg_result += f"{'True negative:':<25}{true_neg:>25}\n"
    msg_result += f"{'False positive:':<25}{false_pos:>25}\n"
    msg_result += f"{'False negative:':<25}{false_neg:>25}\n\n"

    if true_pos + false_pos == 0 or false_neg + true_pos == 0:
        msg = "Precision/Recall computation impossible"
        raise ValueError(msg)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f_score = 2 * (precision * recall) / (precision + recall)
    msg_result += f"{'Precision:':<25}{precision:>25.2f}\n"
    msg_result += f"{'Recall:':<25}{recall:>25.2f}\n"
    msg_result += f"{'F-score:':<25}{f_score:>25.2f}\n\n"

    df_union = intersection_or_union(df, "union")
    df_intersection = intersection_or_union(df, "intersection")
    msg_result += f"{'Union:':<25}{len(df_union):>25.0f}\n"
    msg_result += f"{'Intersection:':<25}{len(df_intersection):>25.0f}\n"

    logging.info(msg_result)

    return precision, recall, f_score


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
