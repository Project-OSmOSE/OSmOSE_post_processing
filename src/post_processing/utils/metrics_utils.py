"""Plot functions used for DataAplose objects."""

from __future__ import annotations

import logging

import numpy as np
from pandas import DataFrame, DatetimeIndex, Timedelta

from post_processing.utils.core_utils import get_count
from post_processing.utils.filtering_utils import (
    get_annotators,
    get_labels,
    get_max_time,
    intersection_or_union,
)


def detection_perf(
    df: DataFrame,
    *,
    ref: tuple[str, str],
    time: DatetimeIndex | None = None,
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
    time: DatetimeIndex
        DatetimeIndex from a specified beginning to end

    Returns
    -------
    precision: float
    recall: float
    f_score: float

    """
    annotators = get_annotators(df)
    if len(annotators) != 2:  # noqa: PLR2004
        msg = f"Two annotators needed, DataFrame contains {len(annotators)} annotators"
        raise ValueError(msg)

    labels = get_labels(df)

    timebin = Timedelta(get_max_time(df), "s")
    df_count = get_count(df, timebin, time)

    # reference annotator and label
    annotator1, label1 = ref
    annotations1 = df[(df["annotator"] == annotator1) & (df["annotation"] == label1)]
    if annotations1.empty:
        msg = f"No detection found for {annotator1}/{label1}"
        raise ValueError(msg)
    vec1 = df_count[f"{label1}-{annotator1}"]

    # second annotator and label
    annotator2 = next(ant for ant in annotators if ant != annotator1)
    label2 = (
        next(lbl for lbl in labels if lbl != label1)
        if len(labels) == 2  # noqa: PLR2004
        else label1
    )
    vec2 = df_count[f"{label2}-{annotator2}"]

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
        selection1=(annotator1, label1),
        selection2=(annotator2, label2),
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
