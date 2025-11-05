"""Functions used to filter APLOSE-formatted DataFrame."""

from __future__ import annotations

import bisect
import csv
from typing import TYPE_CHECKING

from pandas import (
    DataFrame,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    read_csv,
    to_datetime,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dateutil.tz import tzoffset

    from post_processing.dataclass.detection_filter import DetectionFilter


def find_delimiter(file: Path) -> str:
    """Find the proper delimiter for a csv file."""
    with file.open(newline="") as csv_file:
        try:
            temp_lines = csv_file.readline() + "\n" + csv_file.readline()
            dialect = csv.Sniffer().sniff(temp_lines, delimiters=",;")
            delimiter = dialect.delimiter
        except csv.Error as err:
            msg = "Could not determine delimiter"
            raise ValueError(msg) from err
        return delimiter


def filter_by_time(
    df: DataFrame,
    begin: Timestamp | None,
    end: Timestamp | None,
) -> DataFrame:
    """Filter a DataFrame based on begin and/or end timestamps."""
    if begin is not None:
        df = df[df["start_datetime"] >= begin]
        if df.empty:
            msg = f"No detection found after '{begin}'."
            raise ValueError(msg)

    if end is not None:
        df = df[df["end_datetime"] <= end]
        if df.empty:
            msg = f"No detection found after '{end}'."
            raise ValueError(msg)

    return df


def filter_by_annotator(
    df: DataFrame,
    annotator: str | list[str],
) -> DataFrame:
    """Filter a DataFrame based on annotator selection."""
    list_annotators = get_annotators(df)

    if not annotator:
        return df

    if isinstance(annotator, str):
        ensure_in_list(annotator, list_annotators, "annotator")
        return df[df["annotator"] == annotator]

    invalid = [a for a in annotator if a not in list_annotators]
    ensure_no_invalid(invalid, "annotators")

    return df.loc[df["annotator"].isin(annotator)]


def filter_by_label(
    df: DataFrame,
    label: str | list[str],
) -> DataFrame:
    """Filter a DataFrame based on label selection."""
    list_labels = get_labels(df)

    if not label:
        return df

    if isinstance(label, str):
        ensure_in_list(label, list_labels, "label")
        return df[df["annotation"] == label]

    invalid = [lbl for lbl in label if lbl not in list_labels]
    ensure_no_invalid(invalid, "labels")

    return df.loc[df["annotation"].isin(label)]


def filter_by_freq(
    df: DataFrame,
    f_min: int | None,
    f_max: int | None,
) -> DataFrame:
    """Filter a DataFrame based on frequency selection."""
    if f_min is not None:
        df = df[df["start_frequency"] >= f_min]
        if df.empty:
            msg = f"No detection found above {f_min}Hz."
            raise ValueError(msg)

    if f_max is not None:
        df = df[df["end_frequency"] <= f_max]
        if df.empty:
            msg = f"No detection found below {f_max}Hz."
            raise ValueError(msg)
    return df


def filter_by_score(df: DataFrame, score: float) -> DataFrame:
    """Filter a DataFrame based on minimum score."""
    if not score:
        return df

    if "score" not in df.columns:
        msg = "'score' column not present if DataFrame."
        raise ValueError(msg)

    df = df[df["score"] >= score]
    if df.empty:
        msg = f"No detection found with score above {score}."
        raise ValueError(msg)
    return df


def read_dataframe(file: Path, nrows: int | None = None) -> DataFrame:
    """Read csv file."""
    delimiter = find_delimiter(file)
    return (
        read_csv(file,
                 sep=delimiter,
                 parse_dates=["start_datetime", "end_datetime"],
                 nrows=nrows,
                 )
        .drop_duplicates()
        .dropna(subset=["annotation"])
        .sort_values(by=["start_datetime", "end_datetime"])
        .reset_index(drop=True)
    )


def get_annotators(df: DataFrame) -> list[str]:
    """Return list of annotators."""
    return sorted(set(df["annotator"]))


def get_labels(df: DataFrame) -> list[str]:
    """Return list of labels."""
    return sorted(set(df["annotation"]))


def get_max_freq(df: DataFrame) -> float:
    """Return the maximum frequency of DataFrame."""
    return df["end_frequency"].max()


def get_max_time(df: DataFrame) -> float:
    """Return the maximum time of DataFrame."""
    return df["end_time"].max()


def get_dataset(df: DataFrame) -> list[str]:
    """Return list of datasets."""
    datasets = sorted(set(df["dataset"]))
    return datasets if len(datasets) > 1 else datasets[0]


def get_timezone(df: DataFrame) -> tzoffset | list[tzoffset]:
    """Return timezone(s) from DataFrame."""
    timezones = {ts.tz for ts in df["start_datetime"] if ts.tz is not None}
    if len(timezones) == 1:
        return next(iter(timezones))
    return sorted(timezones, key=lambda tz: tz.utcoffset(None))


def reshape_timebin(
    df: DataFrame,
    timestamp_wav: list[Timestamp],
    timebin_new: Timedelta | None,
    timestamp: list[Timestamp] | None = None,
) -> DataFrame:
    """Reshape an APLOSE result DataFrame according to a new time bin.

    Parameters
    ----------
    df: DataFrame
        An APLOSE result DataFrame.
    timebin_new: Timedelta
        The size of the new time bin.
    timestamp: list[Timestamp]
        A list of Timestamp objects corresponding to the shape
        in which the data should be reshaped.
    timestamp_wav: list[Timestamp]
        A list of the start datetime of each wavfile. Length should be the same as df

    Returns
    -------
    df_new_timebin: DataFrame
        The reshaped DataFrame

    """
    if df.empty:
        msg = "DataFrame is empty"
        raise ValueError(msg)

    if not timebin_new:
        return df

    annotators = get_annotators(df)
    labels = get_labels(df)
    max_freq = get_max_freq(df)
    dataset = get_dataset(df)

    if isinstance(get_timezone(df), list):
        df["start_datetime"] = [to_datetime(elem, utc=True)
                                for elem in df["start_datetime"]]
        df["end_datetime"] = [to_datetime(elem, utc=True)
                              for elem in df["end_datetime"]]

    results = []
    for ant in annotators:
        for lbl in labels:
            df_1annot_1label = df[(df["annotator"] == ant) & (df["annotation"] == lbl)]

            if df_1annot_1label.empty:
                continue

            if timestamp is not None:
                # I do not remember if this is a regular case or not
                # might need to be deleted
                origin_timebin = timestamp[1] - timestamp[0]
                step = int(timebin_new / origin_timebin)
                time_vector = timestamp[0::step]
            else:
                t1 = min(df_1annot_1label["start_datetime"]).floor(timebin_new)
                t2 = max(df_1annot_1label["end_datetime"]).ceil(timebin_new)
                time_vector = date_range(start=t1, end=t2, freq=timebin_new)

            ts_detect_beg = df_1annot_1label["start_datetime"].to_list()
            ts_detect_end = df_1annot_1label["end_datetime"].to_list()
            filenames = df_1annot_1label["filename"].to_list()

            # filename_vector
            filename_vector = []
            for ts in time_vector:
                if (bisect.bisect_left(ts_detect_beg, ts) > 0 and
                        bisect.bisect_left(ts_detect_beg, ts) != len(ts_detect_beg)):
                    idx = bisect.bisect_left(ts_detect_beg, ts)
                    filename_vector.append(
                        filenames[idx] if timestamp_wav[idx] <= ts else
                        filenames[idx - 1],
                    )
                elif bisect.bisect_left(ts_detect_beg, ts) == len(ts_detect_beg):
                    filename_vector.append(filenames[-1])
                else:
                    filename_vector.append(filenames[0])

            # detection vector
            detect_vec = [0] * len(time_vector)
            for start, end in zip(ts_detect_beg, ts_detect_end, strict=False):
                idx = bisect.bisect_left(time_vector, start)
                idx = idx if start in time_vector else max(0, idx - 1)
                while idx < len(time_vector) and time_vector[idx] < end:
                    detect_vec[idx] = 1
                    idx += 1

            # rows for dataframe
            start_datetime = [
                time_vector[i] for i in range(len(time_vector)) if detect_vec[i]
            ]
            end_datetime = [t + timebin_new for t in start_datetime]
            file_vector = [
                filename_vector[i] for i in range(len(time_vector)) if detect_vec[i]
            ]

            if start_datetime:
                results.append(
                    DataFrame(
                        {
                            "dataset": [dataset] * len(file_vector),
                            "filename": file_vector,
                            "start_time": [0] * len(file_vector),
                            "end_time": [timebin_new.total_seconds()]
                            * len(file_vector),
                            "start_frequency": [0] * len(file_vector),
                            "end_frequency": [max_freq] * len(file_vector),
                            "annotation": [lbl] * len(file_vector),
                            "annotator": [ant] * len(file_vector),
                            "start_datetime": start_datetime,
                            "end_datetime": end_datetime,
                            "is_box": [0] * len(file_vector),
                        },
                    ),
                )

    return concat(results).sort_values(by=["start_datetime", "end_datetime",
                                           "annotator", "annotation"]).reset_index(drop=True)


def ensure_in_list(value: str, candidates: list[str], label: str) -> None:
    """Check for non-valid elements of a list."""
    if value not in candidates:
        msg = f"'{value}' not present in {label}, upload aborted"
        raise ValueError(msg)


def ensure_no_invalid(invalid: list[str], label: str) -> None:
    """Return non-valid elements of a list."""
    if invalid:
        msg = f"'{invalid}' not present in {label}, upload aborted"
        raise ValueError(msg)


def load_detections(filters: DetectionFilter) -> DataFrame:
    """Load and filter an APLOSE-formatted detection file.

    Parameters
    ----------
    filters : DetectionFilter
        All selection / filtering options.

    Returns
    -------
    DataFrame
        Detections that match the selected filters.

    """
    df = read_dataframe(filters.detection_file)
    df = filter_by_time(df, filters.begin, filters.end)
    df = filter_by_annotator(df, annotator=filters.annotator)
    df = filter_by_label(df, label=filters.annotation)
    df = filter_by_freq(df, filters.f_min, filters.f_max)
    df = filter_by_score(df, filters.score)
    df = reshape_timebin(df, filters.timebin_new)

    annotators = get_annotators(df)
    if len(annotators) > 1 and filters.user_sel in ["union", "intersection"]:
        df = intersection_or_union(df, user_sel=filters.user_sel)

    return df.sort_values(by=["start_datetime", "end_datetime"]).reset_index(drop=True)


def intersection_or_union(df: DataFrame, user_sel: str) -> DataFrame:
    """Compute intersection or union of annotations from multiple annotators."""
    annotators = get_annotators(df)
    if len(annotators) <= 1:
        msg = "Not enough annotators detected"
        raise ValueError(msg)

    if user_sel == "all":
        return df

    if user_sel not in ("intersection", "union"):
        msg = "'user_sel' must be either 'intersection' or 'union'"
        raise ValueError(msg)

    # Count how many annotators marked each (start_datetime, annotation) pair
    counts = df.groupby(["annotation", "start_datetime"])["annotator"].transform(
        "nunique",
    )

    if user_sel == "intersection":
        df_result = df[counts == len(annotators)]
        annotator_name = " ∩ ".join(annotators)
    else:  # union
        df_result = df[counts >= 1]
        annotator_name = " ∪ ".join(annotators)  # noqa: RUF001

    return (
        df_result.drop_duplicates(subset=["annotation", "start_datetime"])
        .assign(annotator=annotator_name)
        .sort_values("start_datetime")
        .reset_index(drop=True)
    )
