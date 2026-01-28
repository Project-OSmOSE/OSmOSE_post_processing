"""Functions used to filter APLOSE-formatted DataFrame."""

from __future__ import annotations

import bisect
import csv
import datetime
from typing import TYPE_CHECKING

import pytz
from osekit.utils.timestamp_utils import strptime_from_text
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

    from post_processing.dataclass.detection_filter import DetectionFilter


def find_delimiter(file: Path) -> str:
    """Find the proper delimiter for a csv file."""
    allowed_delimiters = {",", ";", "\t", "|"}
    try:
        with file.open("r", encoding="utf-8") as f:
            # Read first few lines to detect delimiter
            sample = f.read(4096)

            if not sample.strip():
                msg = f"Could not determine delimiter for '{file}': file is empty"
                raise ValueError(msg)

            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)

            if dialect.delimiter not in allowed_delimiters:
                msg = (f"Could not determine delimiter for '{file}': "
                       f"unsupported delimiter '{dialect.delimiter}'")
                raise ValueError(msg)

            return dialect.delimiter

    except csv.Error as e:
        msg = f"Could not determine delimiter for '{file}': {e}"
        raise ValueError(msg) from e


def filter_strong_detection(
    df: DataFrame,
) -> DataFrame:
    """Filter to keep only weak detections (exclude box/strong annotations).

    This function identifies and removes "strong" or "box" type annotations,
    keeping only "weak" detections. It checks for either an 'is_box' or 'type' column.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame with either 'is_box' or 'type' column.

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only weak detections.

    """
    if "type" in df.columns:
        df = df[df["type"] == "WEAK"]
    elif "is_box" in df.columns:
        df = df[df["is_box"] == 0]
    else:
        msg = "Could not determine annotation type."
        raise ValueError(msg)
    if df.empty:
        msg = "No weak detection found."
        raise ValueError(msg)
    return df


def filter_by_time(
    df: DataFrame,
    begin: Timestamp | None,
    end: Timestamp | None,
) -> DataFrame:
    """Filter detections by time range.

    Parameters
    ----------
    df : DataFrame
        APLOSE DataFrame containing 'start_datetime' and 'end_datetime' columns.
    begin : Timestamp, optional
        Start of time range (inclusive). If None, no lower bound is applied.
    end : Timestamp, optional
        End of time range (inclusive). If None, no upper bound is applied.

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only detections within the specified time range.

    """
    if begin is not None:
        df = df[df["start_datetime"] >= begin]
        if df.empty:
            msg = f"No detection found after '{begin}'."
            raise ValueError(msg)

    if end is not None:
        df = df[df["end_datetime"] <= end]
        if df.empty:
            msg = f"No detection found before '{end}'."
            raise ValueError(msg)

    return df


def filter_by_annotator(
    df: DataFrame,
    annotator: str | list[str] | None,
) -> DataFrame:
    """Filter a DataFrame based on annotator selection.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame containing an 'annotator' column.
    annotator : str or list of str
        Single annotator name or list of annotator names to filter by.

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only detections from the specified annotator(s).

    """
    if annotator is None:
        return df

    list_annotators = get_annotators(df)

    if isinstance(annotator, str):
        ensure_in_list(annotator, list_annotators, "annotator")
        return df[df["annotator"] == annotator]

    invalid = [a for a in annotator if a not in list_annotators]
    ensure_no_invalid(invalid, "annotators")

    return df.loc[df["annotator"].isin(annotator)]


def filter_by_label(
    df: DataFrame,
    label: str | list[str] | None,
) -> DataFrame:
    """Filter a DataFrame based on label selection.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame containing an 'annotation' column.
    label : str or list of str
        Single label or list of labels to filter by.

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only detections with the specified label(s).

    """
    if label is None:
        return df

    list_labels = get_labels(df)

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
    """Filter a DataFrame based on frequency selection.

    Parameters
    ----------
    df : DataFrame
        APLOSE DataFrame containing 'start_frequency' and 'end_frequency' columns.
    f_min : float, optional
        Minimum frequency in Hz (inclusive). If None, no lower bound is applied.
    f_max : float, optional
        Maximum frequency in Hz (inclusive). If None, no upper bound is applied.

    Returns
    -------
    DataFrame
        Filtered DataFrame with only detections within the specified frequency range.

    """
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
    """Filter detections by confidence score.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame containing a 'score' column.
    score : float
        The minimum confidence score threshold (inclusive).

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only detections with score >= min_score.

    """
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


def read_dataframe(file: Path, rows: int | None = None) -> DataFrame:
    """Read an APLOSE-formatted CSV file into a DataFrame."""
    delimiter = find_delimiter(file)
    return (
        read_csv(file,
                 sep=delimiter,
                 parse_dates=["start_datetime", "end_datetime"],
                 nrows=rows,
                 )
        .drop_duplicates()
        .dropna(subset=["annotation"])
        .sort_values(by=["start_datetime", "end_datetime"])
        .reset_index(drop=True)
    )


def get_annotators(df: DataFrame) -> list[str]:
    """Return the annotator list of APLOSE DataFrame."""
    return sorted(set(df["annotator"]))


def get_labels(df: DataFrame) -> list[str]:
    """Return the label list of APLOSE DataFrame."""
    return sorted(set(df["annotation"]))


def get_max_freq(df: DataFrame) -> float:
    """Return the maximum frequency of APLOSE DataFrame."""
    return df["end_frequency"].max()


def get_max_time(df: DataFrame) -> float:
    """Return the maximum time of APLOSE DataFrame."""
    return df["end_time"].max()


def get_dataset(df: DataFrame) -> str | list[str]:
    """Return dataset list  of APLOSE DataFrame."""
    datasets = sorted(set(df["dataset"]))
    return datasets if len(datasets) > 1 else datasets[0]


def get_canonical_tz(tz: datetime.tzinfo) -> pytz.tzinfo.BaseTzInfo:
    """Convert a timezone object to its canonical pytz representation.

    This function ensures compatibility between different timezone implementations
    (pytz, zoneinfo) by converting them to pytz timezone objects.

    Parameters
    ----------
    tz : datetime.tzinfo
        Timezone object (can be pytz timezone or ZoneInfo).

    Returns
    -------
    pytz.tzinfo.BaseTzInfo
        Canonical pytz timezone object.

    """
    if isinstance(tz, datetime.timezone):
        if tz == datetime.UTC:
            return pytz.utc
        offset_minutes = int(tz.utcoffset(None).total_seconds() / 60)
        return pytz.FixedOffset(offset_minutes)
    if hasattr(tz, "zone") and tz.zone:
        return pytz.timezone(tz.zone)
    if hasattr(tz, "key"):
        return pytz.timezone(tz.key)
    msg = f"Unknown timezone: {tz}"
    raise TypeError(msg)


def get_timezone(df: DataFrame)\
        -> pytz.tzinfo.BaseTzInfo | list[pytz.tzinfo.BaseTzInfo]:
    """Return timezone(s) from APLOSE DataFrame.

    Parameters
    ----------
    df: DataFrame
        APLOSE result Dataframe

    Returns
    -------
    tzoffset: list[tzoffset]
        list of timezones

    """
    timezones = {get_canonical_tz(ts.tzinfo) for ts in df["start_datetime"]}

    if len(timezones) == 1:
        return next(iter(timezones))
    return list(timezones)


def check_timestamp(df: DataFrame, timestamp_audio: list[Timestamp]) -> None:
    """Check if a provided timestamp_audio list is correctly formated.

    Parameters
    ----------
    df: DataFrame
        APLOSE results Dataframe.
    timestamp_audio: list[Timestamp]
        A list of timestamps. Each timestamp is
        the start datetime of the corresponding audio file for each detection in df.

    """
    if timestamp_audio is None:
        msg = "`timestamp_wav` is empty"
        raise ValueError(msg)
    if len(timestamp_audio) != len(df):
        msg = "`timestamp_wav` is not the same length as `df`"
        raise ValueError(msg)


def _build_filename_vector(
    time_vector: list[Timestamp],
    ts_detect_beg: list[Timestamp],
    timestamp_audio: list[Timestamp],
    filenames: list[str],
) -> list[str]:
    """Build the filename vector for each time bin."""
    filename_vector = []
    for ts in time_vector:
        idx = bisect.bisect_left(ts_detect_beg, ts)

        if idx == 0:
            filename_vector.append(filenames[0])
        elif idx == len(ts_detect_beg):
            filename_vector.append(filenames[-1])
        else:
            # Choose a filename based on timestamp_audio
            filename_vector.append(
                filenames[idx] if timestamp_audio[idx] <= ts else filenames[idx - 1],
            )

    return filename_vector


def _build_detection_vector(
    time_vector: list[Timestamp],
    ts_detect_beg: list[Timestamp],
    ts_detect_end: list[Timestamp],
) -> list[int]:
    """Build a binary detection vector indicating presence in each time bin."""
    detect_vec = [0] * len(time_vector)

    for start, end in zip(ts_detect_beg, ts_detect_end, strict=False):
        idx = bisect.bisect_left(time_vector, start)
        idx = idx if start in time_vector else max(0, idx - 1)

        while idx < len(time_vector) and time_vector[idx] < end:
            detect_vec[idx] = 1
            idx += 1

    return detect_vec


def _create_result_dataframe(
    file_vector: list[str],
    start_datetime: list[Timestamp],
    timebin_new: Timedelta,
    max_freq: float,
    dataset: str,
    label: str,
    annotator: str,
) -> DataFrame:
    """Create result DataFrame for one annotator-label combination."""
    return DataFrame({
        "dataset": [dataset] * len(file_vector),
        "filename": file_vector,
        "start_time": [0] * len(file_vector),
        "end_time": [timebin_new.total_seconds()] * len(file_vector),
        "start_frequency": [0] * len(file_vector),
        "end_frequency": [max_freq] * len(file_vector),
        "annotation": [label] * len(file_vector),
        "annotator": [annotator] * len(file_vector),
        "start_datetime": start_datetime,
        "end_datetime": [t + timebin_new for t in start_datetime],
        "type": ["WEAK"] * len(file_vector),
    })


def _normalize_timezones(df: DataFrame) -> DataFrame:
    """Convert all timestamps to UTC if multiple timezones are present."""
    if isinstance(get_timezone(df), list):
        df["start_datetime"] = [
            to_datetime(elem, utc=True) for elem in df["start_datetime"]
        ]
        df["end_datetime"] = [
            to_datetime(elem, utc=True) for elem in df["end_datetime"]
        ]
    return df


def _process_annotator_label_pair(
    df: DataFrame,
    annotator: str,
    label: str,
    timebin_new: Timedelta,
    timestamp_audio: list[Timestamp],
    max_freq: float,
    dataset: str,
) -> DataFrame | None:
    """Process detections for one annotator-label combination."""
    df_subset = df[(df["annotator"] == annotator) & (df["annotation"] == label)]

    if df_subset.empty:
        return None

    # Create a time vector
    t1 = min(df_subset["start_datetime"]).floor(timebin_new)
    t2 = max(df_subset["end_datetime"]).ceil(timebin_new)
    time_vector = date_range(start=t1, end=t2, freq=timebin_new)

    # Extract detection data
    ts_detect_beg = df_subset["start_datetime"].to_list()
    ts_detect_end = df_subset["end_datetime"].to_list()
    filenames = df_subset["filename"].to_list()

    # Build vectors
    filename_vector = _build_filename_vector(
        time_vector, ts_detect_beg, timestamp_audio, filenames,
    )
    detect_vec = _build_detection_vector(time_vector, ts_detect_beg, ts_detect_end)

    # Filter to only detected time bins
    start_datetime = [
        time_vector[i] for i, detected in enumerate(detect_vec) if detected
    ]
    file_vector = [
        filename_vector[i] for i, detected in enumerate(detect_vec) if detected
        # filename_vector[i + 1] for i, detected in enumerate(detect_vec) if detected
    ]

    if not start_datetime:
        return None

    return _create_result_dataframe(
        file_vector, start_datetime, timebin_new, max_freq, dataset, label, annotator,
    )


def reshape_timebin(
    df: DataFrame,
    timebin_new: Timedelta | None,
    timestamp_audio: list[Timestamp],
) -> DataFrame:
    """Reshape an APLOSE result DataFrame according to a new time bin.

    Parameters
    ----------
    df: DataFrame
        An APLOSE result DataFrame.
    timebin_new: Timedelta
        The size of the new time bin.
    timestamp_audio: list[Timestamp]
        A list of Timestamp objects corresponding to the shape
        in which the data should be reshaped.

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

    check_timestamp(df, timestamp_audio)

    # Extract metadata
    annotators = get_annotators(df)
    labels = get_labels(df)
    max_freq = get_max_freq(df)
    dataset = get_dataset(df)

    # Normalize timezones if needed
    df = _normalize_timezones(df)

    # Process each annotator-label combination
    results = []
    for ant in annotators:
        for lbl in labels:
            result = _process_annotator_label_pair(
                df, ant, lbl, timebin_new, timestamp_audio, max_freq, dataset,
            )
            if result is not None:
                results.append(result)

    return (
        concat(results)
        .sort_values(by=["start_datetime", "end_datetime", "annotator", "annotation"])
        .reset_index(drop=True)
    )


def get_filename_timestamps(df: DataFrame, date_parser: str) -> list[Timestamp]:
    """Get audio file start timestamps of each detection contained in df.

    Parameters.
    ----------
    df: DataFrame
        An APLOSE result DataFrame.
    date_parser: str
        date parser of the wav file

    Returns
    -------
    List of Timestamps corresponding to the wav files' start timestamps
    of each detection contained in df.

    """
    tz = get_timezone(df)
    timestamps = [
        strptime_from_text(
            ts,
            datetime_template=date_parser,
        ) for ts in df["filename"]
    ]

    if all(t.tz is None for t in timestamps):
        timestamps = [t.tz_localize(tz) for t in timestamps]

    return timestamps


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
    if filters.box:
        df = filter_strong_detection(df)
    df = filter_by_time(df, filters.begin, filters.end)
    df = filter_by_annotator(df, annotator=filters.annotator)
    df = filter_by_label(df, label=filters.annotation)
    df = filter_by_freq(df, filters.f_min, filters.f_max)
    df = filter_by_score(df, filters.score)
    filename_ts = get_filename_timestamps(df, filters.filename_format)
    df = reshape_timebin(df,
                         timebin_new=filters.timebin_new,
                         timestamp_audio=filename_ts,
                         )

    annotators = get_annotators(df)
    if len(annotators) > 1 and filters.user_sel in {"union", "intersection"}:
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

    if user_sel not in {"intersection", "union"}:
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
