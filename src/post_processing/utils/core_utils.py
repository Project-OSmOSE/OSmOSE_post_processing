"""General functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import astral
import easygui
import numpy as np
from astral.sun import sunrise, sunset
from matplotlib import pyplot as plt
from osekit.config import TIMESTAMP_FORMAT_AUDIO_FILE
from osekit.utils.timestamp_utils import strptime_from_text
from pandas import (
    DataFrame,
    DatetimeIndex,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    json_normalize,
    to_datetime,
)
from pandas.tseries import offsets
from pandas.tseries.offsets import BaseOffset

from post_processing.utils.filtering_utils import (
    get_annotators,
    get_dataset,
    get_labels,
    get_max_freq,
    get_max_time,
    get_timezone,
)

if TYPE_CHECKING:
    from datetime import tzinfo
    from pathlib import Path

    import matplotlib.pyplot as plt


def get_season(ts: Timestamp, *, northern: bool = True) -> tuple[str, int]:
    """Determine the meteorological season from a Timestamp.

    In the Northern hemisphere
    Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Autumn: Sep-Nov

    In the Southern hemisphere
    Winter: Jun-Aug, Spring: Sep-Nov, Summer: Dec-Feb, Autumn: Mar-May

    Parameters
    ----------
    northern: boolean, default True.
        Specify if the seasons are northern or austral
    ts: Timestamp
        Considered datetime

    Returns
    -------
    The season and year of ts

    """
    if northern:
        winter = [1, 2, 12]
        spring = [3, 4, 5]
        summer = [6, 7, 8]
        autumn = [9, 10, 11]
    else:
        winter = [6, 7, 8]
        spring = [9, 10, 11]
        summer = [1, 2, 12]
        autumn = [3, 4, 5]

    if ts.month in spring:
        season = "spring"
    elif ts.month in summer:
        season = "summer"
    elif ts.month in autumn:
        season = "autumn"
    elif ts.month in winter:
        season = "winter"
    else:
        msg = "Invalid timestamp"
        raise ValueError(msg)

    return season, ts.year - 1 if ts.month in [1, 2] else ts.year


def get_sun_times(
    start: Timestamp,
    stop: Timestamp,
    lat: float,
    lon: float,
) -> (
    list[float],
    list[float],
):
    """Fetch sunrise and sunset hours for dates between start and stop.

    Parameters
    ----------
    start: Timestamp
        start datetime of when to fetch sun hour
    stop: Timestamp
        end datetime of when to fetch sun hour
    lat: float
        latitude in decimal degrees
    lon: float
        longitude in decimal degrees

    Returns
    -------
    hour_sunrise: list[float]
        Sunrise decimal hours for each day between start and stop

    hour_sunset: list[float]
        Sunset decimal hours for each day between start and stop

    """
    if start.tz is None or stop.tz is None:
        msg = "start and stop must be timezone-aware"
        raise ValueError(msg)

    tz = start.tz

    gps = astral.LocationInfo(latitude=lat, longitude=lon, timezone=tz)

    h_sunrise, h_sunset = [], []

    for date in [ts.date() for ts in date_range(start.normalize(),
                                                stop.normalize(),
                                                freq="D",
                                                )]:
        dt_sunrise = sunrise(gps.observer, date=date, tzinfo=tz)
        dt_sunset = sunset(gps.observer, date=date, tzinfo=tz)

        # Convert sunrise and sunset to decimal hours
        h_sunrise.append(dt_sunrise.hour + dt_sunrise.minute / 60 + dt_sunrise.second / 3600 + dt_sunrise.microsecond / 3_600_000_000)
        h_sunset.append(dt_sunset.hour + dt_sunset.minute / 60 + dt_sunset.second / 3600 + dt_sunset.microsecond / 3_600_000_000)

    return h_sunrise, h_sunset


def get_coordinates() -> tuple:
    """Ask for user input to get GPS coordinates."""
    title = "Coordinates in degree° minute'"
    msg = "latitude (N/S) and longitude (E/W)"
    field_names = ["lat decimal degree", "lon decimal degree"]
    field_values = easygui.multenterbox(msg, title, field_names)

    max_lat = 90
    max_lon = 180

    while True:
        if field_values is None:
            msg = "'get_coordinates()' was cancelled"
            raise TypeError(msg)

        lat, lon = field_values
        errmsg = ""
        try:
            lat_val = float(lat.strip())  # Convert to float for latitude
            if lat_val < -max_lat or lat_val > max_lat:
                errmsg += (
                    f"'{lat}' is not a valid latitude. It must be between -90 and 90.\n"
                )
        except ValueError:
            errmsg += f"'{lat}' is not a valid entry for latitude.\n"

        try:
            lon_val = float(lon.strip())  # Convert to float for longitude
            if lon_val < -max_lon or lon_val > max_lon:
                errmsg += (
                    f"'lon', invalid entry: '{lon}'. It must be between -180 and 180.\n"
                )
        except ValueError:
            errmsg += f"'lon', invalid entry: '{lon}'.\n"

        if errmsg == "":
            break

        field_values = easygui.multenterbox(errmsg, title, field_names)

    lat = float(lat.strip())
    lon = float(lon.strip())

    return lat, lon


def add_weak_detection(
    df: DataFrame,
    datetime_format: str = TIMESTAMP_FORMAT_AUDIO_FILE,
    max_time: Timedelta | None = None,
    max_freq: float | None = None,
) -> DataFrame:
    """Add weak detections APLOSE formatted DataFrame with only strong detections.

    Parameters
    ----------
    df: DataFrame
        An APLOSE formatted DataFrame.
    datetime_format: str
        A string corresponding to the datetime format in the `filename` column
    max_time: Timedelta
        Size of the weak detections
    max_freq: float
        Height of the weak detections

    """
    annotators = get_annotators(df)
    labels = get_labels(df)
    dataset_id = get_dataset(df)
    tz = get_timezone(df)

    if not max_freq:
        max_freq = get_max_freq(df)
    if not max_time:
        max_time = get_max_time(df)

    for ant in annotators:
        for lbl in labels:
            filenames = (
                df[(df["annotator"] == ant) & (df["annotation"] == lbl)]["filename"]
                .drop_duplicates()
                .tolist()
            )
            for f in filenames:
                test = df[(df["filename"] == f) & (df["annotation"] == lbl)]["is_box"]
                if test.any():
                    start_datetime = strptime_from_text(
                        text=f,
                        datetime_template=datetime_format,
                    ).tz_localize(tz)
                    end_datetime = start_datetime + Timedelta(max_time, unit="s")
                    new_line = [
                        dataset_id,
                        f,
                        0,
                        max_time,
                        0,
                        max_freq,
                        lbl,
                        ant,
                        start_datetime,
                        end_datetime,
                        0,
                    ]
                    if "score" in df.columns:
                        new_line.append(np.nan)
                    df.loc[df.index.max() + 1] = new_line

    return df.sort_values(by=["start_datetime", "annotator"]).reset_index(drop=True)


def json2df(json_path: Path) -> DataFrame:
    """Convert a metadatax json file into a DataFrame.

    Parameters
    ----------
    json_path: Path
        Json file path

    """
    with json_path.open(encoding="utf-8") as f:
        metadatax_df = json_normalize(json.load(f))
        metadatax_df["deployment_date"] = to_datetime(
            metadatax_df["deployment_date"],
        )
        metadatax_df["recovery_date"] = to_datetime(metadatax_df["recovery_date"])

    return metadatax_df


def add_season_period(
    ax: plt.Axes,
    bar_height: int = 10,
    *,
    northern: bool = True,
) -> None:
    """Add a bar at the top of the plot to seasons.

    Parameters
    ----------
    northern: boolean, default True.
        Specify if the seasons are northern or austral
    ax: plt.Axes
        Matplotlib Axes to add the bar to.
    bar_height: int
        Bar height in pixels

    """
    if not ax.has_data():
        msg = "Axes have no data"
        raise ValueError(msg)

    bins = date_range(
        start=Timestamp(ax.get_xlim()[0], unit="D").floor("1D"),
        end=Timestamp(ax.get_xlim()[1], unit="D").ceil("1D"),
    )

    season_colors = {
        "winter": "#2ce5e3",
        "spring": "#4fcf50",
        "summer": "#ffcf50",
        "autumn": "#fb9a67",
    }

    bin_centers = [
        (bins[i].timestamp() + bins[i + 1].timestamp()) / 2
        for i in range(len(bins) - 1)
    ]
    bin_centers = [Timestamp(center, unit="s") for center in bin_centers]

    bin_seasons = [get_season(bc, northern=northern)[0] for bc in bin_centers]
    bar_height = set_bar_height(ax, bar_height)
    bar_bottom = ax.get_ylim()[1] + (0.2 * bar_height)

    for i, season in enumerate(bin_seasons):
        ax.bar(
            bin_centers[i],
            height=bar_height,
            bottom=bar_bottom,
            width=(bins[i + 1] - bins[i]),
            color=season_colors[season],
            align="center",
            zorder=0,
            alpha=0.6,
        )

    ax.set_ylim(ax.dataLim.ymin, ax.dataLim.ymax)


def set_bar_height(ax: plt.Axes, pixel_height: int = 10) -> float:
    """Convert pixel height to data coordinates.

    Parameters
    ----------
    ax: Axes
        Figure plot

    pixel_height: int
        In pixel

    """
    if not ax.has_data():
        msg = "Axe has no data"
        raise ValueError(msg)

    display_to_data = ax.transData.inverted().transform
    _, data_bottom = display_to_data((0, 0))  # Bottom of the axis
    _, data_top = display_to_data((0, pixel_height))  # Top of the bar

    return data_top - data_bottom  # Convert pixel height to data scale


def add_recording_period(
    df: DataFrame,
    ax: plt.Axes,
    bar_height: int = 10,
) -> None:
    """Add a bar at the bottom on plot to show recording periods.

    Parameters
    ----------
    df: DataFrame
        Includes the recording campaign deployment
        and recovery dates (typically extracted from Metadatax)

    ax: Axes
        Figure plot

    bar_height: int
        Bar height in pixels

    """
    if not ax.has_data():
        msg = "Axe has no data"
        raise ValueError(msg)

    recorder_intervals = [
        (start, end - start)
        for start, end in zip(df["deployment_date"], df["recovery_date"], strict=False)
    ]

    bar_height = set_bar_height(ax=ax, pixel_height=bar_height)

    ax.broken_barh(
        recorder_intervals,
        (ax.get_ylim()[0] - (1.2 * bar_height), bar_height),
        facecolors="red",
        alpha=0.6,
    )
    ax.set_ylim(ax.dataLim.ymin, ax.dataLim.ymax)


def get_count(df: DataFrame, bin_size: Timedelta | BaseOffset) -> DataFrame:
    """Count observations per label and annotator.

    This function groups a DataFrame of events into uniform time bins and counts the
    number of events for each (label, annotator) pair in each bin.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame.
    bin_size : Timedelta | offsets
        Width or frequency of bins.

    Returns
    -------
    DataFrame
        A DataFrame indexed by the start of each bin (datetime), with columns named
        "<label>-<annotator>", containing the count of observations in that bin.

    """
    if not isinstance(df, DataFrame):
        msg = "`df` must be a DataFrame"
        raise TypeError(msg)

    if df.empty:
        msg = "`df` contains no data"
        raise ValueError(msg)

    datetime_list = list(df["start_datetime"])

    bins, bin_size = get_time_range_and_bin_size(datetime_list, bin_size)

    labels, annotators = get_labels_and_annotators(df)

    series_list = [
        df[(df["annotation"] == label) & (df["annotator"] == annotator)][
            "start_datetime"
        ]
        for label, annotator in zip(labels, annotators, strict=False)
    ]

    counts_df = DataFrame(index=bins[:-1])
    for i, series in enumerate(series_list):
        binned = cut(series, bins=bins, right=False)
        counts_df[f"{labels[i]}-{annotators[i]}"] = binned.value_counts().sort_index()
    return counts_df


def get_labels_and_annotators(df: DataFrame) -> tuple[list, list]:
    """Extract and align annotation labels and annotators from a DataFrame.

    If only one label is present, it is duplicated to match the number of annotators.
    Similarly, if one annotator is present, it is duplicated to match the labels.

    Parameters
    ----------
    df : DataFrame
        The APLOSE-formatted DataFrame.

    Returns
    -------
    tuple[list, list]
        A tuple containing the labels and annotators lists.

    """
    if not isinstance(df, DataFrame):
        msg = "`df` must be a DataFrame"
        raise TypeError(msg)

    if df.empty:
        msg = "`df` contains no data"
        raise ValueError(msg)

    annotators = df["annotator"].unique().tolist()
    labels = df["annotation"].unique().tolist()
    if len(labels) == 1:
        labels = [labels[0]] * len(annotators)
    if len(annotators) == 1:
        annotators = [annotators[0]] * len(labels)

    if len(annotators) != len(labels):
        msg = f"{len(annotators)} annotators and {len(labels)} labels must match."
        raise ValueError(msg)

    return labels, annotators


def localize_timestamps(timestamps: list[Timestamp], tz: tzinfo) -> list[Timestamp]:
    """Localize timestamps if necessary."""
    localized = []
    for ts in timestamps:
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            localized.append(ts.tz_localize(tz))
        else:
            localized.append(ts)
    return localized


def get_time_range_and_bin_size(timestamp_list: list[Timestamp],
                    bin_size: Timedelta | BaseOffset,
                    ) -> (DatetimeIndex, Timedelta):
    """Return time vector given a bin size."""
    if (not isinstance(timestamp_list, list) or
            not all(isinstance(ts, Timestamp) for ts in timestamp_list)):
        msg = "`timestamp_list` must be a list[Timestamp]"
        raise TypeError(msg)

    if len(timestamp_list) == 0:
        msg = "`timestamp_list` is empty"
        raise ValueError(msg)

    start, end, _ = round_begin_end_timestamps(timestamp_list, bin_size)
    timestamp_range = date_range(start=start, end=end, freq=bin_size)
    if isinstance(bin_size, Timedelta):
        return timestamp_range, bin_size
    if isinstance(bin_size, BaseOffset):
        return timestamp_range, timestamp_range[1] - timestamp_range[0]

    msg = "Could not get time range."
    raise ValueError(msg)


def round_begin_end_timestamps(timestamp_list: list[Timestamp],
                    bin_size: Timedelta | BaseOffset,
                    ) -> (Timestamp, Timestamp, Timedelta):
    """Return time vector given a bin size."""
    if (not isinstance(timestamp_list, list) or
            not all(isinstance(ts, Timestamp) for ts in timestamp_list)):
        msg = "timestamp_list must be a list[Timestamp]"
        raise TypeError(msg)

    if not timestamp_list:
        msg = "`timestamp_list` is empty"
        raise ValueError(msg)

    if isinstance(bin_size, Timedelta):
        start = min(timestamp_list).floor(bin_size)
        end = max(timestamp_list).ceil(bin_size)
        return start, end, bin_size

    if isinstance(bin_size, BaseOffset):
        start = bin_size.rollback(min(timestamp_list))
        end = bin_size.rollforward(max(timestamp_list))
        if not isinstance(bin_size, (offsets.Hour, offsets.Minute, offsets.Second)):
            start = Timestamp(start).normalize()
            end = Timestamp(end).normalize() + bin_size
            if start == end:
                end += bin_size

        timestamp_range = date_range(start=start, end=end, freq=bin_size)
        bin_size = timestamp_range[1] - timestamp_range[0]
        return start.floor(bin_size), end.ceil(bin_size), bin_size

    msg = "Could not get start/end timestamps."
    raise ValueError(msg)


def timedelta_to_str(td: Timedelta) -> str:
    """From a Timedelta to corresponding string."""
    seconds = int(td.total_seconds())

    if seconds % 86400 == 0:
        return f"{seconds // 86400}D"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"
