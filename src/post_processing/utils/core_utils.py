"""General functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import astral
import easygui
from astral.sun import sunrise, sunset
from matplotlib import pyplot as plt
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

if TYPE_CHECKING:
    from datetime import tzinfo
    from pathlib import Path

    import matplotlib.pyplot as plt


def get_season(ts: Timestamp, *, northern: bool = True) -> tuple[str, int]:
    """Determine the meteorological season from a Timestamp.

    In the Northern Hemisphere
    Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Autumn: Sep-Nov

    In the Southern Hemisphere
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
        spring = [3, 4, 5]
        summer = [6, 7, 8]
        autumn = [9, 10, 11]
    else:
        spring = [9, 10, 11]
        summer = [1, 2, 12]
        autumn = [3, 4, 5]

    if ts.month in spring:
        season = "spring"
    elif ts.month in summer:
        season = "summer"
    elif ts.month in autumn:
        season = "autumn"
    else:
        season = "winter"

    return season, ts.year - 1 if ts.month in [1, 2] else ts.year


def get_sun_times(
    start: Timestamp,
    stop: Timestamp,
    lat: float,
    lon: float,
) -> tuple[list[float], list[float]]:
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

    for date in [
        ts.date()
        for ts in date_range(
            start.normalize(),
            stop.normalize(),
            freq="D",
        )
    ]:
        dt_sunrise = Timestamp(sunrise(gps.observer, date=date)).tz_convert(tz)
        dt_sunset = Timestamp(sunset(gps.observer, date=date)).tz_convert(tz)

        # Convert sunrise and sunset to decimal hours
        h_sunrise.append(
            dt_sunrise.hour
            + dt_sunrise.minute / 60
            + dt_sunrise.second / 3600
            + dt_sunrise.microsecond / 3_600_000_000,
        )
        h_sunset.append(
            dt_sunset.hour
            + dt_sunset.minute / 60
            + dt_sunset.second / 3600
            + dt_sunset.microsecond / 3_600_000_000,
        )

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
            errmsg += f"'lat', invalid entry: '{lat}'.\n"

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


def json2df(json_path: Path) -> DataFrame:
    """Convert a metadatax JSON file into a DataFrame.

    Parameters
    ----------
    json_path: Path
        JSON file path

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

    patches = ax.patches

    bins = date_range(
        start=Timestamp(ax.get_xlim()[0], unit="D").round("1ms"),
        end=Timestamp(ax.get_xlim()[1], unit="D").round("1ms"),
        freq=Timedelta(patches[0].get_width(), "D").round("1ms"),
    )

    season_colors = {
        "winter": "#84eceb",
        "spring": "#91de92",
        "summer": "#fce097",
        "autumn": "#f9c1a5",
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
            zorder=2,
            alpha=1,
            linewidth=0,
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
        msg = "Axe have no data"
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
        msg = "Axe have no data"
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


def get_count(
    df: DataFrame, bin_size: Timedelta | BaseOffset, time: DatetimeIndex | None = None
) -> DataFrame:
    """Count observations per label and annotator.

    This function groups a DataFrame of events into uniform time bins and counts the
    number of events for each (label, annotator) pair in each bin.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame.
    bin_size : Timedelta | offsets
        Width or frequency of bins.
    time: DatetimeIndex
        DatetimeIndex from a specified beginning to end

    Returns
    -------
    DataFrame
        A DataFrame indexed by the start of each bin (datetime), with columns named
        "<label>-<annotator>", containing the count of observations in that bin.

    """
    if df.empty:
        msg = "`df` contains no data"
        raise ValueError(msg)

    datetime_list = list(df["start_datetime"]) if time is None else time.to_list()

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
    """Extract and align annotation labels and annotators from an APLOSE DataFrame.

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
    """Localise timestamps if necessary."""
    localized = []
    for ts in timestamps:
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            localized.append(ts.tz_localize(tz))
        else:
            localized.append(ts)
    return localized


def get_time_range_and_bin_size(
    timestamp_list: list[Timestamp],
    bin_size: Timedelta | BaseOffset,
) -> tuple[DatetimeIndex, Timedelta]:
    """Return time vector given a bin size."""
    if not isinstance(timestamp_list, list) or not all(
        isinstance(ts, Timestamp) for ts in timestamp_list
    ):
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
    msg = "bin_size must be a Timedelta or BaseOffset."
    raise TypeError(msg)


def round_begin_end_timestamps(
    timestamp_list: list[Timestamp],
    bin_size: Timedelta | BaseOffset,
) -> tuple[Any, Any, Any]:
    """Return time vector given a bin size."""
    if not isinstance(timestamp_list, list) or not all(
        isinstance(ts, Timestamp) for ts in timestamp_list
    ):
        msg = "timestamp_list must be a list[Timestamp]"
        raise TypeError(msg)

    if not timestamp_list:
        msg = "`timestamp_list` is empty"
        raise ValueError(msg)

    if isinstance(bin_size, Timedelta):
        start = min(timestamp_list).floor(bin_size)
        end = max(timestamp_list).floor(bin_size) + bin_size
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
        if bin_size.resolution_string in {"s", "min", "h"}:
            return start.floor(bin_size), end.ceil(bin_size), bin_size
        return start, end, bin_size

    msg = "Could not get start/end timestamps."
    raise ValueError(msg)


def timedelta_to_str(td: Timedelta) -> str:
    """From a Timedelta to the corresponding string."""
    seconds = int(td.total_seconds())

    if seconds % 86400 == 0:
        return f"{seconds // 86400}D"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"
