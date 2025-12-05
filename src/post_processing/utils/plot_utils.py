"""Plot functions used for DataAplose objects."""

from __future__ import annotations

import logging
from collections import Counter
from itertools import cycle
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from matplotlib.dates import num2date
from matplotlib.ticker import PercentFormatter
from numpy import ceil, histogram, polyfit
from pandas import DataFrame, DatetimeIndex, Index, Timedelta, Timestamp, date_range
from pandas.tseries import frequencies
from scipy.stats import pearsonr
from seaborn import scatterplot

from post_processing.utils.core_utils import (
    add_season_period,
    get_coordinates,
    get_labels_and_annotators,
    get_sun_times,
    get_time_range_and_bin_size,
    round_begin_end_timestamps,
    timedelta_to_str,
)
from post_processing.utils.filtering_utils import (
    get_max_time,
    get_timezone,
    filter_by_annotator,
)
from post_processing.utils.metrics_utils import normalize_counts_by_effort

if TYPE_CHECKING:
    from datetime import tzinfo

    from matplotlib.axes import Axes
    from pandas.tseries.offsets import BaseOffset

    from post_processing.dataclass.recording_period import RecordingPeriod

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def histo(
    df: DataFrame,
    ax: plt.Axes,
    bin_size: Timedelta | BaseOffset,
    time_bin: Timedelta,
    **kwargs: bool | str | list[str] | tuple[float, float] | list[Timestamp] | RecordingPeriod,  # noqa: E501
) -> None:
    """Seasonality plot.

    Parameters
    ----------
    df: DataFrame
        Data to plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the histogram.
    bin_size: Timedelta | BaseOffset
        The size of the histogram bins.
    time_bin: Timedelta
        The size of detections.
    **kwargs: Additional keyword arguments depending on the mode.
        - legend: bool
            Whether to show the legend.
        - color: str | list[str]
            Color or list of colors for the histogram bars.
            If not provided, default colors will be used.
        - season: bool
            Whether to show the season.
        - coordinates: tuple[float, float]
            The coordinates of the plotted detections.
        - effort: RecordingPeriod
            Object corresponding to the observation effort.
            If provided, data will be normalized by observation effort.

    """
    labels, annotators = zip(*[col.rsplit("-", 1) for col in df.columns], strict=False)
    labels = list(labels)
    annotators = list(annotators)

    if len(df) <= 1:
        msg = (f"DataFrame with annotators '{', '.join(annotators)}'"
               f" / labels '{', '.join(labels)}'"
               f" do not contains enough detections.")
        logging.warning(msg)
        return

    legend = kwargs.get("legend", False)
    color = kwargs.get("color", False)
    season = kwargs.get("season", False)
    effort = kwargs.get("effort", False)
    lat, lon = kwargs.get("coordinates")

    bin_size_str = get_bin_size_str(bin_size)

    begin, end, bin_size = round_begin_end_timestamps(list(df.index), bin_size)

    color = color or get_colors(df)

    if len(df.columns) > 1 and legend:
        legend_labels = get_legend(labels, annotators)
    else:
        legend_labels = None

    if effort:
        normalize_counts_by_effort(df, effort, time_bin)

    n_groups = len(labels) if legend_labels else 1
    bar_width = bin_size / n_groups
    bin_starts = mdates.date2num(df.index)

    for i in range(n_groups):
        offset = i * bar_width.total_seconds() / 86400

        bar_kwargs = {
            "width": bar_width.total_seconds() / 86400,
            "align": "edge",
            "edgecolor": "black",
            "color": color,
            "zorder": 2,
        }
        if legend_labels:
            bar_kwargs["label"] = legend_labels[i]

        ax.bar(bin_starts + offset, df.iloc[:, i], **bar_kwargs)

    if len(df.columns) > 1 and legend:
        ax.legend(labels=legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")

    y_label = (
        f"Detections{(' normalized by effort' if effort else '')}"
        f"\n(detections: {timedelta_to_str(time_bin)}"
        f" - bin size: {bin_size_str})"
    )
    ax.set_ylabel(y_label)
    set_y_axis_to_percentage(ax) if effort else set_dynamic_ylim(ax, df)
    set_plot_title(ax, annotators, labels)
    ax.set_xlim(begin, end)

    if season:
        if lat is None or lon is None:
            get_coordinates()
        add_season_period(ax, northern=lat >= 0)

    if effort:
        shade_no_effort(
            ax=ax,
            bin_starts=df.index,
            observed=effort,
            bar_width=bin_size,
        )


def _prepare_timeline_plot(
    df: DataFrame,
    ax: Axes,
    *,
    bins: DatetimeIndex = None,
    coordinates: tuple[float, float] | None = None,
    **kwargs: bool,
) -> None:
    """Plot daily detection patterns for a given annotator and label.

    Parameters
    ----------
    df: DataFrame
        data to plot
    ax : plt.Axes
        The matplotlib axis to draw on.
    coordinates : tuple[float, float]
        The latitude and longitude.
    bins : DatetimeIndex
        Pandas date range of the data to plot
    **kwargs: Additional keyword arguments depending on the mode.
        -show_rise_set : bool, default True
            Whether to overlay sunrise and sunset lines.

    """
    lat, lon = coordinates
    if lat is None or lon is None:
        lat, lon = get_coordinates()

    begin = bins[0]
    end = bins[-1]

    show_rise_set = kwargs.get("show_rise_set", False)

    labels, annotators = get_labels_and_annotators(df)

    ax.set_xlim(begin, end)
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 2))
    ax.set_ylabel("Hour")
    ax.set_xlabel("Date")
    ax.grid(color="k", linestyle="-", linewidth=0.2)

    set_plot_title(ax=ax, annotators=annotators, labels=labels)

    if show_rise_set:
        tz = get_timezone(df)
        if isinstance(tz, list):
            msg = "Several timezones not supported."
            raise ValueError(msg)
        add_sunrise_sunset(ax, lat, lon, tz)


def scatter(
    df: DataFrame,
    ax: Axes,
    time_range: DatetimeIndex,
    **kwargs: bool | tuple[float, float],
) -> None:
    """Scatter-plot of detections for a given annotator and label.

    Parameters
    ----------
    df: DataFrame
        data to plot
    ax : plt.Axes
        The matplotlib axis to draw on.
    time_range: DatetimeIndex
        The time range of the heatmap.
    **kwargs: Additional keyword arguments depending on the mode.
        -coordinates: tuple[float, float]
            The latitude and longitude.
        -show_rise_set : bool, default True
            Whether to overlay sunrise and sunset lines.
        -season: bool
            Whether to show the season.

    """
    show_rise_set = kwargs.get("show_rise_set", False)
    season = kwargs.get("season", False)
    coordinates = kwargs.get("coordinates", False)

    _prepare_timeline_plot(
        df=df,
        ax=ax,
        bins=time_range,
        show_rise_set=show_rise_set,
        season=season,
        coordinates=coordinates,
    )

    labels, annotators = get_labels_and_annotators(df)
    for ann in set(annotators):
        for lbl in set(labels):
            group = df[(df["annotator"] == ann) & (df["annotation"] == lbl)]

            if group.empty:
                continue

            detect_time_dec = [
                ts.hour + ts.minute / 60 + ts.second / 3600
                for ts in group["start_datetime"]
            ]

            ax.scatter(
                group["start_datetime"],
                detect_time_dec,
                label=f"{ann} - {lbl}",
                marker="x",
                linewidths=1,
                alpha=0.7,
            )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=True,
        framealpha=0.6,
    )


def heatmap(df: DataFrame,
            ax: Axes,
            bin_size: Timedelta | BaseOffset,
            time_range: DatetimeIndex,
            **kwargs: bool | tuple[float, float],
            ) -> None:
    """Heatmap of detections for a given annotator and label.

    Parameters
    ----------
    df: DataFrame
        data to plot
    ax : plt.Axes
        The matplotlib axis to draw on.
    bin_size: Timedelta | BaseOffset
        The size of the heatmap bins.
        Must be >= 24h.
    time_range: DatetimeIndex
        The time range of the heatmap.
    **kwargs: Additional keyword arguments depending on the mode.
        -coordinates: tuple[float, float]
            The latitude and longitude.
        -show_rise_set : bool, default True
            Whether to overlay sunrise and sunset lines.
        -season: bool
            Whether to show the season.

    """
    datetime_list = list(df["start_datetime"])

    _, bin_size_dt = get_time_range_and_bin_size(datetime_list, bin_size)
    if bin_size_dt < Timedelta("1D"):
        msg = "`bin_size` must be >= 24h for heatmap mode."
        raise ValueError(msg)

    show_rise_set = kwargs.get("show_rise_set", False)
    season = kwargs.get("season", False)
    coordinates = kwargs.get("coordinates", False)

    begin = time_range[0]
    end = time_range[-1]

    # Coarse bins (for display cells)
    cell_bins = date_range(begin, end, freq=bin_size)

    _prepare_timeline_plot(
        df=df,
        ax=ax,
        bins=cell_bins,
        show_rise_set=show_rise_set,
        coordinates=coordinates,
    )

    freq = frequencies.to_offset(Timedelta(get_max_time(df), "s"))

    # Fine bins (for counting detection)
    fine_bins = date_range(begin, end, freq=freq)

    # Assign each timestamp to fine bin
    fine_idx = np.searchsorted(fine_bins, datetime_list, side="right") - 1

    # Map fine bins to coarse cell index
    fine_to_cell = np.searchsorted(cell_bins, fine_bins, side="right") - 1

    mat = np.zeros((24, len(cell_bins) - 1), dtype=int)

    for dt, f_idx in zip(datetime_list, fine_idx, strict=False):
        if 0 <= f_idx < len(fine_bins) - 1:
            c_idx = fine_to_cell[f_idx]
            if 0 <= c_idx < len(cell_bins) - 1:
                mat[dt.hour, c_idx] += 1

    im = ax.imshow(
        mat,
        extent=(begin, end, 0, 24),
        vmin=0,
        vmax=mat.max(),
        aspect="auto",
        origin="lower",
    )

    if coordinates and season:
        lat, lon = coordinates
        add_season_period(ax, northern=lat >= 0)

    bin_size_str = get_bin_size_str(bin_size)
    freq_str = get_bin_size_str(freq)

    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, pad=0.1)
    cbar.ax.set_ylabel(f"{freq_str} detections per hour")
    ax.set_ylabel("Hour of day")
    ax.set_xlabel(f"Time ({bin_size_str} bin)")


def overview(df: DataFrame, annotator: list[str] | None = None) -> None:
    """Overview of an APLOSE formatted DataFrame.

    Parameters
    ----------
    df: DataFrame
        The Dataframe to analyse.
    annotator: list[str]
        List of annotators.

    """
    if annotator is not None:
        df = filter_by_annotator(df, annotator)

    summary_label = (
        df.groupby("annotation")["annotator"]  # noqa: PD010
        .apply(Counter)
        .unstack(fill_value=0)
    )

    summary_annotator = (
        df.groupby("annotator")["annotation"]  # noqa: PD010
        .apply(Counter)
        .unstack(fill_value=0)
    )

    dataset = df["dataset"].iloc[0]

    fig, axs = plt.subplots(2, 1)
    axs[0] = summary_label.plot(
        kind="bar",
        ax=axs[0],
        edgecolor="black",
        linewidth=0.5,
    )
    axs[1] = summary_annotator.plot(
        kind="bar",
        ax=axs[1],
        edgecolor="black",
        linewidth=0.5,
    )

    for a in axs:
        a.legend(
            loc="center left",
            frameon=1,
            framealpha=0.6,
            bbox_to_anchor=(1.01, 0.5),
        )
        a.tick_params(axis="both", rotation=0)
        a.set_ylabel("Number of annotated calls")
        a.yaxis.grid(color="gray", linestyle="--")
        a.set_axisbelow(True)

    # labels
    _wrap_xtick_labels(axs[0], max_chars=10)
    axs[1].set_xlabel("Annotator")

    # titles
    axs[0].set_title("Number of annotations per label")
    axs[1].set_title("Number of annotations per annotator")
    fig.suptitle(f"{dataset}")

    plt.tight_layout()

    # log
    msg = f"""{" Overview ":#^40}"""
    msg += f"\n\n {summary_label}"
    logging.info(msg)


def _wrap_xtick_labels(ax: plt.Axes, max_chars: int = 10) -> None:
    """Wrap x-axis tick labels at max_chars per line."""

    def wrap_text(text: str) -> str:
        lines = []
        while len(text) > max_chars:
            break_index = text.rfind(" ", 0, max_chars + 1)
            if break_index == -1:
                break_index = max_chars  # force break
            lines.append(text[:break_index])
            text = text[break_index:].lstrip()
        lines.append(text)  # remaining part
        return "\n".join(lines)

    new_labels = [wrap_text(label.get_text()) for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels, rotation=0)


def agreement(
    df: DataFrame,
    bin_size: Timedelta | BaseOffset,
    ax: plt.Axes,
) -> None:
    """Compute and visualize agreement between two annotators.

    This function compares annotation timestamps from two annotators over a time range.
    It also fits and plots a linear regression line and displays the coefficient
    of determination (R²) on the plot.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame.
        It must contain The annotations of two annotators.

    bin_size : Timedelta | BaseOffset
        The size of each time bin for aggregating annotation timestamps.

    ax : matplotlib.axes.Axes
        Matplotlib axes object where the scatterplot and regression line will be drawn.

    """
    labels, annotators = get_labels_and_annotators(df)

    datetimes1 = list(
        df[(df["annotator"] == annotators[0]) & (df["annotation"] == labels[0])][
            "start_datetime"
        ],
    )
    datetimes2 = list(
        df[(df["annotator"] == annotators[1]) & (df["annotation"] == labels[1])][
            "start_datetime"
        ],
    )

    # scatter plot
    n_annot_max = bin_size.total_seconds() / df["end_time"].iloc[0]

    start = df["start_datetime"].min()
    stop = df["start_datetime"].max()

    freq = (
        bin_size if isinstance(bin_size, Timedelta) else str(bin_size.n) + bin_size.name
    )

    bins = date_range(
        start=start.floor(bin_size),
        end=stop.ceil(bin_size),
        freq=freq,
    )

    hist1, _ = histogram(datetimes1, bins=bins)
    hist2, _ = histogram(datetimes2, bins=bins)

    df_hist = (
        DataFrame(
            {
                annotators[0]: hist1,
                annotators[1]: hist2,
            },
        )
        / n_annot_max
    )

    scatterplot(data=df_hist, x=annotators[0], y=annotators[1], ax=ax)

    coefficients = polyfit(df_hist[annotators[0]], df_hist[annotators[1]], 1)
    poly = np.poly1d(coefficients)
    ax.plot(df_hist[annotators[0]], poly(df_hist[annotators[0]]), lw=1)

    ax.set_xlabel(f"{annotators[0]}\n{labels[0]}")
    ax.set_ylabel(f"{annotators[1]}\n{labels[1]}")
    ax.grid(linestyle="-", linewidth=0.2)

    # Pearson correlation (R²)
    r, _ = pearsonr(df_hist[annotators[0]], df_hist[annotators[1]])
    ax.text(0.05, 0.85, f"R² = {r**2:.2f}", transform=ax.transAxes)


def timeline(
    df: DataFrame,
    ax: plt.Axes,
    **kwargs: list[str],
) -> None:
    """Plot detections on a timeline.

    Parameters
    ----------
    df: DataFrame
        APLOSE DataFrame
    ax : matplotlib.axes.Axes
        Matplotlib axes object where the scatterplot and regression line will be drawn.
    **kwargs: Additional keyword arguments depending on the mode.
        - color: str | list[str]
            Color or list of colors for the histogram bars.
            If not provided, default colors will be used.

    """
    color = kwargs.get("color")

    labels, _ = get_labels_and_annotators(df)

    color = (
        color or [c for _, c in zip(range(len(labels)), cycle(default_colors))]
    )

    for i, label in enumerate(labels):
        time_det = df[(df["annotation"] == label)]["start_datetime"].to_list()
        l_data = len(time_det)
        x = np.ones((l_data, 1), int) * i
        ax.scatter(time_det, x, color=color[i])

    ax.grid(color="k", linestyle="-", linewidth=0.2)
    ax.set_yticks(np.arange(0, len(labels), 1))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Date")
    ax.set_xlim(
        df["start_datetime"].min().floor("1d"),
        df["end_datetime"].max().ceil("1d"),
    )


def get_colors(df: DataFrame) -> list[str]:
    """Return default plot colors."""
    return [c for _, c in zip(range(len(df.columns)), cycle(default_colors))]


def get_legend(annotators: str | list[str], labels: str | list[str]) -> list[str]:
    """Return plot legend."""
    if len(set(labels)) > 1 and len(set(annotators)) == 1:
        return labels
    if len(set(annotators)) > 1 and len(set(labels)) == 1:
        return annotators
    return [f"{ant}\n{lbl}" for ant, lbl in zip(annotators, labels, strict=False)]


def get_bin_size_str(bin_size: Timedelta | BaseOffset) -> str:
    """Return bin size as a string."""
    if isinstance(bin_size, Timedelta):
        return timedelta_to_str(bin_size)
    return str(bin_size.n) + bin_size.freqstr


def set_y_axis_to_percentage(
    ax: plt.Axes,
) -> None:
    """Set y-axis to percentage."""
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_yticks(np.arange(0, 1.02, 0.2))


def set_dynamic_ylim(ax: plt.Axes,
                     df: DataFrame,
                     padding: float = 0.05,
                     nticks: int = 4,
                     ) -> None:
    """Set y-axis limits and ticks dynamically based on DataFrame values."""
    max_val = np.nanmax(df.to_numpy())
    upper_lim = int(ceil((1 + padding) * max_val))
    ax.set_ylim(0, upper_lim)

    step = int(max(1, ceil(max_val / (nticks - 1))))
    ax.set_yticks(range(0, upper_lim + 1, step))


def set_plot_title(ax: plt.Axes, annotators: list[str], labels: list[str]) -> None:
    """Set plot title."""
    title = (
        f"annotator: {', '.join(set(annotators))}\n"
        f"label: {', '.join(set(labels))}"
    )
    ax.set_title(title)


def shade_no_effort(
    ax: plt.Axes,
    bin_starts: Index,
    observed: RecordingPeriod,
    bar_width: Timedelta,
) -> None:
    """Shade areas of the plot where no observation effort was made.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to draw the shaded regions.
    bin_starts : Index
        A datetime index representing the start times of each bin.
    observed : RecordingPeriod
        A Series with observation counts or flags, indexed by datetime.
        Should be aligned or re-indexable to `bin_starts`.
    bar_width : Timedelta
        Width of each time bin. Used to compute the span of the shaded areas.


    """
    width_days = bar_width.total_seconds() / 86400
    no_effort_bins = bin_starts[observed.counts.reindex(bin_starts) == 0]
    for ts in no_effort_bins:
        start = mdates.date2num(ts)
        ax.axvspan(start, start + width_days, color="grey", alpha=0.08, zorder=1.5)

    x_min, x_max = ax.get_xlim()
    data_min = mdates.date2num(bin_starts[0])
    data_max = mdates.date2num(bin_starts[-1]) + width_days

    if x_min < data_min:
        ax.axvspan(x_min, data_min, color="grey", alpha=0.08, zorder=1.5)
    if x_max > data_max:
        ax.axvspan(data_max, x_max, color="grey", alpha=0.08, zorder=1.5)
    ax.set_xlim(x_min, x_max)


def add_sunrise_sunset(ax: Axes, lat: float, lon: float, tz: tzinfo) -> None:
    """Display sunrise/sunset times on plot."""
    x_min, x_max = ax.get_xlim()
    start_date = Timestamp(num2date(x_min)).tz_convert(tz)
    end_date = Timestamp(num2date(x_max)).tz_convert(tz)

    num_days = (end_date.date() - start_date.date()).days + 1
    dates = [start_date.date() + Timedelta(days=i) for i in range(num_days)]

    sunrise, sunset = get_sun_times(
        start=start_date,
        stop=end_date,
        lat=lat,
        lon=lon,
    )

    ax.plot(dates, sunrise, color="darkorange", label="Sunrise")
    ax.plot(dates, sunset, color="royalblue", label="Sunset")
    ax.legend(
        loc="center left",
        frameon=True,
        framealpha=0.6,
        bbox_to_anchor=(1.01, 0.95),
    )
