"""`data_aplose` module provides the `DataAplose` class.

DataAplose class is used for handling, analyzing, and visualizing
APLOSE-formatted annotation data. It includes utilities to bin detections,
plot time-based distributions, and manage metadata such as annotators and labels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas import (
    DataFrame,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
)
from pandas.tseries import offsets

from post_processing.dataclass.detection_filter import DetectionFilter
from post_processing.utils.core_utils import get_count
from post_processing.utils.filtering_utils import (
    get_annotators,
    get_dataset,
    get_labels,
    get_timezone,
    load_detections,
)
from post_processing.utils.metrics_utils import detection_perf
from post_processing.utils.plot_utils import (
    heatmap,
    histo,
    overview,
    plot_annotator_agreement,
    scatter,
    timeline,
)

if TYPE_CHECKING:
    from datetime import tzinfo
    from pathlib import Path

    from pandas.tseries.offsets import BaseOffset

    from post_processing.dataclass.recording_period import RecordingPeriod

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _get_locator_from_offset(
    offset: int | Timedelta | BaseOffset,
) -> mdates.DateLocator:
    """Map a pandas offset object to the appropriate matplotlib DateLocator."""
    if isinstance(offset, int):
        return mdates.SecondLocator(interval=offset)

    if isinstance(offset, Timedelta):
        total_seconds = int(offset.total_seconds())
        if total_seconds % 3600 == 0:
            return mdates.HourLocator(interval=total_seconds // 3600)
        if total_seconds % 60 == 0:
            return mdates.MinuteLocator(interval=total_seconds // 60)
        return mdates.SecondLocator(interval=total_seconds)

    offset_to_locator = {
        (
            offsets.MonthEnd,
            offsets.MonthBegin,
            offsets.BusinessMonthEnd,
            offsets.BusinessMonthBegin,
        ): lambda offset: mdates.MonthLocator(interval=offset.n),
        (offsets.Week,): lambda offset: mdates.WeekdayLocator(
            byweekday=offset.weekday,
            interval=offset.n,
        ),
        (offsets.Day,): lambda offset: mdates.DayLocator(interval=offset.n),
        (offsets.Hour,): lambda offset: mdates.HourLocator(interval=offset.n),
        (offsets.Minute,): lambda offset: mdates.MinuteLocator(interval=offset.n),
    }

    for offset_classes, locator_fn in offset_to_locator.items():
        if isinstance(offset, offset_classes):
            return locator_fn(offset)

    msg = f"Unsupported offset type: {type(offset)}"
    raise ValueError(msg)


class DataAplose:
    """A class to handle APLOSE formatted data."""

    def __init__(
        self,
        df: DataFrame = None,
        *,
        begin: Timestamp = None,
        end: Timestamp = None,
    ) -> None:
        """Initialize a DataAplose object from a DataFrame.

        Parameters
        ----------
        df: DataFrame
            APLOSE formatted DataFrame.

        """
        self.df = df.sort_values(
            by=[
                "start_datetime",
                "end_datetime",
                "annotator",
                "annotation",
            ],
        ).reset_index(drop=True)
        self.annotators = sorted(set(self.df["annotator"])) if df is not None else None
        self.labels = sorted(set(self.df["annotation"])) if df is not None else None
        self.begin = min(self.df["start_datetime"]) if begin is None else begin
        self.end = max(self.df["end_datetime"]) if end is None else end
        self.dataset = sorted(set(self.df["dataset"])) if df is not None else None
        self.lat = None
        self.lon = None

    def __str__(self) -> str:
        """Return string representation of DataAplose object."""
        return (
            f"begin: {self.begin}\n"
            f"end: {self.end}\n"
            f"annotators: {self.annotators}\n"
            f"labels: {self.labels}\n"
            f"dataset: {self.dataset}"
        )

    def __repr__(self) -> str:
        """Return string representation of DataAplose object."""
        return self.__str__()

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of DataFrame."""
        return self.df.shape

    @property
    def lat(self) -> float:
        """Return latitude."""
        return self._lat

    @lat.setter
    def lat(self, value: float) -> None:
        self._lat = value

    @property
    def lon(self) -> float:
        """Return longitude."""
        return self._lon

    @lon.setter
    def lon(self, value: float) -> None:
        self._lon = value

    @property
    def coordinates(self) -> tuple[float, float]:
        """Coordinates of the audio data."""
        return self.lat, self.lon

    @coordinates.setter
    def coordinates(self, value: tuple[float, float]) -> None:
        if not isinstance(value, tuple) or len(value) != 2:  # noqa: PLR2004
            msg = "Coordinates must be a tuple of two floats: (lat, lon)."
            raise ValueError(msg)
        self.lat, self.lon = value

    def __getitem__(self, item: int) -> Series:
        """Return the row from the underlying DataFrame."""
        return self.df.iloc[item]

    def change_tz(self, tz: str | tzinfo) -> None:
        """Change the timezone of a DataAplose instance.

        Examples
        --------
        >>> import pytz
        >>> data = DataAplose(...)
        >>> data.change_tz(pytz.timezone("Etc/GMT-2"))

        >>> data = DataAplose(...)
        >>> data.change_tz("UTC")

        >>> data = DataAplose(...)
        >>> data.change_tz("UTC+02:00")

        """
        self.df["start_datetime"] = [
            elem.tz_convert(tz) for elem in self.df["start_datetime"]
        ]
        self.df["end_datetime"] = [
            elem.tz_convert(tz) for elem in self.df["end_datetime"]
        ]
        self.begin = self.begin.tz_convert(tz)
        self.end = self.end.tz_convert(tz)

    def filter_df(
        self,
        annotator: str | list[str],
        label: str | list[str],
    ) -> DataFrame:
        """Filter DataFrame based on annotator and label.

        Parameters
        ----------
        annotator: str | list[str]
            The annotator or list of annotators to filter.
        label: str | list[str]
            The label or list of labels to filter.

        Returns
        -------
        The filtered DataFrame.

        Raises
        ------
        ValueError
            If annotator or label are not valid or if filtered Dataframe is empty.

        """
        if isinstance(label, str):
            label = [label] if isinstance(annotator, str) else [label] * len(annotator)
        if isinstance(annotator, str):
            annotator = (
                [annotator] if isinstance(label, str) else [annotator] * len(label)
            )
        if len(annotator) != len(label):
            msg = (
                f"Length of annotator ({len(annotator)}) and"
                f" label ({len(label)}) must match."
            )
            raise ValueError(msg)

        for ant, lbl in zip(annotator, label, strict=False):
            if ant not in self.annotators:
                msg = f'Annotator "{ant}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if lbl not in self.labels:
                msg = f'Label "{lbl}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if self.df[
                (self.df["type"] == "WEAK")
                & (self.df["annotator"] == ant)
                & (self.df["annotation"] == lbl)
            ].empty:
                msg = (
                    f"DataFrame with annotator '{ant}' / label '{lbl}'"
                    f" contains no weak detection."
                )
                raise ValueError(msg)
        config = list(zip(annotator, label, strict=False))
        return self.df[
            self.df[["annotator", "annotation"]].apply(tuple, axis=1).isin(config)
        ].reset_index(drop=True)

    def set_ax(
        self,
        ax: plt.Axes,
        x_ticks_res: Timedelta | offsets.BaseOffset,
        date_format: str,
    ) -> plt.Axes:
        """Configure a Matplotlib axis for time-based plot.

        Sets up x-axis with appropriate limits, tick spacing,
        formatting, and grid styling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object to configure.
        x_ticks_res : Timedelta | offsets.BaseOffset
            Resolution of the x-axis major ticks.
        date_format : str
            Date format string for x-axis tick labels (e.g., "%b", "%Y-%m-%d %H:%M").

        Returns
        -------
        matplotlib.axes.Axes
            The configured Axes object, ready for plotting.

        """
        ax.xaxis.set_major_locator(
            _get_locator_from_offset(offset=x_ticks_res),
        )
        date_formatter = mdates.DateFormatter(fmt=date_format, tz=self.begin.tz)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.grid(linestyle="--", linewidth=0.2, axis="both", zorder=1)

        return ax

    def overview(self, annotator: list[str] | None = None) -> None:
        """Overview of an APLOSE formatted DataFrame."""
        overview(self.df, annotator)

    def detection_perf(
        self,
        annotators: tuple[str, str] | list[str],
        labels: tuple[str, str] | list[str],
    ) -> tuple[float, float, float]:
        """Compute performance metrics for detection.

        Precision and recall are computed in regard to a reference annotator/label pair.

        Parameters
        ----------
        annotators: [str, str]
            List of the two annotators to compare.
            The first annotator is chosen as a reference.
        labels: [str, str]
            List of the two labels to compare.
            The first label is chosen as a reference.

        Returns
        -------
        precision: float
        recall: float
        f_score: float

        """
        df_filtered = self.filter_df(
            annotators,
            labels,
        )
        if isinstance(annotators, str):
            annotators = [annotators]
        if isinstance(labels, str):
            labels = [labels]
        ref = (annotators[0], labels[0])

        if len(set(df_filtered["end_time"])) > 1:
            msg = "Multiple time bins detected in DataFrame."
            raise ValueError(msg)
        timebin = Timedelta(df_filtered["end_time"].iloc[0], "s")

        return detection_perf(
            df=df_filtered,
            ref=ref,
            time=date_range(self.begin, self.end, freq=timebin),
        )

    def plot(
        self,
        mode: str,
        ax: plt.Axes,
        *,
        annotator: str | list[str],
        label: str | list[str],
        **kwargs: bool | Timedelta | BaseOffset | str | list[str] | RecordingPeriod,
    ) -> None:
        """Plot filtered annotation data using the specified mode.

        Supports multiple plot types depending on the mode:
          - "histogram": Plots a histogram of annotation data.
          - "scatter" / "heatmap": Maps detections on a timeline.
          - "agreement": Plots inter-annotator agreement regression.

        Parameters
        ----------
        mode: str
            Type of plot to generate.
            Must be one of {"histogram", "scatter", "heatmap", "agreement"}.
        ax: plt.Axes
            Matplotlib Axes object to plot on.
        annotator: str | list[str]
            The selected annotator or list of annotators.
        label: str | list[str]
            The selected label or list of labels.
        **kwargs: Additional keyword arguments depending on the mode.
            - legend: bool
                Whether to show the legend.
            - season: bool
                Whether to show the season.
            - show_rise_set: bool
                Whether to show sunrise and sunset times.
            - color: str | list[str]
                Color(s) for the bars.
            - bin_size: Timedelta | BaseOffset
                Bin size for the histogram.
            - effort: Series
                The timestamp intervals corresponding to the observation effort.
                If provided, data will be normalized by observation effort.

        """
        df_filtered = self.filter_df(
            annotator,
            label,
        )

        time = date_range(self.begin, self.end)
        bin_size = kwargs.get("bin_size")
        legend = kwargs.get("legend", True)
        color = kwargs.get("color")
        season = kwargs.get("season")
        effort = kwargs.get("effort")
        show_rise_set = kwargs.get("show_rise_set", True)

        if mode == "histogram":
            ax.set_xlim(time[0], time[-1])
            if not bin_size:
                msg = "'bin_size' missing for histogram plot."
                raise ValueError(msg)
            df_counts = get_count(df_filtered, bin_size)
            detection_size = Timedelta(max(df_filtered["end_time"]), "s")
            return histo(
                df=df_counts,
                ax=ax,
                bin_size=bin_size,
                time_bin=detection_size,
                legend=legend,
                color=color,
                season=season,
                effort=effort,
                coordinates=(self.lat, self.lon),
            )

        if mode == "heatmap":
            ax.set_xlim(time[0], time[-1])
            return heatmap(
                df=df_filtered,
                ax=ax,
                bin_size=bin_size,
                time_range=time,
                show_rise_set=show_rise_set,
                season=season,
                coordinates=self.coordinates,
            )

        if mode == "scatter":
            ax.set_xlim(time[0], time[-1])
            return scatter(
                df=df_filtered,
                ax=ax,
                time_range=time,
                show_rise_set=show_rise_set,
                season=season,
                coordinates=self.coordinates,
                effort=effort,
            )

        if mode == "agreement":
            if not bin_size:
                msg = "'bin_size' missing for agreement plot."
                raise ValueError(msg)
            df_counts = get_count(df_filtered, bin_size)
            return plot_annotator_agreement(df=df_counts, bin_size=bin_size, ax=ax)

        if mode == "timeline":
            ax.set_xlim(time[0], time[-1])
            color = kwargs.get("color")
            df_filtered = self.filter_df(
                annotator,
                label,
            )
            return timeline(df=df_filtered, ax=ax, color=color)

        msg = f"Unsupported plot mode: {mode}"
        raise ValueError(msg)

    @classmethod
    def from_yaml(
        cls,
        file: Path,
        *,
        concat: bool = True,
    ) -> DataAplose | list[DataAplose]:
        """Return a DataAplose object from a yaml file.

        Parameters
        ----------
        file: Path
            The path to a yaml configuration file.
        concat: bool
            If set to True, the DataAplose objects will be concatenated.
            If set to False, the DataAplose objects will be returned as a list.

        Returns
        -------
        DataAplose:
        The DataAplose object.

        """
        filters = DetectionFilter.from_yaml(file=file)
        return cls.from_filters(filters, concat=concat)

    @classmethod
    def from_filters(
        cls,
        filters: DetectionFilter | list[DetectionFilter],
        *,
        concat: bool = False,
    ) -> DataAplose | list[DataAplose]:
        """Return a DataAplose object from a yaml file.

        Parameters
        ----------
        filters: DetectionFilter | list[DetectionFilters]
            Object containing the detection filters.
        concat: bool
            If set to True, the DataAplose objects will be concatenated.
            If set to False, the DataAplose objects will be returned as a list.

        Returns
        -------
        DataAplose:
        The DataAplose object.

        """
        if isinstance(filters, DetectionFilter):
            filters = [filters]
        cls_list = [cls(load_detections(fil)) for fil in filters]

        for cls_obj, fil in zip(cls_list, filters, strict=True):
            cls.reshape(cls_obj, fil.begin, fil.end)

        if len(cls_list) == 1:
            return cls_list[0]

        if concat:
            return cls.concatenate(cls_list)
        return cls_list

    @classmethod
    def concatenate(
        cls,
        data_list: list[DataAplose],
    ) -> DataAplose:
        """Concatenate a list of DataAplose objects into one."""
        df_concat = (
            concat(
                [data.df for data in data_list],
                ignore_index=True,
            )
            .sort_values(
                by=[
                    "start_datetime",
                    "end_datetime",
                    "annotator",
                    "annotation",
                ],
            )
            .reset_index(drop=True)
        )

        obj = cls(
            df=df_concat,
            begin=min(obj.begin for obj in data_list),
            end=max(obj.end for obj in data_list),
        )

        if isinstance(get_timezone(df_concat), list):
            obj.change_tz("utc")
            msg = (
                "Several timezones found in DataFrame,"
                " all timestamps are converted to UTC."
            )
            logging.info(msg)
        return obj

    def reshape(self, begin: Timestamp = None, end: Timestamp = None) -> DataAplose:
        """Reshape the DataAplose with a new beginning and/or end."""
        if not any([begin, end]):
            msg = "No begin and end timestamps provided for reshape of DataAplose instance."
            logging.debug(msg)
            return self

        tz = get_timezone(self.df)
        if begin:
            self.begin = begin
            if not begin.tz:
                self.begin = begin.tz_localize(tz)
        if end:
            self.end = end
            if not end.tz:
                self.end = end.tz_localize(tz)

        if self.begin >= self.end:
            msg = "Begin timestamp is not anterior than end timestamp."
            raise ValueError(msg)

        self.df = self.df[
            (self.df["start_datetime"] >= self.begin)
            & (self.df["end_datetime"] <= self.end)
        ]

        if self.df.empty:
            msg = "DataFrame is empty after reshaping."
            raise ValueError(msg)

        self.dataset = get_dataset(self.df)
        self.labels = get_labels(self.df)
        self.annotators = get_annotators(self.df)

        return self
