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

from disclose.dataclass.data_aplose_config import DataAploseConfig
from disclose.utils.core import get_count
from disclose.utils.filtering import (
    get_annotators,
    get_dataset,
    get_labels,
    get_timezone,
    load_detections,
)
from disclose.dataclass.recording_period import RecordingPeriod
from disclose.utils.metric import detection_perf
from disclose.utils.visualisation import (
    heatmap,
    histo,
    overview,
    plot_annotator_agreement,
    scatter,
    timeline,
)

if TYPE_CHECKING:
    from datetime import tzinfo

    from pandas.tseries.offsets import BaseOffset


default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _get_locator_from_offset(
    offset: int | Timedelta | BaseOffset,
) -> mdates.DateLocator:
    """Map a pandas' offset object to the appropriate matplotlib DateLocator."""
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
        df: DataFrame | None = None,
        config: DataAploseConfig | None = None,
    ) -> None:
        """Initialize a DataAplose object from a DataFrame.

        Parameters
        ----------
        df: DataFrame
            APLOSE-formatted DataFrame

        """
        self.config: DataAploseConfig = config

        self.df = df.sort_values(
            by=[
                "start_datetime",
                "end_datetime",
                "annotator",
                "annotation",
            ],
        ).reset_index(drop=True)
        self.annotators: list | None = (
            sorted(set(self.df["annotator"])) if df is not None else None
        )
        self.labels: list | None = (
            sorted(set(self.df["annotation"])) if df is not None else None
        )
        self.start_datetime: Timestamp | None = (
            config.start_datetime
            if config
            else min(self.df["start_datetime"], default=None)
        )
        self.end_datetime: Timestamp | None = (
            config.end_datetime
            if config
            else max(self.df["end_datetime"], default=None)
        )
        self.dataset: list | None = (
            sorted(set(self.df["dataset"])) if df is not None else None
        )
        self.lat: float | None = None
        self.lon: float | None = None

    def __str__(self) -> str:
        """Return string representation of DataAplose object."""
        return (
            f"start_datetime: {self.start_datetime}\n"
            f"end_datetime: {self.end_datetime}\n"
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

    @property
    def config(self) -> DataAploseConfig:
        """Return the config file."""
        return self._config

    @config.setter
    def config(self, value: DataAploseConfig) -> None:
        self._config = value

    def __getitem__(self, item: int) -> Series:
        """Return the row from the underlying DataFrame."""
        return self.df.iloc[item]

    @classmethod
    def from_dict(
        cls, config: dict | list[dict], *, concat: bool = True
    ) -> DataAplose | list[DataAplose]:
        """Create a DataAplose object from a configuration dictionary.

        Parameters
        ----------
        config : dict | list[dict]
            Configuration dictionary or list of configuration dictionaries.

            Required keys:
                detection_file : Path
                filename_format : str

            Optional keys:
                timebin_new : Timedelta | None
                start_datetime : Timestamp | None
                end_datetime : Timestamp | None
                annotator : str | list[str] | None
                annotation : str | list[str] | None
                type : str | None
                recording_file : Path | None
                user_selection : str = "all"
                min_frequency : float | None
                max_frequency : float | None
                confidence : float | None

        concat : bool, default=True
            If True, returns a single concatenated DataAplose object.
            If False, returns a list of DataAplose objects.

        Returns
        -------
        DataAplose or list[DataAplose]
            The constructed object(s).

        """
        conf_list = DataAploseConfig.from_dict(config=config, concat=False)

        cls_list = [cls(load_detections(conf)) for conf in conf_list]
        cls.config = cls_list

        for obj, conf in zip(cls_list, conf_list, strict=True):
            cls.reshape(obj, conf.start_datetime, conf.end_datetime)
            obj.config = conf

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

        # messy, need improvement
        for data in data_list:
            if data.config:
                if not data.config.start_datetime:
                    data.config.start_datetime = min(df_concat["start_datetime"])
                if not data.config.end_datetime:
                    data.config.end_datetime = max(df_concat["end_datetime"])

        config = DataAploseConfig.concat([data.config for data in data_list])

        obj = cls(df=df_concat, config=config)

        if isinstance(get_timezone(df_concat), list):
            obj.change_tz("utc")
            msg = (
                "Several timezones found in DataFrame,"
                " all timestamps are converted to UTC."
            )
            logging.info(msg)
        return obj

    def reshape(
        self, start_datetime: Timestamp = None, end_datetime: Timestamp = None
    ) -> DataAplose:
        """Reshape the DataAplose with a new beginning and/or end."""
        if not any([start_datetime, end_datetime]):
            msg = "No begin/end timestamps provided for reshape of DataAplose instance."
            logging.debug(msg)
            return self

        tz = get_timezone(self.df)
        if start_datetime:
            self.start_datetime = start_datetime
            if not start_datetime.tz:
                self.start_datetime = start_datetime.tz_localize(tz)
        if end_datetime:
            self.end_datetime = end_datetime
            if not end_datetime.tz:
                self.end_datetime = end_datetime.tz_localize(tz)

        if self.start_datetime >= self.end_datetime:
            msg = "Begin timestamp is not anterior than end timestamp."
            raise ValueError(msg)

        self.df = self.df[
            (self.df["start_datetime"] >= self.start_datetime)
            & (self.df["end_datetime"] <= self.end_datetime)
        ]

        if self.df.empty:
            return self

        self.dataset = get_dataset(self.df)
        self.labels = get_labels(self.df)
        self.annotators = get_annotators(self.df)

        return self

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
        self.start_datetime = self.start_datetime.tz_convert(tz)
        self.end_datetime = self.end_datetime.tz_convert(tz)

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
            If annotator or label are not valid or if the filtered Dataframe is empty.

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
                (self.df["annotator"] == ant) & (self.df["annotation"] == lbl)
            ].empty:
                msg = (
                    f"DataFrame with annotator '{ant}' / label '{lbl}'"
                    f" contains no detection."
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
        date_formatter = mdates.DateFormatter(
            fmt=date_format, tz=self.start_datetime.tz
        )
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
            time=date_range(self.start_datetime, self.end_datetime, freq=timebin),
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
          - "histogram": Plot a histogram of annotation data.
          - "scatter" / "heatmap": Map hourly detections on a timeline.
          - "agreement": Plot inter-annotator agreement regression.
          - "timeline": Plot a timeline of annotation data.

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
            - effort: bool
                The timestamp intervals corresponding to the observation effort.
                If provided by the `recording_file` argument, data will be normalized by observation effort.

        """
        df_filtered = self.filter_df(
            annotator,
            label,
        )

        dates = date_range(self.start_datetime, self.end_datetime)
        bin_size = kwargs.get("bin_size")
        legend = kwargs.get("legend", True)
        color = kwargs.get("color")
        season = kwargs.get("season")
        effort = kwargs.get("effort", False)
        if effort:
            effort = RecordingPeriod.from_config(config=self.config, bin_size=bin_size)
        show_rise_set = kwargs.get("show_rise_set", True)

        if mode == "histogram":
            ax.set_xlim(self.start_datetime, self.end_datetime)
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
            ax.set_xlim(self.start_datetime, self.end_datetime)
            return heatmap(
                df=df_filtered,
                ax=ax,
                bin_size=bin_size,
                time_range=dates,
                show_rise_set=show_rise_set,
                season=season,
                coordinates=self.coordinates,
            )

        if mode == "scatter":
            ax.set_xlim(self.start_datetime, self.end_datetime)
            return scatter(
                df=df_filtered,
                ax=ax,
                time_range=dates,
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
            ax.set_xlim(self.start_datetime, self.end_datetime)
            color = kwargs.get("color")
            df_filtered = self.filter_df(
                annotator,
                label,
            )
            return timeline(df=df_filtered, ax=ax, color=color)

        msg = f"Unsupported plot mode: {mode}"
        raise ValueError(msg)
