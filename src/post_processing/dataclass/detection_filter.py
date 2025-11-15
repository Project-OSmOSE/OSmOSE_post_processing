"""`detection_filter` module provides the `DetectionFilter` dataclass.

DetectionFilter class uses criteria applied to APLOSE-formatted DataFrames.
It supports filtering annotations based on time, frequency, annotators,
and other parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pandas import Timedelta, Timestamp

from post_processing.utils.filtering_utils import (
    get_timezone,
    read_dataframe,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import tzinfo


@dataclass(frozen=True)
class DetectionFilter:
    """A class to handle filters applied to APLOSE formatted DataFrame."""

    detection_file: Path = None
    timebin_origin: Timedelta = None
    timebin_new: Timedelta = None
    begin: Timestamp | None = None
    end: Timestamp | None = None
    timezone: tzinfo = None
    annotator: str | Iterable[str] | None = None
    annotation: str | Iterable[str] | None = None
    timestamp_file: Path | None = None
    user_sel: Literal["all", "union", "intersection"] = "all"
    f_min: float | None = None
    f_max: float | None = None
    score: float | None = None
    box: bool = False
    filename_format: str = None

    @classmethod
    def from_yaml(
        cls,
        file: Path,
    ) -> DetectionFilter | list[DetectionFilter]:
        """Return a DetectionFilter object from a yaml file.

        Parameters
        ----------
        file: Path
            The path to a yaml configuration file.

        Returns
        -------
        DataAplose:
        The DataAplose object.

        """
        with file.open(encoding="utf-8") as yaml_file:
            parameters = yaml.safe_load(yaml_file)
            return cls.from_dict(parameters)

    @classmethod
    def from_dict(
            cls,
            parameters: dict,
    ) -> DetectionFilter | list[DetectionFilter]:
        """Return a DetectionFilter object from a dict.

        Parameters
        ----------
        parameters: dict
            The parameters to load DataAplose object.

        Returns
        -------
        DataAplose:
        The DataAplose object.

        """
        filters = []
        for detection_file, filters_dict in parameters.items():
            df_preview = read_dataframe(Path(detection_file), nrows=5)
            filters_dict["timebin_origin"] = Timedelta(
                max(df_preview["end_time"]),
                "s",
            )
            filters_dict["timezone"] = get_timezone(df_preview)
            filters_dict["detection_file"] = Path(detection_file)
            if filters_dict.get("timebin_new"):
                filters_dict["timebin_new"] = Timedelta(
                    filters_dict["timebin_new"],
                    "s",
                )
            if filters_dict.get("begin"):
                filters_dict["begin"] = Timestamp(filters_dict["begin"])
            if filters_dict.get("end"):
                filters_dict["end"] = Timestamp(filters_dict["end"])
            if filters_dict.get("timestamp_file"):
                filters_dict["timestamp_file"] = Path(
                    filters_dict["timestamp_file"])

            filters.append(cls(**filters_dict))

        if len(filters) == 1:
            return filters[0]

        return filters
