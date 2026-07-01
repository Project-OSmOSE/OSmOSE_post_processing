from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pandas import Timedelta, Timestamp

from disclose.utils.filtering import read_dataframe, get_max_time


@dataclass(frozen=False)
class DataAploseConfig:
    """Configuration object for loading and filtering APLOSE-formatted detection data.

    Parameters
    ----------
    detection_file : Path
        Path to the detection file to be loaded.
    timebin_new : Timedelta | None
        Optional resampling or re-binning time resolution.
    start_datetime : Timestamp | None
        Start datetime used to filter detections.
    end_datetime : Timestamp | None
        End datetime used to filter detections.
    annotator : str | list[str] | None
        Filter for one or multiple annotators.
    annotation : str | list[str] | None
        Filter for one or multiple annotation labels.
    type : str | None
        Optional detection type filter.
    recording_file : Path | None
        Optional external recording period file.
    user_selection : str
        Strategy for combining multiple filters. Default is "all".
    min_frequency : float | None
        Minimum frequency threshold for filtering detections.
    max_frequency : float | None
        Maximum frequency threshold for filtering detections.
    confidence : float | None
        Minimum confidence threshold for detections.
    filename_format : str | None
        Optional filename formatting rule.
    timebin_origin : Timedelta | None
        Automatically computed base time bin derived from the detection file.
        This field is set internally and should not be provided manually.

    """

    detection_file: Path
    timebin_new: Timedelta | None = None
    start_datetime: Timestamp | None = None
    end_datetime: Timestamp | None = None
    annotator: str | list[str] | None = None
    annotation: str | list[str] | None = None
    type: str | None = None
    recording_file: Path | None = None
    user_selection: str = "all"
    min_frequency: float | None = None
    max_frequency: float | None = None
    confidence: float | None = None
    filename_format: str | None = None
    timebin_origin: Timedelta | None = None

    def __post_init__(self) -> None:
        """Compute derived configuration fields after initialization."""
        if self.timebin_origin is None:
            df = read_dataframe(self.detection_file)
            object.__setattr__(self, "timebin_origin", get_max_time(df))

    @classmethod
    def from_dict(
        cls, config: dict | list[dict], concat: bool = True
    ) -> DataAploseConfig | list[DataAploseConfig]:
        """Create a DataAploseConfig object from a dictionary."""
        if isinstance(config, dict):
            config = [config]

        conf_list = [cls(**c) for c in config]

        if concat:
            return cls.concat(conf_list)
        else:
            return conf_list

    @classmethod
    def concat(
        cls, config_list: list[DataAploseConfig | None]
    ) -> DataAploseConfig | None:
        """Concatenate configuration instances."""
        if not any(config_list):
            return None

        vars = list(DataAploseConfig.__dataclass_fields__.keys())

        config_cont = {}

        for var in vars:
            values = [
                config.__getattribute__(var)
                for config in config_list
                if config.__getattribute__(var) is not None
            ]

            unique_values = sorted(set(values))

            if len(unique_values) == 1:
                config_cont[var] = unique_values[0]
            elif len(unique_values) > 1:
                if var == "start_datetime":
                    config_cont[var] = min(unique_values)
                elif var == "end_datetime":
                    config_cont[var] = max(unique_values)
                else:
                    config_cont[var] = unique_values
            else:
                config_cont[var] = None

        return cls(**config_cont)
