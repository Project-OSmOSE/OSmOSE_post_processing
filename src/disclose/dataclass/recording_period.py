"""`recording_period` module provides `RecordingPeriod` dataclass.

RecordingPeriod class returns a Timestamp list corresponding to recording periods.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pandas import (
    IntervalIndex,
    Series,
    Timedelta,
    date_range,
    read_csv,
    to_datetime,
    DataFrame,
)

from disclose.dataclass.data_aplose_config import DataAploseConfig
from disclose.utils.core import round_begin_end_timestamps
from disclose.utils.filtering import find_delimiter

if TYPE_CHECKING:
    from pandas.tseries.offsets import BaseOffset


@dataclass(frozen=True)
class RecordingPeriod:
    """Represents recording effort over time, aggregated into bins."""

    counts: Series
    timebin_origin: Timedelta

    @classmethod
    def from_config(
        cls,
        config: DataAploseConfig,
        *,
        bin_size: Timedelta | BaseOffset,
    ) -> RecordingPeriod:
        """Vectorised creation of recording coverage from CSV with start/end datetimes.

        This method reads a CSV with columns:
        - "start_recording"
        - "end_recording"
        - "start_deployment"
        - "end_deployment"

        It computes the **effective recording interval** as the intersection between
        recording and deployment periods, builds a fine-grained timeline at
        `timebin_origin` resolution, and aggregates effort into `bin_size` bins.

        Parameters
        ----------
        config
            DataAploseConfig object.
        bin_size : Timedelta or BaseOffset
            Size of the aggregation bin (e.g. Timedelta("1H") or "1D").

        Returns
        -------
        RecordingPeriod
            Object containing `counts` (Series indexed by IntervalIndex) and
            `timebin_origin`.

        """
        # Read CSV and parse datetime columns
        recording_file = config.recording_file

        if not recording_file:
            raise ValueError("No recording file provided.")

        if not recording_file.exists():
            raise FileNotFoundError(f"File not found: {recording_file}")

        df = cls.from_csv(recording_file)

        # Compute effective recording intervals (intersection)
        df["effective_start_recording"] = df[
            ["start_recording", "start_deployment"]
        ].max(axis=1)

        df["effective_end_recording"] = df[["end_recording", "end_deployment"]].min(
            axis=1
        )

        # Build fine-grained timeline at `timebin_origin` resolution
        origin = config.timebin_origin
        time_index = date_range(
            start=df["effective_start_recording"].min(),
            end=df["effective_end_recording"].max(),
            freq=origin,
        )

        # Initialise effort vector (0 = no recording, 1 = recording)
        # Compare each timestamp to all intervals in a vectorised manner
        effort = Series(0, index=time_index)

        # Vectorised interval coverage
        t_vals = time_index.to_numpy()[:, None]
        start_vals = df["effective_start_recording"].to_numpy()
        end_vals = df["effective_end_recording"].to_numpy()

        # Boolean matrix: True if the timestamp is within any recording interval
        covered = (t_vals >= start_vals) & (t_vals < end_vals)
        effort[:] = covered.any(axis=1).astype(int)

        # Aggregate effort into user-defined bin_size
        counts = effort.resample(bin_size, closed="left", label="left").sum()

        counts.index = IntervalIndex.from_arrays(
            counts.index,
            counts.index + round_begin_end_timestamps(list(counts.index), bin_size)[-1],
            closed="left",
        )

        return cls(counts=counts, timebin_origin=origin)

    @classmethod
    def from_csv(
        cls,
        csv_file: Path,
    ) -> DataFrame:
        """Load recording coverage from CSV."""
        delim = find_delimiter(csv_file)
        df = read_csv(
            csv_file,
            parse_dates=[
                "start_recording",
                "end_recording",
                "start_deployment",
                "end_deployment",
            ],
            delimiter=delim,
        )

        if df.empty:
            msg = "CSV is empty."
            raise ValueError(msg)

        # Normalise timezones: convert to UTC, then remove tz info (naive)
        for col in df.columns:
            df[col] = to_datetime(df[col], utc=True).dt.tz_convert(None)

        return df

    @classmethod
    def from_json(
        cls,
        json_file: Path,
    ) -> DataFrame:
        """Load recording coverage from JSON."""
        with json_file.open() as f:
            data = json.load(f)

        series_list = []
        for datum in data:
            series_list.append(
                Series({
                    "start_recording": datum["channel_configurations"][0][
                        "record_start_date"
                    ],
                    "end_recording": datum["channel_configurations"][0][
                        "record_end_date"
                    ],
                    "start_deployment": datum["deployment_date"],
                    "end_deployment": datum["recovery_date"],
                })
            )

        return DataFrame(series_list)
