"""`recording_period` module provides `RecordingPeriod` dataclass.

RecordingPeriod class returns a Timestamp list corresponding to recording periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pandas import (
    IntervalIndex,
    Series,
    Timedelta,
    date_range,
    read_csv,
    to_datetime,
)

from post_processing.utils.core_utils import round_begin_end_timestamps
from post_processing.utils.filtering_utils import (
    find_delimiter,
)

if TYPE_CHECKING:
    from pandas.tseries.offsets import BaseOffset

    from post_processing.dataclass.detection_filter import DetectionFilter


@dataclass(frozen=True)
class RecordingPeriod:
    """Represents recording effort over time, aggregated into bins."""

    counts: Series
    timebin_origin: Timedelta

    @classmethod
    def from_path(
        cls,
        config: DetectionFilter,
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
            Configuration object containing at least:
            - `timestamp_file`: path to CSV
            - `timebin_origin`: Timedelta resolution of detections
        bin_size : Timedelta or BaseOffset
            Size of the aggregation bin (e.g. Timedelta("1H") or "1D").

        Returns
        -------
        RecordingPeriod
            Object containing `counts` (Series indexed by IntervalIndex) and
            `timebin_origin`.

        """
        # Read CSV and parse datetime columns
        timestamp_file = config.timestamp_file
        delim = find_delimiter(timestamp_file)
        df = read_csv(
            config.timestamp_file,
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

        # Ensure all required columns are present
        required_columns = {
            "start_recording",
            "end_recording",
            "start_deployment",
            "end_deployment",
        }

        missing = required_columns - set(df.columns)

        if missing:
            msg = f"CSV is missing required columns: {', '.join(sorted(missing))}"
            raise ValueError(msg)

        # Normalise timezones: convert to UTC, then remove tz info (naive)
        for col in [
            "start_recording",
            "end_recording",
            "start_deployment",
            "end_deployment",
        ]:
            df[col] = to_datetime(df[col], utc=True).dt.tz_convert(None)

        # Compute effective recording intervals (intersection)
        df["effective_start_recording"] = df[
            ["start_recording", "start_deployment"]
        ].max(axis=1)

        df["effective_end_recording"] = df[
            ["end_recording", "end_deployment"]
        ].min(axis=1)

        # Remove rows with no actual recording interval
        df = df.loc[
            df["effective_start_recording"] < df["effective_end_recording"]
        ].copy()

        if df.empty:
            msg = "No valid recording intervals after deployment intersection."
            raise ValueError(msg)

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
            counts.index +
            round_begin_end_timestamps(list(counts.index), bin_size)[-1],
            closed="left",
        )

        return cls(counts=counts, timebin_origin=origin)
