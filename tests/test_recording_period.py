from pandas import Timedelta, read_csv, to_datetime

from post_processing.dataclass.detection_filter import DetectionFilter
from post_processing.dataclass.recording_period import RecordingPeriod


def test_recording_period_with_gaps(recording_planning_config: DetectionFilter) -> None:
    """RecordingPeriod correctly represents long gaps with no recording effort.

    The planning contains two recording blocks separated by ~3 weeks with no
    recording at all. Weekly aggregation must reflect:
    - weeks with full effort,
    - weeks with partial effort,
    - weeks with zero effort.
    """
    histo_x_bin_size = Timedelta("7D")
    recording_period = RecordingPeriod.from_path(
        config=recording_planning_config,
        bin_size=histo_x_bin_size,
    )

    counts = recording_period.counts
    origin = recording_planning_config.timebin_origin
    nb_timebin_origin_per_histo_x_bin_size = int(histo_x_bin_size / origin)

    # Computes effective recording intervals from recording planning csv
    df_planning = read_csv(
        recording_planning_config.timestamp_file,
        parse_dates=[
            "start_recording",
            "end_recording",
            "start_deployment",
            "end_deployment",
        ],
    )
    for col in [
        "start_recording",
        "end_recording",
        "start_deployment",
        "end_deployment",
    ]:
        df_planning[col] = (
            to_datetime(df_planning[col], utc=True)
            .dt.tz_convert(None)
        )

    df_planning["start"] = df_planning[
        ["start_recording", "start_deployment"]
    ].max(axis=1)
    df_planning["end"] = df_planning[
        ["end_recording", "end_deployment"]
    ].min(axis=1)

    planning = df_planning.loc[df_planning["start"] < df_planning["end"]]
    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------
    assert not counts.empty
    assert counts.index.is_interval()
    assert counts.min() >= 0
    assert counts.max() <= nb_timebin_origin_per_histo_x_bin_size

    # ------------------------------------------------------------------
    # Find overlap (number of timebin_origin) within each effective recording period
    # ------------------------------------------------------------------
    for interval in counts.index:
        bin_start = interval.left
        bin_end = interval.right

        # Compute overlap with all recording intervals
        overlap_start = planning["start"].clip(lower=bin_start, upper=bin_end)
        overlap_end = planning["end"].clip(lower=bin_start, upper=bin_end)

        overlap = (overlap_end - overlap_start).clip(lower=Timedelta(0))
        expected_minutes = int(overlap.sum() / recording_planning_config.timebin_origin)

        assert counts.loc[interval] == expected_minutes, (
            f"Mismatch for bin {interval}: "
            f"expected {expected_minutes}, got {counts.loc[interval]}"
        )
