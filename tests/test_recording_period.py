import json
from pathlib import Path

import pytest
from pandas import Timedelta, read_csv, to_datetime, DataFrame, Timestamp

from disclose.dataclass.data_aplose_config import DataAploseConfig
from disclose.dataclass.recording_period import RecordingPeriod


def test_recording_period_with_gaps(
    sample_csv_result: Path,
    recording_planning_csv: Path,
) -> None:
    """RecordingPeriod correctly represents long gaps with no recording effort.

    The planning contains two recording blocks separated by ~3 weeks with no
    recording at all. Weekly aggregation must reflect:
    - weeks with full effort,
    - weeks with partial effort,
    - weeks with zero effort.
    """
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S_%f%z",
        "recording_file": recording_planning_csv,
    }
    config = DataAploseConfig(**dict_test)
    histo_x_bin_size = Timedelta("7D")

    recording_period = RecordingPeriod.from_config(
        config=config,
        bin_size=histo_x_bin_size,
    )

    counts = recording_period.counts
    origin = config.timebin_origin
    nb_timebin_origin_per_histo_x_bin_size = int(histo_x_bin_size / origin)

    # Computes effective recording intervals from recording planning csv
    df_planning = read_csv(
        recording_planning_csv,
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
        df_planning[col] = to_datetime(df_planning[col], utc=True).dt.tz_convert(None)

    df_planning["start"] = df_planning[["start_recording", "start_deployment"]].max(
        axis=1
    )
    df_planning["end"] = df_planning[["end_recording", "end_deployment"]].min(axis=1)

    planning = df_planning.loc[df_planning["start"] < df_planning["end"]]

    # Structural checks
    assert not counts.empty
    assert counts.index.is_interval()
    assert counts.min() >= 0
    assert counts.max() <= nb_timebin_origin_per_histo_x_bin_size

    # Find overlap (number of timebin_origin) within each effective recording period
    for interval in counts.index:
        bin_start = interval.left
        bin_end = interval.right

        # Compute overlap with all recording intervals
        overlap_start = planning["start"].clip(lower=bin_start, upper=bin_end)
        overlap_end = planning["end"].clip(lower=bin_start, upper=bin_end)

        overlap = (overlap_end - overlap_start).clip(lower=Timedelta(0))
        expected_minutes = int(overlap.sum() / config.timebin_origin)

        assert counts.loc[interval] == expected_minutes, (
            f"Mismatch for bin {interval}: "
            f"expected {expected_minutes}, got {counts.loc[interval]}"
        )


def test_no_recording_file_provided(
    sample_csv_result: DataFrame, recording_planning_csv: DataFrame
) -> None:
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": None,
    }
    config = DataAploseConfig(**dict_test)

    with pytest.raises(ValueError, match=r"No recording file provided."):
        RecordingPeriod.from_config(config=config, bin_size=Timedelta("1h"))


def test_no_recording_wrong_file(
    sample_csv_result: DataFrame, recording_planning_csv: DataFrame
) -> None:
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": Path("fake_file.csv"),
    }
    config = DataAploseConfig(**dict_test)

    with pytest.raises(FileNotFoundError, match=r"File not found"):
        RecordingPeriod.from_config(config=config, bin_size=Timedelta("1h"))


def test_from_json(tmp_path: Path):
    json_file = tmp_path / "recordings.json"

    data = [
        {
            "deployment_date": "2025-01-01T00:00:00Z",
            "recovery_date": "2025-01-02T00:00:00Z",
            "channel_configurations": [
                {
                    "record_start_date": "2025-01-01T00:00:00Z",
                    "record_end_date": "2025-01-01T01:00:00Z",
                }
            ],
        }
    ]

    json_file.write_text(json.dumps(data))
    df = RecordingPeriod.from_json(json_file)

    expected = DataFrame([
        {
            "start_recording": "2025-01-01T00:00:00Z",
            "end_recording": "2025-01-01T01:00:00Z",
            "start_deployment": "2025-01-01T00:00:00Z",
            "end_deployment": "2025-01-02T00:00:00Z",
        }
    ])

    assert df.equals(expected)


def test_recording_period_from_csv_empty(
    sample_csv_result: Path,
    tmp_path: Path,
) -> None:
    empty_file = tmp_path / "recordings.csv"
    data = DataFrame(
        [],
        columns=[
            "start_recording",
            "end_recording",
            "start_deployment",
            "end_deployment",
        ],
    )
    data.to_csv(empty_file, index=False)

    with pytest.raises(ValueError, match=r"CSV is empty."):
        RecordingPeriod.from_csv(empty_file)


def test_recording_period_from_csv_missing_column(
    sample_csv_result: Path,
    tmp_path: Path,
) -> None:
    missing_col_file = tmp_path / "recordings.csv"
    data = DataFrame(
        data=[
            [
                Timestamp("2025-01-01T00:00:00+0000"),
                Timestamp("2026-01-01T00:00:00+0000"),
            ]
        ],
        columns=["start_recording", "end_recording"],
    )
    data.to_csv(missing_col_file, index=False)

    with pytest.raises(ValueError, match=r"Missing column provided"):
        RecordingPeriod.from_csv(missing_col_file)


def test_recording_from_config_empty_file(
    tmp_path: Path, sample_csv_result: DataFrame, recording_planning_csv: DataFrame
) -> None:
    empty_file = tmp_path / "recordings.csv"
    data = DataFrame(
        [],
        columns=[
            "start_recording",
            "end_recording",
            "start_deployment",
            "end_deployment",
        ],
    )
    data.to_csv(empty_file, index=False)
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": empty_file,
    }
    config = DataAploseConfig(**dict_test)

    with pytest.raises(ValueError, match=r"CSV is empty."):
        RecordingPeriod.from_config(config=config, bin_size=Timedelta("1h"))
