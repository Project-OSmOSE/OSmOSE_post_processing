import csv
from datetime import timezone, UTC, datetime
from pathlib import Path
import yaml

import pytest
from pandas import DataFrame, Timedelta, Timestamp, to_datetime, testing, Series

from post_processing.utils.filtering_utils import (
    filter_by_annotator,
    filter_by_freq,
    filter_by_label,
    filter_by_score,
    filter_by_time,
    find_delimiter,
    get_annotators,
    get_dataset,
    get_labels,
    get_max_freq,
    get_max_time,
    get_timezone,
    read_dataframe,
    reshape_timebin, get_filename_timestamps, check_timestamp,
)

# %% find delimiter


@pytest.mark.parametrize(("delimiter", "rows"), [
    (",", [["a", "b", "c"], ["1", "2", "3"]]),
    (";", [["x", "y", "z"], ["4", "5", "6"]]),
])
def test_find_delimiter_valid(tmp_path: Path,
                              delimiter: str,
                              rows: list[list[str]],
                              ) -> None:
    file = tmp_path / "test.csv"
    with file.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(rows)

    detected = find_delimiter(file)
    assert detected == delimiter


def test_find_delimiter_invalid(tmp_path: Path) -> None:
    file = tmp_path / "invalid.csv"
    file.write_text("this is not really&csv&content")
    with pytest.raises(ValueError, match="Could not determine delimiter"):
        find_delimiter(file)


def test_find_delimiter_empty_file(tmp_path: Path) -> None:
    file = tmp_path / "empty.csv"
    file.write_text("")
    with pytest.raises(ValueError, match="Could not determine delimiter"):
        find_delimiter(file)


# %% filter utils

# filter_by_time
def test_filter_by_time_begin(sample_df: DataFrame) -> None:
    ts = sample_df["start_datetime"].iloc[4]
    df = filter_by_time(sample_df, begin=ts, end=None)
    assert (df["start_datetime"] >= ts).all()


def test_filter_by_time_end(sample_df: DataFrame) -> None:
    ts = sample_df["end_datetime"].iloc[4]
    df = filter_by_time(sample_df, begin=None, end=ts)
    assert (df["end_datetime"] <= ts).all()


def test_filter_by_time_out_of_range(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="No detection found after '2050"):
        filter_by_time(sample_df, begin=Timestamp("2050-01-01", tz="utc"), end=None)


# filter_by_annotator
def test_filter_by_annotator_string(sample_df: DataFrame) -> None:
    df = filter_by_annotator(sample_df, "ann1")
    assert (df["annotator"] == "ann1").all()


def test_filter_by_annotator_list(sample_df: DataFrame) -> None:
    df = filter_by_annotator(sample_df, ["ann2"])
    assert set(df["annotator"]) == {"ann2"}


def test_filter_by_annotator_invalid(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="not present in annotator, upload aborted"):
        filter_by_annotator(sample_df, "BbJuni")


# filter_by_label
def test_filter_by_label_string(sample_df: DataFrame) -> None:
    df = filter_by_label(sample_df, "lbl1")
    assert set(df["annotation"]) == {"lbl1"}


def test_filter_by_label_list(sample_df: DataFrame) -> None:
    df = filter_by_label(sample_df, ["lbl2"])
    assert set(df["annotation"]) == {"lbl2"}


def test_filter_by_label_invalid(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="not present in label, upload aborted"):
        filter_by_label(sample_df, "hihi")


# filter_by_freq
def test_filter_by_freq_min(sample_df: DataFrame) -> None:
    freq_min = 500
    df = filter_by_freq(sample_df, f_min=freq_min, f_max=None)
    assert (df["start_frequency"] >= freq_min).all()


def test_filter_by_freq_max(sample_df: DataFrame) -> None:
    freq_max = 60000
    df = filter_by_freq(sample_df, f_min=None, f_max=freq_max)
    assert (df["end_frequency"] <= freq_max).all()


def test_filter_by_freq_no_results(sample_df: DataFrame) -> None:
    freq_min = 144000
    with pytest.raises(ValueError, match=f"No detection found above {int(freq_min)}Hz"):
        filter_by_freq(sample_df, f_min=freq_min, f_max=None)


# filter_by_score
def test_filter_by_score_valid(sample_df: DataFrame) -> None:
    df = filter_by_score(sample_df, 0.5)
    assert (df["score"] >= 0.5).all()


def test_filter_by_score_no_results(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="No detection found with score above 1.0"):
        filter_by_score(sample_df, 1.0)


def test_filter_by_score_missing_column(sample_df: DataFrame) -> None:
    df = sample_df.drop(columns=["score"])
    with pytest.raises(ValueError, match="'score' column not present"):
        filter_by_score(df, 0.5)


def test_get_annotators(sample_df: DataFrame) -> None:
    annotators = get_annotators(sample_df)
    expected = sorted(set(sample_df["annotator"]))
    assert annotators == expected


def test_get_labels(sample_df: DataFrame) -> None:
    labels = get_labels(sample_df)
    expected = sorted(set(sample_df["annotation"]))
    assert labels == expected


def test_get_max_freq(sample_df: DataFrame) -> None:
    assert get_max_freq(sample_df) == sample_df["end_frequency"].max()


def test_get_max_time(sample_df: DataFrame) -> None:
    assert get_max_time(sample_df) == sample_df["end_time"].max()


def test_get_dataset(sample_df: DataFrame) -> None:
    assert get_dataset(sample_df) == sample_df["dataset"].iloc[0]


def test_get_timezone_single(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    assert isinstance(tz, timezone)
    assert tz == UTC


def test_get_filename_timestamp(sample_df: DataFrame, sample_yaml: Path) -> None:
    tz = get_timezone(sample_df)
    with sample_yaml.open(encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    sample_key = next(iter(data_yaml.keys()))
    date_parser = data_yaml[sample_key]["filename_format"]
    assert isinstance(to_datetime(sample_df["filename"][0],
                                  format=date_parser).tz_localize(tz),
                      datetime)
    results = get_filename_timestamps(sample_df, date_parser)
    expected_timestamps = [
        to_datetime(
            ts,
            format=date_parser
        ).tz_localize(tz) for ts in sample_df["filename"]
        ]
    assert len(results) == len(sample_df)
    assert all(isinstance(ts, Timestamp) for ts in results)
    testing.assert_series_equal(Series(results), Series(expected_timestamps),
                                check_names=False)


def test_check_timestamp_none(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="`timestamp_wav` is empty"):
        check_timestamp(sample_df, None)


def test_check_timestamp_wrong_length(sample_df: DataFrame) -> None:
    len_sample_df = len(sample_df) + 1
    timestamps = [
        Timestamp("2025-01-01") + Timedelta(days=i)
        for i in range(len_sample_df)
    ]  # shorter list
    with pytest.raises(ValueError, match="`timestamp_wav` is not the same length as `df`"):
        check_timestamp(sample_df, timestamps)
# %% read DataFrame


def test_read_dataframe_comma_delimiter(tmp_path: Path) -> None:
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "start_datetime,end_datetime,annotation\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"
        "2025-01-01 13:00:00,2025-01-01 13:05:00,dolphin\n",
    )

    df = read_dataframe(csv_file)

    assert list(df.columns) == ["start_datetime", "end_datetime", "annotation"]
    assert len(df) == 2
    assert df.iloc[0]["annotation"] == "whale"


def test_read_dataframe_drop_duplicates_and_na(tmp_path: Path) -> None:
    csv_file = tmp_path / "test_duplicates.csv"
    csv_file.write_text(
        "start_datetime,end_datetime,annotation\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"  # duplicate
        "2025-01-01 13:00:00,2025-01-01 13:05:00,\n",      # NaN annotation
    )

    df = read_dataframe(csv_file)
    assert len(df) == 1


def test_read_dataframe_sorted_by_datetime(tmp_path: Path) -> None:
    csv_file = tmp_path / "test_unsorted.csv"
    csv_file.write_text(
        "start_datetime,end_datetime,annotation\n"
        "2025-01-01 14:00:00,2025-01-01 14:05:00,dolphin\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n",
    )

    df = read_dataframe(csv_file)
    assert list(df["annotation"]) == ["whale", "dolphin"]


def test_read_dataframe_nrows(tmp_path: Path) -> None:
    csv_file = tmp_path / "test_nrows.csv"
    csv_file.write_text(
        "start_datetime,end_datetime,annotation\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"
        "2025-01-01 13:00:00,2025-01-01 13:05:00,dolphin\n",
    )

    df = read_dataframe(csv_file, nrows=1)
    assert len(df) == 1
    assert df.iloc[0]["annotation"] in {"whale", "dolphin"}

# %% reshape_timebin


def test_no_timebin_returns_original(sample_df: DataFrame) -> None:
    df_out = reshape_timebin(sample_df, timebin_new=None)
    assert df_out.equals(sample_df)


def test_no_timebin_original_timebin(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    timestamp_wav = to_datetime(
        sample_df["filename"],
        format="%Y_%m_%d_%H_%M_%S"
    ).dt.tz_localize(tz)
    df_out = reshape_timebin(
        sample_df,
        timestamp_audio=timestamp_wav,
        timebin_new=Timedelta("1min"),
    )
    expected = DataFrame(
        {
            "dataset": ["sample_dataset"] * 18,
            "filename": [
                "2025_01_25_06_20_00",
                "2025_01_25_06_20_00",
                "2025_01_25_06_20_00",
                "2025_01_25_06_20_00",
                "2025_01_25_06_20_00",
                "2025_01_25_06_20_00",
                "2025_01_25_06_20_00",
                "2025_01_25_06_20_10",
                "2025_01_25_06_20_10",
                "2025_01_25_06_20_40",
                "2025_01_25_06_20_30",
                "2025_01_26_06_20_00",
                "2025_01_26_06_20_00",
                "2025_01_26_06_20_00",
                "2025_01_26_06_20_00",
                "2025_01_26_06_20_00",
                "2025_01_26_06_20_00",
                "2025_01_26_06_20_00",
            ],
            "start_time": [0] * 18,
            "end_time": [60.0] * 18,
            "start_frequency": [0] * 18,
            "end_frequency": [72_000.0] * 18,
            "annotation": [
                "lbl1",
                "lbl2",
                "lbl1",
                "lbl2",
                "lbl2",
                "lbl2",
                "lbl1",
                "lbl2",
                "lbl3",
                "lbl1",
                "lbl2",
                "lbl1",
                "lbl2",
                "lbl1",
                "lbl2",
                "lbl2",
                "lbl2",
                "lbl1",

            ],
            "annotator": [
                "ann1",
                "ann1",
                "ann2",
                "ann2",
                "ann3",
                "ann4",
                "ann5",
                "ann5",
                "ann5",
                "ann6",
                "ann6",
                "ann1",
                "ann1",
                "ann2",
                "ann2",
                "ann3",
                "ann4",
                "ann5",

            ],
            "start_datetime": [Timestamp("2025-01-25 06:20:00+00:00")] * 11 +
                              [Timestamp("2025-01-26 06:20:00+00:00")] * 7,
            "end_datetime": [Timestamp("2025-01-25 06:21:00+00:00")] * 11 +
                            [Timestamp("2025-01-26 06:21:00+00:00")] * 7,
            "is_box": [0] * 18,
        },
    )

    assert df_out.equals(expected)


def test_simple_reshape_hourly(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    timestamp_wav = to_datetime(
        sample_df["filename"],
        format="%Y_%m_%d_%H_%M_%S"
    ).dt.tz_localize(tz)
    df_out = reshape_timebin(
        sample_df,
        timestamp_audio=timestamp_wav,
        timebin_new=Timedelta(hours=1),
    )
    assert not df_out.empty
    assert all(df_out["end_time"] == 3600.0)
    assert df_out["end_frequency"].max() == sample_df["end_frequency"].max()
    assert set(df_out["annotation"]) <= set(sample_df["annotation"])
    assert set(df_out["annotator"]) <= set(sample_df["annotator"])


def test_reshape_daily_multiple_bins(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    timestamp_wav = to_datetime(
        sample_df["filename"],
        format="%Y_%m_%d_%H_%M_%S"
    ).dt.tz_localize(tz)
    df_out = reshape_timebin(
        sample_df,
        timestamp_audio=timestamp_wav,
        timebin_new=Timedelta(days=1),
    )
    assert not df_out.empty
    assert all(df_out["end_time"] == 86400.0)
    assert df_out["start_datetime"].min() >= sample_df["start_datetime"].min().floor("D")
    assert df_out["end_datetime"].max() <= sample_df["end_datetime"].max().ceil("D")


def test_empty_result_when_no_matching(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="DataFrame is empty"):
        reshape_timebin(DataFrame())