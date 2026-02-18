from __future__ import annotations

import csv
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import pytz
from pandas import DataFrame, Timedelta, Timestamp, concat, to_datetime

from post_processing.utils.filtering_utils import (
    ensure_no_invalid,
    filter_by_annotator,
    filter_by_freq,
    filter_by_label,
    filter_by_score,
    filter_by_time,
    filter_strong_detection,
    find_delimiter,
    get_annotators,
    get_canonical_tz,
    get_dataset,
    get_labels,
    get_max_freq,
    get_max_time,
    get_timezone,
    read_dataframe,
    reshape_timebin,
    intersection_or_union,
)


# %% find delimiter


@pytest.mark.parametrize(
    ("delimiter", "rows"),
    [
        (",", [["a", "b", "c"], ["1", "2", "3"]]),
        (";", [["x", "y", "z"], ["4", "5", "6"]]),
    ],
)
def test_find_delimiter_valid(
    tmp_path: Path,
    delimiter: str,
    rows: list[list[str]],
) -> None:
    file = tmp_path / "test.csv"
    with file.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(rows)

    detected = find_delimiter(file)
    assert detected == delimiter


def test_find_delimiter_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "bad.csv"
    file.write_text("a,b,c")

    def raise_error(*args, **kwargs):
        raise csv.Error("sniff failed")

    monkeypatch.setattr(csv.Sniffer, "sniff", raise_error)

    with pytest.raises(ValueError, match="Could not determine delimiter"):
        find_delimiter(file)


def test_find_delimiter_empty_file(tmp_path: Path) -> None:
    file = tmp_path / "empty.csv"
    file.write_text("")
    with pytest.raises(ValueError, match="Could not determine delimiter"):
        find_delimiter(file)


def test_find_delimiter_unsupported_delimiter(tmp_path: Path) -> None:
    file = tmp_path / "lame.csv"

    # '&' is consistent and sniffable, but not allowed
    file.write_text("a&b&c\n1&2&3\n")

    with pytest.raises(
        ValueError,
        match=r"unsupported delimiter '&'",
    ):
        find_delimiter(file)


# %% filter utils


# filter_by_time
@pytest.mark.parametrize(
    "begin, end",
    [
        pytest.param(
            Timestamp("2020-01-01", tz="utc"),
            None,
            id="valid_begin_only",
        ),
        pytest.param(
            None,
            Timestamp("2030-01-01", tz="utc"),
            id="valid_end_only",
        ),
        pytest.param(
            Timestamp("2020-01-01", tz="utc"),
            Timestamp("2030-01-01", tz="utc"),
            id="valid_begin_and_end",
        ),
    ],
)
def test_filter_by_time_valid(sample_df: DataFrame, begin, end):
    result = filter_by_time(sample_df, begin=begin, end=end)

    assert not result.empty
    if begin is not None:
        assert (result["start_datetime"] >= begin).all()
    if end is not None:
        assert (result["end_datetime"] <= end).all()


@pytest.mark.parametrize(
    "begin, end, expected_msg",
    [
        pytest.param(
            Timestamp("2050-01-01", tz="utc"),
            None,
            "No detection found after '2050",
            id="out_of_range_begin",
        ),
        pytest.param(
            None,
            Timestamp("1900-01-01", tz="utc"),
            "No detection found before '1900",
            id="out_of_range_end",
        ),
    ],
)
def test_filter_by_time_out_of_range(sample_df: DataFrame, begin, end, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        filter_by_time(sample_df, begin=begin, end=end)


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


@pytest.mark.parametrize(
    "f_min, f_max",
    [
        pytest.param(
            500,  # valid lower bound
            None,
            id="valid_f_min_only",
        ),
        pytest.param(
            None,
            60000,  # valid upper bound
            id="valid_f_max_only",
        ),
        pytest.param(
            500,
            60000,
            id="valid_f_min_and_f_max",
        ),
    ],
)
def test_filter_by_freq_valid(sample_df: DataFrame, f_min, f_max):
    result = filter_by_freq(sample_df, f_min=f_min, f_max=f_max)

    assert not result.empty

    if f_min is not None:
        assert (result["start_frequency"] >= f_min).all()
    if f_max is not None:
        assert (result["end_frequency"] <= f_max).all()


@pytest.mark.parametrize(
    "f_min, f_max, expected_msg",
    [
        pytest.param(
            1e8,
            None,
            "No detection found above",
            id="out_of_range_f_min",
        ),
        pytest.param(
            None,
            1,
            "No detection found below",
            id="out_of_range_f_max",
        ),
    ],
)
def test_filter_by_freq_out_of_range(sample_df: DataFrame, f_min, f_max, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        filter_by_freq(sample_df, f_min=f_min, f_max=f_max)


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


# filter_weak_strong_detection
def test_filter_weak_only_is_box_colum(sample_df: DataFrame) -> None:
    df = filter_strong_detection(sample_df)
    assert set(df["type"]) == {"WEAK"}


def test_filter_weak_only_type_column(sample_df: DataFrame) -> None:
    df = filter_strong_detection(sample_df)
    assert set(df["type"]) == {"WEAK"}


def test_filter_weak_only_invalid() -> None:
    invalid_df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    with pytest.raises(ValueError, match="Could not determine annotation type"):
        filter_strong_detection(invalid_df)


def test_filter_weak_empty(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="No weak detection found"):
        filter_strong_detection(sample_df[sample_df["type"] == "BOX"])


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


@pytest.mark.parametrize(
    "input_tz, expected",
    [
        (pytz.timezone("Europe/Samara"), pytz.timezone("Europe/Samara")),
        (ZoneInfo("Pacific/Pago_Pago"), pytz.timezone("Pacific/Pago_Pago")),
    ],
)
def test_get_canonical_tz_valid(input_tz, expected):
    result = get_canonical_tz(input_tz)
    assert result == expected


def test_get_canonical_tz_raises_on_unknown():
    class DummyTZ:
        pass

    dummy = DummyTZ()
    with pytest.raises(TypeError) as exc:
        get_canonical_tz(dummy)

    assert "Unknown timezone" in str(exc.value)


def test_get_timezone_single(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    assert tz == pytz.utc


def test_get_timezone_several(sample_df: DataFrame) -> None:
    new_row = {
        "dataset": "dataset",
        "filename": "2025_01_26_06_20_00",
        "start_time": 0,
        "end_time": 2,
        "start_frequency": 100,
        "end_frequency": 200,
        "annotation": "annotation",
        "annotator": "annotator",
        "start_datetime": Timestamp("2025-01-27 06:00:00.000000+07:00"),
        "end_datetime": Timestamp("2025-01-27 06:00:00.000000+07:00"),
        "is_box": 1,
        "score": None,
    }
    sample_df = concat(
        [sample_df, DataFrame([new_row])],
        ignore_index=False,
    )
    tz = get_timezone(sample_df)
    assert len(tz) == 2
    assert pytz.UTC in tz
    assert pytz.FixedOffset(420) in tz


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
        "2025-01-01 13:00:00,2025-01-01 13:05:00,\n",  # NaN annotation
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

    df = read_dataframe(csv_file, rows=1)
    assert len(df) == 1
    assert df.iloc[0]["annotation"] in {"whale", "dolphin"}


# %% reshape_timebin


def test_no_timebin_returns_original(sample_df: DataFrame) -> None:
    df_out = reshape_timebin(sample_df, timebin_new=None, timestamp_audio=None)
    assert df_out.equals(sample_df)


def test_no_timebin_several_tz(sample_df: DataFrame) -> None:
    new_row = {
        "dataset": "dataset",
        "filename": "2025_01_26_06_20_00",
        "start_time": 0,
        "end_time": 2,
        "start_frequency": 100,
        "end_frequency": 200,
        "annotation": "annotation",
        "annotator": "annotator",
        "start_datetime": Timestamp("2025-01-27 06:00:00.000000+07:00"),
        "end_datetime": Timestamp("2025-01-27 06:00:00.000000+07:00"),
        "is_box": 1,
        "score": None,
    }
    sample_df = concat(
        [sample_df, DataFrame([new_row])],
        ignore_index=False,
    )
    timestamp_wav = to_datetime(
        sample_df["filename"], format="%Y_%m_%d_%H_%M_%S"
    ).dt.tz_localize(pytz.UTC)
    df_out = reshape_timebin(sample_df, timestamp_audio=timestamp_wav, timebin_new=None)
    assert df_out.equals(sample_df)


def test_no_timebin_original_timebin(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    timestamp_wav = to_datetime(
        sample_df["filename"],
        format="%Y_%m_%d_%H_%M_%S",
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
            "start_datetime": [Timestamp("2025-01-25 06:20:00+00:00")] * 11
            + [Timestamp("2025-01-26 06:20:00+00:00")] * 7,
            "end_datetime": [Timestamp("2025-01-25 06:21:00+00:00")] * 11
            + [Timestamp("2025-01-26 06:21:00+00:00")] * 7,
            "type": ["WEAK"] * 18,
        },
    )

    assert df_out.equals(expected)


def test_simple_reshape_hourly(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    timestamp_wav = to_datetime(
        sample_df["filename"],
        format="%Y_%m_%d_%H_%M_%S",
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
        format="%Y_%m_%d_%H_%M_%S",
    ).dt.tz_localize(tz)
    df_out = reshape_timebin(
        sample_df, timestamp_audio=timestamp_wav, timebin_new=Timedelta(days=1)
    )
    assert not df_out.empty
    assert all(df_out["end_time"] == 86400.0)
    assert df_out["start_datetime"].min() >= sample_df["start_datetime"].min().floor(
        "D"
    )
    assert df_out["end_datetime"].max() <= sample_df["end_datetime"].max().ceil("D")


def test_with_manual_timestamps_vector(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    timestamp_wav = to_datetime(
        sample_df["filename"], format="%Y_%m_%d_%H_%M_%S"
    ).dt.tz_localize(tz)
    df_out = reshape_timebin(
        sample_df,
        timestamp_audio=timestamp_wav,
        timebin_new=Timedelta(hours=1),
    )

    assert not df_out.empty
    assert all(isinstance(t, Timestamp) for t in df_out["start_datetime"])
    assert df_out["end_time"].iloc[0] == 3600.0


def test_empty_result_when_no_matching(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    timestamp_wav = to_datetime(
        sample_df["filename"], format="%Y_%m_%d_%H_%M_%S"
    ).dt.tz_localize(tz)
    with pytest.raises(ValueError, match="DataFrame is empty"):
        reshape_timebin(
            DataFrame(), timestamp_audio=timestamp_wav, timebin_new=Timedelta(hours=1)
        )


# %% ensure_no_invalid


def test_ensure_no_invalid_empty() -> None:
    try:
        ensure_no_invalid([], "label")
    except ValueError:
        pytest.fail("ensure_no_invalid raised ValueError unexpectedly.")


def test_ensure_no_invalid_with_elements() -> None:
    invalid_items = ["foo", "bar"]
    with pytest.raises(ValueError) as exc_info:
        ensure_no_invalid(invalid_items, "columns")

    assert "foo" in str(exc_info.value)
    assert "bar" in str(exc_info.value)
    assert "columns" in str(exc_info.value)


def test_ensure_no_invalid_single_element() -> None:
    invalid_items = ["baz"]
    with pytest.raises(ValueError) as exc_info:
        ensure_no_invalid(invalid_items, "features")
    assert "baz" in str(exc_info.value)
    assert "features" in str(exc_info.value)


# %% intersection / union


def test_intersection(sample_df: DataFrame) -> None:
    df_result = intersection_or_union(
        sample_df[sample_df["annotator"].isin(["ann1", "ann2"])],
        user_sel="intersection",
    )

    assert set(df_result["annotation"]) == {"lbl1 ∩ lbl2"}
    assert set(df_result["annotator"]) == {"ann1 ∩ ann2"}


def test_union(sample_df: DataFrame) -> None:
    df_result = intersection_or_union(
        sample_df[sample_df["annotator"].isin(["ann1", "ann2"])], user_sel="union"
    )

    assert set(df_result["annotation"]) == {"lbl1 ∪ lbl2"}
    assert set(df_result["annotator"]) == {"ann1 ∪ ann2"}


def test_all_user_sel_returns_original(sample_df: DataFrame) -> None:
    df_result = intersection_or_union(sample_df, user_sel="all")

    assert len(df_result) == len(sample_df)


def test_invalid_user_sel_raises(sample_df: DataFrame) -> None:
    with pytest.raises(
        ValueError, match="'user_sel' must be either 'intersection' or 'union'"
    ):
        intersection_or_union(sample_df, user_sel="invalid")


def test_not_enough_annotators_raises() -> None:
    df_single_annotator = DataFrame({
        "annotation": ["cat"],
        "start_datetime": to_datetime(["2025-01-01 10:00"]),
        "end_datetime": to_datetime(["2025-01-01 10:01"]),
        "annotator": ["A"],
    })
    with pytest.raises(ValueError, match="Not enough annotators detected"):
        intersection_or_union(df_single_annotator, user_sel="intersection")
