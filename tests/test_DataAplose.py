import logging
import re
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from typing import ContextManager

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytest
import pytz
from pandas import DataFrame, Timedelta, Timestamp
from pandas.tseries import frequencies

from disclose.dataclass.data_aplose import DataAplose, _get_locator_from_offset
from disclose.utils.filtering import get_timezone


def test_data_aplose_str(sample_df: DataFrame) -> None:
    obj = DataAplose(sample_df)

    expected = (
        f"start_datetime: {obj.start_datetime}\n"
        f"end_datetime: {obj.end_datetime}\n"
        f"annotators: {obj.annotators}\n"
        f"labels: {obj.labels}\n"
        f"dataset: {obj.dataset}"
    )
    assert str(obj) == expected
    assert repr(obj) == str(obj)


def test_data_aplose_init(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    assert isinstance(data.df, DataFrame)
    assert sorted(data.annotators) == ["ann1", "ann2", "ann3", "ann4", "ann5", "ann6"]
    assert sorted(data.labels) == ["lbl1", "lbl2", "lbl3"]
    assert data.dataset == ["sample_dataset"]
    assert data.shape == sample_df.shape
    assert data.start_datetime == sample_df["start_datetime"].min()
    assert data.end_datetime == sample_df["end_datetime"].max()


@pytest.mark.parametrize(
    ("value", "expected_msg"),
    [
        pytest.param((48.39, -4.49), None, id="valid"),
        pytest.param((0.0, 0.0), None, id="valid_zero"),
        pytest.param(
            [48.39, -4.49],
            re.escape("Coordinates must be a tuple of two floats: (lat, lon)."),
            id="invalid_list",
        ),
        pytest.param(
            (48.39,),
            re.escape("Coordinates must be a tuple of two floats: (lat, lon)."),
            id="invalid_missing_lon",
        ),
        pytest.param(
            (48, 5, 6),
            re.escape("Coordinates must be a tuple of two floats: (lat, lon)."),
            id="invalid_long_tuple",
        ),
    ],
)
def test_coordinates_setter(value: tuple, expected_msg: str | None) -> None:
    obj = DataAplose.__new__(DataAplose)

    if expected_msg is None:
        obj.coordinates = value
        assert (obj.lat, obj.lon) == value
    else:
        with pytest.raises(ValueError, match=expected_msg):
            obj.coordinates = value


def test_filter_df_single_pair(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    filtered_data = data.filter_df(annotator="ann1", label="lbl1")
    assert sorted(set(filtered_data["annotation"])) == ["lbl1"]
    assert sorted(set(filtered_data["annotator"])) == ["ann1"]
    expected = sample_df[
        (sample_df["annotator"] == "ann1") & (sample_df["annotation"] == "lbl1")
    ].reset_index(drop=True)
    assert filtered_data.equals(expected)


def test_change_tz(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    new_tz = "Etc/GMT-7"
    data.change_tz(new_tz)
    start_dt = data.df["start_datetime"]
    end_dt = data.df["end_datetime"]
    assert all(ts.tz.zone == new_tz for ts in start_dt), (
        f"The detection start timestamps have to be in {new_tz} timezone"
    )
    assert all(ts.tz.zone == new_tz for ts in end_dt), (
        f"The detection end timestamps have to be in {new_tz} timezone"
    )
    assert data.start_datetime.tz.zone == new_tz, (
        f"The begin value of the DataAplose has to be in {new_tz} timezone"
    )
    assert data.end_datetime.tz.zone == new_tz, (
        f"The end value of the DataAplose has to be in {new_tz} timezone"
    )


def test_filter_df_multiple_pairs(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    filtered_data = data.filter_df(annotator=["ann1", "ann2"], label=["lbl1", "lbl2"])
    assert sorted(set(filtered_data["annotation"])) == ["lbl1", "lbl2"]
    assert sorted(set(filtered_data["annotator"])) == ["ann1", "ann2"]
    pairs = [("ann1", "lbl1"), ("ann2", "lbl2")]
    expected = sample_df[
        sample_df[["annotator", "annotation"]].apply(tuple, axis=1).isin(pairs)
    ].reset_index(drop=True)
    assert filtered_data.equals(expected)


def test_filter_df_invalid_annotator(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    with pytest.raises(
        ValueError,
        match='Annotator "bbjuni" not in APLOSE DataFrame',
    ):
        data.filter_df(annotator="bbjuni", label="lbl1")


def test_filter_df_invalid_label(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    with pytest.raises(
        ValueError,
        match='Label "cool" not in APLOSE DataFrame',
    ):
        data.filter_df(annotator="ann2", label="cool")


def test_filter_df_invalid_combination(
    sample_df: DataFrame,
) -> None:
    data = DataAplose(sample_df)
    with pytest.raises(
        ValueError,
        match=r"DataFrame with annotator 'ann1' /"
        " label 'lbl3' contains no detection.",
    ):
        data.filter_df(annotator="ann1", label="lbl3")


def test_filter_df_invalid_lists_size(
    sample_df: DataFrame,
) -> None:
    data = DataAplose(sample_df)
    with pytest.raises(
        ValueError,
        match=r"Length of annotator \(2\) and label \(1\) must match.",
    ):
        data.filter_df(annotator=["ann1", "ann2"], label=["lbl2"])


def test_getitem(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    assert all(data[0] == sample_df.iloc[0])


def test_set_ax_uses_2hour_locator(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    fig, ax = plt.subplots()
    tick_freq = frequencies.to_offset("2h")

    ax = data.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )

    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, mdates.HourLocator)
    assert locator._get_interval() == 2  # noqa: PLR2004


def test_histo(sample_csv_result: DataFrame, recording_planning_csv: DataFrame) -> None:
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": recording_planning_csv,
    }
    data = DataAplose.from_dict(dict_test)
    _, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")
    ax = data.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )
    data.plot(
        mode="histogram",
        ax=ax,
        annotator="ann1",
        label="lbl1",
        bin_size=bin_size,
        effort=True,
    )


def test_histo_no_recording_file_provided(
    sample_csv_result: DataFrame, recording_planning_csv: DataFrame
) -> None:
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": None,
    }
    data = DataAplose.from_dict(dict_test)
    _, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")
    ax = data.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )
    with pytest.raises(ValueError, match=r"No recording file provided."):
        data.plot(
            mode="histogram",
            ax=ax,
            annotator="ann1",
            label="lbl1",
            bin_size=bin_size,
            effort=True,
        )


def test_histo_no_recording_fake_file_provided(
    sample_csv_result: DataFrame, recording_planning_csv: DataFrame
) -> None:
    dict_test = {
        "detection_file": sample_csv_result,
        "filename_format": "%Y_%m_%d_%H_%M_%S",
        "recording_file": Path("fake_file.csv"),
    }
    data = DataAplose.from_dict(dict_test)
    _, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")
    ax = data.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )
    with pytest.raises(FileNotFoundError, match=r"File not found"):
        data.plot(
            mode="histogram",
            ax=ax,
            annotator="ann1",
            label="lbl1",
            bin_size=bin_size,
            effort=True,
        )


@pytest.mark.parametrize("mode", ["scatter", "heatmap", "timeline"])
def test_plot_scatter_heatmap_timeline(sample_df: DataFrame, mode: str) -> None:
    data = DataAplose(sample_df)
    data.lon = 0
    data.lat = 0
    bin_size = frequencies.to_offset("1d")
    fig, ax = plt.subplots()
    data.plot(
        mode=mode, ax=ax, annotator="ann1", label="lbl1", bin_size=bin_size, color="red"
    )


def test_heatmap_wrong_bin(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    data.lon = 0
    data.lat = 0
    bins = frequencies.to_offset("10s")
    _, ax = plt.subplots()
    with pytest.raises(
        ValueError, match=r"`bin_size` must be >= 24h for heatmap mode."
    ):
        data.plot(
            mode="heatmap",
            ax=ax,
            annotator="ann1",
            label="lbl1",
            bin_size=bins,
            color="red",
        )


def test_plot_invalid_mode(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Unsupported plot mode"):
        data.plot("wrong_mode", ax, annotator="ann1", label="lbl1")


def test_plot_histogram_missing_bin_size(monkeypatch, sample_df: DataFrame) -> None:
    obj = DataAplose(sample_df)

    monkeypatch.setattr(obj, "filter_df", lambda annotator, label: sample_df)
    monkeypatch.setattr(
        "disclose.dataclass.data_aplose.histo", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "disclose.dataclass.data_aplose.get_count", lambda *args, **kwargs: None
    )

    _, ax = plt.subplots()

    with pytest.raises(ValueError, match=r"'bin_size' missing for histogram plot."):
        obj.plot(
            mode="histogram",
            ax=ax,
            annotator="ann1",
            label="lbl1",
        )


def test_plot_agreement_missing_bin_size(monkeypatch, sample_df: DataFrame) -> None:
    obj = DataAplose(sample_df)

    monkeypatch.setattr(obj, "filter_df", lambda annotator, label: sample_df)

    _, ax = plt.subplots()

    with pytest.raises(ValueError, match=r"'bin_size' missing for agreement plot."):
        obj.plot(
            mode="agreement",
            ax=ax,
            annotator="ann1",
            label="lbl1",
            bin_size=None,
        )


def test_plot_agreement(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    fig, ax = plt.subplots()
    data.plot(
        mode="agreement",
        ax=ax,
        annotator=["ann1", "ann2"],
        label="lbl1",
        bin_size=Timedelta("10s"),
    )


def test_set_ax(sample_df: DataFrame) -> None:
    da = DataAplose(sample_df)
    _, ax = plt.subplots()
    ax = da.set_ax(ax, Timedelta("7h"), "%Y-%m-%d")
    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, mdates.HourLocator)


def test_from_dict(
    sample_dict: dict,
    sample_df: DataFrame,
) -> None:
    df_from_dict = DataAplose.from_dict(sample_dict).df
    df_expected = (
        DataAplose(sample_df)
        .filter_df(annotator="ann1", label="lbl1")
        .reset_index(drop=True)
    )
    assert df_from_dict.equals(df_expected)


def test_from_dict_concat(
    sample_dict: dict,
    sample_df: DataFrame,
) -> None:
    data_expected = DataAplose.from_dict(sample_dict)
    start = data_expected.start_datetime
    end = data_expected.end_datetime
    median_time = start + (end - start) / 2
    config = [
        {
            "detection_file": sample_dict["detection_file"],
            "end_datetime": median_time,
            "filename_format": sample_dict["filename_format"],
            "annotator": sample_dict["annotator"],
            "annotation": sample_dict["annotation"],
        },
        {
            "detection_file": sample_dict["detection_file"],
            "start_datetime": median_time,
            "filename_format": sample_dict["filename_format"],
            "annotator": sample_dict["annotator"],
            "annotation": sample_dict["annotation"],
        },
    ]
    data_concat = DataAplose.from_dict(config, concat=True)

    assert all(data_expected.df == data_concat.df)
    assert repr(data_expected) == repr(data_concat)


def test_from_dict_no_concat(
    sample_dict: dict,
    sample_df: DataFrame,
) -> None:
    data_expected = DataAplose.from_dict(sample_dict)
    start = data_expected.start_datetime
    end = data_expected.end_datetime
    median_time = start + (end - start) / 2
    config = [
        {
            "detection_file": sample_dict["detection_file"],
            "end_datetime": median_time,
            "filename_format": sample_dict["filename_format"],
            "annotator": sample_dict["annotator"],
            "annotation": sample_dict["annotation"],
        },
        {
            "detection_file": sample_dict["detection_file"],
            "start_datetime": median_time,
            "filename_format": sample_dict["filename_format"],
            "annotator": sample_dict["annotator"],
            "annotation": sample_dict["annotation"],
        },
    ]
    data_concat = DataAplose.from_dict(config, concat=False)

    assert isinstance(data_concat, list)
    assert len(data_concat) == len(config)
    assert [isinstance(data, DataAplose) for data in data_concat]


def test_concatenate(sample_dict: Path, sample_df: DataFrame) -> None:
    data1 = DataAplose(sample_df.loc[: len(sample_df) / 2])
    data2 = DataAplose(sample_df.loc[len(sample_df) / 2 :])

    data_concat = DataAplose.concatenate([data1, data2])
    expected = DataAplose(sample_df)

    attrs = [
        name
        for name in dir(expected)
        if not name.startswith("_") and not callable(getattr(expected, name))
    ]

    for attr in attrs:
        got = getattr(data_concat, attr)
        exp = getattr(expected, attr)

        if isinstance(exp, DataFrame):
            assert got.equals(exp), f"Mismatch in {attr}"
        else:
            assert got == exp, f"Mismatch in {attr}"


def test_concatenate_change_tz(sample_df: DataFrame, caplog) -> None:
    data1 = DataAplose(sample_df.loc[: len(sample_df) / 2])
    data1.change_tz(pytz.timezone("Etc/GMT-7"))
    data2 = DataAplose(sample_df.loc[len(sample_df) / 2 :])

    with caplog.at_level(logging.INFO):
        data_concat = DataAplose.concatenate([data1, data2])

    assert get_timezone(data_concat.df) == pytz.utc
    assert (
        "Several timezones found in DataFrame, all timestamps are converted to UTC."
        in caplog.text
    )


# %% Overview


def test_data_aplose_overview(monkeypatch, sample_df: DataFrame) -> None:
    obj = DataAplose(sample_df)

    called = {}

    def fake_overview(df: DataFrame, annotator: str) -> None:
        called["df"] = df
        called["annotator"] = annotator

    monkeypatch.setattr(
        "disclose.dataclass.data_aplose.overview",
        fake_overview,
    )

    annotator = ["ann1"]
    obj.overview(annotator)

    assert called["df"] is obj.df
    assert called["annotator"] == annotator


@pytest.mark.parametrize(
    ("annotators", "labels", "expected_ref"),
    [
        pytest.param(
            ("ann1", "ann2"),
            "lbl1",
            ("ann1", "lbl1"),
            id="annotators_tuple_labels_str",
        ),
        pytest.param(
            "ann1",
            ("lbl1", "lbl2"),
            ("ann1", "lbl1"),
            id="annotators_str_labels_tuple",
        ),
    ],
)
def test_data_aplose_detection_perf_wrapper_parametrized(
    monkeypatch,
    sample_df: DataFrame,
    annotators: tuple[str, str] | str,
    labels: tuple[str, str] | str,
    expected_ref: tuple[str, str],
) -> None:
    obj = DataAplose(sample_df[sample_df["type"] == "WEAK"])

    called = {}

    def fake_detection_perf(
        df: DataFrame, ref: tuple[str, str], time
    ) -> tuple[float, float, float]:
        called["df"] = df
        called["ref"] = ref
        called["time"] = time
        return (0.1, 0.2, 0.3)

    monkeypatch.setattr(
        "disclose.dataclass.data_aplose.detection_perf",
        fake_detection_perf,
    )

    result = obj.detection_perf(
        annotators=annotators,
        labels=labels,
    )

    assert result == (0.1, 0.2, 0.3)
    assert called["ref"] == expected_ref


def test_detection_perf_multiple_timebins(sample_df: DataFrame) -> None:
    obj = DataAplose(sample_df)

    with pytest.raises(ValueError, match="Multiple time bins detected"):
        obj.detection_perf(
            annotators=("ann1", "ann2"),
            labels=("lbl1", "lbl2"),
        )


# %% Reshape


@pytest.mark.parametrize(
    ("begin", "end", "expected"),
    [
        pytest.param(
            Timestamp("2025-01-26T06:20:09.999+00:00"),
            None,
            nullcontext(),
            id="new_begin_after_original_end",
        ),
        pytest.param(
            None,
            Timestamp("2025-01-25T06:20:00.001+00:00"),
            nullcontext(),
            id="new_end_before_original_begin",
        ),
        pytest.param(
            Timestamp("2024-12-31"),
            Timestamp("2024-01-01"),
            pytest.raises(
                ValueError, match=r"Begin timestamp is not anterior than end timestamp."
            ),
            id="begin_after_end_inverted_range",
        ),
        pytest.param(
            Timestamp("2050-01-01", tz="UTC"),
            Timestamp("2050-12-31", tz="UTC"),
            nullcontext(),
            id="tz_aware_future_range_no_data",
        ),
        pytest.param(
            Timestamp("1990-01-01", tz="America/New_York"),
            Timestamp("1990-12-31", tz="America/New_York"),
            nullcontext(),
            id="tz_aware_past_range_no_data",
        ),
    ],
)
def test_reshape_errors(
    sample_data_aplose: DataAplose,
    begin: Timestamp | None,
    end: Timestamp | None,
    expected: ContextManager[Exception],
) -> None:
    """Test that reshape function handles error cases appropriately."""
    with expected:
        sample_data_aplose.reshape(begin, end)


@pytest.mark.parametrize(
    ("begin", "end", "should_filter"),
    [
        pytest.param(
            None,
            None,
            False,
            id="no_timestamps_provided",
        ),
        pytest.param(
            Timestamp("1990-01-01", tz="UTC"),
            None,
            True,
            id="tz_aware_begin_only",
        ),
        pytest.param(
            None,
            Timestamp("2050-12-31", tz="UTC"),
            True,
            id="tz_aware_end_only",
        ),
        pytest.param(
            Timestamp("2025-01-24", tz="Europe/Paris"),
            Timestamp("2025-01-27", tz="Europe/Paris"),
            True,
            id="tz_aware_both_timestamps",
        ),
        pytest.param(
            Timestamp("1990-01-01"),
            None,
            True,
            id="tz_naive_begin_only",
        ),
        pytest.param(
            None,
            Timestamp("2050-12-31"),
            True,
            id="tz_naive_end_only",
        ),
        pytest.param(
            Timestamp("1990-01-01"),
            Timestamp("2050-12-31"),
            True,
            id="tz_naive_both_timestamps",
        ),
        pytest.param(
            Timestamp("1990-01-01", tz="Europe/Paris"),
            Timestamp("2050-12-31", tz="America/New_York"),
            True,
            id="tz_aware_different_timezone",
        ),
        pytest.param(
            Timestamp("2025-01-26T00:00:00.000+00:00"),
            Timestamp("2025-01-26T10:00:00.000+00:00"),
            True,
            id="narrowing_detections",
        ),
    ],
)
def test_reshape_valid_cases(
    sample_data_aplose: DataAplose,
    begin: Timestamp | None,
    end: Timestamp | None,
    should_filter: bool,
) -> None:
    """Test that reshape handles valid timestamp cases appropriately."""
    reshaped = copy(sample_data_aplose)
    reshaped.reshape(start_datetime=begin, end_datetime=end)

    original_tz = get_timezone(sample_data_aplose.df)

    # Check timestamps were updated
    if begin is not None:
        if begin.tz is None:
            begin = begin.tz_localize(original_tz)
        assert reshaped.start_datetime != sample_data_aplose.start_datetime
        assert reshaped.start_datetime == begin

    if end is not None:
        if end.tz is None:
            end = end.tz_localize(original_tz)
        assert reshaped.end_datetime != sample_data_aplose.end_datetime
        assert reshaped.end_datetime == end

    # Check timezone was applied for tz-naive timestamps
    assert reshaped.start_datetime.tz is not None
    assert reshaped.end_datetime.tz is not None

    # Check filtering behavior
    if should_filter:
        assert reshaped.shape <= sample_data_aplose.shape
        assert all(reshaped.df["start_datetime"] >= reshaped.start_datetime)
        assert all(reshaped.df["end_datetime"] <= reshaped.end_datetime)


# %%


def test_get_locator_from_offset(monkeypatch) -> None:
    class FakeSecond:
        def __init__(self, interval) -> None:
            self.interval = interval

    class FakeMinute:
        def __init__(self, interval) -> None:
            self.interval = interval

    monkeypatch.setattr(mdates, "SecondLocator", FakeSecond)
    monkeypatch.setattr(mdates, "MinuteLocator", FakeMinute)

    @pytest.mark.parametrize(
        "offset, expected, interval",
        [
            (5, FakeSecond, 5),
            (Timedelta("45s"), FakeSecond, 45),
            (Timedelta("120s"), FakeMinute, 2),
        ],
    )
    def check_valid(offset, expected, interval):
        loc = _get_locator_from_offset(offset)
        assert isinstance(loc, expected)
        assert loc.interval == interval

    for offset, expected, interval in [
        (5, FakeSecond, 5),
        (Timedelta("45s"), FakeSecond, 45),
        (Timedelta("120s"), FakeMinute, 2),
    ]:
        loc = _get_locator_from_offset(offset)
        assert isinstance(loc, expected)
        assert loc.interval == interval


def test_get_locator_from_offset_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported offset type"):
        _get_locator_from_offset(3.14)
