from unittest.mock import patch

import pytest
from matplotlib import pyplot as plt
from pandas import DataFrame, Timedelta, Timestamp, date_range
from pandas.tseries import frequencies
from pytz import timezone

from post_processing.dataclass.data_aplose import DataAplose
from post_processing.utils.core_utils import (
    add_recording_period,
    add_season_period,
    add_weak_detection,
    get_coordinates,
    get_count,
    get_labels_and_annotators,
    get_season,
    get_sun_times,
    get_time_range_and_bin_size,
    json2df,
    localize_timestamps,
    round_begin_end_timestamps,
    set_bar_height,
    timedelta_to_str,
)


def test_coordinates_valid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = ["42", "-71"]

    def fake_box(msg: str, title: str, fields: list[str]) -> list[str]:
        return inputs
    monkeypatch.setattr("easygui.multenterbox", fake_box)
    lat, lon = get_coordinates()
    assert lat == 42  # noqa: PLR2004
    assert lon == -71  # noqa: PLR2004


def test_coordinates_cancelled_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_box(msg: str, title: str, fields: list[str]) -> None:
        return None
    monkeypatch.setattr("easygui.multenterbox", fake_box)
    with pytest.raises(TypeError, match="was cancelled"):
        get_coordinates()


def test_coordinates_invalid_then_valid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = [["900", "50"], ["45", "900"], ["45", "100"]]

    def fake_box(msg: str, title: str, fields: list[str]) -> list[str]:
        return inputs.pop(0)
    monkeypatch.setattr("easygui.multenterbox", fake_box)
    lat, lon = get_coordinates()
    assert lat == 45.0  # noqa: PLR2004
    assert lon == 100.0  # noqa: PLR2004


def test_coordinates_non_numeric_input(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = [["abc", "-20"], ["-20", "abc"], ["10", "-20"]]

    def fake_box(msg: str, title: str, fields: list[str]) -> list[str]:
        return inputs.pop(0)
    monkeypatch.setattr("easygui.multenterbox", fake_box)
    lat, lon = get_coordinates()
    assert lat == 10.0  # noqa: PLR2004
    assert lon == -20.0  # noqa: PLR2004


@pytest.mark.parametrize(
    ("ts", "northern", "expected"),
    [
        # Northern hemisphere
        (Timestamp("2025-03-15"), True, ("spring", 2025)),
        (Timestamp("2025-06-21"), True, ("summer", 2025)),
        (Timestamp("2025-09-01"), True, ("autumn", 2025)),
        (Timestamp("2025-01-10"), True, ("winter", 2024)),
        (Timestamp("2025-02-28"), True, ("winter", 2024)),
        (Timestamp("2024-02-29"), True, ("winter", 2023)),
        (Timestamp("2025-12-25"), True, ("winter", 2025)),

        # Southern hemisphere
        (Timestamp("2025-03-15"), False, ("autumn", 2025)),
        (Timestamp("2025-06-21"), False, ("winter", 2025)),
        (Timestamp("2025-09-01"), False, ("spring", 2025)),
        (Timestamp("2025-01-10"), False, ("summer", 2024)),
        (Timestamp("2025-02-28"), False, ("summer", 2024)),
        (Timestamp("2024-02-29"), False, ("summer", 2023)),
        (Timestamp("2025-12-25"), False, ("summer", 2025)),
    ],
)
def test_get_season(ts: Timestamp, northern: bool, expected: tuple[str, int]) -> None:
    assert get_season(ts, northern=northern) == expected


@pytest.mark.parametrize(
    ("start", "stop", "lat", "lon"),
    [
        (Timestamp("2025-06-01 00:00:00+00:00"),
         Timestamp("2025-06-03 23:59:59+00:00"),
         49.4333,
         -1.5167),
        (Timestamp("2025-12-21 00:00:00+00:00"),
         Timestamp("2025-12-22 23:59:59+00:00"),
         -34.9011,
         -56.1645),
    ],
)
def test_get_sun_times_valid_input(start: Timestamp,
                                   stop: Timestamp,
                                   lat: float,
                                   lon: float,
                                   ) -> None:
    results = get_sun_times(start, stop, lat, lon)
    h_sunrise, h_sunset = results

    n_days = (stop.normalize() - start.normalize()).days + 1

    assert all(len(lst) == n_days for lst in results)
    assert all(isinstance(h, float) for h in h_sunrise)
    assert all(isinstance(h, float) for h in h_sunset)

    for sunrise, sunset in zip(h_sunrise, h_sunset, strict=False):
        assert 0 <= sunrise <= 24  # noqa: PLR2004
        assert 0 <= sunset <= 24  # noqa: PLR2004
        assert sunrise < sunset


@pytest.mark.parametrize(
    ("start", "stop", "lat", "lon"),
    [
        (Timestamp("2025-06-01 00:00:00"),
         Timestamp("2025-06-03 23:59:59"),
         49.4333,
         -1.5167),
    ],
)
def test_get_sun_times_naive_timestamps(
    start: Timestamp,
    stop: Timestamp,
    lat: float,
    lon: float,
) -> None:
    with pytest.raises(ValueError, match="start and stop must be timezone-aware"):
        get_sun_times(start, stop, lat, lon)


# %% get_count

def test_get_count_basic(sample_df: DataFrame) -> None:
    df = DataAplose(sample_df).filter_df(annotator="ann1", label="lbl1")
    result = get_count(df, bin_size=Timedelta("30min"))
    expected = sample_df[
        (sample_df["annotator"] == "ann1") &
        (sample_df["annotation"] == "lbl1")
    ]
    assert list(result.index) == date_range(
        Timestamp("2025-01-25 06:00:00+0000"),
        Timestamp("2025-01-26 06:00:00+0000"),
        freq="30min",
    ).to_list()
    assert result.columns == ["lbl1-ann1"]
    assert sum(result["lbl1-ann1"].tolist()) == len(expected)


def test_get_count_multiple_annotators(sample_df: DataFrame) -> None:
    df = DataAplose(sample_df).filter_df(annotator=["ann1", "ann2"], label="lbl1")
    result = get_count(df, bin_size=Timedelta("1d"))
    expected = sample_df[
        (sample_df["annotator"].isin(["ann1", "ann2"])) &
        (sample_df["annotation"] == "lbl1")
        ]

    assert set(result.columns) == {"lbl1-ann1", "lbl1-ann2"}
    assert result["lbl1-ann1"].sum() == len(expected[expected["annotator"] == "ann1"])
    assert result["lbl1-ann2"].sum() == len(expected[expected["annotator"] == "ann2"])


def test_get_count_multiple_labels(sample_df: DataFrame) -> None:
    df = DataAplose(sample_df).filter_df(annotator="ann5", label=["lbl1", "lbl2", "lbl3"])
    result = get_count(df, bin_size=Timedelta("1day"))
    expected = sample_df[
        (sample_df["annotator"] == "ann5") &
        (sample_df["annotation"].isin(["lbl1", "lbl2", "lbl3"]))
        ]

    assert set(result.columns) == {"lbl1-ann5", "lbl2-ann5", "lbl3-ann5"}
    assert result["lbl1-ann5"].sum() == len(expected[expected["annotation"] == "lbl1"])
    assert result["lbl2-ann5"].sum() == len(expected[expected["annotation"] == "lbl2"])
    assert result["lbl3-ann5"].sum() == len(expected[expected["annotation"] == "lbl3"])


def test_get_count_multiple_labels_annotators(sample_df: DataFrame) -> None:
    df = DataAplose(sample_df).filter_df(annotator=["ann1", "ann2"],
                                         label=["lbl1", "lbl2"],
                                         )
    result = get_count(df, bin_size=Timedelta("1day"))
    assert set(result.columns) == {"lbl1-ann1", "lbl2-ann2"}
    assert result["lbl1-ann1"].sum() == len(sample_df[(sample_df["annotation"] == "lbl1") & (sample_df["annotator"] == "ann1")])
    assert result["lbl2-ann2"].sum() == len(sample_df[(sample_df["annotation"] == "lbl2") & (sample_df["annotator"] == "ann2")])


def test_get_count_empty_df() -> None:
    with pytest.raises(ValueError, match="`df` contains no data"):
        get_count(DataFrame(), Timedelta("1h"))

# %% get_labels_and_annotators


def test_get_labels_and_annotators_valid_entry(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df).filter_df(annotator="ann1", label="lbl1")
    labels, annotators = get_labels_and_annotators(data)
    assert labels == ["lbl1"]
    assert annotators == ["ann1"]


def test_single_label_multiple_annotators(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df).filter_df(annotator=["ann1", "ann2"], label="lbl1")
    labels, annotators = get_labels_and_annotators(data)
    assert sorted(labels) == ["lbl1"] * 2
    assert sorted(annotators) == ["ann1", "ann2"]


def test_single_annotator_multiple_labels(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df).filter_df(annotator="ann1", label=["lbl1", "lbl2"])
    labels, annotators = get_labels_and_annotators(data)
    assert sorted(labels) == ["lbl1", "lbl2"]
    assert sorted(annotators) == ["ann1"] * 2


def test_mismatched_labels_and_annotators() -> None:
    df = DataFrame({
        "annotation": ["lbl1", "lbl2", "lbl3"],
        "annotator": ["ann1", "ann2", "ann2"],
    })
    with pytest.raises(ValueError, match="annotators and .* labels must match"):
        get_labels_and_annotators(df)


def test_get_labels_and_annotators_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="`df` contains no data"):
        get_labels_and_annotators(DataFrame())

# %% localize_timestamps


def test_localize_all_naive() -> None:
    tz = timezone("Europe/Paris")
    timestamps = [Timestamp("2025-01-01 12:00:00"), Timestamp("2025-01-02 13:00:00")]
    localized = localize_timestamps(timestamps, tz)
    for ts in localized:
        assert ts.tzinfo is not None
        assert ts.tz.zone == tz.zone


def test_already_localized() -> None:
    tz = timezone("Europe/Paris")
    timestamps = [
        Timestamp("2025-01-01 12:00:00", tz=tz),
        Timestamp("2025-01-02 13:00:00", tz=tz),
    ]
    localized = localize_timestamps(timestamps, tz)
    assert localized == timestamps


def test_mixed_naive_and_aware() -> None:
    tz = timezone("Europe/Paris")
    timestamps = [
        Timestamp("2025-01-01 12:00:00"),
        Timestamp("2025-01-02 13:00:00", tz=tz),
    ]
    localized = localize_timestamps(timestamps, tz)
    assert localized[0].tzinfo.zone == tz.zone
    assert localized[1].tzinfo.zone == tz.zone

# %% get_time_range_and_bin_size


def test_time_range_timedelta() -> None:
    timestamps = [Timestamp("2025-08-20 12:00:00"), Timestamp("2025-08-20 14:30:00")]
    bin_size = Timedelta("1h")
    time_range, returned_bin = get_time_range_and_bin_size(timestamps, bin_size)

    expected = date_range(start="2025-08-20 12:00:00", end="2025-08-20 15:00:00", freq="1h")
    assert (time_range == expected).all()
    assert returned_bin == bin_size


def test_time_range_baseoffset() -> None:
    timestamps = [Timestamp("2025-08-20 12:00:00"), Timestamp("2025-08-20 14:30:00")]
    bin_size = frequencies.to_offset("1h")
    time_range, returned_bin = get_time_range_and_bin_size(timestamps, bin_size)

    expected = date_range(start="2025-08-20 12:00:00", end="2025-08-20 15:00:00", freq="1h")
    assert (time_range == expected).all()
    assert returned_bin == bin_size


def test_empty_timestamp_list() -> None:
    timestamps = []
    bin_size = Timedelta("1h")
    with pytest.raises(ValueError, match="`timestamp_list` is empty"):
        get_time_range_and_bin_size(timestamps, bin_size)


def test_invalid_timestamp_list_type() -> None:
    timestamps = "not_a_list"
    bin_size = Timedelta("1h")
    with pytest.raises(TypeError, match=r"`timestamp_list` must be a list\[Timestamp\]"):
        get_time_range_and_bin_size(timestamps, bin_size)


def test_invalid_timestamp_list_content() -> None:
    timestamps = [Timestamp("2025-08-20"), "not_a_timestamp"]
    bin_size = Timedelta("1h")
    with pytest.raises(TypeError, match=r"`timestamp_list` must be a list\[Timestamp\]"):
        get_time_range_and_bin_size(timestamps, bin_size)

# %% round_begin_end_timestamps


def test_round_begin_end_timestamps_empty_list() -> None:
    with pytest.raises(ValueError, match="`timestamp_list` is empty"):
        round_begin_end_timestamps([], Timedelta("1h"))


def test_round_begin_end_timestamps_invalid_entry() -> None:
    ts_list = [
        Timestamp("2025-01-01 00:15:00"),
        Timestamp("2025-01-01 01:45:00"),
    ]
    with pytest.raises(ValueError, match="Could not get start/end timestamps."):
        round_begin_end_timestamps(ts_list, "not_a_valid_entry")


def test_round_begin_end_timestamps_invalid_entry_2() -> None:
    with pytest.raises(TypeError, match=r"timestamp_list must be a list\[Timestamp\]"):
        round_begin_end_timestamps("not_a_valid_entry", Timedelta("1h"))


def test_round_begin_end_timestamps_valid_entry() -> None:
    ts_list = [
        Timestamp("2025-01-01 00:15:00"),
        Timestamp("2025-01-01 01:45:00"),
    ]
    start, end, bin_size = round_begin_end_timestamps(ts_list, Timedelta("1h"))

    assert start == Timestamp("2025-01-01 00:00:00")
    assert end == Timestamp("2025-01-01 02:00:00")
    assert bin_size == Timedelta("1h")


def test_round_begin_end_timestamps_valid_entry_2() -> None:
    ts_list = [
        Timestamp("2025-01-01 10:15:00"),
        Timestamp("2025-01-03 18:45:00"),
    ]
    start, end, bin_size = round_begin_end_timestamps(ts_list, frequencies.to_offset("1h"))

    assert start == Timestamp("2025-01-01 10:00:00")
    assert end == Timestamp("2025-01-03 19:00:00")
    assert bin_size == Timedelta("1h")

# %% timedelta_to_str


@pytest.mark.parametrize(
    ("td", "expected"),
    [
        (Timedelta(days=2), "2D"),
        (Timedelta(hours=5), "5h"),
        (Timedelta(minutes=30), "30min"),
        (Timedelta(seconds=45), "45s"),
        (Timedelta(days=1, hours=2), "26h"),
        (Timedelta(minutes=90), "90min"),
    ],
)
def test_timedelta_to_str(td, expected) -> None:
    assert timedelta_to_str(td) == expected


# %% add_weak_detection / json2df


def test_add_wd(sample_df: DataFrame) -> None:
    df_only_wd = sample_df[sample_df["type"] == "BOX"]
    add_weak_detection(
        df=df_only_wd.copy(),
        datetime_format="%Y_%m_%d_%H_%M_%S",
    )


# %% add_season_period

def test_add_season_valid() -> None:
    _, ax = plt.subplots()
    start = Timestamp("2025-01-01")
    stop = Timestamp("2026-01-01")
    freq = Timedelta("1d")
    ts = date_range(start=start, end=stop, freq=freq, tz="UTC")
    values = [date.day for date in ts]
    [ax.bar(loc + freq, height) for loc, height in zip(ts, values, strict=True)]
    add_season_period(ax=ax)


def test_add_season_no_data() -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"have no data"):
        add_season_period(ax=ax)

# %% add_recording_period


def test_add_recording_period_valid() -> None:
    fig, ax = plt.subplots()
    start = Timestamp("2025-01-01T00:00:00+00:00")
    stop = Timestamp("2025-01-02T00:00:00+00:00")

    ts = date_range(start=start, end=stop, freq="H", tz="UTC")
    values = list(range(len(ts)))
    ax.plot(ts, values)

    df = DataFrame(
        data=[
            [
                Timestamp("2025-01-01T00:00:00+00:00"),
                Timestamp("2025-01-02T00:00:00+00:00"),
            ],
        ],
        columns=["deployment_date", "recovery_date"],
    )
    add_recording_period(df=df, ax=ax)


def test_add_recording_period_no_data() -> None:
    fig, ax = plt.subplots()
    df = DataFrame()
    with pytest.raises(ValueError, match=r"have no data"):
        add_recording_period(df=df, ax=ax)

# %% set_bar_height


def test_set_bar_height_valid() -> None:
    fig, ax = plt.subplots()
    start = Timestamp("2025-01-01T00:00:00+00:00")
    stop = Timestamp("2025-01-02T00:00:00+00:00")

    ts = date_range(start=start, end=stop, freq="H", tz="UTC")
    values = list(range(len(ts)))
    ax.plot(ts, values)

    set_bar_height(ax=ax)


def test_set_bar_height_no_data() -> None:
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"have no data"):
        set_bar_height(ax=ax)

# %% json2df


def test_json2df_valid(tmp_path):
    fake_json = {
        "deployment_date": "2025-01-01T00:00:00+00:00",
        "recovery_date": "2025-01-02T00:00:00+00:00",
    }

    json_file = tmp_path / "metadatax.json"
    json_file.write_text("{}", encoding="utf-8")

    with patch("json.load", return_value=fake_json):
        df = json2df(json_file)

    expected = DataFrame(
        data=[
            [
                Timestamp("2025-01-01T00:00:00+00:00"),
                Timestamp("2025-01-02T00:00:00+00:00"),
            ],
        ],
        columns=["deployment_date", "recovery_date"],
    )

    assert df.equals(expected)
