from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytest
from pandas import DataFrame, Timedelta
from pandas.tseries import frequencies

from post_processing.dataclass.data_aplose import DataAplose


def test_data_aplose_init(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    assert isinstance(data.df, DataFrame)
    assert sorted(data.annotators) == ["ann1", "ann2", "ann3", "ann4", "ann5", "ann6"]
    assert sorted(data.labels) == ["lbl1", "lbl2", "lbl3"]
    assert data.dataset == ["sample_dataset"]
    assert data.shape == sample_df.shape
    assert data.begin == sample_df["start_datetime"].min()
    assert data.end == sample_df["end_datetime"].max()


def test_filter_df_single_pair(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    filtered_data = data.filter_df(annotator="ann1", label="lbl1")
    assert sorted(set(filtered_data["annotation"])) == ["lbl1"]
    assert sorted(set(filtered_data["annotator"])) == ["ann1"]
    expected = sample_df[
        (sample_df["annotator"] == "ann1") &
        (sample_df["annotation"] == "lbl1")
    ].reset_index(drop=True)
    assert filtered_data.equals(expected)


def test_change_tz(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    new_tz = "Etc/GMT-7"
    data.change_tz(new_tz)
    start_dt = data.df["start_datetime"]
    end_dt = data.df["end_datetime"]
    assert all(ts.tz.zone == new_tz for ts in start_dt), f"The detection start timestamps have to be in {new_tz} timezone"
    assert all(ts.tz.zone == new_tz for ts in end_dt), f"The detection end timestamps have to be in {new_tz} timezone"
    assert data.begin.tz.zone == new_tz, f"The begin value of the DataAplose has to be in {new_tz} timezone"
    assert data.end.tz.zone == new_tz, f"The end value of the DataAplose has to be in {new_tz} timezone"


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
        match="DataFrame with annotator 'ann1' /"
        " label 'lbl3' contains no weak detection.",
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


def test_histo(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    fig, ax = plt.subplots()
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
    )


@pytest.mark.parametrize("mode", ["scatter", "heatmap", "timeline"])
def test_plot_scatter_heatmap_timeline(sample_df: DataFrame, mode: str) -> None:
    data = DataAplose(sample_df)
    data.lon = 0
    data.lat = 0
    bin_size = frequencies.to_offset("1d")
    fig, ax = plt.subplots()
    data.plot(mode=mode, ax=ax, annotator="ann1", label="lbl1", bin_size=bin_size, color="red")


def test_heatmap_wrong_bin(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    data.lon = 0
    data.lat = 0
    bins = frequencies.to_offset("10s")
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="`bin_size` must be >= 24h for heatmap mode."):
        data.plot(mode="heatmap", ax=ax, annotator="ann1", label="lbl1", bin_size=bins, color="red")


def test_plot_invalid_mode(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Unsupported plot mode"):
        data.plot("wrong_mode", ax, annotator="ann1", label="lbl1")


def test_plot_agreement(sample_df: DataFrame) -> None:
    data = DataAplose(sample_df)
    fig, ax = plt.subplots()
    data.plot(mode="agreement",
              ax=ax, annotator=["ann1", "ann2"],
              label="lbl1",
              bin_size=Timedelta("10s"),
              )


def test_set_ax(sample_df: DataFrame) -> None:
    da = DataAplose(sample_df)
    fig, ax = plt.subplots()
    ax = da.set_ax(ax, Timedelta("7h"), "%Y-%m-%d")
    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, mdates.HourLocator)


def test_from_yaml(
        sample_yaml: Path,
        sample_df: DataFrame,
) -> None:
    df_from_yaml = DataAplose.from_yaml(file=sample_yaml).df
    df_expected = DataAplose(sample_df).filter_df(annotator="ann1", label="lbl1").reset_index(drop=True)
    assert df_from_yaml.equals(df_expected)


def test_concat(sample_yaml: Path, sample_df: DataFrame) -> None:
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
