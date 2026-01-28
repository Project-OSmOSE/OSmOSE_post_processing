import matplotlib as mpl
import pytest
from numpy import arange, linspace

from post_processing.utils.glider_utils import (
    compute_acoustic_diversity,
    export_gpx,
    get_position_from_timestamp,
    load_glider_nav,
    plot_detections_with_nav_data,
    set_trajectory,
)

mpl.use("Agg")  # no GUI for plots

from pathlib import Path

from pandas import DataFrame, Timedelta, date_range

from post_processing.dataclass.trajectory import Trajectory


@pytest.fixture
def nav_df() -> DataFrame:
    return DataFrame({
        "Timestamp": date_range("2023-01-01", periods=5, freq="1h", tz="UTC"),
        "Lat": linspace(10, 15, 5),
        "Lon": linspace(20, 25, 5),
        "Depth": arange(5),
        "NavState": [1, 2, 2, 3, 3],
    })


@pytest.fixture
def df_detections(nav_df: DataFrame) -> DataFrame:
    return DataFrame({
        "annotation": ["whale", "whale", "dolphin"],
        "is_box": [0, 0, 0],
        "start_datetime": nav_df["Timestamp"][:3],
    })


def test_set_trajectory(nav_df: DataFrame) -> None:
    traj = set_trajectory(nav_df)
    assert isinstance(traj, Trajectory)
    assert len(traj.timestamps) == len(nav_df)


def test_get_position_from_timestamp(nav_df: DataFrame) -> None:
    traj = set_trajectory(nav_df)
    times = list(nav_df["Timestamp"])
    lat, lon, ts = get_position_from_timestamp(traj, times)
    assert len(lat) == len(times)
    assert all(isinstance(x, float) for x in lat)


def test_plot_detections_with_nav_data(
        df_detections: DataFrame,
        nav_df: DataFrame,
) -> None:
    plot_detections_with_nav_data(
        df=df_detections,
        nav=nav_df,
        criterion="Depth",
        ticks=Timedelta(seconds=3600),
    )


def test_load_glider_nav() -> None:
    input_dir = Path(__file__).parent.parent / "user_case" / "resource" / "OHAGEODAMS_nav"
    df = load_glider_nav(input_dir)
    assert isinstance(df, DataFrame)
    assert "Lat" in df.columns
    assert not df.empty


def test_load_glider_nav_missing_dir(tmp_path: Path) -> None:
    bad_dir = tmp_path / "doesnotexist"
    with pytest.raises(FileNotFoundError):
        load_glider_nav(bad_dir)


def test_load_glider_nav_no_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_glider_nav(tmp_path)


def test_compute_acoustic_diversity(df_detections: DataFrame,
                                    nav_df: DataFrame) -> None:
    time_vector = list(nav_df["Timestamp"])
    result = compute_acoustic_diversity(df_detections, nav_df, time_vector)
    assert isinstance(result, DataFrame)
    assert "Acoustic Diversity" in result.columns


def test_export_gpx(nav_df: DataFrame, tmp_path: Path) -> None:
    out_file = tmp_path / "trace.gpx"
    export_gpx(nav_df, tmp_path, "trace")
    assert out_file.exists()
    content = out_file.read_text()
    assert "<gpx" in content
