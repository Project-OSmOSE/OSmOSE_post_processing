"""Functions to exploit glider data."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import gpxpy
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Timedelta, Timestamp, concat
from tqdm import tqdm

from post_processing.dataclass.trajectory import Trajectory
from post_processing.glider_config import NAV_STATE

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.lib.npyio import NpzFile


def set_trajectory(nav: pd.DataFrame) -> Trajectory:
    """Create a Trajectory object in order to track data.

    Parameters
    ----------
    nav: DataFrame,
        Navigation data with GPS positions and associated datetimes

    Returns
    -------
    Trajectory object

    """
    traj = Trajectory()
    for _, row in tqdm(nav.iterrows(), total=len(nav), desc="Setting trajectory"):
        traj.add_position(
            timestamp=row["Timestamp"].timestamp(),
            latitude=row["Lat"],
            longitude=row["Lon"],
        )
    return traj


def get_position_from_timestamp(
    traj: Trajectory,
    time_vector: list[Timestamp],
) -> (list[float], list[float], list[float]):
    """Compute location for at a given datetime for a given trajectory.

    Parameters
    ----------
    traj: Trajectory
        Approximate trajectory data using polynomial fitting
    time_vector:  list
        Timestamps to associate a location to

    Returns
    -------
    (latitude, longitude, timestamp)

    """
    time_vector_unix = [ts.timestamp() for ts in time_vector]
    latitudes, longitudes, timestamps = [], [], []

    for ts in time_vector_unix:
        if traj.timestamps.min() <= ts <= traj.timestamps.max():
            lat, lon = traj.get_position(ts)
            latitudes.append(lat)
            longitudes.append(lon)
            timestamps.append(ts)

    return latitudes, longitudes, timestamps


def plot_detections_with_nav_data(
    df: DataFrame,
    nav: DataFrame,
    criterion: str,
    ticks: Timedelta,
    datetime_format: str = "%d/%m/%y",
) -> None:
    """Plot detections of all annotation types according to a navigation data criterion.

    Parameters
    ----------
    df: DataFrame
        APLOSE formatted detection file
    nav: DataFrame
        Navigation data comprised of criteria (latitude, longitude, depth...)
        and associated datetimes
    criterion: str
        User selected navigation parameter from nav (latitude, longitude, depth...)
    ticks: Timedelta
        Resolution of the x-axis major ticks.
    datetime_format : str
        Date format string for x-axis tick labels (e.g., "%b", "%Y-%m-%d %H:%M").

    """
    fig, ax = plt.subplots()
    labels = df["annotation"].unique()

    for annotation in labels:
        df_1label = df[(df["annotation"] == annotation) & (df["is_box"] == 0)]

        glider_timestamps_numeric = [int(ts.timestamp()) for ts in nav["Timestamp"]]
        detections_timestamps_numeric = [
            int(ts.timestamp()) for ts in df_1label["start_datetime"]
        ]
        matching_depths = np.interp(
            detections_timestamps_numeric,
            glider_timestamps_numeric,
            nav[criterion].astype("float"),
        )

        plt.scatter(
            df_1label["start_datetime"],
            matching_depths,
            label=annotation,
            zorder=2,
            s=8,
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    plt.plot(
        nav["Timestamp"],
        nav[criterion],
        label=criterion,
        zorder=1,
        linewidth=0.5,
        color="grey",
    )

    locator = mdates.SecondLocator(interval=int(ticks.total_seconds()))
    ax.xaxis.set_major_locator(locator)

    formatter = mdates.DateFormatter(datetime_format)
    ax.xaxis.set_major_formatter(formatter)

    plt.grid(color="k", linestyle="--", linewidth=0.2, zorder=0)
    plt.xlim(nav["Timestamp"].iloc[0], nav["Timestamp"].iloc[-1])
    plt.ylabel(criterion)
    plt.tight_layout()


def load_glider_nav(directory: Path) -> DataFrame:
    """Load the navigation data from glider output files in a specified directory.

    This function searches the given directory for compressed `.gz` files
    containing the substring 'gli' in their filenames.

    Parameters
    ----------
    directory : Path
        The path to the directory containing the glider output files.

    Returns
    -------
    DataFrame
        Combined navigation data from all matching glider output files.
        The DataFrame includes:
        - Columns from the original CSV files;
        - A 'yo' column representing the file number extracted from the filename;
        - A 'file' column indicating the source file for each row;
        - A 'Lat DD' column for latitude in decimal degrees;
        - A 'Lon DD' column for longitude in decimal degrees;
        - A 'Depth' column with depth values adjusted to be positive.

    """
    if not directory.exists():
        msg = f"Directory '{directory}' does not exist."
        raise FileNotFoundError(msg)

    file = [f for f in directory.rglob("*.gz") if "gli" in f.name]

    if not len(file) > 0:
        msg = f"Directory '{directory}' does not contain '.gz' files."
        raise FileNotFoundError(msg)

    all_rows = []  # Initialize an empty list to store the contents of all CSV files
    yo = []  # List to store the file numbers
    data = []

    first_file = True
    file_number = 1  # Initialize the file number

    for f in tqdm(file, desc="Reading navigation data", unit="file"):
        with gzip.open(f, "rt") as gz_file:
            delimiter = ";"  # Specify the desired delimiter
            gz_reader = pd.read_csv(gz_file, delimiter=delimiter)
            # If it's the first file, append the header row
            if first_file:
                all_rows.append(gz_reader.columns.tolist())
                first_file = False
            # Add the rows from the current CSV file to the all_rows list
            all_rows.extend(gz_reader.to_numpy().tolist())
            # Add yo number to the yo list
            yo.extend([str(f).split(".")[-2]] * len(gz_reader))
            data.extend([f.name] * len(gz_reader))
            file_number += 1  # Increment the file number for the next file

    # Create a DataFrame from the combined data
    df_nav = pd.DataFrame(all_rows)
    df_nav.columns = df_nav.iloc[0]  # set 1st row as headers
    df_nav = df_nav.iloc[1:, 0:-1]  # delete last column and 1st row

    # Add the yo number to the DataFrame
    df_nav["yo"] = [int(x) for x in yo]

    df_nav["file"] = data
    df_nav = df_nav.drop(
        df_nav[(df_nav["Lat"] == 0) & (df_nav["Lon"] == 0)].index,
    ).reset_index(drop=True)

    df_nav["Lat"] = [
        int(x) + (((x - int(x)) / 60) * 100) if not np.isnan(x) else np.nan
        for x in df_nav["Lat"] / 100
    ]
    df_nav["Lon"] = [
        int(x) + (((x - int(x)) / 60) * 100) if not np.isnan(x) else np.nan
        for x in df_nav["Lon"] / 100
    ]
    df_nav["Depth"] = -df_nav["Depth"]
    df_nav["NavState"] = df_nav["NavState"].replace(NAV_STATE)
    df_nav["Timestamp"] = [
        pd.to_datetime(ts, format="%d/%m/%Y %H:%M:%S").tz_localize("UTC")
        for ts in df_nav["Timestamp"]
    ]

    return df_nav.sort_values(by=["Timestamp"]).reset_index(drop=True)


def plot_nav_state(df: DataFrame, npz: NpzFile) -> None:
    """Plot the LTAS from a npz file and the associated glider state of navigation.

    Parameters
    ----------
    df: pd.DataFrame
        Glider navigation data

    npz: NpzFile,
        Npz file containing the LTAS matrix, its associated frequency and time vectors

    """
    f = npz["Freq"]
    sxx = npz["LTAS"]
    t = npz["time"]

    fig, (ax, cax) = plt.subplots(2, gridspec_kw={"height_ratios": [1, 0.05]})
    im = ax.imshow(
        sxx[1:-1],
        aspect="auto",
        extent=[t[0], t[-1], f[1], f[-1]],
        origin="lower",
        vmin=40,
        vmax=100,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt="%H:%M"))
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=3600 * 4))
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Date")

    cbar = fig.colorbar(im, orientation="horizontal", cax=cax)
    cbar.ax.set_xlabel("dB ref 1µPa/Hz")

    ax2 = ax.twinx()
    ax2.plot(
        df[(df["Timestamp"] >= t[0]) & (df["Timestamp"] <= t[-1])]["Timestamp"],
        df[(df["Timestamp"] >= t[0]) & (df["Timestamp"] <= t[-1])]["NavState"],
        color="white",
        linewidth=1,
        alpha=0.7,
        zorder=3,
    )
    ax2.set_ylabel("Navigation state")

    plt.tight_layout()


def compute_acoustic_diversity(
    df: DataFrame,
    nav: DataFrame,
    time_vector: list[Timestamp],
) -> DataFrame:
    """Compute the number of different annotations at given positions and timestamps.

    Parameters
    ----------
    df: DataFrame
        APLOSE formatted result file
    nav: DataFrame
        Navigation data comprised of positions and associated timestamps
    time_vector: list[Timestamp]
        List of timestamps used to check for annotations from df.
        For APLOSE user, this can be constructed from task status files.

    Returns
    -------
    DataFrame comprised of timestamps, associated position and acoustic diversity

    """
    # track_data: glider positions at every timestamp
    track_data = nav[["Timestamp", "Lat", "Lon", "Depth"]]

    # compute trajectory object from glider navigation data
    trajectory = set_trajectory(nav=track_data)

    # compute localisation of each detection
    df_acoustic_diversity = DataFrame(
        columns=["Timestamp", "Latitude", "Longitude", "Acoustic Diversity"],
    )

    lat_time_vector, lon_time_vector, ts_time_vector = get_position_from_timestamp(
        traj=trajectory,
        time_vector=time_vector,
    )

    # delete duplicate detection in case several users annotated the same segment
    det = df.drop_duplicates(subset=["annotation", "start_datetime"])

    # unix time of detections
    time_det_unix = [ts.timestamp() for ts in det["start_datetime"]]

    acoustic_diversity = np.zeros(len(time_vector), dtype=int)
    for i, ts in enumerate(time_vector[: -(len(time_vector) - len(ts_time_vector))]):
        for ts_det in time_det_unix:
            if (
                ts.timestamp() <= ts_det <= ts.timestamp() + 1
                and trajectory.timestamps.min() <= ts_det <= trajectory.timestamps.max()
            ):
                acoustic_diversity[i] += 1

        new_row = DataFrame(
            {
                "Timestamp": [ts],
                "Latitude": [lat_time_vector[i]],
                "Longitude": [lon_time_vector[i]],
                "Acoustic Diversity": [acoustic_diversity[i]],
            },
        )

        df_acoustic_diversity = (
            new_row
            if df_acoustic_diversity.empty
            else concat([df_acoustic_diversity, new_row], ignore_index=True)
        )

    return df_acoustic_diversity


def export_gpx(nav: DataFrame, output_dir: Path, output_file: str = "trace") -> None:
    """Export a navigation DataFrame to a GPX file.

    Creates a GPX track from latitude, longitude, depth, and timestamp values
    contained in a navigation DataFrame. A waypoint is added at the start position.

    Parameters
    ----------
    nav : DataFrame
        Navigation data with required columns: ["Lat", "Lon", "Depth", "Timestamp"].
    output_dir : Path
        Directory where the GPX file will be saved.
    output_file : str, optional
        Base name of the GPX file (default is "trace").

    """
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name="trace")
    segment = gpxpy.gpx.GPXTrackSegment()

    for _, row in nav.iterrows():
        point = gpxpy.gpx.GPXTrackPoint(
            float(row["Lat"]),
            float(row["Lon"]),
            float(row["Depth"]),
            time=row["Timestamp"],
        )
        segment.points.append(point)

    track.segments.append(segment)
    gpx.tracks.append(track)

    waypoint = gpxpy.gpx.GPXWaypoint(
        latitude=nav["Lat"].iloc[0],
        longitude=nav["Lon"].iloc[0],
        name="trace_start",
    )
    gpx.waypoints.append(waypoint)

    with (output_dir / (output_file + ".gpx")).open("w") as f:
        f.write(gpx.to_xml())
