"""FPOD/ CPOD processing functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from numpy import (
    argsort,
    dtype,
    exp,
    float64,
    linspace,
    log,
    nan,
    ndarray,
    sort,
    sqrt,
    zeros,
)
from osekit.utils.timestamp import strftime_osmose_format, strptime_from_text
from pandas import (
    DataFrame,
    DateOffset,
    Series,
    Timedelta,
    Timestamp,
    concat,
    notna,
    read_csv,
    to_datetime,
    to_numeric,
)
from scipy import stats
from sklearn import mixture

from post_processing.utils.filtering_utils import find_delimiter
from user_case.config import season_color, site_colors

if TYPE_CHECKING:
    from pathlib import Path

    import pytz
    from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def pod2aplose(
    df: DataFrame,
    tz: pytz.timezone,
    dataset_name: str,
    annotation: str,
    annotator: str,
    bin_size: Timedelta,
) -> DataFrame:
    """Format PODs DataFrame to match an APLOSE format.

    Parameters
    ----------
    df: DataFrame
        FPOD result dataframe
    tz: pytz.timezone
        Timezone object to get non-naïve datetime.
    dataset_name: str
        dataset name.
    annotation: str
        annotation name.
    annotator: str
        annotator name.
    bin_size: Timedelta
        Duration of the detections in seconds.

    Returns
    -------
    DataFrame
        An APLOSE formatted DataFrame.

    """
    fpod_start_dt = [tz.localize(entry) for entry in df["Datetime"]]

    data = {
        "dataset": [dataset_name] * len(df),
        "filename": list(fpod_start_dt),
        "start_time": [0] * len(df),
        "end_time": [bin_size.total_seconds()] * len(df),
        "start_frequency": [0] * len(df),
        "end_frequency": [0] * len(df),
        "annotation": [annotation] * len(df),
        "annotator": [annotator] * len(df),
        "start_datetime": [
            strftime_osmose_format(entry.floor(bin_size)) for entry in fpod_start_dt
        ],
        "end_datetime": [
            strftime_osmose_format(entry.ceil(bin_size)) for entry in fpod_start_dt
        ],
        "type": ["WEAK"] * len(df),
        "deploy": df["Deploy"].tolist(),
    }

    return DataFrame(data)


def load_pod_folder(
    folder: Path,
    ext: str,
) -> DataFrame:
    """Read POD's result files from a folder.

    Parameters
    ----------
    folder: Path
        Folder's place.
    ext: str
        File extension of result files.

    Returns
    -------
    DataFrame
        Concatenated data.

    Raises
    ------
    ValueError
        If no result files are found.

    """
    if ext not in {"csv", "txt"}:
        msg = f"Invalid file extension: {ext}"
        raise ValueError(msg)

    all_files = sorted(folder.rglob(f"*.{ext}"))

    if not all_files:
        msg = f"No .{ext} files found in {folder}"
        raise ValueError(msg)

    all_data = []
    for file in all_files:
        sep = find_delimiter(file)
        df = read_csv(
            file,
            sep=sep,
            dtype={"microsec": "Int32"},
            usecols=lambda col: col not in {"SmoothedICI", "ICIslope"},
        ).dropna()

        df["Deploy"] = file.stem.strip().lower().replace(" ", "_")
        all_data.append(df)

    data = concat(all_data, ignore_index=True)

    if ext == "csv":
        return _process_csv_data(data)
    if ext == "txt":
        return _process_txt_data(data)

    msg = f"Could not load {ext} result folder"
    raise ValueError(msg)


def _process_csv_data(data: DataFrame) -> DataFrame:
    """Process CSV data with filtering and datetime conversion."""
    data_filtered = _filter_csv_data(data)
    data_filtered["Datetime"] = [
        strptime_from_text(dt, "%d/%m/%Y %H:%M") for dt in data_filtered["ChunkEnd"]
    ]
    return data_filtered.sort_values(by=["Datetime"]).reset_index(drop=True)


def _filter_csv_data(data: DataFrame) -> DataFrame:
    """Filter CSV data based on available columns."""
    if "%TimeLost" in data.columns:
        data_filtered = data[data["File"].notna()].copy()
        data_filtered = data_filtered[data_filtered["Nall/m"].notna()]
    else:
        data_filtered = data[data["DPM"] > 0].copy()
        data_filtered = data_filtered[data_filtered["MinsOn"].notna()]

    return data_filtered


def _process_txt_data(data: DataFrame) -> DataFrame:
    """Process TXT data with datetime conversion."""
    data["Datetime"] = data.apply(get_feeding_buzz_datetime, axis=1)
    return data.drop_duplicates().sort_values(by=["Datetime"]).reset_index(drop=True)


def get_feeding_buzz_datetime(row: Series) -> Timestamp:
    """Convert feeding buzz timestamp into a standard Timestamp.

    The conversion method differs based on the POD type.
    """
    exceptions = []
    try:
        return (
            Timestamp("1899-12-30")
            + Timedelta(minutes=row["Minute"])
            + Timedelta(microseconds=row["microsec"])
        )
    except (KeyError, TypeError, ValueError) as e:
        exceptions.append(e)

    try:
        return (strptime_from_text(row["Minute"], "%-d/%-m/%Y %H:%M") +
                Timedelta(microseconds=row["microsec"]))
    except (KeyError, TypeError, ValueError) as e:
        exceptions.append(e)

    msg = "Could not convert feeding buzz timestamp."
    raise ExceptionGroup(msg, exceptions)


def process_feeding_buzz(
    df: DataFrame,
    species: str,
) -> DataFrame:
    """Process a POD feeding buzz detection DataFrame.

    Give the feeding buzz duration, depending on the studied species
    (`delphinid`, `porpoise` or `commerson`).

    Parameters
    ----------
    df: DataFrame
        Path to cpod.exe feeding buzz file
    species: str
        Select the species to use between porpoise and Commerson's dolphin

    Returns
    -------
    DataFrame
        Containing all ICIs for every positive minute to click

    """
    df["ICI"] = df["Datetime"].diff()
    df["Datetime"] = df["Datetime"].dt.floor("min")

    if species.lower() == "delphinid":  # Herzing et al., 2014
        df["Buzz"] = (
            df["ICI"]
            .between(
                Timedelta(0),
                Timedelta(seconds=0.02),
            )
            .astype(int)
        )
    elif species.lower() == "porpoise":  # Nuuttila et al., 2013
        df["Buzz"] = (
            df["ICI"]
            .between(
                Timedelta(0),
                Timedelta(seconds=0.01),
            )
            .astype(int)
        )
    elif species.lower() == "commerson":  # Reyes Reyes et al., 2015
        df["Buzz"] = (
            df["ICI"]
            .between(
                Timedelta(0),
                Timedelta(seconds=0.005),
            )
            .astype(int)
        )
    else:
        msg = "This species is not supported"
        raise ValueError(msg)

    df_buzz = df.groupby(["Datetime"])["Buzz"].sum().reset_index()
    df_buzz["Foraging"] = to_numeric(
        df_buzz["Buzz"] != 0,
        downcast="integer",
    ).astype(int)

    return df_buzz


def compute_ici(df: DataFrame) -> DataFrame:
    """Calculate Inter-Click Intervals (in minutes) from feeding buzz timestamps."""
    df = df.copy()
    df["ICI_minutes"] = df["Datetime"].diff().dt.total_seconds() / 60
    return df[df["ICI_minutes"] > 0].dropna(subset=["ICI_minutes"])


def fit_gmm(df: DataFrame, comp: int) -> tuple[DataFrame, ndarray, GaussianMixture]:
    """Fit a GMM on log-transformed ICIs and label clusters by ascending mean.

    Parameters
    ----------
    df: DataFrame
        POD loaded dataframe
    comp: int
        Number of components to apply to the GMM.

    Returns
    -------
    tuple
    Returns the enriched DataFrame, the log-ICI array, and the fitted GMM.

    """
    df = compute_ici(df)
    ici_log = log(df["ICI_minutes"].to_numpy()).reshape(-1, 1)

    gmm = mixture.GaussianMixture(
        n_components=comp, covariance_type="full", random_state=42, n_init=20,
    )
    labels = gmm.fit_predict(ici_log)

    rank = argsort(argsort(gmm.means_.flatten()))
    df["cluster"] = rank[labels]

    return df, ici_log, gmm


def cluster_info(gmm: GaussianMixture) -> list[dict]:
    """Extract per-component statistics from a fitted GMM, sorted by ascending mean."""
    component_names = ["Buzz ICIs", "Regular ICIs", "Long ICIs"]
    sorted_means = sort(gmm.means_, axis=0)

    return [
        {
            "name": component_names[i],
            "id": i,
            "mean_log": sorted_means[i][0],
            "std_log": sqrt(gmm.covariances_[i][0][0]),
            "mean_minutes": exp(sorted_means[i][0]),
            "mean_ms": exp(sorted_means[i][0]) * 60 * 1000,
        }
        for i in range(gmm.n_components)
    ]


def _mixture_density(gmm: GaussianMixture, x_range: ndarray) -> ndarray:
    """Compute the total GMM mixture density over x_range."""
    density = zeros(len(x_range))
    for idx in range(gmm.n_components):
        mean = gmm.means_[idx][0]
        std = sqrt(gmm.covariances_[idx][0][0])
        density += gmm.weights_[idx] * stats.norm.pdf(x_range, mean, std)
    return density


def gmm_feeding_buzz(df: DataFrame, comp: int) -> DataFrame:
    """Categorize ICIs with a GMM and aggregate foraging activity per minute.

    Parameters
    ----------
    df: DataFrame
        POD loaded dataframe
    comp: int
        Number of components to apply to the GMM.

    Returns
    -------
    DataFrame
        A DataFrame of two columns : minute positive to feeding buzz or not and number of buzzes.

    """
    df, _, _ = fit_gmm(df, comp)

    df["Buzz"] = nan
    df.loc[df["cluster"] == 0, "Buzz"] = 1
    df["start_datetime"] = df["Datetime"].dt.floor("min")

    df_buzz = df.groupby("start_datetime")["Buzz"].sum().reset_index()
    df_buzz["Foraging"] = to_numeric(df_buzz["Buzz"] != 0, downcast="integer").astype(int)
    return df_buzz


def plot_gmm_ici(df: DataFrame, comp: int) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of log ICIs overlaid with GMM components and total mixture."""
    df, ici_log, gmm = fit_gmm(df, comp)

    x_flat = sort(ici_log.flatten())
    x_range = linspace(ici_log.min(), ici_log.max(), 2000)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(
        ici_log, bins=200, histtype="bar", density=True,
        alpha=0.6, color="lightgray", edgecolor="black", linewidth=0.5,
    )

    lines = []
    for idx in range(comp):
        mean, std, weight = gmm.means_[idx, 0], sqrt(gmm.covariances_[idx, 0, 0]), gmm.weights_[idx]
        (line,) = ax.plot(
            x_flat, weight * stats.norm.pdf(x_flat, mean, std),
            label=f"(μ={mean:.2f}, σ={std:.2f})",
        )
        lines.append(line)

    (mix_line,) = ax.plot(
        x_range, _mixture_density(gmm, x_range),
        linewidth=2, color="black", linestyle="--", label="Total mixture", alpha=0.7,
    )
    lines.append(mix_line)

    ax.set(xlabel="Log ICI (log minutes)", ylabel="Density", title="GMM clustering of Inter-Click Intervals")
    ax.legend(handles=lines)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()
    return fig, ax


def process_timelost(df: DataFrame, threshold: int = 0) -> Series[Any]:
    """Process TimeLost DataFrame.

    Returns relevant columns and reshape into hourly data.

    Parameters
    ----------
    df: DataFrame
        All your Environmental data files.
    threshold: float
        TimeLost threshold.

    Returns
    -------
    %TimeLost DataFrame.

    """
    if threshold not in range(101):
        msg = "Threshold must integer between 0 and 100."
        raise ValueError(msg)

    df["Datetime"] = df["Datetime"].dt.floor("h")
    cols_to_drop = [
        col
        for col in df.columns
        if col
        not in {
            "File",
            "Datetime",
            "Temp",
            "Angle",
            "%TimeLost",
            "Deploy",
        }
    ]
    return (
        df[df["%TimeLost"] <= threshold]
        .drop(
            columns=cols_to_drop,
        )
        .sort_values(["Datetime"])
        .reset_index(drop=True)
    )


def create_matrix(
    df: DataFrame,
    group_cols: list,
    agg_cols: list,
) -> DataFrame:
    """Create a stats matrix (mean & std).

    Parameters
    ----------
    df : DataFrame
        Extended frame with raw data to calculate stats for
    group_cols : list
        Additional columns to group by
    agg_cols : list
        Columns to aggregate

    Returns
    -------
    Give a matrix of the data in [agg_cols] grouped by [group_cols].

    """
    matrix = df.groupby(group_cols).agg({col: ["mean", "std"] for col in agg_cols})
    matrix = matrix.reset_index()

    matrix.columns = group_cols + [
        f"{col}_{stat}" for col in agg_cols for stat in ["mean", "std"]
    ]
    return matrix


def percent_calc(
    data: DataFrame,
    time_unit: str | None = None,
) -> DataFrame:
    """Calculate the percentage of clicks, feeding buzzes and positive hours to detection.

    Computed on the entire effort and for every site.

    Parameters
    ----------
    data: DataFrame
        All values concatenated

    time_unit: str
        Time unit you want to group your data in

    Returns
    -------
    DataFrame

    """
    df = (
        data
        .groupby(time_unit)
        .agg(
            {
                "DPh": "sum",
                "dpm_count": "sum",
                "Day": "size",
                "Foraging": "sum",
            },
        )
        .reset_index()
    )

    df["%click"] = df["dpm_count"] * 100 / (df["Day"] * 60)
    df["%DPh"] = df["DPh"] * 100 / df["Day"]
    df["FBR"] = df.apply(
        lambda row: (row["Foraging"] * 100 / row["dpm_count"])
        if row["dpm_count"] > 0
        else 0,
        axis=1,
    )
    df["%buzzes"] = df["Foraging"] * 100 / (df["Day"] * 60)
    return df


def percent_barplot(df: DataFrame, unit: str, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection for every site.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site
    unit: str
        Time unit the data are grouped in
    metric: str
        Type of percentage shown on the graph

    """
    fig, ax = plt.subplots()
    ax.bar(df[unit].astype(str), df[metric], color="#0072b2")
    ax.set_title(f"{metric} per {unit}")
    ax.set_ylabel(f"{metric}")
    ax.set_xlabel(f"{unit}")
    if metric in {"%buzzes", "FBR"}:
        for _, bar in enumerate(ax.patches):
            bar.set_hatch("/")
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()


def calendar(
    data: DataFrame,
) -> None:
    """Produce the calendar of the given data. Deployments and actual collection of data.

    Parameters
    ----------
    data: DataFrame
        Custom file containing all beginning and end of deployment and recordings.

    """
    for i in data["Site"].unique():
        mask = data["Site"] == i
        data["start_recording"] = to_datetime(data["start_recording"])
        data["end_recording"] = to_datetime(data["end_recording"])
        data["start_deployment"] = to_datetime(data["start_deployment"])
        data["end_deployment"] = to_datetime(data["end_deployment"])

        data.loc[
            mask & (data["start_recording"] < data["start_deployment"]),
            "start_recording",
        ] = data.loc[
            mask & (data["start_recording"] < data["start_deployment"]),
            "start_deployment",
        ]

        data.loc[
            mask & (data["end_recording"] > data["end_deployment"]),
            "end_recording"] = data.loc[
            mask & (data["end_recording"] > data["end_deployment"]), "end_deployment"]

        data.loc[mask & (data["start_recording"] > data["end_recording"]),
        ["start_recording", "end_recording"]] = None
        data = data.sort_values(["Phase", "start_deployment"]).reset_index(drop=True)

    data["color"] = data["Site"].map(site_colors)

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 4))

    sites = sorted(data["Site"].unique(), reverse=True)
    site_mapping = {site: idx for idx, site in enumerate(sites)}

    for _, row in data.iterrows():
        y_pos = site_mapping[row["Site"]]
        ax.broken_barh(
            [
                (
                    row["start_deployment"],
                    row["end_deployment"] - row["start_deployment"],
                ),
            ],
            (y_pos - 0.3, 0.6),
            facecolors="#F5F5F5",
            edgecolors="black",
            linewidth=0.8,
        )

        if (notna(row["start_recording"]) and notna(row["end_recording"]) and
                row["end_recording"] > row["start_recording"]):
            ax.broken_barh(
                [(row["start_recording"],
                  row["end_recording"] - row["start_recording"])],
                (y_pos - 0.15, 0.3),
                facecolors=row["color"],
                edgecolors="black",
                linewidth=0.8,
            )

    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites, fontsize=15)

    plt.xticks(fontsize=15)
    plt.tight_layout()
    plt.show()


def matrice_hist(df: DataFrame, unit: str, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection for every site.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site
    unit: str
        Time unit you want to group your data in
    metric: str
        Type of percentage you want to show on the graph

    """
    fig, ax = plt.subplots()
    ax.bar(df[unit], df[f"{metric}_mean"], color="#0072b2")
    ax.set_xlabel(f"{unit}")
    ax.set_ylabel(f"{metric}")
    plt.errorbar(df[unit], df[f"{metric}_mean"], df[f"{metric}_std"],
                 fmt=".", color="Black", elinewidth=2, capthick=10,
                 errorevery=1, alpha=0.5, ms=4, capsize=2)
    ax.set_ylim(0, max(df[f"{metric}_mean"] + df[f"{metric}_std"]) * 1.1)
    if metric in {"%buzzes", "FBR"}:
        for _, bar in enumerate(ax.patches):
            bar.set_hatch("/")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.show()
