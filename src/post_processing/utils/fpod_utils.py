"""FPOD/ CPOD processing functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import pytz
import seaborn as sns
from matplotlib import patches
from matplotlib import pyplot as plt
from osekit.utils.timestamp_utils import strftime_osmose_format, strptime_from_text
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
    to_timedelta,
)

from post_processing.utils.filtering_utils import find_delimiter
from user_case.config import season_color, site_colors

if TYPE_CHECKING:
    from pathlib import Path

    import pytz

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
    fpod_start_dt = [
        tz.localize(entry)
        for entry in df["Datetime"]
    ]

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
        )

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
        strptime_from_text(dt, "%d/%m/%Y %H:%M")
        for dt in data_filtered["ChunkEnd"]
    ]
    return data_filtered.sort_values(by=["Datetime"]).reset_index(drop=True)


def _filter_csv_data(data: DataFrame) -> DataFrame:
    """Filter CSV data based on available columns."""
    if "%TimeLost" in data.columns:
        data_filtered = data[data["File"].notna()].copy()
        data_filtered = data_filtered[data_filtered["Nall/m"].notna()]
    else:
        data_filtered = data[data["DPM"] > 0].copy()
        data_filtered = data_filtered[data_filtered["Nall"].notna()]

    return data_filtered


def _process_txt_data(data: DataFrame) -> DataFrame:
    """Process TXT data with datetime conversion."""
    data["Datetime"] = data.apply(get_feeding_buzz_datetime, axis=1)
    return data.drop_duplicates().sort_values(by=["Datetime"]).reset_index(drop=True)


def get_feeding_buzz_datetime(row: Series) -> Timestamp:
    """Convert feeding buzz timestamp into a standard Timestamp.

    The conversion method differs based on the POD type.
    """
    try:
        return (
            to_datetime("1900-01-01")
            + to_timedelta(row["Minute"], unit="min")
            + to_timedelta(row["microsec"] / 1e6, unit="sec")
            - to_timedelta(2, unit="D")
        )
    except (KeyError, TypeError, ValueError):
        pass

    try:
        return strptime_from_text(
            f"{row['Minute']}:{int(str(row['microsec'])[0]):02d}.{int(str(row['microsec'])[1:])}",
            "%-d/%-m/%Y %H:%M:%S.%f",
        )
    except (KeyError, TypeError, ValueError):
        pass

    msg = "Could not convert feeding buzz timestamp."
    raise ValueError(msg)


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
        df["Buzz"] = df["ICI"].between(
            Timedelta(0),
            Timedelta(seconds=0.02),
        ).astype(int)
    elif species.lower() == "porpoise":  # Nuuttila et al., 2013
        df["Buzz"] = df["ICI"].between(
            Timedelta(0),
            Timedelta(seconds=0.01),
        ).astype(int)
    elif species.lower() == "commerson":  # Reyes Reyes et al., 2015
        df["Buzz"] = df["ICI"].between(
            Timedelta(0),
            Timedelta(seconds=0.005),
        ).astype(int)
    else:
        msg = "This species is not supported"
        raise ValueError(msg)

    df_buzz = df.groupby(["Datetime"])["Buzz"].sum().reset_index()
    df_buzz["Foraging"] = to_numeric(
        df_buzz["Buzz"] != 0, downcast="integer",
    ).astype(int)

    return df_buzz


def process_timelost(df: DataFrame, threshold: int = 0) -> DataFrame:
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
        col for col in df.columns if col not in {
            "File", "Datetime", "Temp", "Angle", "%TimeLost", "Deploy",
        }
    ]
    return df[df["%TimeLost"] <= threshold].drop(
        columns=cols_to_drop,
    ).sort_values(["Datetime"]).reset_index(drop=True)


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
    matrix = df.groupby(group_cols).agg({
        col: ["mean", "std"] for col in agg_cols
    })
    matrix = matrix.reset_index()

    matrix.columns = group_cols + [f"{col}_{stat}"
                                    for col in agg_cols
                                    for stat in ["mean", "std"]]
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
    group_cols = ["site.name"]
    if time_unit is not None:
        group_cols.insert(0, time_unit)

    # Aggregate and compute metrics
    df = (
        data.groupby(group_cols)
        .agg(
            {
                "DPh": "sum",
                "DPM": "sum",
                "Day": "size",
                "Foraging": "sum",
            },
        )
        .reset_index()
    )

    df["%click"] = df["DPM"] * 100 / (df["Day"] * 60)
    df["%DPh"] = df["DPh"] * 100 / df["Day"]
    df["FBR"] = df.apply(
        lambda row: (row["Foraging"] * 100 / row["DPM"]) if row["DPM"] > 0 else 0,
        axis=1)
    df["%buzzes"] = df["Foraging"] * 100 / (df["Day"] * 60)
    return df


def site_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection for every site.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site
    metric: str
        Type of percentage you want to show on the graph

    """
    ax = sns.barplot(
        data=df,
        x="site.name",
        y=metric,
        hue="site.name",
        dodge=False,
        palette=site_colors,
    )
    ax.set_title(f"{metric} per site")
    ax.set_ylabel(f"{metric}")
    if metric in {"%buzzes", "FBR"}:
        for _, bar in enumerate(ax.patches):
            bar.set_hatch("/")
    plt.show()


def year_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection per site/year.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and year
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        ax.bar(
            site_data["Year"],
            site_data[metric],
            label=f"Site {site}",
            color=site_colors.get(site, "gray"),
        )
        ax.set_title(f"{site}")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Year")
        if metric in {"%buzzes", "FBR"}:
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per year", fontsize=16)
    plt.show()


def ym_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of DPM per site/month-year.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and month per year
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        bar_colors = site_data["Season"].map(season_color).fillna("gray")
        ax.bar(
            site_data["YM"],
            site_data[metric],
            label=f"Site {site}",
            color=bar_colors,
            width=25,
        )
        ax.set_title(f"{site}")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Months")
        if metric in {"%buzzes", "FBR"}:
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    legend_elements = [
        patches.Patch(facecolor=col, edgecolor="black", label=season.capitalize())
        for season, col in season_color.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        title="Seasons",
        bbox_to_anchor=(0.95, 0.95),
    )
    fig.suptitle(f"{metric} per month", fontsize=16)
    plt.show()


def week_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of DPM per site/month-year.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and month per year
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(15, 3 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]

    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site].copy()
        ax = axs[i]

        # Masque pour identifier les NAs
        na_mask = site_data["DPM"].isna()

        # Définir la limite Y
        ymax = max(df[metric].dropna()) + 0.2 if not df[metric].dropna().empty else 1
        ax.set_ylim(0, ymax)

        # Tracer les rectangles pour les périodes de NAs
        na_dates = site_data.loc[na_mask, "start_datetime"]
        if len(na_dates) > 0:
            na_groups = []
            current_group = [na_dates.iloc[0]]

            for j in range(1, len(na_dates)):
                # Vérifier si les semaines sont consécutives (~7 jours)
                if (na_dates.iloc[j] - current_group[-1]).days < 10:
                    current_group.append(na_dates.iloc[j])
                else:
                    na_groups.append(current_group)
                    current_group = [na_dates.iloc[j]]
            na_groups.append(current_group)

            # Créer les rectangles
            for group in na_groups:
                start = group[0] - DateOffset(days=3.5)  # Centrer sur la semaine
                width = len(group) * 7 + 2  # Largeur en jours
                rect = patches.Rectangle(
                    (mdates.date2num(start), 0),
                    width,
                    ymax,
                    linewidth=1,
                    edgecolor="gray",
                    facecolor="lightgray",
                    alpha=0.3,
                    label="Pas de données"
                    if (i == 0 and group == na_groups[0])
                    else "",
                )
                ax.add_patch(rect)

        # Tracer les barres avec données
        bar_colors = site_data.loc[~na_mask, "Season"].map(season_color).fillna("gray")
        bars = ax.bar(
            site_data.loc[~na_mask, "start_datetime"],
            site_data.loc[~na_mask, metric],
            label=f"Site {site}",
            color=bar_colors,
            width=6,  # Largeur adaptée pour les semaines
        )

        # Ajouter des hachures si nécessaire
        if metric in {"%buzzes", "FBR"}:
            for bar in bars:
                bar.set_hatch("/")

        ax.set_title(f"{site}")
        ax.set_ylabel(metric)
        if i != n_sites - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Week")

    # Légende des saisons
    legend_elements = [
        patches.Patch(facecolor=col, edgecolor="black", label=season.capitalize())
        for season, col in season_color.items()
    ]

    # Ajouter "Pas de données" à la légende si des NAs existent
    if df["DPM"].isna().any():
        legend_elements.append(
            patches.Patch(
                facecolor="lightgray",
                edgecolor="gray",
                alpha=0.3,
                label="Pas de données"))

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        title="Seasons",
        bbox_to_anchor=(0.95, 0.95),
    )
    fig.suptitle(f"{metric} per week", fontsize=16)

    # Formatage de l'axe X
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def month_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection per site/month.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and month
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        ax.bar(
            site_data["Month"],
            site_data[metric],
            label=f"Site {site}",
            color=site_colors.get(site, "gray"),
        )
        ax.set_title(f"{site} - Percentage of minutes positive to detection per month")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        ax.set_xticks(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Agu",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
        )
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Months")
        if metric in {"%buzzes", "FBR"}:
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per month", fontsize=16)
    plt.show()


def day_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of DPM per site/month-year.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and month per year
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        bar_colors = site_data["Season"].map(season_color).fillna("gray")
        ax.bar(
            site_data["Date"],
            site_data[metric],
            label=f"Site {site}",
            color=bar_colors,
        )
        ax.set_title(f"{site}")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Months")
        if metric in {"%buzzes", "FBR"}:
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    legend_elements = [
        patches.Patch(facecolor=col, edgecolor="black", label=season.capitalize())
        for season, col in season_color.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        title="Seasons",
        bbox_to_anchor=(0.95, 0.95),
    )
    fig.suptitle(f"{metric} per month", fontsize=16)
    plt.show()


def hour_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection per site/hour.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and hour
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        ax.bar(
            site_data["Hour"],
            site_data[metric],
            label=f"Site {site}",
            color=site_colors.get(site, "gray"),
        )
        ax.set_title(
            f"Site {site} - Percentage of minutes positive to detection per hour",
        )
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Hour")
        if metric in {"%buzzes", "FBR"}:
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per hour", fontsize=16)
    plt.show()


def calendar(
    meta: DataFrame,
    data: DataFrame,
) -> None:
    """Produce the calendar of the given data.

    Parameters
    ----------
    meta: DataFrame
        metadatax file
    data: DataFrame
        cpod file from all sites and phases

    """
    # format the dataframe
    meta["deployment_date"] = to_datetime(meta["deployment_date"])
    meta["recovery_date"] = to_datetime(meta["recovery_date"])
    meta = meta.sort_values(["deploy.name", "deployment_date"]).reset_index(drop=True)
    data = data.sort_values(["deploy.name", "Deb"]).reset_index(drop=True)
    df_fusion = data.merge(
        meta[["deploy.name", "deployment_date", "recovery_date"]],
        on=["deploy.name"],
        how="outer",
    )

    df_fusion["Deb"] = df_fusion["Deb"].fillna(df_fusion["deployment_date"])
    df_fusion["Fin"] = df_fusion["Fin"].fillna(df_fusion["deployment_date"])

    df_fusion[["Site", "Phase"]] = df_fusion["deploy.name"].str.split("_", expand=True)
    df_fusion["color"] = df_fusion["Site"].map(site_colors)

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 4))

    sites = sorted(df_fusion["Site"].unique(), reverse=True)
    site_mapping = {site: idx for idx, site in enumerate(sites)}

    for _, row in df_fusion.iterrows():
        y_pos = site_mapping[row["Site"]]
        ax.broken_barh(
            [(row["deployment_date"], row["recovery_date"] - row["deployment_date"])],
            (y_pos - 0.3, 0.6),
            facecolors="#F5F5F5",
            edgecolors="black",
            linewidth=0.8,
        )

        if notna(row["Deb"]) and notna(row["Fin"]) and row["Fin"] > row["Deb"]:
            ax.broken_barh(
                [(row["Deb"], row["Fin"] - row["Deb"])],
                (y_pos - 0.15, 0.3),
                facecolors=row["color"],
                edgecolors="black",
                linewidth=0.8,
            )

    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites, fontsize=12)

    legend_elements = [
        patches.Patch(facecolor="#F5F5F5", edgecolor="black", label="Deployment"),
    ]
    for site, color in site_colors.items():
        if site in sites:
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor="black", label=f"{site}"),
            )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, frameon=True)
    # Layout final
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def hist_mean_m(
    df: DataFrame,
    metric_mean: str,
    metric_std: str,
    y_lab: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Produce a histogram of the given data.

    It shows mean and standard deviation of the metric.

    Parameters
    ----------
    df: DataFrame
        All data grouped by site and month
    metric_mean: str
        Column name for the mean values (e.g., "%click_mean")
    metric_std: str
        Column name for the standard deviation values (e.g., "%click_std")
    y_lab: str, optional
        Label for y-axis. If None, uses metric_mean
    title_suffix: str, optional
        Suffix for the main title. If None, uses metric_mean

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 3 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]

    # Calculate max for y-axis scaling
    max_value = max(df[metric_mean] + df[metric_std])

    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]

        ax.bar(
            x=site_data["Month"],
            height=site_data[metric_mean],
            yerr=site_data[metric_std],
            capsize=4,
            color=site_colors.get(site, "gray"),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=f"Site {site}")

        ax.set_title(f"{site}", fontsize=12)
        ax.set_ylim(0, max_value * 1.1)
        ax.set_ylabel(y_lab or metric_mean, fontsize=10)

        # Only set x-label on last subplot
        if i == n_sites - 1:
            ax.set_xlabel("Mois", fontsize=10)
            ax.set_xticks(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [
                    "Jan",
                    "Fev",
                    "Mar",
                    "Avr",
                    "Mai",
                    "Jun",
                    "Jul",
                    "Aou",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
            )
        if metric_mean in {"%buzzes_mean", "FBR_mean"}:
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")

    fig.suptitle(
        f"{title_suffix or metric_mean} per month",
        fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def hist_mean_h(
    df: DataFrame,
    metric_mean: str,
    metric_std: str,
    y_lab: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Produce a histogram of the given data.

    It shows mean and standard deviation of the metric.

    Parameters
    ----------
    df: DataFrame
        All data grouped by site and month
    metric_mean: str
        Column name for the mean values (e.g., "%click_mean")
    metric_std: str
        Column name for the standard deviation values (e.g., "%click_std")
    y_lab: str, optional
        Label for y-axis. If None, uses metric_mean
    title_suffix: str, optional
        Suffix for the main title. If None, uses metric_mean

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]

    # Calculate max for y-axis scaling
    max_value = max(df[metric_mean] + df[metric_std])

    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]

        ax.bar(
            x=site_data["Hour"],
            height=site_data[metric_mean],
            yerr=site_data[metric_std],
            capsize=4,
            color=site_colors.get(site, "gray"),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=f"Site {site}",
        )

        ax.set_title(f"{site}", fontsize=12)
        ax.set_ylim(0, max_value * 1.1)
        ax.set_ylabel(y_lab or metric_mean, fontsize=10)
        ax.set_xticks(range(24))

        # Only set x-label on last subplot
        if i == n_sites - 1:
            ax.set_xlabel("Heure", fontsize=10)
        if metric_mean in {"%buzzes_mean", "FBR_mean"}:
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")

    fig.suptitle(f"{title_suffix or metric_mean} per hour", fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def hist_mean_s(
    df: DataFrame,
    metric_mean: str,
    metric_std: str,
    y_lab: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Plot bar chart with mean values and error bars (std) per site.

    Parameters
    ----------
    df: DataFrame
        All data grouped by site
    metric_mean: str
        Column name for the mean values (e.g., "FBR_mean")
    metric_std: str
        Column name for the standard deviation values (e.g., "FBR_std")
    y_lab: str, optional
        Label for y-axis. If None, uses metric_mean
    title_suffix: str, optional
        Suffix for the title. If None, uses metric_mean

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by site and calculate means if needed
    plot_data = df.groupby("site.name")[[metric_mean, metric_std]].mean().reset_index()

    x_pos = range(len(plot_data))

    # Create bars
    ax.bar(
    x=x_pos,
    height=plot_data[metric_mean],
    color=[site_colors.get(site, "gray") for site in plot_data["site.name"]],
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5)

    # Add hatching if requested
    if metric_mean in {"%buzzes_mean", "FBR_mean"}:
        for _, bar in enumerate(ax.patches):
            bar.set_hatch("/")

    # Add error bars
    for i, (_, row) in enumerate(plot_data.iterrows()):
        # Ensure error bar doesn't go below zero
        yerr_lower = min(row[metric_mean], row[metric_std])
        yerr_upper = row[metric_std]
        ax.errorbar(
            i,
            row[metric_mean],
            yerr=[[yerr_lower], [yerr_upper]],
            fmt="none",
            color="black",
            capsize=5,
            linewidth=2,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_data["site.name"])
    ax.set_title(f"{title_suffix or metric_mean} per site",
                 fontsize=12)
    ax.set_ylabel(y_lab or metric_mean, fontsize=10)
    ax.set_xlabel("Site", fontsize=10)

    plt.tight_layout()
    plt.show()


def hist_mean_season(
    df: DataFrame,
    metric_mean: str,
    metric_std: str,
    y_lab: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Produce a histogram of the given data.

    It shows mean and standard deviation of the metric.

    Parameters
    ----------
    df: DataFrame
        All data grouped by site and month
    metric_mean: str
        Column name for the mean values (e.g., "%click_mean")
    metric_std: str
        Column name for the standard deviation values (e.g., "%click_std")
    y_lab: str, optional
        Label for y-axis. If None, uses metric_mean
    title_suffix: str, optional
        Suffix for the main title. If None, uses metric_mean

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]

    # Calculate max for y-axis scaling
    max_value = max(df[metric_mean] + df[metric_std])

    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]

        ax.bar(
            x=site_data["Season"],
            height=site_data[metric_mean],
            yerr=site_data[metric_std],
            capsize=4,
            color=site_colors.get(site, "gray"),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=f"Site {site}",
        )

        ax.set_title(f"{site}", fontsize=12)
        ax.set_ylim(0, max_value * 1.1)
        ax.set_ylabel(y_lab or metric_mean, fontsize=10)

        # Only set x-label on last subplot
        if i == n_sites - 1:
            ax.set_xlabel("Season", fontsize=10)
        if metric_mean in {"%buzzes_mean", "FBR_mean"}:
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")

    fig.suptitle(f"{title_suffix or metric_mean} per season", fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()