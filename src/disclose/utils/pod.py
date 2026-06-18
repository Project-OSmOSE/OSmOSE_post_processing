from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytz
import seaborn as sns
from matplotlib import pyplot as plt
from osekit.config import TIMESTAMP_FORMAT_AUDIO_FILE
from osekit.utils.timestamp_utils import strftime_osmose_format, strptime_from_text
from pandas import (
    DataFrame,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    notna,
    read_csv,
    read_excel,
    to_datetime,
)

from disclose import logger
from disclose.utils.core import get_coordinates, get_sun_times

if TYPE_CHECKING:
    import pytz


def fpod2aplose(
    df: DataFrame,
    tz: pytz.timezone,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
) -> DataFrame:
    """Format FPOD DataFrame to match APLOSE format.

    Parameters
    ----------
    df: DataFrame
        FPOD result dataframe
    tz: pytz.timezone
        Timezone object to get non-naïve datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int
        Duration of the detections in seconds

    Returns
    -------
    DataFrame
        An APLOSE formatted DataFrame

    """
    fpod_start_dt = sorted(
        [
            tz.localize(strptime_from_text(entry, "%d/%m/%Y %H:%M"))
            for entry in df["Date heure"]
        ],
    )

    fpod_end_dt = sorted(
        [entry + Timedelta(seconds=bin_size) for entry in fpod_start_dt],
    )

    data = {
        "dataset": [dataset_name] * len(df),
        "filename": [""] * len(df),
        "start_time": [0] * len(df),
        "end_time": [bin_size] * len(df),
        "min_frequency": [0] * len(df),
        "max_frequency": [0] * len(df),
        "annotation": [annotation] * len(df),
        "annotator": ["FPOD"] * len(df),
        "start_datetime": [strftime_osmose_format(entry) for entry in fpod_start_dt],
        "end_datetime": [strftime_osmose_format(entry) for entry in fpod_end_dt],
        "type": [0] * len(df),
    }

    return DataFrame(data)


def cpod2aplose(
    df: DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
    extra_columns: list | None = None,
) -> DataFrame:
    """Format CPOD DataFrame to match APLOSE format.

    Parameters
    ----------
    df: DataFrame
        CPOD result dataframe
    tz: pytz.BaseTzInfo
        Timezone object to get non-naïve datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int, optional
        Duration of the detections in seconds
    extra_columns: list, optional
        Additional columns added from df to data

    Returns
    -------
    DataFrame
        An APLOSE formatted DataFrame

    """
    df_cpod = df.rename(columns={"ChunkEnd": "Date heure"})

    # remove lines where the C-POD stopped working
    df_cpod = df_cpod.drop(
        df_cpod.loc[df_cpod["Date heure"] == " at minute "].index,
    )
    data = fpod2aplose(df_cpod, tz, dataset_name, annotation, bin_size)
    data["annotator"] = data.loc[data["annotator"] == "FPOD"] = "CPOD"
    if extra_columns:
        for col in extra_columns:
            if col in df_cpod.columns:
                data[col] = df_cpod[col].tolist()
            else:
                msg = f"Column '{col}' does not exist and will be ignored."
                logger.warning(msg)

    return DataFrame(data)


def usable_data_phase(
    d_meta: DataFrame,
    df: DataFrame,
    dpl: str,
) -> DataFrame:
    """Calculate the percentage of usable data.

    Considering the deployment dates and the collected data.

    Parameters
    ----------
    df: DataFrame
        CPOD result DataFrame
    d_meta: DataFrame
        Metadata DataFrame with deployments information (previously exported as json)
    dpl: str
        Deployment of interest where percentage of usable data will be calculated

    Returns
    -------
    DataFrame
        Returns the percentage of usable datas in the chosen phase

    """
    d_meta.loc[:, ["deployment_date", "recovery_date"]] = d_meta[
        ["deployment_date", "recovery_date"]
    ].apply(
        to_datetime,
    )
    df["start_datetime"] = to_datetime(df["start_datetime"])

    phase = d_meta.loc[d_meta["name"] == dpl].reset_index()
    data = df.loc[df["name"] == dpl].reset_index()
    start_date = phase.loc[0, "deployment_date"]
    end_date = phase.loc[0, "recovery_date"]

    # Calculate the percentage of collected data on the phase length of time
    if data.empty:
        percentage_data = 0
        msg = "No data for this phase"
    else:
        df_end = data.loc[data.index[-1], "start_datetime"]
        df_start = data.loc[data.index[0], "start_datetime"]
        act_length = df_end - df_start
        p_length = end_date - start_date
        percentage_data = act_length * 100 / p_length
        msg = f"Percentage of usable data : {percentage_data}%"

    logger.info(msg)
    return percentage_data


def meta_cut_aplose(
    d_meta: DataFrame,
    df: DataFrame,
) -> DataFrame:
    """From APLOSE DataFrame with all rows to filtered DataFrame.

    Parameters
    ----------
    df: DataFrame
        CPOD result dataframe
    d_meta: DataFrame
        Metadata dataframe with deployments information (previously exported as json)

    Returns
    -------
    DataFrame
        An APLOSE DataFrame with data from beginning to end of each deployment.
        Returns the percentage of usable datas.

    """
    d_meta.loc[:, ["deployment_date", "recovery_date"]] = d_meta[
        ["deployment_date", "recovery_date"]
    ].apply(to_datetime)
    df["start_datetime"] = to_datetime(
        df["start_datetime"],
        format=TIMESTAMP_FORMAT_AUDIO_FILE,
    )

    # Add DPM column
    df["DPM"] = (df["Nfiltered"] > 0).astype(int)

    # Extract corresponding line
    campaign = df.iloc[0]["dataset"]
    phase = d_meta.loc[d_meta["name"] == campaign].reset_index()
    start_date = phase.loc[0, "deployment_date"]
    end_date = phase.loc[0, "recovery_date"]
    df = df[
        (df["start_datetime"] >= start_date) & (df["start_datetime"] <= end_date)
    ].copy()

    # Calculate the percentage of collected data on the phase length of time
    if df.empty:
        msg = "No data for this phase"
    else:
        df_end = df.loc[df.index[-1], "start_datetime"]
        df_start = df.loc[df.index[0], "start_datetime"]
        act_length = df_end - df_start
        p_length = end_date - start_date
        percentage_data = act_length * 100 / p_length
        on = int(df.loc[df.MinsOn == 1, "MinsOn"].count())
        percentage_on = percentage_data * (on / len(df))
        msg = f"Percentage of usable data : {percentage_on}%"

    logger.info(msg)
    return df


def format_calendar(path: Path) -> DataFrame:
    """Format calendar.

    Parameters
    ----------
    path: Path
        Excel calendar path

    """
    df_calendar = read_excel(path)
    df_calendar = df_calendar[df_calendar["Site group"] == "Data"].copy()

    return df_calendar.rename(
        columns={
            "Start": "start_datetime",
            "Stop": "end_datetime",
            "Site": "site.name",
        },
    )


def dpm_to_dph(
    df: DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 3600,
    extra_columns: list | None = None,
) -> DataFrame:
    """From CPOD result DataFrame to APLOSE formatted DataFrame.

    Parameters
    ----------
    df: DataFrame
        CPOD result DataFrame
    tz: pytz.BaseTzInfo
        Timezone object to get timezone-aware datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int
        Duration of the detections in seconds
    extra_columns: list, optional
        Additional columns added from df to data

    Returns
    -------
    DataFrame
        An APLOSE DataFrame

    """
    df["start_datetime"] = to_datetime(df["start_datetime"], utc=True)
    df["end_datetime"] = to_datetime(df["end_datetime"], utc=True)
    df["Date heure"] = df["start_datetime"].dt.floor("h")
    dph = df.groupby(["Date heure"])["DPM"].sum().reset_index()
    dph["Date heure"] = dph["Date heure"].apply(
        lambda x: Timestamp(x).strftime(format="%d/%m/%Y %H:%M:%S"),
    )

    return cpod2aplose(dph, tz, dataset_name, annotation, bin_size, extra_columns)


def assign_phase(
    meta: DataFrame,
    data: DataFrame,
    site: str,
) -> DataFrame:
    """Add a column to an APLOSE DataFrame to specify the name of the phase.

    The name of the phase is attributed according to metadata.

    Parameters
    ----------
    meta: DataFrame
        Metadata dataframe with deployments information (previously exported as json).
    data: DataFrame
        Contain positive hours to detections.
    site: str
        Name of the site you wish to assign phases to.

    Returns
    -------
    DataFrame
        The same dataframe with the column Phase.

    """
    data["start_datetime"] = to_datetime(data["start_datetime"], utc=True)
    meta["deployment_date"] = to_datetime(meta["deployment_date"], utc=True)
    meta["recovery_date"] = to_datetime(meta["recovery_date"], utc=True)

    meta = meta[meta["site.name"] == site].copy()

    data["name"] = None
    for _, meta_row in meta.iterrows():
        j = 0
        while j < len(data):
            if (
                meta_row["deployment_date"]
                <= data.loc[j, "start_datetime"]
                < meta_row["recovery_date"]
            ):
                data.loc[j, "name"] = meta_row["name"]
            j += 1
    return data


def assign_phase_simple(
    meta: DataFrame,
    data: DataFrame,
) -> DataFrame:
    """Add column to an Aplose DataFrame to specify the phase, according to metadata.

    Parameters
    ----------
    meta: DataFrame
        Metadata dataframe with deployments information (previously exported as json).
    data: DataFrame
        Contain positive hours to detections.

    Returns
    -------
    DataFrame
        The same dataframe with the column Phase.

    """
    data["start_datetime"] = to_datetime(data["start_datetime"], utc=True)
    data["end_datetime"] = to_datetime(data["end_datetime"], dayfirst=True, utc=True)
    meta["deployment_date"] = to_datetime(meta["deployment_date"], utc=True)
    meta["recovery_date"] = to_datetime(meta["recovery_date"], utc=True)
    meta["deployment_date"] = meta["deployment_date"].dt.floor("d")
    meta["recovery_date"] = meta["recovery_date"].dt.floor("d")

    data["name"] = None
    for site in data["site.name"].unique():
        site_meta = meta[meta["site.name"] == site]
        site_data = data[data["site.name"] == site]

        for _, meta_row in site_meta.iterrows():
            time_filter = (
                meta_row["deployment_date"] <= site_data["start_datetime"]
            ) & (site_data["start_datetime"] < meta_row["recovery_date"])
            data.loc[site_data.index[time_filter], "name"] = meta_row["name"]

    return data


def generate_hourly_detections(meta: DataFrame, site: str) -> DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    meta: DataFrame
        Metadata dataframe with deployments information (previously exported as json)
    site: str
        A way to isolate the site you want to work on.

    Returns
    -------
    DataFrame
        A full period of time with positive and negative hours to detections.

    """
    df_meta = meta[meta["site.name"] == site].copy()
    df_meta["deployment_date"] = to_datetime(df_meta["deployment_date"])
    df_meta["recovery_date"] = to_datetime(df_meta["recovery_date"])
    df_meta["deployment_date"] = df_meta["deployment_date"].dt.floor("h")
    df_meta["recovery_date"] = df_meta["recovery_date"].dt.floor("h")
    df_meta = df_meta.sort_values(by=["deployment_date"])

    records = [
        {"name": row["name"], "start_datetime": date}
        for _, row in df_meta.iterrows()
        for date in date_range(
            start=row["deployment_date"],
            end=row["recovery_date"],
            freq="h",
        )
    ]

    return DataFrame(records)


def merging_tab(meta: DataFrame, data: DataFrame) -> DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    meta: DataFrame
        Metadata with deployments information (previously exported as json)
    data: DataFrame
        Contain positive hours to detections

    Returns
    -------
    DataFrame
        A full period of time with positive and negative hours to detections.

    """
    data["start_datetime"] = to_datetime(data["start_datetime"], utc=True)
    meta["start_datetime"] = to_datetime(meta["start_datetime"], utc=True)

    deploy_detec = data["name"].unique()
    df_filtered = meta[meta["name"].isin(deploy_detec)]

    output = df_filtered.merge(
        data[["name", "start_datetime", "DPM", "Nfiltered"]],
        on=["name", "start_datetime"],
        how="outer",
    )
    output["DPM"] = output["DPM"].fillna(0)
    output["Nfiltered"] = output["Nfiltered"].fillna(0)

    output["Day"] = output["start_datetime"].dt.day
    output["Month"] = output["start_datetime"].dt.month
    output["Year"] = output["start_datetime"].dt.year
    output["hour"] = output["start_datetime"].dt.hour

    return output


def feeding_buzz(df: DataFrame, species: str) -> DataFrame:
    """Process a CPOD/FPOD feeding buzz detection file.

    Gives the feeding buzz duration, depending on the studied species.

    Parameters
    ----------
    df: DataFrame
        Path to cpod.exe feeding buzz file
    species: str
        Select the species to use between porpoise and Commerson's dolphin

    Returns
    -------
    DataFrame
        Containing all ICIs for every positive minutes to clicks

    """
    df.columns = df.columns.str.upper()
    df["MICROSEC"] = df["MICROSEC"] / 1e6
    col = "DATE HEURE MINUTE"
    col2 = "HEURE MINUTE"
    if col in df.columns:
        df[["DATE", "HEURE", "MINUTE"]] = df[col].str.split(" ", expand=True)
        df["Time"] = (
            df["DATE"].astype(str)
            + " "
            + df["HEURE"].astype(str)
            + ":"
            + df["MINUTE"].astype(str)
            + ":"
            + df["MICROSEC"].astype(str)
        )
        df["Time"] = to_datetime(df["Time"], dayfirst=True)
    elif col2 in df.columns:
        df[["HEURE", "MINUTE"]] = df[col2].str.split(" ", expand=True)
        df["Time"] = (
            df["DATE"].astype(str)
            + " "
            + df["HEURE"].astype(str)
            + ":"
            + df["MINUTE"].astype(str)
            + ":"
            + df["MICROSEC"].astype(str)
        )
        df["Time"] = to_datetime(df["Time"], dayfirst=True)
    else:
        df["Time"] = df["MINUTE"].astype(str) + ":" + df["MICROSEC"].astype(str)
        df["Time"] = to_datetime(df["Time"], dayfirst=True)

    df = df.sort_values(by="Time").reset_index(drop=True)
    df["ICI"] = df["Time"].diff().dt.total_seconds()

    df["Buzz"] = 0
    if species == "Porpoise":
        feeding_idx = df.index[df["ICI"] < 0.01]
    else:
        feeding_idx = df.index[df["ICI"] >= 0.005]

    df.loc[feeding_idx, "Buzz"] = 1
    df.loc[feeding_idx - 1, "Buzz"] = 1
    df.loc[df.index < 0, "Buzz"] = 0

    df["start_datetime"] = df["Time"].dt.floor("min")
    df["start_datetime"] = to_datetime(df["start_datetime"], dayfirst=False, utc=True)
    f = df.groupby(["start_datetime"])["Buzz"].sum().reset_index()

    f["Foraging"] = (f["Buzz"] != 0).astype(int)

    return f


def assign_daytime(
    df: DataFrame,
) -> DataFrame:
    """Assign datetime categories to events.

    Categorize daytime of the detection (among 4 categories).

    Parameters
    ----------
    df: DataFrame
        Contains positive hours to detections.

    Returns
    -------
    DataFrame
        The same dataframe with the column daytime.

    """
    start = df.iloc[0]["Time"]
    stop = df.iloc[-1]["Time"]
    lat, lon = get_coordinates()
    _, _, dawn, day, dusk, night = get_sun_times(start, stop, lat, lon)
    dawn = Series(dawn, name="dawn")
    day = Series(day, name="day")
    dusk = Series(dusk, name="dusk")
    night = Series(night, name="night")
    jour = concat([day, night, dawn, dusk], axis=1)

    for i, row in df.iterrows():
        dpm_i = row["Time"]
        if notna(dpm_i):  # Check if time is not NaN
            jour_i = jour[
                (jour["dusk"].dt.year == dpm_i.year)
                & (jour["dusk"].dt.month == dpm_i.month)
                & (jour["dusk"].dt.day == dpm_i.day)
            ]
            if not jour_i.empty:  # Ensure there"s a matching row
                jour_i = jour_i.iloc[0]  # Extract first match
                if dpm_i <= jour_i["day"]:
                    df.loc[i, "REGIME"] = 1
                elif dpm_i < jour_i["dawn"]:
                    df.loc[i, "REGIME"] = 2
                elif dpm_i < jour_i["dusk"]:
                    df.loc[i, "REGIME"] = 3
                elif dpm_i > jour_i["night"]:
                    df.loc[i, "REGIME"] = 1
                elif dpm_i > jour_i["dusk"]:
                    df.loc[i, "REGIME"] = 4
                else:
                    df.loc[i, "REGIME"] = 1

    return df


def process_files_in_folder(folder_path: Path, species: str) -> DataFrame:
    """Process a folder containing all CPOD/FPOD feeding buzz detection files.

    Apply the feeding buzz function to these files.

    Parameters
    ----------
    folder_path: Path
        Path to the folder.
    species: str
        Select the species to use between porpoise and Commerson's dolphin

    Returns
    -------
    DataFrame
       Compiled feeding buzz detection positive minutes.

    """
    all_files = list(Path(folder_path).rglob("*.txt"))
    all_data = []

    for file in all_files:
        file_path = folder_path / file
        df = read_csv(file_path, sep="\t")
        processed_df = feeding_buzz(df, species)
        processed_df["file"] = file
        all_data.append(processed_df)

    return concat(all_data, ignore_index=True)


colors = {
    "DY1": "#118B50",
    "DY2": "#5DB996",
    "DY3": "#B0DB9C",
    "DY4": "#E3F0AF",
    "CA4": "#5EABD6",
    "Walde": "#FFB4B4",
}


def extract_site(df: DataFrame) -> DataFrame:
    """Create new columns: site.name and campaign.name, in order to match the metadata.

    Parameters
    ----------
    df: DataFrame
        All values concatenated

    Returns
    -------
    DataFrame
        The same dataframe with two additional columns.

    """
    df[["site.name", "campaign.name"]] = df["name"].str.split("_", expand=True)
    return df


def percent_calc(data: DataFrame, time_unit: str | None = None) -> DataFrame:
    """Calculate percentage of clicks, feeding buzzes and positive hours to detection.

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
        .agg({
            "DPH": "sum",
            "DPM": "sum",
            "Day": "size",
            "Foraging": "sum",
        })
        .reset_index()
    )

    df["%click"] = df["DPM"] * 100 / (df["Day"] * 60)
    df["%DPH"] = df["DPH"] * 100 / df["Day"]
    df["FBR"] = df["Foraging"] * 100 / df["DPM"]
    df["%buzz"] = df["Foraging"] * 100 / (df["Day"] * 60)
    return df


def site_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with percentage of minutes positive to detection for every site.

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
        palette=colors,
    )
    ax.set_title(f"{metric} per site")
    ax.set_ylabel(f"{metric}")
    if metric == "%buzzes":
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
            color=colors.get(site, "gray"),
        )
        ax.set_title(f"Site {site}")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Year")
        if metric == "%buzzes":
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per year", fontsize=16)
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
            color=colors.get(site, "gray"),
        )
        ax.set_title(f"{site} - Percentage of postitive to detection minutes per month")
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
        if metric == "%buzzes":
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
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
            site_data["hour"],
            site_data[metric],
            label=f"Site {site}",
            color=colors.get(site, "gray"),
        )
        ax.set_title(f"Site {site} - Percentage of positive to detection per hour")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Hour")
        if metric == "%buzzes":
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per hour", fontsize=16)
    plt.show()
