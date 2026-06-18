"""Functions to manipulate Sylence data."""

from osekit.utils.timestamp_utils import strptime_from_text
from pandas import DataFrame


def clean_pamguard_false_detection(df: DataFrame) -> DataFrame:
    """Clean PAMGuard false detections.

    The first detection on each audio file corresponds to the detection of
    an electronic buzz made by the recorder (might be very specific to Sylence data).
    This function identifies and removes these false detections by checking if
    a detection occurs within the first five seconds of the corresponding audio file.

    Parameters
    ----------
    df: DataFrame
        APLOSE DataFrame (presumably from PAMGuard)

    Returns
    -------
    A cleaned DataFrame with false detections removed

    """
    filenames = df["filename"]
    tz_data = df["start_datetime"][0].tz
    filename_datetimes = [
        strptime_from_text(fn, "%Y-%m-%d_%H-%M-%S").tz_localize(tz_data)
        for fn in filenames
    ]

    start_datetimes = df["start_datetime"]

    # compare date of filename detection and date of detection and delete all lines
    # for which the detection happens in the first second of the file
    max_time = 1
    idx_false_detections = []
    for i in range(len(start_datetimes)):
        d = (start_datetimes[i] - filename_datetimes[i]).total_seconds()
        if d < max_time:
            idx_false_detections.append(i)

    return df.drop(labels=idx_false_detections)
