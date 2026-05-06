from pathlib import Path

from osekit.core.audio_data import AudioData
from osekit.utils.timestamp import strftime_osmose_format
from pandas import DataFrame, Timedelta, Timestamp
from pypamguard import load_pamguard_binary_folder
from pypamguard.core.filters import DateFilter, Filters
from pypamguard.logger import Verbosity, logger
from tqdm import tqdm

logger.set_verbosity(Verbosity.ERROR)


def process_binary(
    audio: AudioData,
    binary: Path,
    dataset: str,
    annotation: str,
) -> DataFrame:
    r"""Process PAMGuard binary files into APLOSE DataFrame.

    Parameters
    ----------
    audio : AudioData
        Osekit AudioData object built from audio files
    binary : Path
        Path to the PAMGuard binary files
    dataset : str
        Dataset name
    annotation : str
        Annotation label

    Returns
    -------
    DataFrame
        APLOSE-formatted DataFrame containing PAMGuard detections

    Examples
    --------
    >>> from pathlib import Path
    >>> from osekit.core_api.audio_file import AudioFile
    >>> from osekit.core_api.audio_data import AudioData

    >>> audio_path = Path(r"path/to/audio/folder")
    >>> binary_path = Path(r"path/to/binary/folder")

    >>> dataset = "dataset_name"
    >>> annotation = "label_name"
    >>> datetime_format = "%Y-%m-%dT%H:%M:%S"

    >>> begin = Timestamp("2025-05-29T00:00:00+0000")
    >>> end = Timestamp("2025-05-30T00:00:00+0000")

    >>> audio_files = [
    ...     AudioFile(path=f,
    ...               strptime_format=datetime_format,
    ...               timezone=begin.tz)
    ...     for f in audio_path.rglob("*/*.wav")
    ... ]

    >>> ad = AudioData.from_files(files=audio_files, begin=begin, end=end)

    >>> df = process_binary(ad, binary_path, dataset, annotation)

    """
    filter_obj = Filters(
        {
            "daterange": DateFilter(
                start_date=audio.begin, end_date=audio.end, ordered=True
            ),
        },
    )

    data, _, _ = load_pamguard_binary_folder(binary, r"**/*.pgdf", filters=filter_obj)

    (
        start_datetimes,
        start_times,
        end_datetimes,
        end_times,
        durations,
        freq_min,
        freq_max,
        filenames,
        annotator,
    ) = ([], [], [], [], [], [], [], [], [])

    for d in tqdm(data, desc="Creating DataFrame"):
        begin = Timestamp(d.date)
        start_datetimes.append(begin)

        matching_file = None
        for f in audio.files:
            if f.begin <= begin <= f.end:
                matching_file = f
                break

        duration = Timedelta(d.sample_duration / matching_file.sample_rate, "s")
        durations.append(duration)
        end_datetimes.append(begin + duration)

        freq_min.append(d.freq_limits[0])
        freq_max.append(d.freq_limits[1])

        start_time = (begin - matching_file.begin).total_seconds()
        end_time = round(start_time + duration.total_seconds(), 3)
        start_times.append(start_time)
        end_times.append(end_time)

        filenames.append(matching_file.path.name)

        annotator.append(type(d).__name__)

    return DataFrame({
        "dataset": dataset,
        "filename": filenames,
        "start_time": start_times,
        "end_time": end_times,
        "start_frequency": freq_min,
        "end_frequency": freq_max,
        "annotation": annotation,
        "annotator": annotator,
        "start_datetime": [strftime_osmose_format(beg) for beg in start_datetimes],
        "end_datetime": [strftime_osmose_format(end) for end in end_datetimes],
        "is_box": True,
    }).sort_values("start_datetime")
