import pytest
from pandas import DataFrame, Timedelta, Timestamp, date_range

from post_processing.utils.formatting_utils import aplose2raven


@pytest.fixture
def aplose_dataframe() -> DataFrame:
    data = DataFrame(
        {
            "dataset": ["dataset_test", "dataset_test", "dataset_test", "dataset_test"],
            "filename": ["file1.wav", "file2.wav", "file3.wav", "file4.wav"],
            "start_time": [0, 0, 5.9, 0],
            "end_time": [30, 30, 8.1, 30],
            "start_frequency": [0, 0, 18500.0, 0],
            "end_frequency": [96000, 96000, 53000.0, 96000],
            "annotation": ["boat", "boat", "boat", "boat"],
            "annotator": ["bbjuni", "bbjuni", "bbjuni", "bbjuni"],
            "start_datetime": [
                Timestamp("2020-05-29T11:30:00.000+00:00"),
                Timestamp("2020-05-29T11:31:00.000+00:00"),
                Timestamp("2020-05-29T11:31:05.900+00:00"),
                Timestamp("2020-05-29T11:32:50.000+00:00"),
            ],
            "end_datetime": [
                Timestamp("2020-05-29T11:30:30.000+00:00"),
                Timestamp("2020-05-29T11:31:30.000+00:00"),
                Timestamp("2020-05-29T11:31:08.100+00:00"),
                Timestamp("2020-05-29T11:33:20.000+00:00"),
            ],
            "is_box": [0, 0, 1, 0],
        },
    )

    return data.reset_index(drop=True)


@pytest.fixture
def audio_timestamps() -> list:
    return list(
        date_range(
            start="2020-05-29T11:30:00.000+00:00",
            end="2020-05-29T11:35:00.000+00:00",
            freq="1min",
        ),
    )


@pytest.fixture
def audio_durations(audio_timestamps: list[Timestamp]) -> list:
    return [Timedelta("30s")] * len(audio_timestamps)


@pytest.mark.unit
def test_aplose2raven(
    aplose_dataframe: DataFrame,
    audio_timestamps: list[Timestamp],
    audio_durations: list[Timedelta],
) -> None:
    raven_dataframe = aplose2raven(
        aplose_result=aplose_dataframe,
        list_audio_begin_time=audio_timestamps,
        audio_durations=audio_durations,
    )

    expected_raven_dataframe = DataFrame(
        {
            "Selection": [1, 2, 3, 4],
            "View": [1, 1, 1, 1],
            "Channel": [1, 1, 1, 1],
            "Begin Time (s)": [0.0, 30.0, 35.9, 90.0],
            "End Time (s)": [30.0, 60.0, 38.1, 110.0],
            "Low Freq (Hz)": [0.0, 0.0, 18500.0, 0.0],
            "High Freq (Hz)": [96000.0, 96000.0, 53000.0, 96000.0],
            "Begin Date Time Real": aplose_dataframe["start_datetime"],
        },
    )

    assert expected_raven_dataframe.equals(raven_dataframe)
