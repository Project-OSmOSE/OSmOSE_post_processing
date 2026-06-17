from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from osekit.core_api.audio_data import AudioData
from pandas import DataFrame, Timestamp
from pypamguard.chunks.generics import GenericModule

from post_processing.utils.pamguard import process_binary


@pytest.fixture
def fake_audio() -> AudioData:
    file = MagicMock()
    file.begin = Timestamp("2025-05-29T00:00:00+0000")
    file.end = Timestamp("2025-05-29T01:00:00+0000")
    file.sample_rate = 48000
    file.path.name = "fake.wav"

    audio = MagicMock()
    audio.begin = file.begin
    audio.end = file.end
    audio.files = [file]
    return audio


@pytest.fixture
def fake_detection() -> GenericModule:
    det = MagicMock()
    det.date = "2025-05-29T00:10:00+0000"
    det.sample_duration = 48000
    det.freq_limits = (1000, 5000)
    return det


def test_process_binary_basic(
    fake_audio: AudioData, fake_detection: GenericModule
) -> None:
    with patch(
        "post_processing.utils.pamguard.load_pamguard_binary_folder"
    ) as mock_loader:
        mock_loader.return_value = ([fake_detection], None, None)

        df = process_binary(fake_audio, Path("/fake/binary"), "Dataset", "Label")

        assert isinstance(df, DataFrame)
        expected_cols = {
            "dataset",
            "filename",
            "start_time",
            "end_time",
            "min_frequency",
            "max_frequency",
            "annotation",
            "annotator",
            "start_datetime",
            "end_datetime",
            "type",
        }
        assert set(df.columns) == expected_cols

        row = df.iloc[0]
        assert row["dataset"] == "Dataset"
        assert row["filename"] == "fake.wav"
        assert row["min_frequency"] == 1000
        assert row["max_frequency"] == 5000
        assert row["annotation"] == "Label"
        assert row["type"]


def test_process_binary_no_detections(fake_audio: AudioData) -> None:
    with patch(
        "post_processing.utils.pamguard.load_pamguard_binary_folder"
    ) as mock_loader:
        mock_loader.return_value = ([], None, None)
        df = process_binary(fake_audio, Path("/fake/binary"), "Dataset", "Label")
        assert df.empty


def test_process_binary_detection_outside_audio(
    fake_audio: AudioData, fake_detection: GenericModule
) -> None:
    fake_detection.date = "2025-05-28T23:59:00+0000"

    with patch(
        "post_processing.utils.pamguard.load_pamguard_binary_folder"
    ) as mock_loader:
        mock_loader.return_value = ([fake_detection], None, None)
        with pytest.raises(AttributeError):
            process_binary(fake_audio, Path("/fake/binary"), "Dataset", "Label")
