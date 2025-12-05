from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from post_processing.utils.audio_utils import create_raven_file_list, normalize_audio


def test_normalize_audio_default_folder(sample_audio: Path, tmp_path: Path) -> None:
    normalize_audio(sample_audio)

    normalized_file = tmp_path / (sample_audio.stem + "_normalized.wav")
    assert normalized_file.exists()

    data, _ = sf.read(normalized_file)
    assert np.isclose(np.max(np.abs(data)), 1.0, rtol=1e-4)


def test_normalize_audio_custom_folder(sample_audio: Path, tmp_path: Path) -> None:
    out_folder = tmp_path / "output"
    normalize_audio(sample_audio, output_folder=out_folder)

    normalized_file = out_folder / sample_audio.name
    assert normalized_file.exists()

    data, _ = sf.read(normalized_file)
    assert np.isclose(np.max(np.abs(data)), 1.0, rtol=1e-4)


def test_normalize_audio_invalid_file(tmp_path: Path) -> None:
    invalid_file = tmp_path / "invalid.wav"
    invalid_file.write_text("not audio data")

    with pytest.raises(ValueError, match="An error occurred while reading file"):
        normalize_audio(invalid_file)


def test_create_raven_file_list(tmp_audio_dir: Path) -> None:
    create_raven_file_list(tmp_audio_dir)

    file_list_path = tmp_audio_dir / "Raven_file_list.txt"
    assert file_list_path.exists(), "Raven_file_list.txt should exist"

    with file_list_path.open() as f:
        lines = [line.strip() for line in f.readlines()]

    expected_files = [
        tmp_audio_dir / "file1.wav",
        tmp_audio_dir / "file2.wav",
        tmp_audio_dir / "nested" / "file3.wav",
        tmp_audio_dir / "nested" / "file4.wav",
    ]
    expected_files_str = [str(f) for f in expected_files]

    assert set(lines) == set(expected_files_str), "All WAV files should be listed"
    assert all(line.endswith(".wav") for line in lines), "Only WAV files should be listed"


def test_create_raven_file_list_empty_dir(tmp_path: Path) -> None:
    create_raven_file_list(tmp_path)

    file_list_path = tmp_path / "Raven_file_list.txt"
    assert file_list_path.exists(), "Raven_file_list.txt should exist"
    with file_list_path.open() as f:
        lines = f.readlines()
    assert lines == [], "Empty directory should produce empty file list"
