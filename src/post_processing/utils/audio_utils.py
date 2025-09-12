from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from post_processing import logger

if TYPE_CHECKING:
    from pathlib import Path


def normalize_audio(file: Path, output_folder: Path | None = None) -> None:
    """Normalize the audio data of a file.

    Parameters
    ----------
    file : Path
        The path of the audio file to normalize
    output_folder : Path
        The path to output destination

    """
    try:
        data, fs = sf.read(file)
    except sf.LibsndfileError as e:
        msg = f"An error occurred while reading file {file}: {e}"
        raise ValueError(msg) from e

    if not output_folder:
        output_folder = file.parent

    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    format_file = sf.info(file).format
    subtype = sf.info(file).subtype

    data_norm = np.transpose(np.array([data / np.max(np.abs(data))]))[:, 0]

    new_fn = (
        file.stem + ".wav"
        if output_folder != file.parent
        else file.stem + "_normalized.wav"
    )
    new_path = output_folder / new_fn
    sf.write(
        file=new_path,
        data=data_norm,
        samplerate=fs,
        subtype=subtype,
        format=format_file,
    )

    logger.info("File '%s' exported in '%s'", new_fn, file.parent)


def create_raven_file_list(directory: Path) -> None:
    """Create a text file with reference to audio in directory.

    The test file contained the paths of audio files in directory and all subfolders.
    This is useful to open several audio located in different subfolders in Raven.

    Parameters
    ----------
    directory: Path
        Folder containing all audio data

    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # get file list. Don't take the 1ko files into account, because Raven can't open them (makes it crash)
    files = [f for f in directory.rglob("*.wav") if f.stat().st_size > 1024]

    # save file list as a txt file
    filename = directory / "Raven_file_list.txt"
    with filename.open(mode="w") as f:
        for item in files:
            f.write(f"{item}\n")

    logger.info("File list saved in '%s'", directory)
