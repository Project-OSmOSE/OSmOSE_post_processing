from pathlib import Path

from post_processing.dataclass.detection_filter import DetectionFilter


def test_from_yaml(sample_yaml: Path,
                   sample_csv_result: Path,
                   sample_csv_timestamp: Path,
                   ) -> None:
    config = DetectionFilter.from_yaml(sample_yaml)

    param = {
        f"{sample_csv_result}": {
            "timebin_new": None,
            "begin": None,
            "end": None,
            "annotator": "ann1",
            "annotation": "lbl1",
            "box": False,
            "timestamp_file": f"{sample_csv_timestamp}",
            "user_sel": "all",
            "f_min": None,
            "f_max": None,
            "score": None,
        },
    }

    expected = DetectionFilter.from_dict(param)
    assert config == expected
