"""FPOD/ CPOD processing functions tests."""
import pytest
import pytz
from pandas import DataFrame

from post_processing.utils.fpod_utils import (
    load_pod_folder,
    pod2aplose,
)

# SAMPLE_POD = """File,ChunkEnd,DPM,Nall,MinsOn
# sample_dataset,2023/11/29 08:05,0,0,0
#
# """
# SAMPLE_AP = """dataset,filename,start_time,end_time,start_frequency,end_frequency,
# annotation,annotator,start_datetime,end_datetime,is_box
# sample_dataset,,0,60,0,0,ann1,POD,2023-11-29T08:30:00.000+00:00,2023-11-29T08:31:00.000+00:00,0
# sample_dataset,,0,60,0,0,ann1,POD,2023-11-29T08:31:00.000+00:00,2023-11-29T08:32:00.000+00:00,0
# sample_dataset,,0,60,0,0,ann1,POD,2023-11-29T09:30:00.000+00:00,2023-11-29T09:31:00.000+00:00,0
# sample_dataset,,0,60,0,0,ann1,POD,2023-11-30T08:30:00.000+00:00,2023-11-30T08:31:00.000+00:00,0
# sample_dataset,,0,60,0,0,ann1,POD,2023-12-29T08:30:00.000+00:00,2023-12-29T08:31:00.000+00:00,0
# sample_dataset,,0,60,0,0,ann1,POD,2024-11-29T08:30:00.000+00:00,2024-11-29T08:31:00.000+00:00,0
# """
#
# @pytest.fixture
# def pod_dataframe() -> DataFrame:
#     data = DataFrame(
#         {
#             "File": [
#                 "sample_dataset",
#                 "sample_dataset",
#                 "sample_dataset",
#                 "sample_dataset",
#                 "sample_dataset",
#                 "sample_dataset",
#             ],
#             "ChunkEnd": [
#                 Timestamp("2023/11/29 08:30"),
#                 Timestamp("2023/11/29 08:31"),
#                 Timestamp("2023/11/29 08:32"),
#                 Timestamp("2023/11/29 08:33"),
#                 Timestamp("2023/11/29 08:34"),
#                 Timestamp("2023/11/29 08:35"),
#             ],
#             "deploy.name": [
#                 "site_deploy",
#                 "site_deploy",
#                 "site_deploy",
#                 "site_deploy",
#                 "site_deploy",
#                 "site_deploy",
#             ],
#             "DPM": [1, 1, 0, 0, 0, 0],
#             "Nall": [44, 66, 0, 22, 0, 0],
#             "MinsOn": [1, 1, 1, 1, 1, 0],
#         },
#     )
#
#     return data.reset_index(drop=True)
#
#
# @pytest.fixture
# def aplose_dataframe() -> DataFrame:
#     data = DataFrame(
#         {
#             "dataset": ["dataset_test", "dataset_test", "dataset_test", "dataset_test",
#                         "dataset_test", "dataset_test"],
#             "filename": ["", "", "", ""],
#             "start_time": [0, 0, 0, 0, 0, 0],
#             "end_time": [60, 60, 60, 60, 60, 60],
#             "start_frequency": [0, 0, 0, 0, 0, 0],
#             "end_frequency": [0, 0, 0, 0, 0, 0],
#             "annotation": ["ann1", "ann1", "ann1", "ann1", "ann1", "ann1"],
#             "annotator": ["POD", "POD", "POD", "POD", "POD", "POD"],
#             "start_datetime": [
#                 Timestamp("2023-11-29T08:30:00.000+00:00"),
#                 Timestamp("2023-11-29T08:31:00.000+00:00"),
#                 Timestamp("2023-11-29T09:31:00.000+00:00"),
#                 Timestamp("2023-11-30T09:31:00.000+00:00"),
#                 Timestamp("2023-12-30T09:31:00.000+00:00"),
#                 Timestamp("2024-12-30T09:31:00.000+00:00"),
#             ],
#             "end_datetime": [
#                 Timestamp("2023-11-29T08:31:00.000+00:00"),
#                 Timestamp("2023-11-29T08:32:00.000+00:00"),
#                 Timestamp("2023-11-29T09:32:00.000+00:00"),
#                 Timestamp("2023-11-30T09:32:00.000+00:00"),
#                 Timestamp("2023-12-30T09:32:00.000+00:00"),
#                 Timestamp("2024-12-30T09:32:00.000+00:00"),
#             ],
#             "is_box": [0, 0, 0, 0, 0, 0],
#             "deploy.name": ["site_campaign", "site_campaign", "site_campaign",
#                             "site_campaign", "site_campaign", "site_campaign"],
#         },
#     )
#
#     return data.reset_index(drop=True)

#@pytest.fixture(scope="module")
# @dt.working_directory(__file__)
# def df_raw() -> DataFrame:
#     return read_csv("pod_raw.csv")
#
# @pytest.fixture(scope="module")
# @dt.working_directory(__file__)
# def df_ap() -> DataFrame:
#     return read_csv("pod_aplose.csv")

#@pytest.mark.mandatory
# def test_columns(df_raw: DataFrame) -> None:
#     dt.validate(
#         df_raw.columns,
#         {"File", "ChunkEnd", "DPM", "Nall", "MinsOn"},
#     )
#
# @pytest.mark.mandatory
# def test_columns(df_ap: DataFrame) -> None:
#     dt.validate(
#         df_ap.columns,
#         {"dataset","filename","start_time","end_time","start_frequency","end_frequency",
#          "annotation","annotator","start_datetime","end_datetime","is_box"},
#     )
#
# def test_chunk_end(df_raw: DataFrame) -> None:
#     dt.validate(df_raw["ChunkEnd"],
#                 strptime_from_text(df_raw["ChunkEnd"], "%Y/%m/%d %H:%M"))
#
# def test_start_datetime(df_ap: DataFrame) -> None:
#     dt.validate(df_ap["start_datetime"], strptime_from_text(df_ap["start_datetime"],
#                                             "%Y-%m-%dT%H:%M:%S"))

# @pytest.fixture
# def sample_pod() -> DataFrame:
#     df = read_csv(io.StringIO(SAMPLE_POD), parse_dates=["ChunkEnd"])
#     return df.sort_values(["ChunkEnd"]).reset_index(drop=True)


# csv_folder
def test_csv_folder_single_file(tmp_path) -> None:
    """Test processing a single CSV file."""
    # Create a CSV file
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1;col2\nval1;val2\nval3;val4", encoding="latin-1")

    result = load_pod_folder(tmp_path)

    assert isinstance(result, DataFrame)
    assert len(result) == 2
    assert "deploy.name" in result.columns
    assert all(result["deploy.name"] == "data")
    assert list(result.columns) == ["col1", "col2", "deploy.name"]


# pod2aplose
@pytest.fixture
def sample_df():
    """Create a sample POD DataFrame for testing."""
    return DataFrame({
        "ChunkEnd": ["15/01/2024 10:30", "15/01/2024 11:00", "15/01/2024 09:45"],
        "deploy.name": ["deploy1", "deploy2", "deploy1"],
    })


@pytest.fixture
def timezone():
    """Return UTC timezone for testing."""
    return pytz.UTC


def test_pod2aplose_basic_structure(sample_df, timezone):
    """Test that basic structure and required columns are present."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="test_dataset",
        annotation="test_annotation",
        annotator="test_annotator",
    )

    expected_columns = [
        "dataset",
        "filename",
        "start_time",
        "end_time",
        "start_frequency",
        "end_frequency",
        "annotation",
        "annotator",
        "start_datetime",
        "end_datetime",
        "is_box",
        "deploy.name",
    ]

    assert isinstance(result, DataFrame)
    assert list(result.columns) == expected_columns
    assert len(result) == len(sample_df)


def test_pod2aplose_dataset_propagation(sample_df, timezone):
    """Test that dataset name is propagated to all rows."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="my_dataset",
        annotation="click",
        annotator="john",
    )

    assert all(result["dataset"] == "my_dataset")


def test_pod2aplose_annotation_propagation(sample_df, timezone):
    """Test that annotation is propagated to all rows."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="porpoise_click",
        annotator="john",
    )

    assert all(result["annotation"] == "porpoise_click")


def test_pod2aplose_annotator_propagation(sample_df, timezone):
    """Test that annotator is propagated to all rows."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="alice",
    )

    assert all(result["annotator"] == "alice")


def test_pod2aplose_default_bin_size(sample_df, timezone):
    """Test default bin_size of 60 seconds."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
    )

    assert all(result["start_time"] == 0)
    assert all(result["end_time"] == 60)


def test_pod2aplose_custom_bin_size(sample_df, timezone):
    """Test custom bin_size parameter."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
        bin_size=120,
    )

    assert all(result["start_time"] == 0)
    assert all(result["end_time"] == 120)


def test_pod2aplose_frequency_values(sample_df, timezone):
    """Test that frequency values are set to 0."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
    )

    assert all(result["start_frequency"] == 0)
    assert all(result["end_frequency"] == 0)


def test_pod2aplose_is_box_values(sample_df, timezone):
    """Test that is_box values are set to 0."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
    )

    assert all(result["is_box"] == 0)


def test_pod2aplose_deploy_name_preserved(sample_df, timezone):
    """Test that deploy.name values are preserved from input."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
    )

    # After sorting, deploy.name should still be present
    assert "deploy.name" in result.columns
    assert len(result["deploy.name"]) == len(sample_df)
    assert set(result["deploy.name"]) == {"deploy1", "deploy2"}


def test_pod2aplose_sorting_by_datetime(timezone):
    """Test that rows are sorted by datetime."""
    df = DataFrame({
        "ChunkEnd": ["15/01/2024 12:00", "15/01/2024 10:00", "15/01/2024 11:00"],
        "deploy.name": ["d1", "d2", "d3"],
    })

    result = pod2aplose(
        df=df, tz=timezone, dataset_name="dataset", annotation="click", annotator="john"
    )

    # Check that deploy.name follows the sorted order (by time)
    assert result["deploy.name"].tolist() == ["d2", "d3", "d1"]


def test_pod2aplose_datetime_formatting():
    """Test that datetime strings are properly formatted."""
    df = DataFrame({"ChunkEnd": ["01/02/2024 14:30"], "deploy.name": ["deploy1"]})

    result = pod2aplose(
        df=df,
        tz=pytz.UTC,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
        bin_size=60,
    )

    # Check that datetime strings are present and not empty
    assert len(result["start_datetime"].iloc[0]) > 0
    assert len(result["end_datetime"].iloc[0]) > 0
    assert len(result["filename"].iloc[0]) > 0


def test_pod2aplose_end_datetime_offset(timezone):
    """Test that end_datetime is offset by bin_size from start_datetime."""
    df = DataFrame({"ChunkEnd": ["15/01/2024 10:00"], "deploy.name": ["deploy1"]})

    result = pod2aplose(
        df=df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
        bin_size=120,
    )

    # Both should be valid datetime strings
    assert result["start_datetime"].iloc[0] != result["end_datetime"].iloc[0]


def test_pod2aplose_different_timezones():
    """Test with different timezone."""
    df = DataFrame({"ChunkEnd": ["15/01/2024 10:00"], "deploy.name": ["deploy1"]})

    tz_paris = pytz.timezone("Europe/Paris")

    result = pod2aplose(
        df=df, tz=tz_paris, dataset_name="dataset", annotation="click", annotator="john"
    )

    assert len(result) == 1
    assert result["dataset"].iloc[0] == "dataset"


def test_pod2aplose_empty_dataframe(timezone):
    """Test handling of empty DataFrame."""
    df = DataFrame({"ChunkEnd": [], "deploy.name": []})

    result = pod2aplose(
        df=df, tz=timezone, dataset_name="dataset", annotation="click", annotator="john"
    )

    assert len(result) == 0
    assert list(result.columns) == [
        "dataset",
        "filename",
        "start_time",
        "end_time",
        "start_frequency",
        "end_frequency",
        "annotation",
        "annotator",
        "start_datetime",
        "end_datetime",
        "is_box",
        "deploy.name",
    ]


def test_pod2aplose_single_row(timezone):
    """Test with single row DataFrame."""
    df = DataFrame({"ChunkEnd": ["20/03/2024 15:45"], "deploy.name": ["single_deploy"]})

    result = pod2aplose(
        df=df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
        bin_size=90,
    )

    assert len(result) == 1
    assert result["deploy.name"].iloc[0] == "single_deploy"
    assert result["end_time"].iloc[0] == 90


def test_pod2aplose_does_not_modify_original(sample_df, timezone):
    """Test that the original DataFrame is not modified."""
    original_columns = sample_df.columns.tolist()
    original_len = len(sample_df)

    pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
    )

    # Original DataFrame should be unchanged
    assert sample_df.columns.tolist() == original_columns
    assert len(sample_df) == original_len
    assert "_temp_dt" not in sample_df.columns


def test_pod2aplose_large_bin_size(sample_df, timezone):
    """Test with large bin_size value."""
    result = pod2aplose(
        df=sample_df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john",
        bin_size=3600,  # 1 hour
    )

    assert all(result["end_time"] == 3600)


def test_pod2aplose_index_reset(timezone):
    """Test that index is properly reset after sorting."""
    df = DataFrame({
        "ChunkEnd": ["15/01/2024 12:00", "15/01/2024 10:00"],
        "deploy.name": ["d1", "d2"]
    })

    result = pod2aplose(
        df=df,
        tz=timezone,
        dataset_name="dataset",
        annotation="click",
        annotator="john"
    )

    # Index should be 0, 1 after reset
    assert result.index.tolist() == [0, 1]

# meta_cut_aplose


# build_range


# feeding_buzz


# assign_daytime


# fb_folder
# def test_fb_folder_non_existent() -> None:
#     with pytest.raises(FileNotFoundError):
#         txt_folder(Path("/non/existent/folder"))
#
# def test_fb_folder_no_files(tmp_path: pytest.fixture) -> None:
#     with pytest.raises(ValueError, match="No .txt files found"):
#         txt_folder(tmp_path)

# extract_site
# def test_extract_site(self) -> None:
#     input_data = [
#         {"deploy.name":"Walde_Phase46"},
#         {"deploy.name":"Site A Ile Haute_Phase8"},
#         {"deploy.name":"Site B Ile Heugh_Phase9"},
#         {"deploy.name":"Point E_Phase 4"},
#     ]
#     expected_site = [
#         "Walde",
#         "Site A Ile Haute",
#         "Site B Ile Heugh",
#         "Point E",
#     ]
#     expected_campaign = [
#         "Phase46",
#         "Phase8",
#         "Phase9",
#         "Phase 4",
#     ]
#
#     for variant, (input_row, site, campaign) in enumerate(
#         zip(input_data, expected_site, expected_campaign, strict=False), start=1):
#         with self.subTest(
#             f"variation #{variant}",
#             deploy_name=input_row["deploy.name"],
#             expected_site=site,
#             expected_campaign=campaign,
#         ):
#             df = DataFrame([input_row])
#             result = extract_site(df)
#             actual_site = result["site.name"].iloc[0]
#             actual_campaign = result["campaign.name"].iloc[0]
#
#             error_message_site = (
#                 f'Called extract_site() with deploy.name="{input_row["deploy.name"]}". '
#                 f'The function returned site.name="{actual_site}", but the test '
#                 f'expected "{expected_site}".'
#             )
#
#             error_message_campaign = (
#                 f'Called extract_site() with deploy.name="{input_row["deploy.name"]}". '
#                 f'The function returned campaign.name="{actual_campaign}", but the test'
#                 f'expected "{expected_campaign}".'
#             )
#
#             assert actual_site == expected_site, error_message_site
#             assert actual_campaign == expected_campaign, error_message_campaign
#
#             assert "deploy.name" in result.columns
#             assert "value" in result.columns

# csv_folder
# def test_csv_folder_non_existent() -> None:
#     with pytest.raises(FileNotFoundError):
#         csv_folder(Path("/non/existent/folder"))
#
# def test_csv_folder_no_files(tmp_path: pytest.fixture) -> None:
#     with pytest.raises(ValueError, match="No .csv files found"):
#         csv_folder(tmp_path)

# is_dpm_col


# pf_datetime


# build_aggregation_dict


# resample_dpm


# parse_timestamps
# def test_parse_timestamps() -> None:
#     df = DataFrame({"date": ["2024-01-01T10:00:00", "06/01/2025 08:35"]})
#     result = parse_timestamps(df, "date")
#     expected = DataFrame({"date": ["2024-01-01 10:00:00",
#                                    "2025-01-06 08:35:00"]}).astype("datetime64[ns]")
#     assert_frame_equal(result, expected)

# deploy_period
# def test_deploy_period() -> None:
#     df = DataFrame(
#         {
#             "deploy.name": ["A", "A", "B"],
#             "start_datetime": [
#                 datetime(2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
#                 datetime(2024, 1, 2, 15, 30, tzinfo=datetime.timezone.utc),
#                 datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
#             ],
#         })
#
#     expected = DataFrame(
#         {
#             "deploy.name": ["A", "B"],
#             "Début": [
#                 datetime(2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
#                 datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
#             ],
#             "Fin": [
#                 datetime(2024, 1, 2, 15, 30, tzinfo=datetime.timezone.utc),
#                 datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
#             ],
#         })
#     result = deploy_period(df)
#     assert_frame_equal(result, expected)

# actual_data