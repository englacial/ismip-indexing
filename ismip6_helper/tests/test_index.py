"""Tests for ismip6_helper.index — file indexing from S3."""

import pytest
import pandas as pd
from ismip6_helper import get_file_index, parse_ismip6_path, SOURCE_DATA_URL


@pytest.fixture(scope="module")
def file_index() -> pd.DataFrame:
    """Load the file index (uses cache if available, otherwise fetches from S3)."""
    return get_file_index()


# --- parse_ismip6_path unit tests ---


def test_parse_standard_path():
    path = "ismip6/Projection-AIS/AWI/PISM1/exp13/acabf_AIS_AWI_PISM1_exp13.nc"
    result = parse_ismip6_path(path)
    assert result is not None
    assert result["variable"] == "acabf"
    assert result["ice_sheet"] == "AIS"
    assert result["institution"] == "AWI"
    assert result["model_name"] == "PISM1"
    assert result["experiment"] == "exp13"
    assert result["url"] == f"{SOURCE_DATA_URL}/Projection-AIS/AWI/PISM1/exp13/acabf_AIS_AWI_PISM1_exp13.nc"


def test_parse_full_s3_path():
    """parse_ismip6_path should handle a full s3:// URL."""
    url = f"{SOURCE_DATA_URL}/Projection-AIS/AWI/PISM1/exp13/acabf_AIS_AWI_PISM1_exp13.nc"
    result = parse_ismip6_path(url)
    assert result is not None
    assert result["variable"] == "acabf"
    assert result["url"] == url


def test_parse_ucijpl_corrects_experiment_prefix():
    """UCIJPL/ISSM files have the experiment name prepended to the variable."""
    path = "ismip6/Projection-AIS/UCIJPL/ISSM/exp13/exp13acabf_AIS_UCIJPL_ISSM_exp13.nc"
    result = parse_ismip6_path(path)
    assert result is not None
    assert result["variable"] == "acabf"


def test_parse_invalid_path_returns_none():
    assert parse_ismip6_path("not/a/valid/path.nc") is None
    assert parse_ismip6_path("ismip6/SomethingElse/foo.nc") is None


# --- build_file_index / get_file_index tests ---


class TestFileIndex:
    """Tests that verify expected properties of the full file index."""

    def test_total_file_count(self, file_index):
        assert len(file_index) == 10034

    def test_columns(self, file_index):
        expected = {"variable", "ice_sheet", "institution", "model_name",
                    "experiment", "url", "size_bytes"}
        assert set(file_index.columns) == expected

    def test_all_urls_are_s3(self, file_index):
        assert file_index["url"].str.startswith("s3://").all()

    def test_all_urls_are_nc(self, file_index):
        assert file_index["url"].str.endswith(".nc").all()

    def test_single_ice_sheet(self, file_index):
        assert list(file_index["ice_sheet"].unique()) == ["AIS"]

    def test_institution_count(self, file_index):
        assert file_index["institution"].nunique() == 14

    def test_model_count(self, file_index):
        assert file_index["model_name"].nunique() == 15

    def test_experiment_count(self, file_index):
        assert file_index["experiment"].nunique() == 94

    def test_variable_count(self, file_index):
        assert file_index["variable"].nunique() == 37

    def test_no_null_values(self, file_index):
        assert not file_index[["variable", "ice_sheet", "institution",
                                "model_name", "experiment", "url"]].isnull().any().any()

    def test_all_sizes_non_negative(self, file_index):
        assert (file_index["size_bytes"] >= 0).all()
