"""Unit tests for configuration, CLI parsing, and backend inference logic."""

import os
import tempfile

import pytest
import yaml

from unittest.mock import patch, MagicMock

# Import the module under test -- it lives at the repo root, not in a package,
# so we import by manipulating sys.path.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import virtualize_with_lithops_combine_variables as virt


# ---------------------------------------------------------------------------
# infer_cloud_backend
# ---------------------------------------------------------------------------

class TestInferCloudBackend:
    def _write_config(self, tmp_path, backend_value):
        config_file = os.path.join(tmp_path, "test_config.yaml")
        with open(config_file, "w") as f:
            yaml.dump({"lithops": {"backend": backend_value}}, f)
        return config_file

    def test_aws_lambda(self, tmp_path):
        cfg = self._write_config(tmp_path, "aws_lambda")
        assert virt.infer_cloud_backend(cfg) == "aws"

    def test_gcp_functions(self, tmp_path):
        cfg = self._write_config(tmp_path, "gcp_functions")
        assert virt.infer_cloud_backend(cfg) == "gcp"

    def test_google_cloud_functions(self, tmp_path):
        cfg = self._write_config(tmp_path, "google_cloud_functions")
        assert virt.infer_cloud_backend(cfg)  == "gcp"

    def test_localhost(self, tmp_path):
        cfg = self._write_config(tmp_path, "localhost")
        assert virt.infer_cloud_backend(cfg) == "local"

    def test_unknown_raises(self, tmp_path):
        cfg = self._write_config(tmp_path, "azure_functions")
        with pytest.raises(ValueError, match="Cannot infer cloud backend"):
            virt.infer_cloud_backend(cfg)


# ---------------------------------------------------------------------------
# get_repo_kwargs
# ---------------------------------------------------------------------------

class TestGetRepoKwargs:
    @patch("icechunk.s3_storage")
    @patch("icechunk.RepositoryConfig")
    @patch("icechunk.containers_credentials")
    def test_aws_backend(self, mock_creds, mock_config, mock_s3):
        mock_config.default.return_value = MagicMock()
        mock_creds.return_value = MagicMock()
        mock_s3.return_value = MagicMock()

        result = virt.get_repo_kwargs(local_storage=False, cloud_backend="aws")

        mock_s3.assert_called_once_with(
            bucket="us-west-2.opendata.source.coop",
            prefix="englacial/ismip6/icechunk-ais",
            region="us-west-2",
            from_env=True,
        )
        assert result['storage'] == mock_s3.return_value
        # AWS gets higher concurrency
        assert mock_config.default.return_value.max_concurrent_requests == 10

    @patch("icechunk.s3_storage")
    @patch("icechunk.RepositoryConfig")
    @patch("icechunk.containers_credentials")
    def test_aws_backend_with_write_creds(self, mock_creds, mock_config, mock_s3, tmp_path):
        import json
        mock_config.default.return_value = MagicMock()
        mock_creds.return_value = MagicMock()
        mock_s3.return_value = MagicMock()

        write_creds = tmp_path / "creds.json"
        write_creds.write_text(json.dumps({
            "aws_access_key_id": "AKID",
            "aws_secret_access_key": "SECRET",
            "aws_session_token": "TOKEN",
        }))

        result = virt.get_repo_kwargs(local_storage=False, cloud_backend="aws", write_creds=str(write_creds))

        mock_s3.assert_called_once_with(
            bucket="us-west-2.opendata.source.coop",
            prefix="englacial/ismip6/icechunk-ais",
            region="us-west-2",
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
        )
        assert result['storage'] == mock_s3.return_value

    @patch("icechunk.gcs_storage")
    @patch("icechunk.RepositoryConfig")
    @patch("icechunk.containers_credentials")
    def test_gcp_backend(self, mock_creds, mock_config, mock_gcs):
        mock_config.default.return_value = MagicMock()
        mock_creds.return_value = MagicMock()
        mock_gcs.return_value = MagicMock()

        result = virt.get_repo_kwargs(local_storage=False, cloud_backend="gcp")

        mock_gcs.assert_called_once_with(
            bucket="ismip6-icechunk",
            prefix="combined-variables-2025-12-19-v2",
            from_env=True,
        )
        assert result['storage'] == mock_gcs.return_value
        # GCP gets lower concurrency
        assert mock_config.default.return_value.max_concurrent_requests == 3

    @patch("icechunk.local_filesystem_storage")
    @patch("icechunk.RepositoryConfig")
    @patch("icechunk.containers_credentials")
    def test_local_storage(self, mock_creds, mock_config, mock_local):
        mock_config.default.return_value = MagicMock()
        mock_creds.return_value = MagicMock()
        mock_local.return_value = MagicMock()

        result = virt.get_repo_kwargs(local_storage=True, cloud_backend="aws")

        mock_local.assert_called_once_with("test-output/test_icechunk")
        assert result['storage'] == mock_local.return_value


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    """Test that argparse produces expected values for various invocations."""

    def _parse(self, args_list):
        """Simulate CLI parsing by calling the argparse parser directly."""
        parser = virt.argparse.ArgumentParser()
        parser.add_argument("--config", default="lithops_aws.yaml")
        parser.add_argument("--local-storage", action="store_true")
        parser.add_argument("--local-execution", action="store_true")
        parser.add_argument("--test-model", default=None)
        parser.add_argument("--test-experiment", default=None)
        parser.add_argument("--sequential-writes", action="store_true")
        parser.add_argument("--store-type", default="all",
                            choices=["all", "combined", "state", "flux"])
        return parser.parse_args(args_list)

    def test_defaults(self):
        args = self._parse([])
        assert args.config == "lithops_aws.yaml"
        assert args.local_storage is False
        assert args.local_execution is False
        assert args.test_model is None
        assert args.test_experiment is None
        assert args.sequential_writes is False
        assert args.store_type == "all"

    def test_gcp_config(self):
        args = self._parse(["--config", "lithops_gcp.yaml"])
        assert args.config == "lithops_gcp.yaml"

    def test_test_filters(self):
        args = self._parse(["--test-model", "fETISh_32km", "--test-experiment", "ctrl_proj_std"])
        assert args.test_model == "fETISh_32km"
        assert args.test_experiment == "ctrl_proj_std"

    def test_sequential_writes_flag(self):
        args = self._parse(["--sequential-writes"])
        assert args.sequential_writes is True

    def test_local_flags(self):
        args = self._parse(["--local-storage", "--local-execution", "--config", "lithops_local.yaml"])
        assert args.local_storage is True
        assert args.local_execution is True

    def test_store_type_all(self):
        args = self._parse(["--store-type", "all"])
        assert args.store_type == "all"

    def test_store_type_state(self):
        args = self._parse(["--store-type", "state"])
        assert args.store_type == "state"

    def test_store_type_flux(self):
        args = self._parse(["--store-type", "flux"])
        assert args.store_type == "flux"

    def test_store_type_combined(self):
        args = self._parse(["--store-type", "combined"])
        assert args.store_type == "combined"

    def test_store_type_invalid(self):
        with pytest.raises(SystemExit):
            self._parse(["--store-type", "invalid"])


# ---------------------------------------------------------------------------
# Test filter logic
# ---------------------------------------------------------------------------

class TestFilterLogic:
    """Test that model/experiment filters work correctly on a DataFrame."""

    def _make_df(self):
        import pandas as pd
        return pd.DataFrame({
            'model_name': ['ModelA', 'ModelA', 'ModelB', 'ModelB'],
            'experiment': ['exp1', 'exp2', 'exp1', 'exp2'],
            'institution': ['Inst1', 'Inst1', 'Inst2', 'Inst2'],
            'url': ['gs://a', 'gs://b', 'gs://c', 'gs://d'],
        })

    def test_no_filter(self):
        df = self._make_df()
        assert len(df) == 4

    def test_model_filter(self):
        df = self._make_df()
        df = df[df['model_name'] == 'ModelA']
        assert len(df) == 2
        assert set(df['model_name']) == {'ModelA'}

    def test_experiment_filter(self):
        df = self._make_df()
        df = df[df['experiment'] == 'exp1']
        assert len(df) == 2
        assert set(df['experiment']) == {'exp1'}

    def test_both_filters(self):
        df = self._make_df()
        df = df[df['model_name'] == 'ModelA']
        df = df[df['experiment'] == 'exp1']
        assert len(df) == 1


# ---------------------------------------------------------------------------
# _parse_variable_from_url
# ---------------------------------------------------------------------------

class TestParseVariable:
    def test_standard_url(self):
        url = "gs://ismip6/output/AIS/AWI/PISM1/exp05/acabf_AIS_AWI_PISM1_exp05.nc"
        assert virt._parse_variable_from_url(url) == "acabf"

    def test_multi_underscore(self):
        url = "gs://bucket/path/to/lithk_AIS_UCIJPL_ISSM_ctrl.nc"
        assert virt._parse_variable_from_url(url) == "lithk"


# ---------------------------------------------------------------------------
# get_id_from_batch
# ---------------------------------------------------------------------------

class TestGetIdFromBatch:
    def test_basic(self):
        batch = {
            'institution_id': 'ULB',
            'source_id': 'fETISh_32km',
            'experiment_id': 'ctrl_proj_std',
        }
        assert virt.get_id_from_batch(batch) == "ULB_fETISh_32km/ctrl_proj_std"


# ---------------------------------------------------------------------------
# STORE_TYPE_CONFIG
# ---------------------------------------------------------------------------

class TestStoreTypeConfig:
    def test_has_all_three_entries(self):
        assert set(virt.STORE_TYPE_CONFIG.keys()) == {"combined", "state", "flux"}

    def test_unified_prefix(self):
        """All store types share a single unified prefix."""
        for key in ("combined", "state", "flux"):
            assert virt.STORE_TYPE_CONFIG[key]["prefix"] == virt.UNIFIED_STORE_PREFIX

    def test_combined_config(self):
        cfg = virt.STORE_TYPE_CONFIG["combined"]
        assert cfg["prefix"] == "englacial/ismip6/icechunk-ais"
        assert cfg["group_prefix"] == "combined"
        assert cfg["bin_time"] is True
        assert cfg["filter"] is None

    def test_state_config(self):
        cfg = virt.STORE_TYPE_CONFIG["state"]
        assert cfg["prefix"] == "englacial/ismip6/icechunk-ais"
        assert cfg["group_prefix"] == "state"
        assert cfg["bin_time"] is False
        assert cfg["filter"] == "ST"

    def test_flux_config(self):
        cfg = virt.STORE_TYPE_CONFIG["flux"]
        assert cfg["prefix"] == "englacial/ismip6/icechunk-ais"
        assert cfg["group_prefix"] == "flux"
        assert cfg["bin_time"] is False
        assert cfg["filter"] == "FL"


# ---------------------------------------------------------------------------
# get_repo_kwargs with store_type
# ---------------------------------------------------------------------------

class TestGetRepoKwargsUnifiedPrefix:
    """All store types now share a single repo, so get_repo_kwargs always uses the
    unified prefix regardless of which store type will be written."""

    @patch("icechunk.s3_storage")
    @patch("icechunk.RepositoryConfig")
    @patch("icechunk.containers_credentials")
    def test_always_uses_unified_prefix(self, mock_creds, mock_config, mock_s3):
        mock_config.default.return_value = MagicMock()
        mock_creds.return_value = MagicMock()
        mock_s3.return_value = MagicMock()

        virt.get_repo_kwargs(local_storage=False, cloud_backend="aws")

        mock_s3.assert_called_once_with(
            bucket="us-west-2.opendata.source.coop",
            prefix="englacial/ismip6/icechunk-ais",
            region="us-west-2",
            from_env=True,
        )

    @patch("icechunk.s3_storage")
    @patch("icechunk.RepositoryConfig")
    @patch("icechunk.containers_credentials")
    def test_no_store_type_param(self, mock_creds, mock_config, mock_s3):
        """get_repo_kwargs no longer accepts store_type."""
        mock_config.default.return_value = MagicMock()
        mock_creds.return_value = MagicMock()
        mock_s3.return_value = MagicMock()

        import inspect
        sig = inspect.signature(virt.get_repo_kwargs)
        assert "store_type" not in sig.parameters
