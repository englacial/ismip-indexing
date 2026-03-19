"""Integration tests that build local Icechunk stores and compare against the
remote production store on source.coop.

Each test virtualizes a small model, writes it to a temporary local store, then
compares every group against the remote store (ignoring annotation attributes
added by the separate --annotate-only post-processing step).

These tests hit the network (source.coop S3) and take a few minutes each.

Run with:
    uv run --extra dev pytest virtualize_ismip6/tests/test_build_and_compare.py -v -m integration

Skip in CI unit-test jobs:
    uv run --extra dev pytest -m "not integration"
"""

import os
import sys

import pytest

# Make the virtualize_ismip6 directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import virtualize_with_lithops_combine_variables as virt
from compare_local_remote_store import (
    compare_group,
    list_groups,
    open_local_repo,
    open_remote_repo,
)

pytestmark = pytest.mark.integration

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..")


def _build_model(output_dir: str, model_name: str, store_type: str = "combined"):
    """Run the virtualization pipeline for a single model into a local store."""
    import icechunk

    # Monkey-patch the local storage path so we write to our temp dir
    original_get_repo_kwargs = virt.get_repo_kwargs

    def patched_get_repo_kwargs(local_storage=False, cloud_backend="aws", write_creds=None):
        kwargs = original_get_repo_kwargs(local_storage=local_storage,
                                          cloud_backend=cloud_backend,
                                          write_creds=write_creds)
        if local_storage:
            kwargs["storage"] = icechunk.local_filesystem_storage(output_dir)
        return kwargs

    # Lithops localhost spawns worker processes that need the module importable
    env_key = "PYTHONPATH"
    old_pythonpath = os.environ.get(env_key, "")
    script_dir = os.path.dirname(os.path.abspath(virt.__file__))
    os.environ[env_key] = script_dir + (os.pathsep + old_pythonpath if old_pythonpath else "")

    virt.get_repo_kwargs = patched_get_repo_kwargs
    try:
        virt.process_all_files(
            local_storage=True,
            local_execution=True,
            config_file=os.path.join(SCRIPT_DIR, "lithops_local.yaml"),
            concurrent_writes=False,
            test_model=model_name,
            store_type=store_type,
        )
    finally:
        virt.get_repo_kwargs = original_get_repo_kwargs
        if old_pythonpath:
            os.environ[env_key] = old_pythonpath
        else:
            os.environ.pop(env_key, None)


def _compare_local_to_remote(local_path: str, group_prefix: str = "combined"):
    """Compare all groups in the local store against the remote store.

    Returns (n_common, diffs_by_group) where diffs_by_group is a dict
    mapping group paths to their list of differences.
    """
    local_repo = open_local_repo(local_path)
    local_session = local_repo.readonly_session(branch="main")
    local_store = local_session.store

    remote_repo = open_remote_repo()
    remote_session = remote_repo.readonly_session(branch="main")
    remote_store = remote_session.store

    local_groups = [g for g in list_groups(local_store) if g.startswith(group_prefix)]
    remote_groups = set(list_groups(remote_store))

    common = sorted(g for g in local_groups if g in remote_groups)
    only_local = sorted(g for g in local_groups if g not in remote_groups)

    diffs_by_group = {}
    for group_path in common:
        diffs = compare_group(local_store, remote_store, group_path,
                              ignore_annotations=True)
        if diffs:
            diffs_by_group[group_path] = diffs

    if only_local:
        for g in only_local:
            diffs_by_group[g] = ["Group exists only in local store"]

    return len(common), diffs_by_group


class TestBuildAndCompare:
    """Build small models locally and verify they match the remote store."""

    def test_bisicles_combined(self, tmp_path):
        """CPOM/BISICLES: smallest model (3 experiments, 93 files).

        Tests the pipeline with a model that uses non-standard time encoding
        and has unusual grid metadata.
        """
        output_dir = str(tmp_path / "icechunk")
        _build_model(output_dir, "BISICLES", store_type="combined")

        n_common, diffs = _compare_local_to_remote(output_dir)
        assert n_common == 3, f"Expected 3 common groups, got {n_common}"
        assert diffs == {}, f"Differences found:\n" + "\n".join(
            f"  {g}: {d}" for g, d in diffs.items()
        )

    def test_imauice1_combined(self, tmp_path):
        """IMAU/IMAUICE1: small model with good coverage (10 experiments, 142 files).

        Tests the pipeline with a well-behaved model that has standard
        NetCDF4/HDF5 files and typical variable coverage.
        """
        output_dir = str(tmp_path / "icechunk")
        _build_model(output_dir, "IMAUICE1", store_type="combined")

        n_common, diffs = _compare_local_to_remote(output_dir)
        assert n_common == 10, f"Expected 10 common groups, got {n_common}"
        assert diffs == {}, f"Differences found:\n" + "\n".join(
            f"  {g}: {d}" for g, d in diffs.items()
        )
