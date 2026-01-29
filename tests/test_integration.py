"""Integration tests that exercise real cloud storage and Lithops Lambda invocations.

These require AWS credentials in the environment and a deployed Lithops runtime.
Run with: uv run python -m pytest tests/test_integration.py -v -m integration

To skip in CI unit-test jobs: uv run python -m pytest -m "not integration"
"""

import os
import uuid

import pytest
import icechunk
import zarr

# Skip entire module if AWS creds are not available
pytestmark = pytest.mark.integration

# Use a unique test prefix per run to avoid collisions and ease cleanup
TEST_PREFIX = f"test-integration-{uuid.uuid4().hex[:8]}"
BUCKET = "ismip6-icechunk"
REGION = "us-west-2"


def _has_aws_credentials():
    """Check if AWS credentials are available via env or default chain."""
    try:
        import boto3
        sts = boto3.client("sts", region_name=REGION)
        sts.get_caller_identity()
        return True
    except Exception:
        return False


def _make_repo_kwargs(prefix: str) -> dict:
    """Create icechunk repo kwargs pointing at a test S3 prefix."""
    storage = icechunk.s3_storage(
        bucket=BUCKET,
        prefix=prefix,
        region=REGION,
        from_env=True,
    )
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store()
        )
    )
    config.max_concurrent_requests = 10
    credentials = icechunk.containers_credentials({"gs://ismip6/": None})
    return {
        "storage": storage,
        "config": config,
        "authorize_virtual_chunk_access": credentials,
    }


def _cleanup_prefix(prefix: str):
    """Delete all objects under a test prefix in S3."""
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        objects = page.get("Contents", [])
        if objects:
            s3.delete_objects(
                Bucket=BUCKET,
                Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
            )


@pytest.fixture(scope="module", autouse=True)
def check_credentials():
    if not _has_aws_credentials():
        pytest.skip("AWS credentials not available")


@pytest.fixture(scope="module")
def test_prefix():
    """Provide a unique S3 prefix and clean up after the test module."""
    yield TEST_PREFIX
    _cleanup_prefix(TEST_PREFIX)


class TestIcechunkS3:
    """Test that we can create, write to, and read from an Icechunk repo on S3."""

    def test_create_repo(self, test_prefix):
        repo_kwargs = _make_repo_kwargs(test_prefix)
        repo = icechunk.Repository.open_or_create(**repo_kwargs)
        assert repo is not None

    def test_write_and_read_group(self, test_prefix):
        import numpy as np
        import xarray as xr

        repo_kwargs = _make_repo_kwargs(test_prefix + "/write-read")
        repo = icechunk.Repository.open_or_create(**repo_kwargs)

        # Write a simple dataset
        session = repo.writable_session("main")
        store = session.store
        root = zarr.open(store, mode="w")
        root.create_group("test_group")
        root["test_group"].create_array("data", data=np.arange(10))
        session.commit("test write")

        # Read it back
        session2 = repo.readonly_session(branch="main")
        root2 = zarr.open(session2.store, mode="r")
        assert "test_group" in root2
        assert list(root2["test_group"]["data"][:]) == list(range(10))

    def test_concurrent_writes_with_rebase(self, test_prefix):
        """Test that two concurrent commits to disjoint groups both succeed via rebase."""
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor, as_completed

        repo_kwargs = _make_repo_kwargs(test_prefix + "/concurrent")
        repo = icechunk.Repository.open_or_create(**repo_kwargs)

        def write_group(group_name):
            session = repo.writable_session("main")
            store = session.store
            root = zarr.open(store, mode="w")
            root.create_group(group_name)
            root[group_name].create_array("values", data=np.ones(5))
            commit_id = session.commit(
                f"Added {group_name}",
                rebase_with=icechunk.ConflictDetector(),
                rebase_tries=20,
            )
            return group_name, commit_id

        results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(write_group, "group_a"),
                pool.submit(write_group, "group_b"),
            ]
            for f in as_completed(futures):
                name, cid = f.result()
                results[name] = cid

        assert "group_a" in results
        assert "group_b" in results

        # Verify both groups exist
        session = repo.readonly_session(branch="main")
        root = zarr.open(session.store, mode="r")
        assert "group_a" in root
        assert "group_b" in root


class TestLithopsLambda:
    """Test that Lithops can invoke a function on AWS Lambda."""

    def test_hello_function(self, test_prefix):
        """Verify Lithops can run a trivial function on Lambda."""
        import lithops

        def hello(x):
            return x * 2

        fexec = lithops.FunctionExecutor(config_file="lithops_aws.yaml")
        future = fexec.call_async(hello, 21)
        result = fexec.get_result(future)
        fexec.clean()
        assert result == 42

    def test_virtualize_single_batch(self, test_prefix):
        """Run the actual virtualization function on Lambda for one small batch."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import virtualize_with_lithops_combine_variables as virt
        import lithops

        # A small batch: one model, one experiment, a few files
        batch = {
            "path": "ULB_fETISh_32km/ctrl_proj_std",
            "experiment_id": "ctrl_proj_std",
            "institution_id": "ULB",
            "source_id": "fETISh_32km",
            "urls": [
                "gs://ismip6/output/AIS/ULB/fETISh_32km/ctrl_proj_std/acabf_AIS_ULB_fETISh_32km_ctrl_proj_std.nc",
                "gs://ismip6/output/AIS/ULB/fETISh_32km/ctrl_proj_std/lithk_AIS_ULB_fETISh_32km_ctrl_proj_std.nc",
            ],
        }

        fexec = lithops.FunctionExecutor(config_file="lithops_aws.yaml")
        futures = fexec.map(virt.batch_virt_func, [{"batch": batch}])
        results = fexec.get_result(futures)
        fexec.clean()

        assert len(results) == 1
        r = results[0]
        assert r["success"] is True
        assert r["batch"]["path"] == "ULB_fETISh_32km/ctrl_proj_std"
        vds = r["virtual_dataset"]
        # Should have the variables we requested
        assert "acabf" in vds or "lithk" in vds

    def test_virtualize_and_write_to_icechunk(self, test_prefix):
        """End-to-end: virtualize on Lambda, write to S3 Icechunk, read back."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import virtualize_with_lithops_combine_variables as virt
        import lithops

        batch = {
            "path": "ULB_fETISh_32km/ctrl_proj_std",
            "experiment_id": "ctrl_proj_std",
            "institution_id": "ULB",
            "source_id": "fETISh_32km",
            "urls": [
                "gs://ismip6/output/AIS/ULB/fETISh_32km/ctrl_proj_std/acabf_AIS_ULB_fETISh_32km_ctrl_proj_std.nc",
            ],
        }

        # Virtualize on Lambda
        fexec = lithops.FunctionExecutor(config_file="lithops_aws.yaml")
        futures = fexec.map(virt.batch_virt_func, [{"batch": batch}])
        results = fexec.get_result(futures)
        fexec.clean()

        assert results[0]["success"]
        vds = results[0]["virtual_dataset"]

        # Write to Icechunk on S3
        write_prefix = test_prefix + "/e2e-write"
        repo_kwargs = _make_repo_kwargs(write_prefix)
        repo = icechunk.Repository.open_or_create(**repo_kwargs)
        write_result = virt.batch_write_func(batch, vds, repo)
        assert write_result["success"], f"Write failed: {write_result.get('error')}"

        # Read back
        session = repo.readonly_session(branch="main")
        root = zarr.open(session.store, mode="r")
        assert "ULB_fETISh_32km" in root
        assert "ctrl_proj_std" in root["ULB_fETISh_32km"]
