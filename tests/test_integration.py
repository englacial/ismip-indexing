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

    def test_sequential_writes_with_rebase(self, test_prefix):
        """Test that two sequential commits to disjoint groups both succeed.

        NOTE: Truly concurrent writes to the same icechunk repo can fail with
        RebaseFailedError when both sessions modify the root group structure.
        BasicConflictSolver cannot resolve structural group conflicts.
        The production pipeline handles this via retries and the "skip existing
        groups" logic on re-runs.
        """
        import numpy as np

        repo_kwargs = _make_repo_kwargs(test_prefix + "/sequential")
        repo = icechunk.Repository.open_or_create(**repo_kwargs)

        for group_name in ["group_a", "group_b"]:
            session = repo.writable_session("main")
            store = session.store
            root = zarr.open(store, mode="a")
            root.create_group(group_name)
            root[group_name].create_array("values", data=np.ones(5))
            session.commit(f"Added {group_name}")

        # Verify both groups exist
        session = repo.readonly_session(branch="main")
        root = zarr.open(session.store, mode="r")
        assert "group_a" in root
        assert "group_b" in root


class TestLithopsLambda:
    """Test that Lithops can invoke a function on AWS Lambda.

    NOTE: These tests require a deployed Lithops Lambda runtime.
    The runtime is built and deployed locally (not in CI) via:
        uv run lithops runtime build -f Dockerfile.lithops -c lithops_aws.yaml ismip6-icechunk
    If the runtime is not deployed, these tests will fail.
    """

    def test_hello_function(self, test_prefix):
        """Verify Lithops can run a trivial function on Lambda."""
        import lithops

        def hello(x):
            return x * 2

        fexec = lithops.FunctionExecutor(config_file="lithops_aws.yaml")
        futures = fexec.map(hello, [21])
        result = fexec.get_result(futures)
        fexec.clean()
        assert result == [42]

    def test_virtualize_single_batch(self, test_prefix):
        """Run the actual virtualization function on Lambda for one small batch."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import virtualize_with_lithops_combine_variables as virt
        import lithops

        # A small batch: one model, one experiment, a few files
        batch = {
            "path": "AWI_PISM1/ctrl_proj_std",
            "experiment_id": "ctrl_proj_std",
            "institution_id": "AWI",
            "source_id": "PISM1",
            "urls": [
                "gs://ismip6/Projection-AIS/AWI/PISM1/ctrl_proj_std/acabf_AIS_AWI_PISM1_ctrl_proj_std.nc",
                "gs://ismip6/Projection-AIS/AWI/PISM1/ctrl_proj_std/lithk_AIS_AWI_PISM1_ctrl_proj_std.nc",
            ],
        }

        fexec = lithops.FunctionExecutor(config_file="lithops_aws.yaml")
        futures = fexec.map(virt.batch_virt_func, [{"batch": batch}])
        results = fexec.get_result(futures)
        fexec.clean()

        assert len(results) == 1
        r = results[0]
        assert r["success"] is True, f"Virtualization failed: {r.get('error', 'unknown')}"
        assert r["batch"]["path"] == "AWI_PISM1/ctrl_proj_std"
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
            "path": "AWI_PISM1/ctrl_proj_std",
            "experiment_id": "ctrl_proj_std",
            "institution_id": "AWI",
            "source_id": "PISM1",
            "urls": [
                "gs://ismip6/Projection-AIS/AWI/PISM1/ctrl_proj_std/acabf_AIS_AWI_PISM1_ctrl_proj_std.nc",
            ],
        }

        # Virtualize on Lambda
        fexec = lithops.FunctionExecutor(config_file="lithops_aws.yaml")
        futures = fexec.map(virt.batch_virt_func, [{"batch": batch}])
        results = fexec.get_result(futures)
        fexec.clean()

        assert results[0]["success"], f"Virtualization failed: {results[0].get('error', 'unknown')}"
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
        assert "AWI_PISM1" in root
        assert "ctrl_proj_std" in root["AWI_PISM1"]
