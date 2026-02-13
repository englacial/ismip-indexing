import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import dataclasses
import gc
import logging
import os
import tempfile
import threading

import lithops
import icechunk
import ismip6_helper
import numpy as np
import obstore
import xarray as xr
import yaml

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

import zarr
from typing import Dict, Tuple, Any, List, Union

zarr.config.set({
    'async.concurrency': 20,
    'threading.max_workers': 10
})

logger = logging.getLogger(__name__)

# AWS Lambda pricing (us-west-2): $0.0000166667 per GB-second
LAMBDA_COST_PER_GB_SECOND = 0.0000166667
# S3 PUT request cost: $0.005 per 1,000 requests
S3_PUT_COST_PER_REQUEST = 0.000005

# Source data bucket (public, anonymous read)
SOURCE_BUCKET = "s3://us-west-2.opendata.source.coop/englacial/ismip6"

# Unified store prefix: all store types live under a single icechunk repo
UNIFIED_STORE_PREFIX = "englacial/ismip6/icechunk-ais"

# Store type configurations: controls group prefix, time binning, and variable filtering
STORE_TYPE_CONFIG = {
    "combined": {"prefix": UNIFIED_STORE_PREFIX, "group_prefix": "combined", "bin_time": True,  "filter": None},
    "state":    {"prefix": UNIFIED_STORE_PREFIX, "group_prefix": "state",    "bin_time": False, "filter": "ST"},
    "flux":     {"prefix": UNIFIED_STORE_PREFIX, "group_prefix": "flux",     "bin_time": False, "filter": "FL"},
}


def load_skip_list(path: str = "skip_list.txt") -> set:
    """Load the skip list from a text file.

    Each non-empty, non-comment line is a substring matched against URLs.
    """
    skip_set = set()
    if not os.path.exists(path):
        logger.warning("Skip list not found at %s", path)
        return skip_set
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                skip_set.add(line)
    logger.info("Loaded %d entries from skip list %s", len(skip_set), path)
    return skip_set


def filter_urls_by_skip_list(urls: List[str], skip_list: set) -> tuple:
    """Filter URLs against the skip list. Returns (kept, skipped) lists."""
    kept = []
    skipped = []
    for url in urls:
        if any(pattern in url for pattern in skip_list):
            skipped.append(url)
        else:
            kept.append(url)
    return kept, skipped


def _parse_variable_from_url(url: str) -> str:
    return url.split('/')[-1].split('_')[0]


def infer_cloud_backend(config_file: str) -> str:
    """Infer the cloud backend ('aws' or 'gcp') from a lithops config file."""
    with open(config_file) as f:
        config = yaml.safe_load(f)
    backend = config.get('lithops', {}).get('backend', '')
    if 'aws' in backend:
        return 'aws'
    elif 'gcp' in backend or 'google' in backend:
        return 'gcp'
    elif backend == 'localhost':
        return 'local'
    else:
        raise ValueError(f"Cannot infer cloud backend from lithops config backend: {backend!r}")


def _rewrite_manifest_paths(vds: xr.Dataset, old_path: str, new_path: str) -> xr.Dataset:
    """Rewrite manifest paths from a local file path to an S3 URL.

    When NetCDF3 files are downloaded and parsed locally, the manifest entries
    point to ``file:///tmp/...``. This replaces those paths with the original
    S3 URL so icechunk can resolve virtual chunks at read time.
    """
    new_vars = {}
    for name in vds.data_vars:
        var = vds[name]
        if not isinstance(var.data, ManifestArray):
            continue
        marr = var.data
        entries = list(marr.manifest.values())
        if not entries or old_path not in entries[0]["path"]:
            continue
        shape = marr.manifest.shape_chunk_grid
        paths = np.empty(shape, dtype=np.dtypes.StringDType())
        offsets = np.zeros(shape, dtype=np.dtype("uint64"))
        lengths = np.zeros(shape, dtype=np.dtype("uint64"))
        for idx, entry in zip(np.ndindex(shape), entries):
            paths[idx] = entry["path"].replace(old_path, new_path)
            offsets[idx] = entry["offset"]
            lengths[idx] = entry["length"]
        new_manifest = ChunkManifest.from_arrays(paths=paths, offsets=offsets, lengths=lengths, validate_paths=False)
        new_marr = ManifestArray(metadata=marr.metadata, chunkmanifest=new_manifest)
        new_vars[name] = xr.Variable(dims=var.dims, data=new_marr, attrs=var.attrs, encoding=var.encoding)
    if new_vars:
        vds = vds.assign(new_vars)
    return vds


def _open_netcdf3_via_download(url: str, s3_store, registry,
                                loadable_variables: list) -> xr.Dataset:
    """Open a NetCDF3 file by downloading via obstore and parsing locally.

    Bypasses virtualizarr's ManifestStore/ObjectStoreRegistry entirely by
    using kerchunk + manifestgroup_from_kerchunk_refs directly. This avoids
    the obstore LocalStore that fails on Lambda's async runtime.
    """
    from kerchunk.netCDF3 import NetCDF3ToZarr
    from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs

    # Extract the S3 key from the full URL
    bucket_prefix = SOURCE_BUCKET
    s3_key = url[len(bucket_prefix) + 1:]  # strip "s3://bucket/" prefix

    # Download file via obstore (works on Lambda without s3fs)
    data = obstore.get(s3_store, s3_key)
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, os.path.basename(url))
    with open(tmp_path, 'wb') as f:
        f.write(data.bytes())

    try:
        # Parse with kerchunk (uses fsspec, not obstore)
        refs = NetCDF3ToZarr(tmp_path, inline_threshold=0).translate()
        manifest_group = manifestgroup_from_kerchunk_refs(refs)

        # Build virtual dataset — bypasses ManifestStore and registry entirely
        vds = manifest_group.to_virtual_dataset()

        virtual_vars = [v for v in vds.data_vars if isinstance(vds[v].data, ManifestArray)]
        logger.info("NC3 bypass %s: virtual_vars=%s, all_vars=%s",
                     os.path.basename(url), virtual_vars, list(vds.data_vars))

        # Load coordinate/loadable variables with scipy
        with xr.open_dataset(tmp_path, engine='scipy', decode_times=False) as real_ds:
            vars_to_load = [v for v in loadable_variables if v in real_ds and v in vds]
            logger.info("NC3 bypass %s: loading %s (requested %s, in real_ds %s, in vds %s)",
                         os.path.basename(url), vars_to_load, loadable_variables,
                         list(real_ds.data_vars) + list(real_ds.coords), list(vds.variables))
            if vars_to_load:
                real_keep = real_ds[vars_to_load].load()
                vds_keep = vds.drop_vars(vars_to_load, errors='ignore')
                vds = xr.merge([real_keep, vds_keep])

        # Rewrite paths: file:///tmp/foo.nc → s3://source.coop/...
        old_path = f"file://{tmp_path}"
        # Check manifest path format before rewrite
        for v in vds.data_vars:
            if isinstance(vds[v].data, ManifestArray):
                sample = list(vds[v].data.manifest.values())[0]
                logger.info("NC3 bypass %s: manifest path=%s, old_path=%s, match=%s",
                             os.path.basename(url), sample["path"], old_path,
                             old_path in sample["path"])
                break
        vds = _rewrite_manifest_paths(vds, old_path, url)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return vds


def virtualize_and_combine_batch(urls: List[str], registry: ObjectStoreRegistry, bin_time: bool = True) -> Dict[str, Any]:
    # Take a batch of datasets that belong to a single simulation (same model + experiment) and merge them into
    # a single virtual dataset

    # create virtual datasets (can we speed this up in parallel if we group them? Not a prio right now)
    # Time is included in loadable_variables so we can normalize its encoding (transform values
    # to a standard epoch/calendar). Time arrays are small so the overhead is negligible.
    loadable_variables = ['x', 'y', 'lat', 'lon', 'latitude', 'longitude', 'nv4', 'lon_bnds', 'lat_bnds', 'time', 'bnds']

    # S3 store for downloading NetCDF3 files (obstore works without s3fs)
    s3_store = obstore.store.from_url(SOURCE_BUCKET, skip_signature=True)

    # virtualize and append all needed metadata (for now just the variable)
    vdatasets = []
    nc3_fallbacks = []
    for url in urls:
        # Try HDF5 (NetCDF4) first, fall back to NetCDF3. Some batches contain
        # a mix of formats (e.g. LSCE/GRISLI2/hist_std has both HDF5 and NetCDF3 files).
        # NetCDF3 fallback downloads the file via obstore and parses locally,
        # because s3fs conflicts with Lambda's built-in botocore.
        try:
            vds_var = open_virtual_dataset(
                url=url,
                parser=HDFParser(),
                registry=registry,
                loadable_variables=loadable_variables,
                decode_times=False
            )
        except Exception as hdf5_err:
            logger.warning("HDF5 parse failed for %s: %s — falling back to NetCDF3 download",
                           os.path.basename(url), hdf5_err)
            nc3_fallbacks.append(os.path.basename(url))
            vds_var = _open_netcdf3_via_download(url, s3_store, registry, loadable_variables)
        # apply ismip specific fixer functions
        # fix_time_encoding cleans up malformed attributes (typos, missing calendar, etc.)
        # normalize_time_encoding transforms values to standard epoch/calendar - this works
        # because 'time' is in loadable_variables so it's real data, not a ManifestArray
        vds_var_fixed_time = ismip6_helper.fix_time_encoding(vds_var)
        vds_var_normalized_time = ismip6_helper.normalize_time_encoding(vds_var_fixed_time)
        if bin_time:
            vds_var_normalized_time = ismip6_helper.bin_time_to_year(vds_var_normalized_time)
        vds_var_fixed_grid = ismip6_helper.correct_grid_coordinates(vds_var_normalized_time, _parse_variable_from_url(url))
        vdatasets.append(vds_var_fixed_grid)

    # Rechunk, pad to union time axis, and merge
    vds = ismip6_helper.merge_virtual_datasets(vdatasets)

    return vds, nc3_fallbacks


def batch_virt_func(batch: Tuple[Tuple[str], List[Dict[str, Union[str, int]]]]) -> Dict[str, Any]:
    """Wrap batch virtualization in error handling"""
    try:
        bucket = SOURCE_BUCKET
        store = obstore.store.from_url(bucket, skip_signature=True)
        registry = ObjectStoreRegistry({bucket: store})
        bin_time = batch.get('bin_time', True)
        vds, nc3_fallbacks = virtualize_and_combine_batch(batch['urls'], registry, bin_time=bin_time)
        return {
                'success': True,
                'batch': batch,
                'virtual_dataset': vds,
                'nc3_fallbacks': nc3_fallbacks,
            }

    except Exception as e:
        return {
            "success": False,
            "batch": batch,
            "error": str(e),
        }


def _single_write_attempt(repo, vds, path, commit_msg):
    """Execute a single write+commit attempt. Runs in a disposable thread so
    that a Rust-level deadlock (poisoned connection pool) can be detected via
    timeout rather than hanging the caller forever."""
    session = repo.writable_session("main")
    vds.vz.to_icechunk(session.store, group=path)
    commit_id = session.commit(
        commit_msg,
        rebase_with=icechunk.BasicConflictSolver(),
        rebase_tries=5,
    )
    return commit_id


# Per-attempt timeout: if a single write+commit takes longer than this,
# assume the icechunk Rust runtime is deadlocked (poisoned connection pool).
WRITE_ATTEMPT_TIMEOUT_S = 120


def batch_write_func(batch: Tuple[Tuple[str], List[Dict[str, Union[str, int]]]], vds: xr.Dataset, repo: icechunk.Repository, local_storage: bool = False, max_retries: int = 50) -> Dict[str, Any]:
    """Wrap writing to icechunk in error handling. Retries with fresh sessions on stale parent errors.

    Each attempt runs in a separate thread with a timeout to detect Rust-level
    deadlocks caused by connection pool poisoning (icechunk issue #1586).
    """
    import time as _time
    import random as _random
    path = batch['path']
    commit_msg = f"Added {path}"
    last_error = None
    for attempt in range(max_retries):
        # Run the write attempt in a disposable thread with a timeout.
        # If the icechunk Rust runtime is deadlocked, the thread will hang
        # but we won't block — we'll detect it via timeout and report it.
        result_holder = {}
        error_holder = {}

        def _attempt():
            try:
                result_holder['commit_id'] = _single_write_attempt(
                    repo, vds, path, commit_msg
                )
            except BaseException as e:
                error_holder['error'] = e

        t = threading.Thread(target=_attempt, daemon=True)
        t.start()
        t.join(timeout=WRITE_ATTEMPT_TIMEOUT_S)

        if t.is_alive():
            # Thread is stuck — likely a poisoned Rust mutex / deadlock.
            # We can't kill the thread, but we can stop retrying and report.
            msg = (
                f"DEADLOCK DETECTED: write attempt for {path} did not complete "
                f"within {WRITE_ATTEMPT_TIMEOUT_S}s (attempt {attempt + 1}). "
                f"This is likely caused by a poisoned icechunk connection pool "
                f"(see https://github.com/earth-mover/icechunk/issues/1586). "
                f"The Rust runtime is unrecoverable — remaining batches in this "
                f"process will also fail. Restart the pipeline to resume."
            )
            print(f"    {path}: {msg}")
            return {
                "success": False,
                "batch": batch,
                "error": msg,
                "deadlock": True,
            }

        if 'commit_id' in result_holder:
            if attempt > 0:
                print(f"    {path}: succeeded on attempt {attempt + 1}")

            return {
                'success': True,
                'batch': batch,
                'virtual_dataset': vds,
                'commit_id': result_holder['commit_id'],
                'commit_msg': commit_msg,
            }

        # An exception was raised
        e = error_holder.get('error')
        last_error = e
        err_str = str(e)
        retryable = (
            "expected parent" in err_str
            or "Rebase failed" in err_str
            or "dispatch failure" in err_str
            or "Timeout" in err_str
            or "PanicException" in type(e).__name__
        )
        if retryable:
            # Exponential backoff with jitter: 0.2s, 0.4s, 0.8s, ... capped at 10s
            delay = min(0.2 * (2 ** attempt), 10.0) + _random.uniform(0, 0.5)
            _time.sleep(delay)
            continue
        # Non-retryable error
        return {
            "success": False,
            "batch": batch,
            "error": err_str,
        }
    # All retries exhausted
    return {
        "success": False,
        "batch": batch,
        "error": f"Failed after {max_retries} retries: {last_error}",
    }


def _load_store_credentials(write_creds: str) -> dict:
    """Load icechunk store write credentials from a JSON file.

    Expected format:
    {
        "aws_access_key_id": "...",
        "aws_secret_access_key": "...",
        "aws_session_token": "..."
    }
    """
    import json
    with open(write_creds) as f:
        creds = json.load(f)
    required = ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]
    missing = [k for k in required if k not in creds]
    if missing:
        raise ValueError(f"Credentials file {write_creds} missing keys: {missing}")
    return creds


def get_repo_kwargs(local_storage: bool = False, cloud_backend: str = "aws", write_creds: str = None) -> dict:
    """Build icechunk Repository kwargs for the given storage backend.

    All store types share a single unified icechunk repo at UNIFIED_STORE_PREFIX.

    Args:
        local_storage: Use local filesystem storage instead of cloud.
        cloud_backend: 'aws' or 'gcp'. Determines which object store and
            concurrency settings to use.
        write_creds: Path to JSON file with source.coop store write credentials.
            If None, credentials are read from the environment.
    """
    s3_prefix = UNIFIED_STORE_PREFIX
    if local_storage:
        storage = icechunk.local_filesystem_storage("test-output/test_icechunk")
    elif cloud_backend == "aws":
        storage_kwargs = dict(
            bucket="us-west-2.opendata.source.coop",
            prefix=s3_prefix,
            region="us-west-2",
        )
        if write_creds:
            creds = _load_store_credentials(write_creds)
            storage_kwargs.update(
                access_key_id=creds["aws_access_key_id"],
                secret_access_key=creds["aws_secret_access_key"],
                session_token=creds["aws_session_token"],
            )
        else:
            storage_kwargs["from_env"] = True
        storage = icechunk.s3_storage(**storage_kwargs)
    else:  # gcp
        storage = icechunk.gcs_storage(
            bucket="ismip6-icechunk",
            prefix="combined-variables-2025-12-19-v2",
            from_env=True,
        )

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            SOURCE_BUCKET + "/",
            store=icechunk.s3_store(region="us-west-2", anonymous=True)
        )
    )
    # S3 handles higher concurrency than GCS
    config.max_concurrent_requests = 10 if cloud_backend == "aws" else 3

    credentials = icechunk.containers_credentials({
        SOURCE_BUCKET + "/": None
    })
    return {
        'storage': storage,
        'config': config,
        'authorize_virtual_chunk_access': credentials
    }


def compute_lambda_cost(futures) -> dict:
    """Compute estimated AWS Lambda cost from Lithops futures.

    Returns a dict with total_duration_s, total_gb_seconds, lambda_cost,
    num_invocations, and avg_duration_s.
    """
    total_duration = 0.0
    total_gb_seconds = 0.0
    count = 0
    for f in futures:
        try:
            exec_time = f.stats.get('worker_exec_time', 0)
            memory_mb = f.runtime_memory or 2048
            gb_seconds = (memory_mb / 1024) * exec_time
            total_duration += exec_time
            total_gb_seconds += gb_seconds
            count += 1
        except (AttributeError, TypeError):
            continue

    lambda_cost = total_gb_seconds * LAMBDA_COST_PER_GB_SECOND
    return {
        'num_invocations': count,
        'total_duration_s': round(total_duration, 2),
        'avg_duration_s': round(total_duration / max(count, 1), 2),
        'total_gb_seconds': round(total_gb_seconds, 2),
        'lambda_cost_usd': round(lambda_cost, 4),
    }


def print_cost_summary(cost_stats: dict, s3_put_count: int = 0):
    """Print a human-readable cost summary."""
    s3_cost = s3_put_count * S3_PUT_COST_PER_REQUEST
    total = cost_stats['lambda_cost_usd'] + s3_cost
    print("\n--- Cost Summary ---")
    print(f"Lambda invocations: {cost_stats['num_invocations']}")
    print(f"Total Lambda duration: {cost_stats['total_duration_s']}s "
          f"(avg {cost_stats['avg_duration_s']}s per invocation)")
    print(f"Total GB-seconds: {cost_stats['total_gb_seconds']}")
    print(f"Lambda compute cost: ${cost_stats['lambda_cost_usd']:.4f}")
    if s3_put_count:
        print(f"S3 PUT requests (~{s3_put_count}): ${s3_cost:.4f}")
    print(f"Estimated total cost: ${total:.4f}")
    print("--------------------")


def annotate_all_groups(
    local_storage: bool = False,
    cloud_backend: str = "aws",
    write_creds: str = None,
    store_type: str = "combined",
):
    """Walk all groups in the store and annotate ignore_value attrs.

    Designed to run as a separate pass after the write pipeline completes,
    with fresh credentials. Skips variables that already have ignore_value set.
    """
    store_config = STORE_TYPE_CONFIG[store_type]
    group_prefix = store_config["group_prefix"]

    repo_kwargs = get_repo_kwargs(
        local_storage=local_storage,
        cloud_backend=cloud_backend,
        write_creds=write_creds,
    )
    repo = icechunk.Repository.open_or_create(**repo_kwargs)
    session = repo.readonly_session(branch="main")
    root = zarr.open(session.store, mode="r")

    # Discover all experiment groups: group_prefix/Model_Name/experiment
    try:
        top = root[group_prefix]
    except KeyError:
        print(f"No '{group_prefix}' group found in store, nothing to annotate.")
        return

    groups_to_annotate = []
    for model_name in top.keys():
        model_group = top[model_name]
        if not isinstance(model_group, zarr.Group):
            continue
        for exp_name in model_group.keys():
            groups_to_annotate.append(f"{group_prefix}/{model_name}/{exp_name}")

    print(f"Annotating ignore_value for {len(groups_to_annotate)} groups in '{group_prefix}'...")
    annotated_count = 0
    skipped_count = 0
    failed_count = 0

    for group_path in groups_to_annotate:
        try:
            result = ismip6_helper.annotate_store_group(repo, group_path)
            if result:
                annotated_count += 1
                print(f"  [ANNOTATED] {group_path}")
            else:
                skipped_count += 1
        except Exception as e:
            failed_count += 1
            print(f"  [FAIL] {group_path}: {e}")

    print(f"\nAnnotation complete: {annotated_count} annotated, "
          f"{skipped_count} already done/nothing to do, {failed_count} failed")


def get_id_from_batch(batch):
    return f"{batch['institution_id']}_{batch['source_id']}/{batch['experiment_id']}"


def process_all_files(
    local_storage: bool = False,
    local_execution: bool = False,
    config_file: str = "lithops_aws.yaml",
    cloud_backend: str = "aws",
    concurrent_writes: bool = True,
    test_model: str = None,
    test_experiment: str = None,
    write_creds: str = None,
    store_type: str = "combined",
):
    """Process all files using Lithops serverless executor.

    Strategy:
    1. Build file index and group by model and experiment
    2. For each group (batch):
       - Virtualize files (~20) in serial and combine into single xarray dataset
       - Write virtual dataset to Icechunk in one commit (with conflict resolution)
    3. Process batches (~440) in parallel using lithops
    """
    import warnings
    # Silence the specific Zarr warning
    warnings.filterwarnings('ignore', module='zarr')

    if not local_execution and local_storage:
        raise ValueError("Local storage requires `local_execution=True`")

    # Step 1: Build file index
    print("Step 1/3: Building file index...")
    files_df = ismip6_helper.build_file_index()
    print(f"Step 1/3 Complete: Built file index with {len(files_df)} files!")

    # Optional test filters
    if test_model:
        files_df = files_df[files_df['model_name'] == test_model]
        print(f"  Filtered to model: {test_model} ({len(files_df)} files)")
    if test_experiment:
        files_df = files_df[files_df['experiment'] == test_experiment]
        print(f"  Filtered to experiment: {test_experiment} ({len(files_df)} files)")

    # Filter by variable type for state/flux stores
    store_config = STORE_TYPE_CONFIG[store_type]
    if store_config["filter"] is not None:
        if store_config["filter"] == "ST":
            allowed_vars = ismip6_helper.get_state_variables()
        else:
            allowed_vars = ismip6_helper.get_flux_variables()
        files_df = files_df[files_df['variable'].isin(allowed_vars)]
        print(f"  Filtered to {store_config['filter']} variables: {len(files_df)} files ({len(allowed_vars)} variable names)")

    # Step 2: Group by model and experiment and create
    print("\nStep 2/3: Grouping files by model and experiment...")
    grouped = files_df.groupby(['institution', 'model_name', 'experiment'])

    # skip groups already in the zarr store
    repo_kwargs = get_repo_kwargs(local_storage=local_storage, cloud_backend=cloud_backend, write_creds=write_creds)
    print(f"Opening icechunk repo at {repo_kwargs['storage']}")
    repo = icechunk.Repository.open_or_create(**repo_kwargs)
    session = repo.readonly_session(branch="main")
    group_prefix = store_config["group_prefix"]

    try:
        root = zarr.open(session.store, mode='r')
        print('Found groups in the store. Skipping all existing groups.')
        print(root.tree(level=1))
        batches_raw = []
        for name, group in grouped:
            path = f"{group_prefix}/{name[0]}_{name[1]}/{name[2]}"
            if path not in root:
                print(f'{path} does not exist')
                batches_raw.append((name, group.to_dict('records')))
            else:
                print(f'{path} exists. Skipping')
    except zarr.errors.GroupNotFoundError:
        # Store is empty, process all groups
        print("  Icechunk store is empty, processing all groups...")
        batches_raw = [
            (name, group.to_dict('records'))
            for name, group in grouped
        ]
    # Load skip list
    skip_list = load_skip_list()

    batches = []
    total_skipped_files = 0
    for batch_raw in batches_raw:
        # parse batch input
        ((institution_id, source_id, experiment_id), batch_files) = batch_raw
        # Transform GCS URLs to source.coop S3 URLs
        urls = [bf['url'].replace('gs://ismip6', SOURCE_BUCKET) for bf in batch_files]
        # Filter against skip list
        urls, skipped = filter_urls_by_skip_list(urls, skip_list)
        for s in skipped:
            logger.info("Skipped (skip list): %s", s)
        total_skipped_files += len(skipped)
        if not urls:
            logger.info("Batch %s_%s/%s: all files skipped", institution_id, source_id, experiment_id)
            continue
        path = f"{group_prefix}/{institution_id}_{source_id}/{experiment_id}"
        batches.append({
            'path': path,
            'experiment_id': experiment_id,
            'institution_id': institution_id,
            'source_id': source_id,
            'urls': urls,
            'bin_time': store_config['bin_time'],
        })

    if total_skipped_files:
        print(f"  Skipped {total_skipped_files} files from skip list")
    print(f"Step 2/3 Complete: Created {len(batches)} batches")

    # Increase boto read timeout for downloading large result objects.
    # NetCDF3 batches return bigger virtual datasets that can exceed the 60s default.
    import botocore.config
    os.environ.setdefault('AWS_MAX_ATTEMPTS', '5')
    boto_config = botocore.config.Config(read_timeout=300, retries={'max_attempts': 5})

    # Initialize Lithops executor
    fexec = lithops.FunctionExecutor(config_file=config_file)

    # Patch the storage backend's S3 client to use the extended timeout
    if hasattr(fexec, 'internal_storage') and hasattr(fexec.internal_storage, 'storage'):
        storage = fexec.internal_storage.storage
        if hasattr(storage, 'storage_handler') and hasattr(storage.storage_handler, 's3_client'):
            import boto3
            storage.storage_handler.s3_client = boto3.client(
                's3', region_name='us-west-2', config=boto_config,
            )

    # Step 3: Virtualize files with maximum concurrency
    virt_futures = fexec.map(batch_virt_func, [{'batch': b} for b in batches])
    virtualization_results = fexec.get_result(virt_futures)

    # Compute and display cost
    cost_stats = compute_lambda_cost(virt_futures)
    print_cost_summary(cost_stats)

    # Clean up Lithops executor
    fexec.clean()

    # Separate successful and failed virtualizations
    successful_results = [r for r in virtualization_results if r.get("success")]
    failed_results = [r for r in virtualization_results if not r.get("success")]
    print(f"Virtualization: {len(successful_results)} successful, {len(failed_results)} failed")
    for r in failed_results:
        print(f"  FAIL: {get_id_from_batch(r['batch'])}: {r.get('error', '')[:200]}")

    # Report NetCDF3 fallbacks (each one downloads the full file — major perf hit)
    total_nc3 = sum(len(r.get('nc3_fallbacks', [])) for r in successful_results)
    if total_nc3 > 0:
        print(f"\nNetCDF3 fallbacks: {total_nc3} files across batches (each downloads full file to Lambda):")
        for r in successful_results:
            fb = r.get('nc3_fallbacks', [])
            if fb:
                print(f"  {get_id_from_batch(r['batch'])}: {len(fb)} files — {', '.join(fb)}")

    # Step 4: Write virtual datasets to icechunk
    # Group by model prefix (e.g. "AWI_PISM1") so experiments within the same
    # model are written sequentially, avoiding icechunk rebase conflicts on
    # shared parent group nodes. Different models run in parallel.
    writing_results = []
    if concurrent_writes:
        # Group successful results by model prefix
        # Paths are now "group_prefix/Model_Name/experiment", so model is at index 1
        from collections import defaultdict
        by_model = defaultdict(list)
        for r in successful_results:
            model_prefix = r['batch']['path'].split('/')[1]
            by_model[model_prefix].append(r)

        # Pre-create the top-level group (combined/, state/, or flux/) and all
        # model groups in a single commit so parallel writers don't conflict
        # on root group metadata
        print(f'Pre-creating top-level group and {len(by_model)} model groups...')
        if by_model:
            session = repo.writable_session("main")
            store = session.store
            root = zarr.open(store, mode="a")
            root.require_group(group_prefix)
            for model_name in by_model:
                root.require_group(f"{group_prefix}/{model_name}")
            if session.has_uncommitted_changes:
                session.commit("Pre-create model groups for parallel writes")
            else:
                print("  All model groups already exist, skipping pre-create commit")
        print(f'Writing to icechunk: {len(by_model)} models in parallel, '
              f'experiments sequential within each model')

        # Shared flag: set when any thread detects a deadlock, signaling
        # all other threads to stop submitting new write attempts.
        _deadlock_detected = threading.Event()

        def write_model_group(model_results):
            """Write all experiments for one model sequentially."""
            results = []
            for r in model_results:
                if _deadlock_detected.is_set():
                    results.append({
                        "success": False,
                        "batch": r['batch'],
                        "error": "Skipped: deadlock detected in another thread",
                    })
                    print(f"  [SKIP] {r['batch']['path']} (deadlock in another thread)")
                    continue
                res = batch_write_func(r['batch'], r['virtual_dataset'], repo, local_storage)
                results.append(res)
                status = 'OK' if res.get('success') else 'FAIL'
                print(f"  [{status}] {res['batch']['path']}")
                if not res.get('success') and res.get('error'):
                    print(f"    Error: {res['error'][:200]}")
                if res.get('deadlock'):
                    _deadlock_detected.set()
            return results

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(write_model_group, model_results): model_prefix
                for model_prefix, model_results in by_model.items()
            }
            for future in as_completed(futures):
                writing_results.extend(future.result())

        if _deadlock_detected.is_set():
            print("\n*** DEADLOCK DETECTED ***")
            print("The icechunk Rust runtime entered an unrecoverable state.")
            print("See https://github.com/earth-mover/icechunk/issues/1586")
            print("Successfully written batches are safe. Re-run the pipeline")
            print("to resume — it will skip already-written experiments.")
    else:
        print('Writing to icechunk sequentially')
        for r in successful_results:
            batch = r['batch']
            vds = r['virtual_dataset']
            print(f"  Writing: {batch['path']}")
            res = batch_write_func(batch, vds, repo, local_storage=local_storage)
            writing_results.append(res)
            status = 'OK' if res.get('success') else 'FAIL'
            print(f"  [{status}] {batch['path']}")
            if not res.get('success') and res.get('error'):
                print(f"    Error: {res['error'][:200]}")

    successful_writes = [r for r in writing_results if r.get("success")]
    failed_writes = [r for r in writing_results if not r.get("success")]
    print(f"Writing: {len(successful_writes)} successful, {len(failed_writes)} failed")

    # Explicitly release heavy objects before returning so they don't
    # linger until the next store-type pass when running --store-type all
    del virtualization_results, successful_results, virt_futures
    del repo, writing_results
    gc.collect()

    return {
        'successful': successful_writes,
        'failed': failed_writes
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Virtualize ISMIP6 data and write to an Icechunk store via Lithops."
    )
    parser.add_argument(
        "--config", default="lithops_aws.yaml",
        help="Lithops config file (default: lithops_aws.yaml). "
             "Use lithops_gcp.yaml for Google Cloud, lithops_local.yaml for local execution."
    )
    parser.add_argument(
        "--local-storage", action="store_true",
        help="Use local filesystem for icechunk store instead of cloud."
    )
    parser.add_argument(
        "--local-execution", action="store_true",
        help="Use localhost lithops backend (implied by --config lithops_local.yaml)."
    )
    parser.add_argument(
        "--test-model", default=None,
        help="Filter to a single model name for testing (e.g. fETISh_32km)."
    )
    parser.add_argument(
        "--test-experiment", default=None,
        help="Filter to a single experiment for testing (e.g. ctrl_proj_std)."
    )
    parser.add_argument(
        "--sequential-writes", action="store_true",
        help="Disable concurrent writes (use serial loop). Default is concurrent on AWS."
    )
    parser.add_argument(
        "--write-creds", default=None,
        help="Path to JSON file with source.coop store write credentials. "
             "Expected keys: aws_access_key_id, aws_secret_access_key, aws_session_token. "
             "If not provided, credentials are read from the environment."
    )
    parser.add_argument(
        "--store-type", default="all",
        choices=["all", "combined", "state", "flux"],
        help="Type of icechunk store to build: 'all' (runs combined, state, and flux sequentially), "
             "'combined' (all variables, year-binned), 'state' (ST variables only), or "
             "'flux' (FL variables only). Default: all."
    )
    parser.add_argument(
        "--annotate-only", action="store_true",
        help="Skip the write pipeline and only run ignore_value annotation on existing groups. "
             "Use this after the main pipeline completes, with fresh credentials."
    )
    args = parser.parse_args()

    # Infer backend from config, or default based on config filename
    if args.local_storage or args.config == "lithops_local.yaml":
        cloud_backend = "local"
        local_execution = True
    else:
        cloud_backend = infer_cloud_backend(args.config)
        local_execution = args.local_execution

    if args.store_type == "all":
        store_types = ["combined", "state", "flux"]
    else:
        store_types = [args.store_type]

    # --annotate-only: skip the write pipeline, just annotate existing groups
    if args.annotate_only:
        for st in store_types:
            print(f"\n{'='*60}\nAnnotating store type: {st}\n{'='*60}")
            annotate_all_groups(
                local_storage=args.local_storage,
                cloud_backend=cloud_backend,
                write_creds=args.write_creds,
                store_type=st,
            )
        import sys
        sys.exit(0)

    # Default: concurrent writes on AWS, sequential on GCP (due to rate limits)
    concurrent_writes = not args.sequential_writes
    if cloud_backend == "gcp" and not args.sequential_writes:
        print("Note: GCP backend detected. Using sequential writes by default "
              "(GCS rate-limits ref file mutations). Pass --sequential-writes=false to override.")
        concurrent_writes = False

    if len(store_types) > 1:
        # Run each store type in a separate subprocess to ensure full cleanup
        # of Lithops threads, icechunk Rust runtime, and boto3 connection pools
        # between passes. Without this, CPU usage spikes after the first pass.
        import subprocess
        import sys
        for st in store_types:
            print(f"\n{'='*60}\nProcessing store type: {st} (subprocess)\n{'='*60}")
            cmd = [sys.executable, __file__,
                   "--config", args.config,
                   "--store-type", st]
            if args.local_storage:
                cmd.append("--local-storage")
            if args.local_execution:
                cmd.append("--local-execution")
            if args.test_model:
                cmd.extend(["--test-model", args.test_model])
            if args.test_experiment:
                cmd.extend(["--test-experiment", args.test_experiment])
            if args.sequential_writes:
                cmd.append("--sequential-writes")
            if args.write_creds:
                cmd.extend(["--write-creds", args.write_creds])
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"\nERROR: store type '{st}' exited with code {proc.returncode}")
                sys.exit(proc.returncode)
    else:
        st = store_types[0]
        print(f"\n{'='*60}\nProcessing store type: {st}\n{'='*60}")
        result = process_all_files(
            local_storage=args.local_storage,
            local_execution=local_execution,
            config_file=args.config,
            cloud_backend=cloud_backend,
            concurrent_writes=concurrent_writes,
            test_model=args.test_model,
            test_experiment=args.test_experiment,
            write_creds=args.write_creds,
            store_type=st,
        )

        successes = [get_id_from_batch(r['batch']) for r in result['successful']]
        print(f"\nSuccessful: {successes}")

        if result['failed']:
            print("\nFailures by error:")
            fails_by_error = {}
            for r in result['failed']:
                err = r.get('error', 'unknown')
                fails_by_error.setdefault(err, []).append(get_id_from_batch(r['batch']))
            for err, ids in fails_by_error.items():
                print(f"  {err[:200]}")
                for batch_id in ids:
                    print(f"    - {batch_id}")
