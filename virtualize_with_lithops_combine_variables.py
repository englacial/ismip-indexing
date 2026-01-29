import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading

import lithops
import icechunk
import ismip6_helper
import obstore
import xarray as xr
import yaml

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser, NetCDF3Parser
from virtualizarr.registry import ObjectStoreRegistry

import zarr
from typing import Dict, Tuple, Any, List, Union

zarr.config.set({
    'async.concurrency': 20,
    'threading.max_workers': 10
})

# AWS Lambda pricing (us-west-2): $0.0000166667 per GB-second
LAMBDA_COST_PER_GB_SECOND = 0.0000166667
# S3 PUT request cost: $0.005 per 1,000 requests
S3_PUT_COST_PER_REQUEST = 0.000005


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


def virtualize_and_combine_batch(urls: List[str], parser: Union[HDFParser, NetCDF3Parser], registry: ObjectStoreRegistry) -> Dict[str, Any]:
    # Take a batch of datasets that belong to a single simulation (same model + experiment) and merge them into
    # a single virtual dataset

    # create virtual datasets (can we speed this up in parallel if we group them? Not a prio right now)
    loadable_variables = ['x', 'y', 'lat', 'lon', 'latitude', 'longitude', 'nv4', 'lon_bnds', 'lat_bnds']

    # virtualize and append all needed metadata (for now just the variable)
    vdatasets = []
    for url in urls:
        vds_var = open_virtual_dataset(
            url=url,
            parser=parser,
            registry=registry,
            loadable_variables=loadable_variables,
            decode_times=False
        )
        # apply ismip specific fixer functions
        vds_var_fixed_time = ismip6_helper.fix_time_encoding(vds_var)
        vds_var_fixed_grid = ismip6_helper.correct_grid_coordinates(vds_var_fixed_time, _parse_variable_from_url(url))
        vds_preprocessed = vds_var_fixed_grid
        vdatasets.append(vds_preprocessed)

    vds = xr.merge(vdatasets, join='override', compat='override')

    # Split single-chunk contiguous arrays into per-time-slice virtual chunks
    vds = ismip6_helper.rechunk_contiguous_time_axis(vds)

    return vds


def batch_virt_func(batch: Tuple[Tuple[str], List[Dict[str, Union[str, int]]]]) -> Dict[str, Any]:
    """Wrap batch virtualization in error handling"""
    try:
        # There are some NetCDF3 files in the mix.
        # TODO: The better way to choose the parser would be to dynamically pick it based on a head request
        # But for now this should do.
        # I manually confirmed that *all* files of this model fail with the same issue.
        parser = NetCDF3Parser() if "SICOPOLIS1" in batch['source_id'] else HDFParser()
        bucket = "gs://ismip6"
        store = obstore.store.from_url(bucket, skip_signature=True)
        registry = ObjectStoreRegistry({bucket: store})
        vds = virtualize_and_combine_batch(batch['urls'], parser, registry)
        return {
                'success': True,
                'batch': batch,
                'virtual_dataset': vds
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


def get_repo_kwargs(local_storage: bool = False, cloud_backend: str = "aws") -> dict:
    """Build icechunk Repository kwargs for the given storage backend.

    Args:
        local_storage: Use local filesystem storage instead of cloud.
        cloud_backend: 'aws' or 'gcp'. Determines which object store and
            concurrency settings to use.
    """
    if local_storage:
        storage = icechunk.local_filesystem_storage("test-output/test_icechunk")
    elif cloud_backend == "aws":
        storage = icechunk.s3_storage(
            bucket="ismip6-icechunk",
            prefix="combined-variables-v3",
            region="us-west-2",
            from_env=True,
        )
    else:  # gcp
        storage = icechunk.gcs_storage(
            bucket="ismip6-icechunk",
            prefix="combined-variables-2025-12-19-v2",
            from_env=True,
        )

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store()
        )
    )
    # S3 handles higher concurrency than GCS
    config.max_concurrent_requests = 10 if cloud_backend == "aws" else 3

    credentials = icechunk.containers_credentials({
        "gs://ismip6/": None
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

    # Step 2: Group by model and experiment and create
    print("\nStep 2/3: Grouping files by model and experiment...")
    grouped = files_df.groupby(['institution', 'model_name', 'experiment'])

    # skip groups already in the zarr store
    repo_kwargs = get_repo_kwargs(local_storage=local_storage, cloud_backend=cloud_backend)
    print(f"Opening icechunk repo at {repo_kwargs['storage']}")
    repo = icechunk.Repository.open_or_create(**repo_kwargs)
    session = repo.readonly_session(branch="main")
    try:
        root = zarr.open(session.store, mode='r')
        print('Found groups in the store. Skipping all existing groups.')
        print(root.tree(level=1))
        batches_raw = []
        for name, group in grouped:
            path = f"{name[0]}_{name[1]}/{name[2]}"
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
    batches = []
    for batch_raw in batches_raw:
        # parse batch input
        ((institution_id, source_id, experiment_id), batch_files) = batch_raw
        urls = [bf['url'] for bf in batch_files]
        path = f"{institution_id}_{source_id}/{experiment_id}"
        batches.append({
            'path': path,
            'experiment_id': experiment_id,
            'institution_id': institution_id,
            'source_id': source_id,
            'urls': urls
        })

    print(f"Step 2/3 Complete: Created {len(batches)} batches")

    # Initialize Lithops executor
    fexec = lithops.FunctionExecutor(config_file=config_file)

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

    # Step 4: Write virtual datasets to icechunk
    # Group by model prefix (e.g. "AWI_PISM1") so experiments within the same
    # model are written sequentially, avoiding icechunk rebase conflicts on
    # shared parent group nodes. Different models run in parallel.
    writing_results = []
    if concurrent_writes:
        # Group successful results by model prefix
        from collections import defaultdict
        by_model = defaultdict(list)
        for r in successful_results:
            model_prefix = r['batch']['path'].split('/')[0]
            by_model[model_prefix].append(r)

        # Pre-create all model groups in a single commit so parallel writers
        # don't conflict on root group metadata
        print(f'Pre-creating {len(by_model)} model groups...')
        session = repo.writable_session("main")
        store = session.store
        root = zarr.open(store, mode="a")
        for model_prefix in by_model:
            root.require_group(model_prefix)
        session.commit("Pre-create model groups for parallel writes")
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

    return {
        'successful': successful_writes,
        'failed': failed_writes
    }


if __name__ == "__main__":
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
    args = parser.parse_args()

    # Infer backend from config, or default based on config filename
    if args.local_storage or args.config == "lithops_local.yaml":
        cloud_backend = "local"
        local_execution = True
    else:
        cloud_backend = infer_cloud_backend(args.config)
        local_execution = args.local_execution

    # Default: concurrent writes on AWS, sequential on GCP (due to rate limits)
    concurrent_writes = not args.sequential_writes
    if cloud_backend == "gcp" and not args.sequential_writes:
        print("Note: GCP backend detected. Using sequential writes by default "
              "(GCS rate-limits ref file mutations). Pass --sequential-writes=false to override.")
        concurrent_writes = False

    result = process_all_files(
        local_storage=args.local_storage,
        local_execution=local_execution,
        config_file=args.config,
        cloud_backend=cloud_backend,
        concurrent_writes=concurrent_writes,
        test_model=args.test_model,
        test_experiment=args.test_experiment,
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
