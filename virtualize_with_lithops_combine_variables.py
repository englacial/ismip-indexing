import lithops
import icechunk
import ismip6_helper
import obstore
import xarray as xr
# import json
# import time
# import random
# from datetime import datetime, timezone
# from functools import wraps

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser, NetCDF3Parser
from virtualizarr.registry import ObjectStoreRegistry

import zarr
from typing import Dict, Tuple, Any, List, Union

import zarr
# TODO: Some of the attempts to overcome the write limits (seems to only work locally so far)
zarr.config.set({
    'async.concurrency': 5,
    'threading.max_workers': 5
})
# print(f"{zarr.config.get("async.concurrency")=}")

# print(f"{icechunk.__version__=}")
# import virtualizarr
# print(f"{virtualizarr.__version__=}")

def _parse_variable_from_url(url:str) -> str:
    return url.split('/')[-1].split('_')[0]

# # TODO: Some of the attempts to overcome the write limits (seems to only work locally so far)
# def retry_with_backoff(base_delay=2.0, backoff_factor=2.0, max_delay=30.0, timeout_budget=450.0):
#     """
#     Decorator that retries a function with exponential backoff on GCS-related errors.
#     Tracks elapsed time and stops retrying before exceeding the timeout budget.

#     Args:
#         base_delay: Initial delay in seconds
#         backoff_factor: Multiplier for exponential backoff
#         max_delay: Maximum delay between retries in seconds
#         timeout_budget: Maximum time budget in seconds (default: 450s = 83% of 540s GCF timeout)
#                        Leaves buffer for function execution time before first attempt
#     """
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             start_time = time.time()
#             last_exception = None
#             attempt = 0

#             while True:
#                 try:
#                     result = func(*args, **kwargs)
#                     # Add retry info to result if this succeeded after retries
#                     if attempt > 0 and isinstance(result, dict):
#                         result['retries'] = attempt
#                         result['elapsed_time'] = time.time() - start_time
#                     return result

#                 except Exception as e:
#                     last_exception = e
#                     error_str = str(e).lower()

#                     # Check for retryable errors
#                     retryable = any(keyword in error_str for keyword in [
#                         'exceeded the rate limit', 'too many requests', 'reduce your request rate',
#                         'slowdown', '429', '503', 'timeout', 'deadline',
#                         'transaction not found', "not found", "object store error", "error performing get"  # also retry on icechunk rebase issues
#                     ])

#                     if not retryable:
#                         # Don't retry on non-retryable errors
#                         raise

#                     # Calculate delay with exponential backoff and jitter
#                     delay = min(
#                         base_delay * (backoff_factor ** attempt) + random.uniform(0, 1),
#                         max_delay
#                     )

#                     # Check if we have time budget for another retry
#                     elapsed = time.time() - start_time
#                     if elapsed + delay >= timeout_budget:
#                         print(f"  [Timeout Budget] No time for retry (elapsed: {elapsed:.1f}s, need: {delay:.1f}s, budget: {timeout_budget}s)")
#                         raise last_exception

#                     attempt += 1
#                     print(f"  [Retry {attempt}] Error: {str(e)[:100]}... Retrying in {delay:.1f}s (elapsed: {elapsed:.1f}s)")
#                     time.sleep(delay)

#         return wrapper
#     return decorator



def virtualize_and_combine_batch(urls: List[str], parser: Union[HDFParser, NetCDF3Parser], registry: ObjectStoreRegistry) -> Dict[str, Any]:
    # Take a batch of datasets that belong to a single simulation (same model + experiment) and merge them into
    # a single virtual dataset

    # create virtual datasets (can we speed this up in parallel if we group them? Not a prio right now)
    loadable_variables = ['time', 'x', 'y', 'lat', 'lon', 'latitude', 'longitude', 'nv4', 'lon_bnds', 'lat_bnds']
    
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
        # appy ismip specific fixer functions
        vds_var_fixed_time = ismip6_helper.fix_time_encoding(vds_var)
        vds_var_fixed_grid = ismip6_helper.correct_grid_coordinates(vds_var_fixed_time, _parse_variable_from_url(url))
        vds_preprocessed = vds_var_fixed_grid
        vdatasets.append(vds_preprocessed)


    vds = xr.merge(vdatasets, join='override', compat='override')

    return vds 

def batch_virt_func(batch: Tuple[Tuple[str], List[Dict[str, Union[str, int]]]]) -> Dict[str, Any]:
    """Wrap batch virtualization in error handling"""
    try:
        # There are some NetCDF3 files in the mix. 
        # TODO: The better way to choese the parser would be to dynamically pick it based on a head request
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

# @retry_with_backoff(base_delay=0.5, backoff_factor=2.0, max_delay=30.0, timeout_budget=300.0)
def batch_write_func(batch: Tuple[Tuple[str], List[Dict[str, Union[str, int]]]], vds:xr.Dataset, repo:icechunk.Repository, local_storage:bool=False) -> Dict[str, Any]:
    """Wrap writing to icechunk in error handling"""
    try:
        session = repo.writable_session("main")
        path = batch['path']

        vds.vz.to_icechunk(session.store, group=path)
        commit_msg = f"Added {path}"
        # TODO: Remanent from when I was writing icechunk in parallel too. Might be helpful for debugging later.
        # retries rebase until successful (https://icechunk.io/en/stable/howto/#commit-with-automatic-rebasing)
        # commit_id = session.commit(commit_msg, rebase_with=icechunk.ConflictDetector(), rebase_tries=10) # try less often here so the backoff retry kicks in
        commit_id = session.commit(commit_msg)
        return {
            'success': True,
            'batch': batch,
            'virtual_dataset':vds,
            'commit_id': commit_id,
            'commit_msg': commit_msg,
        }
        
    except Exception as e:
        return {
            "success": False,
            "batch": batch,
            "error": str(e),
        }

def get_repo_kwargs(local_storage:bool=False) -> icechunk.Storage:
    if local_storage:
        storage = icechunk.local_filesystem_storage("/Users/juliusbusecke/Code/ismip-indexing/test-output/test_icechunk")
    else:
        storage = icechunk.gcs_storage(bucket="ismip6-icechunk", prefix="combined-variables-2025-12-19-v2", from_env=True)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store()
        )
    )
    config.max_concurrent_requests=3

    credentials = icechunk.containers_credentials({
        "gs://ismip6/": None
    })
    return {
        'storage':storage,
        'config':config,
        'authorize_virtual_chunk_access':credentials
    }

def get_id_from_batch(batch):
    return f"{batch['institution_id']}_{batch['source_id']}/{batch['experiment_id']}"    

def process_all_files(local_storage:bool=False, local_execution:bool=False):
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
        # not tested, but I am pretty sure this does not work
        # local execution and cloud storage does work for me. 
        raise ValueError("Local storage requires `local_execution=True`")

    # Step 1: Build file index
    print("Step 1/3: Building file index...")
    files_df = ismip6_helper.build_file_index()
    print(f"Step 1/3 Complete: Built file index with {len(files_df)} files!")

    # For TESTING: filter a single model and/or experiment
    files_df = files_df[files_df['model_name'] == "fETISh_32km"]
    files_df = files_df[files_df['experiment'] == "ctrl_proj_std"]

    # Step 2: Group by model and experiment and create
    print("\nStep 2/3: Grouping files by model and experiment...")
    grouped = files_df.groupby(['institution', 'model_name', 'experiment'])

    # skip groups already in the zarr store
    repo_kwargs = get_repo_kwargs(local_storage=local_storage)
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
        ((institution_id, source_id, experiment_id),batch_files) = batch_raw
        urls = [bf['url'] for bf in batch_files]
        path = f"{institution_id}_{source_id}/{experiment_id}"
        batches.append({
            'path': path,
            'experiment_id': experiment_id,
            'institution_id': institution_id,
            'source_id': source_id,
            'urls': urls
        }
        )

    print(f"Step 2/3 Complete: Created {len(batches)} batches")

    # Initialize Lithops executor
    if local_execution:
        config_file = 'lithops_local.yaml'
    else:
        config_file = 'lithops.yaml'
    fexec = lithops.FunctionExecutor(config_file=config_file)

    # Step 3: Virtualize files with maximum concurrency (this takes the longest IIRC)
    futures = fexec.map(batch_virt_func, [{'batch':b} for b in batches])
    virtualization_results = fexec.get_result(futures)

    # Clean up Lithops executor
    fexec.clean()

    # Separate successful and failed virtualizations
    successful_results = [r for r in virtualization_results if r.get("success")]
    failed_results = [r for r in virtualization_results if not r.get("success")]
    print(f"Virtualization: ||{len(successful_results)} successful ✅|| and ||{len(failed_results)} failed ❌||")
    print([(get_id_from_batch(r['batch']), r['error']) for r in failed_results])

    # Step 4: Write virtual datasets to icechunk (for now in serial)
    print('Write out to icechunk in serial')
    writing_results = []
    for r in successful_results:
        batch = r['batch']
        vds = r['virtual_dataset']
        print(f"Writing: {batch["path"]}")
        res = batch_write_func(batch, vds, repo, local_storage=local_storage)
        writing_results.append(res)
        print(f"Done {'✅' if res.get('success') else '❌'} {batch["path"]} ")
        if not res.get('success'):
            if r.get('error'):
                print(f"Error: {r.get('error')[0:200]}")

    successful_results = [r for r in writing_results if r.get("success")]
    failed_results = [r for r in writing_results if not r.get("success")]
    print(f"Writing: ||{len(successful_results)} successful ✅|| and ||{len(failed_results)} failed ❌||")

    #TODO: Better result reporting and returning. Its very basic and not nicely formatted for now
    return {
        'successful': successful_results,
        'failed': failed_results
    }

if __name__ == "__main__":
    # NOTES: 
    # - I was able to write a large part of the data with both local storage and compute! So the logic works in principal.
    # 
    # - Now testing with local execution + cloud storage (at least that way the user does not know the difference). Yeah also does not work. 
    #.  Its the write to GCS that is fundmentally not working. I do not understand why
    # My previous attempts did write from the lithops functions directly and showed the same issues, which leads me to believe that there is something 
    # broken with icechunk writing to GCS. 

    # Full error trace: [0m session error: object store error Generic GCS error: Error performing PUT https://storage.googleapis.com/ismip6%2Dicechunk/\n  \x1b[31m│\x1b[0m combined%2Dvariables%2D2025%2D12%2D19%2Dv2%2Frefs%2Fbranch%2Emain%2Fref%2Ejson in 3.898954958s, after 9 retries, max_retries: 9, retry_timeout: 300s  - Server returned non-2xx status code: 429\n  \x1b[31m│\x1b[0m Too Many Requests: <?xml version=\'1.0\' encoding=\'UTF-8\'?><Error><Code>SlowDown</Code><Message>The object exceeded the rate limit for object mutation operations (create, update, and delete).\n  \x1b[31m│\x1b[0m Please reduce your request rate. See https://cloud.google.com/storage/docs/gcs429.</Message><Details>The object ismip6-icechunk/combined-variables-2025-12-19-v2/refs/branch.main/ref.json\n  \x1b[31m│\x1b[0m exceeded the rate limit for object mutation operations (create, update, and delete). Please reduce your request rate. See https://cloud.google.com/storage/docs/gcs429.</Details></Error>\n  \x1b[31m│\x1b[0m \n  \x1b[31m│\x1b[0m context:\n  \x1b[31m│\x1b[0m    0: icechunk::storage::object_store::write_ref\n  \x1b[31m│\x1b[0m            with ref_key="branch.main/ref.json" previous_version=VersionInfo { etag: Some(ETag("\\"bca5b65bdadff9b7a58b11f9586fe1c9\\"")), generation: Some(Generation("1766201379785026")) }\n  \x1b[31m│\x1b[0m              at icechunk/src/storage/object_store.rs:528\n  \x1b[31m│\x1b[0m    1: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m    2: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m    3: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m    4: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m    5: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m    6: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m    7: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m    8: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m    9: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   10: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   11: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   12: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   13: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   14: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   15: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   16: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   17: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   18: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   19: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   20: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   21: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   22: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   23: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   24: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   25: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   26: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   27: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   28: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   29: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   30: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   31: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   32: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   33: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   34: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   35: icechunk::refs::update_branch\n  \x1b[31m│\x1b[0m            with name="main" new_snapshot=B26JW2A14PX1D4Q5YS5G current_snapshot=Some(0T8K5249691YH9KF2WMG)\n  \x1b[31m│\x1b[0m              at icechunk/src/refs.rs:173\n  \x1b[31m│\x1b[0m   36: icechunk::session::_commit\n  \x1b[31m│\x1b[0m            with Added ULB_fETISh_32km/ctrl_proj_std rewrite_manifests=false\n  \x1b[31m│\x1b[0m              at icechunk/src/session.rs:1042\n  \x1b[31m│\x1b[0m   37: icechunk::session::commit\n  \x1b[31m│\x1b[0m            with Added ULB_fETISh_32km/ctrl_proj_std\n  \x1b[31m│\x1b[0m              at icechunk/src/session.rs:970\n  \x1b[31m│\x1b[0m \n'
    result = process_all_files(local_storage=False, local_execution=True)

    successes = []
    for r in result['successful']:
        successes.append(get_id_from_batch(r['batch']))
    print(f"Successful {successes}")
    
    print(f"Parsing errors")
    # sort results by error type
    fails_by_error = {}
    for r in result['failed']:
        if not r['success']:
            err = r['error']
            if err in fails_by_error.keys():
                fails_by_error[err].append(r)
            else:
                fails_by_error[err] = [r]
            
    id_by_error = {}
    for err, results in fails_by_error.items():
        id_by_error[err] = [get_id_from_batch(r['batch']) for r in results]

    print(f"{id_by_error}")