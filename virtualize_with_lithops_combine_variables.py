import lithops
import icechunk
import ismip6_helper
import obstore
import xarray as xr
import json
from datetime import datetime, timezone

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser, NetCDF3Parser
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.writers.icechunk import virtual_dataset_to_icechunk

import zarr
from typing import Dict, Tuple, Any, List, Union

def _parse_variable_from_url(url:str) -> str:
    return url.split('/')[-1].split('_')[0]

def write_vds_to_icechunk(vds: xr.Dataset, path:str):
    try:
        repo = open_or_create_repo()
        session = repo.writable_session("main")
        vds.vz.to_icechunk(session.store, group=path)
        commit_msg = f"Added {path}"
        # retries rebase until successful (https://icechunk.io/en/stable/howto/#commit-with-automatic-rebasing)
        commit_id = session.commit(commit_msg, rebase_with=icechunk.ConflictDetector())

        return {
            "success": True,
            "commit_id": commit_id,
            "path": path,
            "repo": repo,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "path": path,
            "repo": repo
        }

def virtualize_and_combine_batch(urls: List[str], parser: Union[HDFParser, NetCDF3Parser], registry: ObjectStoreRegistry) -> Dict[str, Any]:
    # Take a batch of datasets that belong to a single simulation (same model + experiment) and merge them into
    # a single virtual dataset

    # create virtual datasets (can we speed this up in parallel if we group them? Not a prio right now)
    loadable_variables = ['time', 'x', 'y', 'lat', 'lon', 'latitude', 'longitude', 'nv4', 'lon_bnds', 'lat_bnds']

    try:
    
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
            
            # # move all coordinates out of data variables
            # # TODO: We could set all coords except the one defined in the dataframe variable
            # # Revise that if this does not solve the issue 
            # vds_preprocessed = vds_var_fixed_grid.set_coords(
            #     [co for co in loadable_variables if co in vds_var_fixed_grid.data_vars]
            #     )
            vds_preprocessed = vds_var_fixed_grid
            # append the url to the dataset for easier debugging
            vds_preprocessed.attrs['url'] = url
            vdatasets.append(vds_preprocessed)


        vds = xr.merge(vdatasets, join='override', compat='override')

        return {
            "success": True,
            "dataset": vds,
            "count": len(urls)
        }
    except Exception as e:
        return {
            'success': False,
            "error": str(e),
            'single_datasets': vdatasets,
            'urls': urls
        }

def batch_func(batch: Tuple[Tuple[str], List[Dict[str, Union[str, int]]]]) -> Dict[str, Any]:
    """Wrapper around virtualization and writing for batch of files to icechunk"""
    try:
        # Virtualizing is costly, check if group exists before
        repo = open_or_create_repo()
        session = repo.readonly_session(branch="main")
        if len(list(session.store.list_dir(batch['path']))) != 0:
            raise ValueError('Group already exists')
        
        # virtualize and write in one go

        # There are some NetCDF3 files in the mix. 
        # TODO: The better way to choese the parser would be to dynamically pick it based on a head request
        # But for now this should do.
        # I manually confirmed that *all* files of this model fail with the same issue. 
        parser = NetCDF3Parser() if "SICOPOLIS1" in batch['source_id'] else HDFParser()
        bucket = "gs://ismip6"
        store = obstore.store.from_url(bucket, skip_signature=True)
        registry = ObjectStoreRegistry({bucket: store})

        virt_results = virtualize_and_combine_batch(batch['urls'], parser, registry)
        if not virt_results['success']:
            raise ValueError(f"Virtualizing failed with {virt_results['error']}")

        write_results = write_vds_to_icechunk(virt_results['dataset'], batch['path'])
        if not write_results['success']:
            raise ValueError(f"Writing failed with {virt_results['error']}")
        return {
            'success': True,
            'batch': batch,
            'results': {
                'virtualization_results':virt_results,
                'writing_results':write_results,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "batch": batch
        }


def open_or_create_repo() -> icechunk.Repository:
    # Setup Icechunk config
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store()
        )
    )

    # Use None for anonymous/public access to source data
    credentials = icechunk.containers_credentials({
        "gs://ismip6/": None
    })
    # FOR LOCAL TESTING
    # icechunk_storage = icechunk.gcs_storage(bucket="ismip6-icechunk", prefix="combined-variables-2025-12-12", from_env=True)
    icechunk_storage = icechunk.local_filesystem_storage("/Users/juliusbusecke/Code/ismip-indexing/test-output/test_icechunk")
    return icechunk.Repository.open_or_create(icechunk_storage, config=config, authorize_virtual_chunk_access=credentials)    


# def write_failures_to_bucket(failed_results: list):
#     """Write failed results to the bucket immediately.

#     Args:
#         failed_results: List of dicts with failure information
#     """
#     if not failed_results:
#         return

#     failure_log_bucket = "gs://ismip6-icechunk"
#     failure_log_path = f"failures/virtualization_failures_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

#     try:
#         failure_store = obstore.store.from_url(failure_log_bucket)
#         failure_data = json.dumps(failed_results, indent=2).encode('utf-8')
#         failure_store.put(failure_log_path, failure_data)
#         print(f"  ✓ Logged {len(failed_results)} failures to {failure_log_bucket}/{failure_log_path}")
#     except Exception as e:
#         print(f"  ✗ Failed to log failures to bucket: {e}")
#         print("  Failed results:")
#         print(json.dumps(failed_results, indent=2))
        


def process_all_files():
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

    # Step 1: Build file index
    print("Step 1/3: Building file index...")
    files_df = ismip6_helper.build_file_index()
    print(f"Step 1/3 Complete: Built file index with {len(files_df)} files!")

    # # For TESTING: filter a single model and experiment
    # files_df = files_df[files_df['model_name'] == "MALI"]
    # files_df = files_df[files_df['experiment'] == "ctrl_proj_std"]

    # Step 2: Group by model and experiment and create
    print("\nStep 2/3: Grouping files by model and experiment...")
    grouped = files_df.groupby(['institution', 'model_name', 'experiment'])

    # skip groups already in the zarr store
    repo = open_or_create_repo()
    session = repo.readonly_session(branch="main")
    root = zarr.open(session.store, mode='r')
    batches_raw = [
        (name, group.to_dict('records')) 
        for name, group in grouped 
        if f"{name[0]}_{name[1]}/{name[2]}" not in root
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
    # FOR LOCAL TESTING
    config_file = '../lithops_local.yaml'
    # config_file = 'lithops.yaml'
    fexec = lithops.FunctionExecutor(config_file=config_file)

    # Step 3: Process each batch sequentially
    print("\nStep 3/3: Processing batches...")
    total_successful = 0
    total_failed = 0

    # Parallelize virtualization for this batch
    print(f"  Virtualizing {len(batches)} files in parallel...")
    futures = fexec.map(batch_func, [{'batch':b} for b in batches])
    virtualization_results = fexec.get_result(futures)

    # Separate successful and failed virtualizations
    successful_results = [r for r in virtualization_results if r.get("success")]
    failed_results = [r for r in virtualization_results if not r.get("success")]

    # Clean up Lithops executor
    fexec.clean()

    return {
        'successful': successful_results,
        'failed': failed_results
    }

if __name__ == "__main__":
    result = process_all_files()


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
        def get_id_from_batch(result):
            b = result['batch']
            return f"{b['institution_id']}_{b['source_id']}/{b['experiment_id']}"
        id_by_error[err] = [get_id_from_batch(r) for r in results]

    id_by_error
    print(f"Processing complete: {result}")
