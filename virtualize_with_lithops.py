import lithops
import icechunk
import ismip6_helper
import obstore
import xarray as xr
import json
from datetime import datetime, timezone

from virtualizarr import open_virtual_dataset, VirtualiZarrDataTreeAccessor
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

import zarr

def virtualize_file(row: dict):
    """Virtualize a single file. Executed as a serverless function."""
    parser = HDFParser()
    bucket = "gs://ismip6"
    store = obstore.store.from_url(bucket, skip_signature=True)
    registry = ObjectStoreRegistry({bucket: store})

    try:
        p = f'{row["institution"]}_{row["model_name"]}/{row["experiment"]}/{row["variable"]}' # DataTree path

        vds = open_virtual_dataset(
            url=row["url"],
            parser=parser,
            registry=registry,
            loadable_variables=['time', 'x', 'y'],
            decode_times=False
        )
        vds_fix_time = ismip6_helper.fix_time_encoding(vds)
        vds_fix_coords = ismip6_helper.correct_grid_coordinates(vds_fix_time, row["variable"])
        return {"success": True, "data": (p, vds_fix_coords), "url": row["url"]}
    except Exception as e:
        return {"success": False, "error": str(e), "url": row["url"], "row": row}

def combine_and_write(virtual_datasets: tuple[(str, xr.Dataset)]):
    zarr.config.set({
        'async': {'concurrency': 100, 'timeout': None},
        'threading': {'max_workers': None}
    })
    # refs list is list of tuples [(key, dataset), ...]
    # first create the dict
    datasets = {k: dataset for k, dataset in virtual_datasets}
    # then create the data tree
    ismip6_dt = xr.DataTree.from_dict(datasets)
    # then the virtualizarr accessor
    vzdt = VirtualiZarrDataTreeAccessor(ismip6_dt)
    # then write to icechunk
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store()
        )
    )

    # When opening the repo, use None for anonymous/public access
    credentials = icechunk.containers_credentials({
        "gs://ismip6/": None  # None uses anonymous credentials if allowed
    })

    icechunk_storage = icechunk.gcs_storage(bucket="ismip6-icechunk", prefix="12-06-2025-test", from_env=True)
    repo = icechunk.Repository.create(icechunk_storage, config, authorize_virtual_chunk_access=credentials)
    session = repo.writable_session("main")
    vzdt.to_icechunk(session.store)
    return print(session.commit("Created virtual store"))


def process_all_files():
    """Process all files using Lithops serverless executor."""
    import warnings
    # Silence the specific Zarr warning
    warnings.filterwarnings('ignore', module='zarr')

    # Step 1: Build file index
    print("Step 1/3: Building file index...")
    files_df = ismip6_helper.build_file_index()
    files = [{'row': f } for f in files_df.iloc[:100].to_dict('records')]
    print(f"Step 1/3 Complete: Built file index with {len(files)} files!")

    # Step 2: Process files in parallel using Lithops
    print(f"\nStep 2/3: Virtualizing files using Lithops serverless functions...")

    # Initialize Lithops executor with config file
    fexec = lithops.FunctionExecutor(config_file='lithops.yaml')

    # Map the virtualize_file function across all files
    futures = fexec.map(virtualize_file, files)

    # Wait for all results
    all_results = fexec.get_result(futures)
    fexec.clean()

    # Filter and log results
    successful_vdss = []
    failed_results = []

    for result in all_results:
        if result["success"]:
            successful_vdss.append(result["data"])
            print(f"✓ Successfully virtualized: {result['data'][0]}")
        else:
            failed_results.append({
                "url": result["url"],
                "error": result["error"],
                "row": result["row"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            print(f"✗ Failed to virtualize {result['url']}: {result['error']}")

    print(f"\nStep 2/3 Complete: Virtualized {len(successful_vdss)}/{len(files)} files")
    print(f"Failed: {len(failed_results)} files")

    # Log failures to bucket
    if failed_results:
        print("\nLogging failures to bucket...")
        failure_log_bucket = "gs://ismip6-icechunk"
        failure_log_path = f"failures/virtualization_failures_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

        try:
            failure_store = obstore.store.from_url(failure_log_bucket, from_env=True)
            failure_data = json.dumps(failed_results, indent=2).encode('utf-8')
            failure_store.put(failure_log_path, failure_data)
            print(f"✓ Logged {len(failed_results)} failures to {failure_log_bucket}/{failure_log_path}")
        except Exception as e:
            print(f"✗ Failed to log failures to bucket: {e}")
            print("Failed results:")
            print(json.dumps(failed_results, indent=2))

    fexec = lithops.FunctionExecutor(config_file='lithops.yaml')
    fexec.call_async(combine_and_write, successful_vdss)
    commit_code = fexec.get_result()
    print("Step 3/3 Complete: Combined virtual datasets to data tree and wrote icechunk store!")

    # Clean up Lithops executor
    fexec.clean()

    return commit_code

if __name__ == "__main__":
    result = process_all_files()
    print(f"Processing complete: {result}")
