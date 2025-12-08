import lithops
import icechunk
import ismip6_helper
import obstore
import xarray as xr
import json
from datetime import datetime, timezone

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.writers.icechunk import virtual_dataset_to_icechunk

import zarr

def virtualize_file(row: dict):
    """Virtualize a single file. Executed as a serverless function."""
    parser = HDFParser()
    bucket = "gs://ismip6"
    store = obstore.store.from_url(bucket, skip_signature=True)
    registry = ObjectStoreRegistry({bucket: store})

    path = f'{row["institution"]}_{row["model_name"]}/{row["experiment"]}/{row["variable"]}' # DataTree path

    vds = open_virtual_dataset(
        url=row["url"],
        parser=parser,
        registry=registry,
        loadable_variables=['time', 'x', 'y'],
        decode_times=False
    )
    vds_fix_time = ismip6_helper.fix_time_encoding(vds)
    vds_fix_coords = ismip6_helper.correct_grid_coordinates(vds_fix_time, row["variable"])
    return path, vds_fix_coords

def write_dataset(path: str, dataset: xr.Dataset):
    """Write a virtual datasets to Icechunk. Executed as a serverless function."""
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

    icechunk_storage = icechunk.gcs_storage(bucket="ismip6-icechunk", prefix="12-07-2025-test2", from_env=True)

    repo = icechunk.Repository.open_or_create(icechunk_storage, config=config, authorize_virtual_chunk_access=credentials)
    session = repo.writable_session("main")

    virtual_dataset_to_icechunk(dataset, session.store, group=path)
    commit_msg = f"Added new node at {path}"
    commit_id = session.commit(commit_msg)
    return commit_id, commit_msg

def process_and_write(row):
    try:
        path, vds = virtualize_file(row)
        commit_id, commit_msg = write_dataset(path, vds)
        return {
            "success": True,
            "url": row["url"],
            "commit_id": commit_id,
            "message": commit_msg
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": row["url"]
        }


def process_all_files():
    """Process all files using Lithops serverless executor."""
    import warnings
    # Silence the specific Zarr warning
    warnings.filterwarnings('ignore', module='zarr')

    # Step 1: Build file index
    print("Step 1/3: Building file index...")
    files_df = ismip6_helper.build_file_index()
    files = [{'row': f } for f in files_df.iloc[5:6].to_dict('records')]
    print(f"Step 1/2 Complete: Built file index with {len(files)} files!")

    # Step 2: Process files in parallel using Lithops
    print(f"\nStep 2/2: Virtualizing files using Lithops serverless functions...")

    # Initialize Lithops executor with config file
    fexec = lithops.FunctionExecutor(config_file='lithops.yaml')

    # Map the virtualize_file function across all files
    futures = fexec.map(process_and_write, files)

    # Wait for all results
    all_results = fexec.get_result(futures)
    fexec.clean()

    # Filter and log results
    successful_vdss = []
    failed_results = []

    for result in all_results:
        if result["success"]:
            successful_vdss.append(result["message"])
            print(f"✓ Successfully virtualized: {result["url"]}")
        else:
            failed_results.append({
                **result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            print(f"✗ Failed to virtualize {result['url']}: {result['error']}")

    print(f"\nStep 2/2 Complete: Virtualized {len(successful_vdss)}/{len(files)} files")
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

if __name__ == "__main__":
    result = process_all_files()
    print(f"Processing complete: {result}")
