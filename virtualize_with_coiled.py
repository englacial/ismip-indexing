import coiled
import icechunk
import ismip6_helper
import obstore
import xarray as xr
import json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from virtualizarr import open_virtual_dataset, VirtualiZarrDataTreeAccessor
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

import zarr

# Create cluster at module level
cluster = coiled.Cluster(
    name="ismip6-virtualization",
    n_workers=1,
    region="us-west1",
    worker_cpu=8,
    spot_policy="on-demand",
)

def virtualize_file(row):
    """Virtualize a single file. Must be a module-level function for multiprocessing."""
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

    icechunk_storage = icechunk.gcs_storage(bucket="ismip6-icechunk", prefix="12-05-2025-test", from_env=True)
    repo = icechunk.Repository.create(icechunk_storage, config, authorize_virtual_chunk_access=credentials)
    session = repo.writable_session("main")
    vzdt.to_icechunk(session.store)
    return print(session.commit("Createed virtual store"))



def process_all_files():
    """Process all files on a large instance with threading."""
    import warnings
    # Silence the specific Zarr warning
    warnings.filterwarnings('ignore', module='zarr')
    
    # Step 1: Build file index
    print("Step 1/3: Building file index...")
    files_df = ismip6_helper.build_file_index()
    files = files_df.iloc[:1000].to_dict('records')
    print(f"Step 1/3 Complete: Built file index with {len(files)} files!")

    # Step 2: Process files in parallel on this instance
    # Using ThreadPoolExecutor instead of ProcessPoolExecutor because this function
    # runs on a Dask worker (daemon process) which cannot spawn child processes
    print(f"\nStep 2/3: Virtualizing files using {cpu_count()} threads...")

    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        all_results = list(executor.map(virtualize_file, files))

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

    commit_code = combine_and_write(successful_vdss)
    print("Step 3/3 Complete: Combined virtual datasets to data tree and wrote icechunk store!")
    return commit_code

if __name__ == "__main__":
    client = cluster.get_client()
    print(f"Cluster dashboard: {client.dashboard_link}")

    # Submit the main processing function to run on the cluster
    future = client.submit(process_all_files)
    result = future.result()
    print(f"Processing complete: {result}")

    cluster.close()
