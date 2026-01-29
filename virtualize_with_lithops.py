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
    """Virtualize a single file. Executed as a serverless function.

    Returns:
        dict with 'path', 'dataset', and 'url' keys
    """
    parser = HDFParser()
    bucket = "gs://ismip6"
    store = obstore.store.from_url(bucket, skip_signature=True)
    registry = ObjectStoreRegistry({bucket: store})

    path = f'{row["institution"]}_{row["model_name"]}/{row["experiment"]}/{row["variable"]}' # DataTree path

    vds = open_virtual_dataset(
        url=row["url"],
        parser=parser,
        registry=registry,
        loadable_variables=['time', 'x', 'y', 'lat', 'lon', 'latitude', 'longitude'],
        decode_times=False
    )
    vds_fix_time = ismip6_helper.fix_time_encoding(vds)
    print(row)
    vds_fix_coords = ismip6_helper.correct_grid_coordinates(vds_fix_time, row["variable"])

    return {
        "success": True,
        "path": path,
        "dataset": vds_fix_coords,
        "url": row["url"]
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

    icechunk_storage = icechunk.gcs_storage(bucket="ismip6-icechunk", prefix="12-07-2025", from_env=True)

    return icechunk.Repository.open_or_create(icechunk_storage, config=config, authorize_virtual_chunk_access=credentials)    

def write_batch_to_icechunk(batch_results: list):
    """Write a batch of virtual datasets to Icechunk in a single commit.

    Args:
        batch_results: List of dicts with 'path', 'dataset', and 'url' keys

    Returns:
        dict with commit_id and list of written paths
    """
    repo = open_or_create_repo()
    session = repo.writable_session("main")

    # Write all datasets in the batch
    written_paths = []
    for result in batch_results:
        virtual_dataset_to_icechunk(result["dataset"], session.store, group=result["path"])
        written_paths.append(result["path"])

    # Single commit for the entire batch
    commit_msg = f"Added {len(written_paths)} datasets: {', '.join(written_paths)}"
    commit_id = session.commit(commit_msg)

    return {
        "commit_id": commit_id,
        "paths": written_paths,
        "count": len(written_paths)
    }

def safe_virtualize_file(row):
    """Wrapper around virtualize_file with error handling."""
    try:
        return virtualize_file(row)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": row["url"]
        }

def write_failures_to_bucket(failed_results: list):
    """Write failed results to the bucket immediately.

    Args:
        failed_results: List of dicts with failure information
    """
    if not failed_results:
        return

    failure_log_bucket = "gs://ismip6-icechunk"
    failure_log_path = f"failures/virtualization_failures_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    try:
        failure_store = obstore.store.from_url(failure_log_bucket)
        failure_data = json.dumps(failed_results, indent=2).encode('utf-8')
        failure_store.put(failure_log_path, failure_data)
        print(f"  ✓ Logged {len(failed_results)} failures to {failure_log_bucket}/{failure_log_path}")
    except Exception as e:
        print(f"  ✗ Failed to log failures to bucket: {e}")
        print("  Failed results:")
        print(json.dumps(failed_results, indent=2))


def process_all_files():
    """Process all files using Lithops serverless executor.

    Strategy:
    1. Build file index and group by model and experiment
    2. For each group (batch):
       - Parallelize virtualization using Lithops
       - Write all successful virtualizations to Icechunk in one commit
    3. Process batches sequentially to avoid Icechunk conflicts
    """
    import warnings
    # Silence the specific Zarr warning
    warnings.filterwarnings('ignore', module='zarr')

    # Step 1: Build file index
    print("Step 1/3: Building file index...")
    files_df = ismip6_helper.build_file_index()
    print(f"Step 1/3 Complete: Built file index with {len(files_df)} files!")

    # Step 2: Group by model and experiment
    print("\nStep 2/3: Grouping files by model and experiment...")
    grouped = files_df.groupby(['institution', 'model_name', 'experiment'])
    # skip groups already in the zarr store
    repo = open_or_create_repo()
    session = repo.readonly_session(branch="main")
    root = zarr.open(session.store, mode='r')
    batches = [
        (name, group.to_dict('records')) 
        for name, group in grouped 
        if f"{name[0]}_{name[1]}/{name[2]}" not in root
    ]
    print(f"Step 2/3 Complete: Created {len(batches)} batches")

    # Initialize Lithops executor
    fexec = lithops.FunctionExecutor(config_file='lithops_gcp.yaml')

    # Step 3: Process each batch sequentially
    print("\nStep 3/3: Processing batches...")
    total_successful = 0
    total_failed = 0

    for batch_idx, ((_, model_name, experiment), batch_files) in enumerate(batches, 1):
        print(f"\n{'='*80}")
        print(f"Batch {batch_idx}/{len(batches)}: {model_name} / {experiment}")
        print(f"Files in batch: {len(batch_files)}")
        print(f"{'='*80}")

        # Parallelize virtualization for this batch
        print(f"  Virtualizing {len(batch_files)} files in parallel...")
        batch_files_as_row = [{'row': file} for file in batch_files]
        futures = fexec.map(safe_virtualize_file, batch_files_as_row)
        virtualization_results = fexec.get_result(futures)

        # Separate successful and failed virtualizations
        successful_vds = [r for r in virtualization_results if r.get("success")]
        failed_vds = [r for r in virtualization_results if not r.get("success")]

        print(f"  ✓ Virtualized: {len(successful_vds)}/{len(batch_files)}")
        if failed_vds:
            print(f"  ✗ Failed virtualization: {len(failed_vds)}")

        # Write successful virtualizations to Icechunk
        if successful_vds:
            try:
                print(f"  Writing {len(successful_vds)} datasets to Icechunk...")
                write_result = write_batch_to_icechunk(successful_vds)
                print(f"  ✓ Committed {write_result['count']} datasets")
                print(f"  Commit ID: {write_result['commit_id']}")
                total_successful += len(successful_vds)
            except Exception as e:
                print(f"  ✗ Failed to write batch to Icechunk: {e}")
                # Treat all successful virtualizations as failed if write fails
                for result in successful_vds:
                    failed_vds.append({
                        "success": False,
                        "url": result["url"],
                        "error": f"Icechunk write failed: {str(e)}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

        # Write failures immediately if any occurred
        if failed_vds:
            batch_failures = [
                {
                    **result,
                    "model_name": model_name,
                    "experiment": experiment,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                for result in failed_vds
            ]
            write_failures_to_bucket(batch_failures)
            total_failed += len(failed_vds)

    # Clean up Lithops executor
    fexec.clean()

    print(f"\n{'='*80}")
    print(f"Step 3/3 Complete!")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"{'='*80}")

    return {
        "total_files": len(files_df),
        "total_batches": len(batches),
        "successful": total_successful,
        "failed": total_failed
    }

if __name__ == "__main__":
    result = process_all_files()
    print(f"Processing complete: {result}")
