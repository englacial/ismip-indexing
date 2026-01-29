"""
Rechunk contiguous uncompressed arrays in the existing Icechunk store.

Surgical approach: reads existing zarr metadata + HDF5 headers to compute
per-time-slice virtual chunk references, then updates the store in a single
commit. No full re-virtualization needed.

The pre-rechunk snapshot remains accessible for A/B testing.

Usage:
    python rechunk_existing_store.py [--dry-run] [--store {s3,gcs}]
"""

import argparse
import asyncio
import warnings

import fsspec
import h5py
import icechunk
import ismip6_helper
import numpy as np
import zarr

warnings.filterwarnings("ignore", module="zarr")

# Models that have contiguous uncompressed 3D arrays in both stores
AFFECTED_MODELS = ["DOE_MALI", "JPL1_ISSM", "UCIJPL_ISSM"]

STORE_CONFIGS = {
    "s3": {
        "description": "Production S3 store (combined-variables-v3)",
        "storage_fn": lambda: icechunk.s3_storage(
            bucket="ismip6-icechunk",
            prefix="combined-variables-v3",
            from_env=True,
            region="us-west-2",
        ),
    },
    "gcs": {
        "description": "Dev GCS store (12-07-2025)",
        "storage_fn": lambda: icechunk.gcs_storage(
            bucket="ismip6-icechunk",
            prefix="12-07-2025",
            from_env=True,
        ),
    },
}

# Remote filesystem for reading HDF5 headers
_https_fs = fsspec.filesystem("https")


def get_repo(store_key: str):
    store_cfg = STORE_CONFIGS[store_key]
    storage = store_cfg["storage_fn"]()
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store(),
        )
    )
    config.max_concurrent_requests = 3
    credentials = icechunk.containers_credentials({"gs://ismip6/": None})

    repo = icechunk.Repository.open(
        storage, config=config, authorize_virtual_chunk_access=credentials
    )
    print(f"Opened: {store_cfg['description']}")
    return repo


def get_source_urls(model_name: str):
    """Build a mapping of (experiment, variable) -> source GCS URL from the file index."""
    files_df = ismip6_helper.build_file_index()
    parts = model_name.split("_", 1)
    institution, model = parts[0], parts[1]

    model_files = files_df[
        (files_df["institution"] == institution) & (files_df["model_name"] == model)
    ]

    url_map = {}
    for _, row in model_files.iterrows():
        url_map[(row["experiment"], row["variable"])] = row["url"]

    return url_map, model_files["experiment"].unique().tolist()


def get_hdf5_offset(gcs_url: str, variable: str) -> int:
    """Read just the HDF5 header to get the byte offset of a dataset."""
    https_url = gcs_url.replace("gs://ismip6/", "https://storage.googleapis.com/ismip6/")
    with _https_fs.open(https_url, "rb") as f:
        h5 = h5py.File(f, "r")
        ds = h5[variable]
        offset = ds.id.get_offset()
        h5.close()
    return offset


def rechunk_array_in_store(session, array_path, source_url, var_name, dry_run=False):
    """
    Surgically rechunk a single array: read its metadata from the store,
    compute per-time-slice virtual refs, delete the old array, recreate it
    with new chunk shape, and set the new virtual refs.

    Returns True if the array was rechunked, False if skipped.
    """
    root = zarr.open(session.store, mode="r" if dry_run else "r+")
    arr = root[array_path]

    shape = arr.shape
    chunks = arr.chunks
    dtype = arr.dtype

    # Only rechunk 3D arrays where the entire time axis is one chunk
    if len(shape) != 3:
        return False
    n_time, ny, nx = shape
    if chunks[0] != n_time:
        return False  # Already chunked along time

    if dry_run:
        print(f"    {array_path}: {shape} chunks={chunks} -> ({n_time},1,1) chunks of (1,{ny},{nx})")
        return True

    itemsize = dtype.itemsize
    slice_bytes = ny * nx * itemsize

    # Get the byte offset from the HDF5 header
    offset = get_hdf5_offset(source_url, var_name)

    # Preserve metadata before deleting
    metadata = arr.metadata

    # Delete the existing array
    asyncio.run(session.store.delete_dir(array_path))

    # Recreate with new chunk shape
    new_arr = root.create_array(
        array_path,
        shape=shape,
        chunks=(1, ny, nx),
        dtype=dtype,
        fill_value=metadata.fill_value,
        compressors=None,
        filters=None,
        serializer="auto",
        attributes=metadata.attributes,
        dimension_names=metadata.dimension_names,
        overwrite=True,
    )

    # Build per-time-slice virtual chunk refs
    chunk_specs = []
    for t in range(n_time):
        chunk_specs.append(
            icechunk.VirtualChunkSpec(
                index=(t, 0, 0),
                location=source_url,
                offset=offset + t * slice_bytes,
                length=slice_bytes,
            )
        )

    failed = session.store.set_virtual_refs(
        f"/{array_path}", chunk_specs, validate_containers=True
    )
    if failed:
        raise RuntimeError(f"Failed to set virtual refs for chunks: {failed}")

    print(f"    {array_path}: {shape} -> {n_time} chunks of (1, {ny}, {nx})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Rechunk affected models in icechunk store")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--store", choices=["s3", "gcs"], default="s3", help="Which store to rechunk (default: s3)")
    args = parser.parse_args()

    print(f"Opening icechunk repository ({args.store})...")
    repo = get_repo(args.store)

    # Log pre-rechunk snapshot for A/B testing
    session = repo.readonly_session("main")
    pre_snapshot = session.snapshot_id
    print(f"Pre-rechunk snapshot: {pre_snapshot}")
    print("(use this snapshot ID in the viewer's ref field to compare against the rechunked version)")

    # Open a single writable session for all changes
    if not args.dry_run:
        session = repo.writable_session("main")
    else:
        session = repo.readonly_session("main")

    total_rechunked = 0

    for model_name in AFFECTED_MODELS:
        print(f"\nProcessing {model_name}...")
        url_map, experiments = get_source_urls(model_name)
        print(f"  {len(experiments)} experiments, {len(url_map)} source files")

        root = zarr.open(session.store, mode="r")

        if model_name not in root:
            print(f"  {model_name} not found in store, skipping")
            continue

        model_group = root[model_name]
        for exp_name in sorted(model_group.group_keys()):
            exp_group = model_group[exp_name]
            for var_name in sorted(exp_group.array_keys()):
                source_url = url_map.get((exp_name, var_name))
                if source_url is None:
                    continue

                array_path = f"{model_name}/{exp_name}/{var_name}"
                try:
                    rechunked = rechunk_array_in_store(
                        session, array_path, source_url, var_name, dry_run=args.dry_run
                    )
                    if rechunked:
                        total_rechunked += 1
                except Exception as e:
                    print(f"    ERROR {array_path}: {e}")

    print(f"\n{'Would rechunk' if args.dry_run else 'Rechunked'} {total_rechunked} arrays total.")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made.")
        return

    commit_id = session.commit(
        f"Rechunked {total_rechunked} arrays for {', '.join(AFFECTED_MODELS)} (time=1 chunks)"
    )
    print(f"\nDone!")
    print(f"  Pre-rechunk snapshot:  {pre_snapshot}")
    print(f"  Post-rechunk commit:   {commit_id}")
    print(f"  A/B test: set viewer ref to either snapshot ID")


if __name__ == "__main__":
    main()
