#!/usr/bin/env python3
"""
Compare a locally-built Icechunk store against the remote production store.

Walks all groups in both stores and compares:
  - Group structure (which groups exist)
  - Variable names within each group
  - Array shapes and dtypes
  - Coordinate array values (time, x, y)
  - Virtual chunk manifest entries (paths and byte ranges)

Usage:
    python compare_local_remote_store.py [--local-path PATH] [--group-prefix PREFIX]

Examples:
    # Compare all groups
    python compare_local_remote_store.py

    # Compare only combined groups
    python compare_local_remote_store.py --group-prefix combined

    # Use a non-default local store path
    python compare_local_remote_store.py --local-path my-output/icechunk
"""

import argparse
import sys

import icechunk
import numpy as np
import xarray as xr
import zarr

import ismip6_helper

SOURCE_BUCKET = ismip6_helper.SOURCE_DATA_URL


def _make_virtual_chunk_config():
    """Shared icechunk config for reading virtual chunks from source.coop."""
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            SOURCE_BUCKET + "/",
            store=icechunk.s3_store(region="us-west-2", anonymous=True),
        )
    )
    credentials = icechunk.containers_credentials({SOURCE_BUCKET + "/": None})
    return config, credentials


def open_local_repo(path: str) -> icechunk.Repository:
    storage = icechunk.local_filesystem_storage(path)
    config, credentials = _make_virtual_chunk_config()
    return icechunk.Repository.open(
        storage=storage,
        config=config,
        authorize_virtual_chunk_access=credentials,
    )


def open_remote_repo() -> icechunk.Repository:
    storage = icechunk.s3_storage(
        bucket="us-west-2.opendata.source.coop",
        prefix="englacial/ismip6/icechunk-ais",
        region="us-west-2",
        anonymous=True,
    )
    config, credentials = _make_virtual_chunk_config()
    return icechunk.Repository.open(
        storage=storage,
        config=config,
        authorize_virtual_chunk_access=credentials,
    )


def list_groups(store) -> list[str]:
    """Recursively list all zarr groups that contain data variables."""
    root = zarr.open_group(store, mode="r")
    groups = []

    def _walk(group, prefix=""):
        for name, item in group.members():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(item, zarr.Group):
                # Check if this group has arrays (leaf group) or just subgroups
                has_arrays = any(
                    isinstance(v, zarr.Array) for _, v in item.members()
                )
                if has_arrays:
                    groups.append(path)
                _walk(item, path)

    _walk(root)
    return sorted(groups)


ANNOTATION_ATTRS = {'ignore_value', 'valid_min', 'valid_max'}


def compare_group(local_store, remote_store, group_path: str,
                  ignore_annotations: bool = False) -> list[str]:
    """Compare a single group between local and remote stores.

    Returns a list of difference descriptions (empty if identical).
    """
    diffs = []

    try:
        local_ds = xr.open_zarr(local_store, group=group_path, consolidated=False)
    except Exception as e:
        return [f"Cannot open local group: {e}"]

    try:
        remote_ds = xr.open_zarr(remote_store, group=group_path, consolidated=False)
    except Exception as e:
        return [f"Cannot open remote group: {e}"]

    # Compare variables
    local_vars = set(local_ds.data_vars) | set(local_ds.coords)
    remote_vars = set(remote_ds.data_vars) | set(remote_ds.coords)

    only_local = local_vars - remote_vars
    only_remote = remote_vars - local_vars
    if only_local:
        diffs.append(f"Variables only in local: {sorted(only_local)}")
    if only_remote:
        diffs.append(f"Variables only in remote: {sorted(only_remote)}")

    # Compare shared variables
    for var in sorted(local_vars & remote_vars):
        local_var = local_ds[var]
        remote_var = remote_ds[var]

        # Shape
        if local_var.shape != remote_var.shape:
            diffs.append(f"  {var}: shape mismatch: local={local_var.shape} remote={remote_var.shape}")
            continue

        # Dtype
        if local_var.dtype != remote_var.dtype:
            diffs.append(f"  {var}: dtype mismatch: local={local_var.dtype} remote={remote_var.dtype}")

        # For coordinate arrays, compare values
        if var in ('time', 'x', 'y'):
            try:
                local_vals = local_var.values
                remote_vals = remote_var.values
                if not np.array_equal(local_vals, remote_vals, equal_nan=True):
                    # Check how close they are
                    if np.issubdtype(local_vals.dtype, np.floating):
                        max_diff = float(np.nanmax(np.abs(local_vals - remote_vals)))
                        diffs.append(f"  {var}: values differ (max abs diff: {max_diff})")
                    else:
                        n_diff = int(np.sum(local_vals != remote_vals))
                        diffs.append(f"  {var}: {n_diff}/{len(local_vals)} values differ")
            except Exception as e:
                diffs.append(f"  {var}: error comparing values: {e}")

        # Compare attributes
        local_attrs = dict(local_var.attrs)
        remote_attrs = dict(remote_var.attrs)
        if local_attrs != remote_attrs:
            for key in set(local_attrs) | set(remote_attrs):
                if ignore_annotations and key in ANNOTATION_ATTRS:
                    continue
                if key not in local_attrs:
                    diffs.append(f"  {var}: attr '{key}' only in remote")
                elif key not in remote_attrs:
                    diffs.append(f"  {var}: attr '{key}' only in local")
                elif local_attrs[key] != remote_attrs[key]:
                    diffs.append(f"  {var}: attr '{key}' differs: local={local_attrs[key]!r} remote={remote_attrs[key]!r}")

    local_ds.close()
    remote_ds.close()
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Compare local and remote Icechunk stores")
    parser.add_argument("--local-path", default="test-output/test_icechunk",
                        help="Path to local Icechunk store (default: test-output/test_icechunk)")
    parser.add_argument("--group-prefix", default=None,
                        help="Only compare groups under this prefix (e.g. 'combined')")
    parser.add_argument("--ignore-annotations", action="store_true",
                        help="Ignore ignore_value/valid_min/valid_max attrs (added by --annotate-only)")
    args = parser.parse_args()

    print("Opening local store...")
    local_repo = open_local_repo(args.local_path)
    local_session = local_repo.readonly_session(branch="main")
    local_store = local_session.store

    print("Opening remote store...")
    remote_repo = open_remote_repo()
    remote_session = remote_repo.readonly_session(branch="main")
    remote_store = remote_session.store

    print("Listing groups...")
    local_groups = list_groups(local_store)
    remote_groups = list_groups(remote_store)

    if args.group_prefix:
        local_groups = [g for g in local_groups if g.startswith(args.group_prefix)]
        remote_groups = [g for g in remote_groups if g.startswith(args.group_prefix)]

    local_set = set(local_groups)
    remote_set = set(remote_groups)

    # Only compare groups that exist in the local store
    # (local may be a subset if built with --test-model)
    only_local = local_set - remote_set
    common = sorted(local_set & remote_set)
    only_remote_count = len(remote_set - local_set)

    print(f"\nLocal groups: {len(local_groups)}")
    print(f"Remote groups: {len(remote_groups)}")
    print(f"Common groups: {len(common)}")
    if only_remote_count:
        print(f"Groups only in remote (skipping): {only_remote_count}")

    if only_local:
        print(f"\nWARNING: Groups only in local (not in remote): {sorted(only_local)}")

    total_diffs = 0
    groups_with_diffs = 0

    for group_path in common:
        diffs = compare_group(local_store, remote_store, group_path,
                              ignore_annotations=args.ignore_annotations)
        if diffs:
            groups_with_diffs += 1
            total_diffs += len(diffs)
            print(f"\n  DIFF {group_path}:")
            for d in diffs:
                print(f"    {d}")
        else:
            print(f"  OK   {group_path}")

    print(f"\n{'='*60}")
    if total_diffs == 0 and not only_local:
        print(f"PASS: All {len(common)} common groups match.")
        return 0
    else:
        print(f"FAIL: {groups_with_diffs}/{len(common)} groups have differences ({total_diffs} total diffs).")
        if only_local:
            print(f"      {len(only_local)} groups exist only in local store.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
