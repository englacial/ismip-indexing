"""Make the public ISMIP6 Icechunk store (englacial/ismip6/icechunk-ais) renderable by gridlook.

gridlook finds a dataset's CRS through the data variable's ``grid_mapping``
attribute, the parent group's ``grid_mapping`` attribute, a ``spatial_ref``
entry in ``coordinates``, or an array literally named ``crs``. The store has a
projection scalar in 1,137 of its 1,317 groups (named ``polar_stereographic``,
``crs`` or ``mapping``) but no ``grid_mapping`` attribute anywhere, and 180
groups have no projection scalar at all, so gridlook treats x/y as lon/lat.

This script adds, per group, a group attribute ``grid_mapping = <name>`` and
the same attribute on every data variable, pointing at the existing projection
scalar; where there is none it creates a ``crs`` scalar with the EPSG:3031
attributes and WKT from ``ismip6_helper.grid_utils`` (the step a fresh build now applies
in ``ensure_grid_mapping``, so a rebuilt store needs no fix-up). Chunk data is
untouched; one commit. Run it with the icechunk this repository pins (1.x), so
the repository stays at Icechunk spec v1.

Usage::

    python -m virtualize_ismip6.fix_crs_existing_store --dry-run
    python -m virtualize_ismip6.fix_crs_existing_store --apply --write-creds sc_creds.json

``sc_creds.json`` holds the source.coop write keys for ``englacial`` (STS-proxy
keys from source.coop, about an hour of life): ``aws_access_key_id``,
``aws_secret_access_key``, ``aws_session_token``, ``endpoint_url``
(``https://data.source.coop``) and ``region_name``. Through that proxy the S3
bucket is the account name and the key starts with the repository name.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import zarr

from ismip6_helper.grid_utils import CRS_ATTRS_BY_REGION, CRS_NAMES, gets_grid_mapping, region_of_crs

BUCKET = "us-west-2.opendata.source.coop"
PREFIX = "englacial/ismip6/icechunk-ais"
SOURCE_PREFIX = f"s3://{BUCKET}/englacial/ismip6/"

def load_sc_creds(path: str) -> dict[str, Any]:
    """The ``icechunk.s3_storage`` keyword arguments for a source.coop ``sc_creds.json``."""
    raw = json.load(open(path))
    raw = raw.get("Credentials", raw)
    return {
        "access_key_id": raw["aws_access_key_id"],
        "secret_access_key": raw["aws_secret_access_key"],
        "session_token": raw.get("aws_session_token"),
        "endpoint_url": raw["endpoint_url"],
        "region": raw.get("region_name", "us-east-1"),
    }


def open_repo(credentials: dict[str, Any] | None = None) -> Any:
    import icechunk

    if credentials:
        # the source.coop proxy: bucket = account, key = <repository>/...
        account, _, repo_prefix = PREFIX.partition("/")
        kw: dict[str, Any] = {"bucket": account, "prefix": repo_prefix, "force_path_style": True}
        kw.update(credentials)
    else:
        kw = {"bucket": BUCKET, "prefix": PREFIX, "region": "us-west-2", "anonymous": True}
    storage = icechunk.s3_storage(**kw)
    creds = icechunk.containers_credentials({SOURCE_PREFIX: icechunk.s3_anonymous_credentials()})
    return icechunk.Repository.open(storage=storage, authorize_virtual_chunk_access=creds)


def plan_group(group: zarr.Group) -> dict[str, Any]:
    """What to change in one group: {'crs_name': ..., 'create_crs': bool, 'vars': [...]}."""
    members = dict(group.members())
    crs_name = next((n for n in CRS_NAMES if n in members), None)
    name = crs_name or "crs"
    arrays = {n: a for n, a in members.items() if isinstance(a, zarr.Array)}
    fields = [n for n, a in arrays.items() if gets_grid_mapping(n, a.metadata.dimension_names)]
    todo = [n for n in fields if arrays[n].attrs.get("grid_mapping") != name]
    # pointers a previous run put on bounds arrays and other non-fields
    stray = [n for n, a in arrays.items() if n not in fields and "grid_mapping" in a.attrs]
    coords = str(group.attrs.get("coordinates", "")).split()
    return {
        "crs_name": name,
        "create_crs": crs_name is None,
        "vars": todo,
        "stray": stray,
        "group_attr": group.attrs.get("grid_mapping") != name,
        # a fresh build lists the scalar in ``coordinates``, so xarray opens it as a coordinate
        "coordinates": " ".join(coords + [name]) if coords and name not in coords else None,
    }


def apply_group(group: zarr.Group, plan: dict[str, Any], region: str = "AIS") -> None:
    name = plan["crs_name"]
    if plan["create_crs"]:
        arr = group.create_array(name, shape=(), dtype="int32", fill_value=0)
        arr[...] = np.int32(0)
        arr.attrs.update(CRS_ATTRS_BY_REGION[region])
    else:
        arr = group[name]
        if "crs_wkt" not in arr.attrs:
            own = region_of_crs(dict(arr.attrs)) or region
            arr.attrs["crs_wkt"] = CRS_ATTRS_BY_REGION[own]["crs_wkt"]
    for v in plan["vars"]:
        group[v].attrs["grid_mapping"] = name
    for v in plan["stray"]:
        del group[v].attrs["grid_mapping"]
    if plan["group_attr"]:
        group.attrs["grid_mapping"] = name
    if plan["coordinates"]:
        group.attrs["coordinates"] = plan["coordinates"]


def iter_groups(root: zarr.Group):
    """Yield (path, group) for every group holding arrays (depth-first)."""
    stack = [("", root)]
    while stack:
        path, g = stack.pop()
        members = dict(g.members())
        if any(isinstance(a, zarr.Array) for a in members.values()):
            yield path, g
        for n, child in members.items():
            if isinstance(child, zarr.Group):
                stack.append((f"{path}/{n}" if path else n, child))


def run(repo: Any, *, apply: bool, branch: str = "main", region: str = "AIS") -> dict[str, Any]:
    import icechunk

    session = repo.writable_session(branch) if apply else repo.readonly_session(branch)
    root = zarr.open_group(session.store, mode="r+" if apply else "r")
    stats: dict[str, Any] = {"groups": 0, "changed": 0, "created_crs": 0, "vars": 0, "stray": 0, "coordinates": 0}
    for _path, g in iter_groups(root):
        stats["groups"] += 1
        plan = plan_group(g)
        if not any((plan["vars"], plan["stray"], plan["group_attr"], plan["create_crs"], plan["coordinates"])):
            continue
        stats["changed"] += 1
        stats["created_crs"] += int(plan["create_crs"])
        stats["vars"] += len(plan["vars"])
        stats["stray"] += len(plan["stray"])
        stats["coordinates"] += int(bool(plan["coordinates"]))
        if apply:
            apply_group(g, plan, region)
    if apply and stats["changed"]:
        stats["snapshot"] = session.commit(
            f"Add grid_mapping attributes and crs scalars for viewers ({stats['changed']} groups)",
            rebase_with=icechunk.ConflictDetector(),
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="read-only report")
    g.add_argument("--apply", action="store_true", help="write the attributes and commit once")
    ap.add_argument("--write-creds", help="source.coop sc_creds.json (with --apply)")
    ap.add_argument("--region", default="AIS", choices=sorted(CRS_ATTRS_BY_REGION), help="projection for groups without a scalar")
    args = ap.parse_args(argv)
    creds = None
    if args.apply:
        if not args.write_creds:
            ap.error("--apply needs --write-creds")
        creds = load_sc_creds(args.write_creds)
    repo = open_repo(creds)
    print("main before:", repo.lookup_branch("main"), file=sys.stderr)
    stats = run(repo, apply=args.apply, region=args.region)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
