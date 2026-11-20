"""The CRS fix on a local repo shaped like the public ISMIP6 store."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from virtualize_ismip6 import fix_crs_existing_store as fix

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def ismip6_like_repo(tmp_path):
    import icechunk

    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(tmp_path / "repo")))
    s = repo.writable_session("main")
    root = zarr.open_group(s.store, mode="w")
    for path, crs_name in (
        ("combined/JPL1_ISSM/exp05", "polar_stereographic"),
        ("state/AWI_PISM1/ctrl_proj_std", "mapping"),
        ("flux/DOE_MALI/exp13", None),
    ):
        g = root.create_group(path)
        g.create_array("x", shape=(4,), dtype="f8", dimension_names=("x",))[...] = np.arange(4.0)
        g.create_array("y", shape=(4,), dtype="f8", dimension_names=("y",))[...] = np.arange(4.0)
        g.create_array("time", shape=(2,), dtype="f8", dimension_names=("time",))[...] = [0.0, 365.0]
        for v in ("lithk", "acabf"):
            a = g.create_array(
                v, shape=(2, 4, 4), dtype="f4", chunks=(1, 4, 4), dimension_names=("time", "y", "x")
            )
            a[...] = 1.0
            a.attrs["units"] = "m"
        if crs_name:
            c = g.create_array(crs_name, shape=(), dtype="i4")
            c[...] = 0
            c.attrs.update(
                {"grid_mapping_name": "polar_stereographic", "latitude_of_projection_origin": -90.0}
            )
        if path.startswith("combined"):  # a bounds array, and a stray pointer from an earlier run
            b = g.create_array("lat_bnds", shape=(4, 4, 4), dtype="f8", dimension_names=("y", "x", "nv4"))
            b[...] = 0.0
            b.attrs["grid_mapping"] = "polar_stereographic"
        g.attrs["coordinates"] = "time x y"
    s.commit("ismip6-like")
    return repo


def test_dry_run_counts_without_changing(ismip6_like_repo):
    stats = fix.run(ismip6_like_repo, apply=False)
    assert stats == {"groups": 3, "changed": 3, "created_crs": 1, "vars": 6, "stray": 1, "coordinates": 3}
    assert len(list(ismip6_like_repo.ancestry(branch="main"))) == 2  # root + one commit


def test_apply_adds_grid_mapping_and_crs(ismip6_like_repo):
    stats = fix.run(ismip6_like_repo, apply=True)
    assert stats["changed"] == 3 and "snapshot" in stats
    root = zarr.open_group(ismip6_like_repo.readonly_session("main").store, mode="r")
    g = root["combined/JPL1_ISSM/exp05"]
    assert g.attrs["grid_mapping"] == "polar_stereographic"
    assert g["lithk"].attrs["grid_mapping"] == "polar_stereographic"
    assert "Antarctic Polar Stereographic" in g["polar_stereographic"].attrs["crs_wkt"]
    g2 = root["state/AWI_PISM1/ctrl_proj_std"]
    assert g2["acabf"].attrs["grid_mapping"] == "mapping"
    g3 = root["flux/DOE_MALI/exp13"]
    assert g3["crs"].attrs["grid_mapping_name"] == "polar_stereographic"
    assert g3["crs"].attrs["latitude_of_projection_origin"] == -90.0
    assert g3["lithk"].attrs["grid_mapping"] == "crs" and g3.attrs["grid_mapping"] == "crs"
    # data untouched, and a second run is a no-op
    assert float(g3["lithk"][0, 0, 0]) == 1.0
    assert fix.run(ismip6_like_repo, apply=False)["changed"] == 0


def test_gridlook_lookup_order_is_satisfied(ismip6_like_repo):
    """gridlook: variable grid_mapping -> group grid_mapping -> 'crs'. All three now hold."""
    fix.run(ismip6_like_repo, apply=True)
    root = zarr.open_group(ismip6_like_repo.readonly_session("main").store, mode="r")
    for _path, g in fix.iter_groups(root):
        name = g.attrs["grid_mapping"]
        members = dict(g.members())
        assert name in members
        assert name in g.attrs["coordinates"].split()
        for v in ("lithk", "acabf"):
            assert g[v].attrs["grid_mapping"] == name
        for v in ("x", "y", "time", name):
            assert "grid_mapping" not in g[v].attrs
        if "lat_bnds" in members:
            assert "grid_mapping" not in g["lat_bnds"].attrs  # the stray pointer was removed


def test_load_sc_creds_maps_the_source_coop_shape(tmp_path):
    p = tmp_path / "sc_creds.json"
    p.write_text(
        '{"aws_access_key_id": "A", "aws_secret_access_key": "S", "aws_session_token": "T",'
        ' "region_name": "us-east-1", "endpoint_url": "https://data.source.coop"}'
    )
    assert fix.load_sc_creds(str(p)) == {
        "access_key_id": "A",
        "secret_access_key": "S",
        "session_token": "T",
        "endpoint_url": "https://data.source.coop",
        "region": "us-east-1",
    }


def test_region_token_of_a_file_name():
    from virtualize_ismip6.virtualize_with_lithops_combine_variables import _parse_region_from_url

    assert _parse_region_from_url("s3://b/Projection-AIS/AWI/PISM1/exp13/acabf_AIS_AWI_PISM1_exp13.nc") == "AIS"
    assert _parse_region_from_url("s3://b/Projection-GIS/AWI/ISSM1/exp05/lithk_GIS_AWI_ISSM1_exp05.nc") == "GrIS"
    assert _parse_region_from_url("s3://b/x/lithk_GrIS_X_Y_exp05.nc") == "GrIS"
    assert _parse_region_from_url("s3://b/Projection-GIS/x/y/z/lithk_AIS_mislabelled.nc") == "GrIS"
    import pytest

    with pytest.raises(ValueError):
        _parse_region_from_url("s3://b/x/scalar_file.nc")
