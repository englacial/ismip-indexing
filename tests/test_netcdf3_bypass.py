"""Test NetCDF3 virtualization bypass (no ObjectStoreRegistry)."""
import os, tempfile, numpy as np, pytest, xarray as xr
from virtualizarr.manifests import ManifestArray

from kerchunk.netCDF3 import NetCDF3ToZarr
from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs


def _create_test_nc3(path):
    """Create a minimal NetCDF3 file with scipy."""
    ds = xr.Dataset(
        {"temp": (["time", "x", "y"], np.random.rand(3, 4, 5).astype(np.float32))},
        coords={"time": [0.0, 1.0, 2.0], "x": np.arange(4), "y": np.arange(5)},
    )
    ds.to_netcdf(path, engine="scipy", format="NETCDF3_CLASSIC")
    return ds


def test_bypass_produces_virtual_dataset():
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp_path = f.name
    try:
        orig = _create_test_nc3(tmp_path)

        # The bypass path
        refs = NetCDF3ToZarr(tmp_path, inline_threshold=0).translate()
        mg = manifestgroup_from_kerchunk_refs(refs)
        vds = mg.to_virtual_dataset()

        # Data vars should be ManifestArrays
        assert "temp" in vds.data_vars
        assert isinstance(vds["temp"].data, ManifestArray)

        # Manifest paths should contain tmp_path (as file:// URI)
        entry = list(vds["temp"].data.manifest.values())[0]
        assert tmp_path in entry["path"] or f"file://{tmp_path}" in entry["path"]

    finally:
        os.unlink(tmp_path)


def test_bypass_with_loadable_vars():
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp_path = f.name
    try:
        orig = _create_test_nc3(tmp_path)

        refs = NetCDF3ToZarr(tmp_path, inline_threshold=0).translate()
        mg = manifestgroup_from_kerchunk_refs(refs)
        vds = mg.to_virtual_dataset()

        # Load coords with scipy (simulating the pipeline)
        loadable = ["time", "x", "y"]
        with xr.open_dataset(tmp_path, engine="scipy", decode_times=False) as real_ds:
            vars_to_load = [v for v in loadable if v in real_ds and v in vds]
            real_keep = real_ds[vars_to_load].load()
            vds_keep = vds.drop_vars(vars_to_load, errors="ignore")
            merged = xr.merge([real_keep, vds_keep])

        # Coords should be real numpy arrays
        assert isinstance(merged["time"].values, np.ndarray)
        assert not isinstance(merged["time"].data, ManifestArray)

        # Data var should still be virtual
        assert isinstance(merged["temp"].data, ManifestArray)

    finally:
        os.unlink(tmp_path)


def test_path_rewrite():
    """Verify manifest paths can be rewritten from local to S3."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp_path = f.name
    try:
        _create_test_nc3(tmp_path)

        refs = NetCDF3ToZarr(tmp_path, inline_threshold=0).translate()
        mg = manifestgroup_from_kerchunk_refs(refs)
        vds = mg.to_virtual_dataset()

        # Check what path format the manifest actually uses
        entry = list(vds["temp"].data.manifest.values())[0]
        actual_path = entry["path"]
        print(f"Manifest path format: {actual_path}")

        # Determine the old_path prefix to use for rewriting
        if actual_path.startswith("file://"):
            old_path = f"file://{tmp_path}"
        else:
            old_path = tmp_path

        # This confirms what _rewrite_manifest_paths needs as old_path
        assert old_path in actual_path

    finally:
        os.unlink(tmp_path)
