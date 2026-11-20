"""ensure_grid_mapping: the CRS pointers a viewer needs, whatever the source files carried."""

import numpy as np
import xarray as xr

import pytest

from ismip6_helper.grid_utils import EPSG_3031_ATTRS, EPSG_3413_ATTRS, ensure_grid_mapping, region_of_crs


def _ds(crs_name=None, as_coord=False):
    data = {v: (("time", "y", "x"), np.ones((2, 4, 4), dtype="f4")) for v in ("lithk", "acabf")}
    coords = {"x": np.arange(4.0), "y": np.arange(4.0), "time": [0.0, 365.0]}
    if crs_name:
        scalar = (("",), np.int32(0)) if False else ((), np.int32(0), {"grid_mapping_name": "polar_stereographic"})
        (coords if as_coord else data)[crs_name] = scalar
    return xr.Dataset(data, coords=coords)


def test_creates_crs_when_the_source_had_none():
    out = ensure_grid_mapping(_ds())
    assert "crs" in out.coords and out["crs"].attrs == EPSG_3031_ATTRS
    assert out.attrs["grid_mapping"] == "crs"
    assert all(out[v].attrs["grid_mapping"] == "crs" for v in ("lithk", "acabf"))


def test_points_at_an_existing_scalar_and_adds_wkt():
    for name, as_coord in (("polar_stereographic", True), ("mapping", False)):
        out = ensure_grid_mapping(_ds(name, as_coord))
        assert name in out.coords and name not in out.data_vars
        assert out.attrs["grid_mapping"] == name
        assert out["lithk"].attrs["grid_mapping"] == name
        assert "Antarctic Polar Stereographic" in out[name].attrs["crs_wkt"]
        assert out[name].attrs["grid_mapping_name"] == "polar_stereographic"


def test_coordinates_bounds_and_the_scalar_itself_get_no_pointer():
    ds = _ds("crs", True)
    ds["lat_bnds"] = (("y", "x", "nv4"), np.zeros((4, 4, 4)))
    ds["latitude"] = (("y", "x"), np.zeros((4, 4)))
    before = {v: dict(ds[v].attrs) for v in ds.variables}
    out = ensure_grid_mapping(ds)
    for v in ("x", "y", "time", "crs", "lat_bnds", "latitude"):
        assert "grid_mapping" not in out[v].attrs
    assert {v: dict(ds[v].attrs) for v in ds.variables} == before  # the input is untouched
    again = ensure_grid_mapping(out)
    assert again.attrs["grid_mapping"] == "crs" and again["lithk"].attrs == out["lithk"].attrs


def test_a_second_projection_scalar_becomes_a_coordinate_not_data():
    ds = _ds("polar_stereographic", True)
    ds["mapping"] = ((), np.int32(0), {"grid_mapping_name": "polar_stereographic"})
    out = ensure_grid_mapping(ds)
    assert out.attrs["grid_mapping"] == "polar_stereographic"
    assert "mapping" in out.coords and "mapping" not in out.data_vars


def test_greenland_gets_epsg_3413():
    out = ensure_grid_mapping(_ds(), region="GrIS")
    assert out["crs"].attrs == EPSG_3413_ATTRS and out["crs"].attrs["spatial_ref"] == "EPSG:3413"
    with pytest.raises(ValueError):
        ensure_grid_mapping(_ds(), region="Mars")


def test_an_existing_scalar_keeps_its_own_hemisphere():
    ds = _ds("polar_stereographic", True)
    ds["polar_stereographic"].attrs["latitude_of_projection_origin"] = 90.0
    out = ensure_grid_mapping(ds, region="AIS")  # wrong hint; the scalar says north
    assert "Polar Stereographic North" in out["polar_stereographic"].attrs["crs_wkt"]
    assert region_of_crs({"latitude_of_projection_origin": -90.0}) == "AIS"
    assert region_of_crs({"spatial_ref": "EPSG:3413"}) == "GrIS"
    assert region_of_crs({}) is None


# --- correct_grid_coordinates: x/y rebuilt from the array shape, per ice sheet ---

from ismip6_helper.grid_utils import correct_grid_coordinates, detect_grid_resolution  # noqa: E402


def _bare(ny, nx):
    """A dataset with no x/y coordinates, as some source files come."""
    return xr.Dataset({"lithk": (("time", "ny", "nx"), np.zeros((1, ny, nx), dtype="f4"))})


def test_ais_761_reconstruction_is_pinned():
    out = correct_grid_coordinates(_bare(761, 761), "lithk")
    expected = np.linspace(-3040000.0, 3040000.0, 761)  # the pre-Greenland output, bit for bit
    assert np.array_equal(out.x.values, expected) and out.x.dtype == expected.dtype
    assert np.array_equal(out.y.values, expected)
    assert out.x.values[1] - out.x.values[0] == 8000.0
    assert out.x.attrs == {"units": "m", "long_name": "x coordinate (EPSG:3031)",
                           "standard_name": "projection_x_coordinate"}
    assert out.y.attrs["long_name"] == "y coordinate (EPSG:3031)"
    assert out.attrs["grid_mapping"] == "polar_stereographic"
    assert out.coords["polar_stereographic"].attrs == EPSG_3031_ATTRS
    assert out.lithk.dims == ("time", "y", "x")


def test_every_ais_shape_keeps_the_square_grid():
    for n, res in ((191, 32000.0), (381, 16000.0), (1521, 4000.0), (3041, 2000.0)):
        out = correct_grid_coordinates(_bare(n, n), "lithk")
        assert np.array_equal(out.x.values, np.linspace(-3040000.0, 3040000.0, n))
        assert np.array_equal(out.y.values, np.linspace(-3040000.0, 3040000.0, n))
        assert out.x.values[1] - out.x.values[0] == res


def test_greenland_4km_reconstruction():
    out = correct_grid_coordinates(_bare(721, 421), "lithk")  # GrIS grids are taller than wide
    x, y = out.x.values, out.y.values
    assert (x[0], x[-1]) == (-720000.0, 960000.0) and len(x) == 421
    assert (y[0], y[-1]) == (-3450000.0, -570000.0) and len(y) == 721
    assert np.allclose(np.diff(x), 4000.0) and np.allclose(np.diff(y), 4000.0)
    assert out.x.attrs["long_name"] == "x coordinate (EPSG:3413)"
    assert out.attrs["grid_mapping"] == "polar_stereographic"
    assert out.coords["polar_stereographic"].attrs == EPSG_3413_ATTRS
    assert out.lithk.dims == ("time", "y", "x") and out.lithk.shape == (1, 721, 421)


def test_greenland_1km_reconstruction():
    out = correct_grid_coordinates(_bare(2881, 1681), "lithk")
    x, y = out.x.values, out.y.values
    assert (x[0], x[-1], y[0], y[-1]) == (-720000.0, 960000.0, -3450000.0, -570000.0)
    assert np.allclose(np.diff(x), 1000.0) and np.allclose(np.diff(y), 1000.0)
    assert out.coords["polar_stereographic"].attrs["spatial_ref"] == "EPSG:3413"


def test_greenland_32km_stops_short_of_the_east_edge():
    # 1680 km is not a whole number of 32 km cells: 53 columns end at 944 km, spacing stays 32 km
    dx, dy, region = detect_grid_resolution(53, 91)
    assert (dx, dy, region) == (32000.0, 32000.0, "GrIS")
    out = correct_grid_coordinates(_bare(91, 53), "lithk")
    assert (out.x.values[0], out.x.values[-1]) == (-720000.0, 944000.0)
    assert (out.y.values[0], out.y.values[-1]) == (-3450000.0, -570000.0)
    assert np.allclose(np.diff(out.x.values), 32000.0)


def test_detect_grid_resolution_names_the_ice_sheet():
    assert detect_grid_resolution(761, 761) == (8000.0, 8000.0, "AIS")
    assert detect_grid_resolution(6081, 6081) == (1000.0, 1000.0, "AIS")
    assert detect_grid_resolution(1681, 2881) == (1000.0, 1000.0, "GrIS")
    assert detect_grid_resolution(337, 577) == (5000.0, 5000.0, "GrIS")


def test_unknown_shape_raises_instead_of_guessing_antarctica():
    for nx, ny in ((760, 761), (2881, 1681), (500, 500), (421, 421)):  # off by one, transposed GrIS, odd, square GrIS
        with pytest.raises(ValueError, match="not a standard ISMIP6 grid"):
            detect_grid_resolution(nx, ny)
    with pytest.raises(ValueError, match="nx=500, ny=500"):
        correct_grid_coordinates(_bare(500, 500), "lithk")
