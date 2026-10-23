"""NonLinLoc grid files and the SIMPLE transform."""

from __future__ import annotations

import numpy as np
import pytest

from osv.io.nlloc import (
    DEG2KM,
    NllocGrid,
    read_nlloc_grid,
    simple_latlon_to_xy,
    simple_xy_to_latlon,
    write_nlloc_grid,
)


def test_simple_transform_roundtrip_and_nlloc_formula() -> None:
    lat0, lon0 = 14.1, 121.21
    lat = np.array([14.1, 14.2, 14.0, 14.15])
    lon = np.array([121.21, 121.3, 121.1, 121.25])
    x, y = simple_latlon_to_xy(lat, lon, lat0, lon0)
    # latlon2rect: x uses the point's own latitude, km per degree pi * 6371.0087714 / 180
    assert DEG2KM == pytest.approx(111.19508, abs=1e-5)
    np.testing.assert_allclose(x, (lon - lon0) * DEG2KM * np.cos(np.deg2rad(lat)))
    np.testing.assert_allclose(y, (lat - lat0) * DEG2KM)
    la, lo = simple_xy_to_latlon(x, y, lat0, lon0)
    np.testing.assert_allclose(la, lat, atol=1e-12)
    np.testing.assert_allclose(lo, lon, atol=1e-12)


def test_grid_roundtrip_layout_and_header(tmp_path) -> None:
    v = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    g = NllocGrid(v, (-1.0, -2.0, -0.5), (0.1, 0.1, 0.1), "SLOW_LEN", 14.1, 121.21)
    hdr, buf = write_nlloc_grid(tmp_path / "m.P.mod", g)
    lines = hdr.read_text().splitlines()
    first = lines[0].split()
    assert first[:3] == ["2", "3", "4"] and first[-2:] == ["SLOW_LEN", "FLOAT"]
    assert lines[1].split()[:2] == ["TRANSFORM", "SIMPLE"] and "RotCW" in lines[1]
    # z fastest, then y, x outermost (array[ix][iy][iz])
    raw = np.fromfile(buf, dtype="<f4")
    assert raw[1] == v[0, 0, 1] and raw[4] == v[0, 1, 0] and raw[12] == v[1, 0, 0]
    back = read_nlloc_grid(tmp_path / "m.P.mod")
    np.testing.assert_array_equal(back.values, v)
    assert (back.lat0, back.lon0, back.rot_cw) == (14.1, 121.21, 0.0)
    assert back.axes()[2][-1] == pytest.approx(-0.5 + 0.3)


def test_write_rejects_bad_grids(tmp_path) -> None:
    good = np.ones((2, 2, 2), np.float32)
    with pytest.raises(ValueError, match="grid type"):
        write_nlloc_grid(tmp_path / "a", NllocGrid(good, (0, 0, 0), (1, 1, 1), "TIME", 0, 0))
    bad = good.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write_nlloc_grid(tmp_path / "b", NllocGrid(bad, (0, 0, 0), (1, 1, 1), "SLOW_LEN", 0, 0))
