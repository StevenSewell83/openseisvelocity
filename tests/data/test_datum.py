"""Tests for osv.data.datum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osv.data.datum import apply_depth_datum


def _events() -> pd.DataFrame:
    return pd.DataFrame({"lat": [-38.5, -38.6], "lon": [176.1, 176.2], "depth_km": [1.0, 3.0]})


def test_sea_level_is_identity() -> None:
    ev = _events()
    out = apply_depth_datum(ev)
    np.testing.assert_allclose(out["depth_km_bsl"], [1.0, 3.0])
    assert out.attrs["depth_datum"] == "sea_level"
    assert "depth_km_bsl" not in ev  # input untouched


def test_reference() -> None:
    out = apply_depth_datum(_events(), "reference", reference_elevation_m=600.0)
    np.testing.assert_allclose(out["depth_km_bsl"], [0.4, 2.4])
    assert out.attrs["depth_datum"] == "reference"
    assert out.attrs["depth_reference_elevation_m"] == 600.0
    assert out.attrs["depth_km_bsl_datum"] == "sea_level"


def test_surface_array_and_callable() -> None:
    out = apply_depth_datum(_events(), "surface", surface_elevation_m=[400.0, 200.0])
    np.testing.assert_allclose(out["depth_km_bsl"], [0.6, 2.8])
    out2 = apply_depth_datum(
        _events(), "surface", surface_elevation_m=lambda lat, lon: np.full(lat.shape, 1000.0)
    )
    np.testing.assert_allclose(out2["depth_km_bsl"], [0.0, 2.0])


def test_errors() -> None:
    with pytest.raises(ValueError, match="surface elevations"):
        apply_depth_datum(_events(), "surface")
    with pytest.raises(ValueError, match="shape"):
        apply_depth_datum(_events(), "surface", surface_elevation_m=[1.0])
    with pytest.raises(ValueError, match="reference_elevation_m"):
        apply_depth_datum(_events(), "reference")
