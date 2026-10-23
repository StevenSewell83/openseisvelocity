"""NonLinLoc export of a synthetic project (no binaries)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.config import OSVConfig
from osv.fmtomo.nlloc_export import NllocExportError, export_nlloc
from osv.io.nlloc import read_nlloc_grid, simple_latlon_to_xy


def _project(tmp_path, anomaly=0.3):
    depth = -1.0 + 0.5 * np.arange(21)
    lat = 14.05 + 0.0045 * np.arange(23)
    lon = 121.16 + 0.0046 * np.arange(22)
    D, LA, LO = np.meshgrid(depth, lat, lon, indexing="ij")
    vp = 3.0 + 0.3 * D
    centre = (np.abs(LA - 14.1) < 0.01) & (np.abs(LO - 121.21) < 0.01) & (np.abs(D - 2) < 1)
    vp_final = vp + anomaly * centre
    coords = {"depth": depth, "lat": lat, "lon": lon}
    dims = ("depth", "lat", "lon")
    (tmp_path / "06_invert").mkdir(parents=True)
    xr.Dataset({"vp": (dims, vp_final), "vp_start": (dims, vp)}, coords=coords).to_netcdf(
        tmp_path / "06_invert" / "final_model.nc"
    )
    (tmp_path / "04_build_model").mkdir()
    xr.Dataset({"vp": (dims, vp), "vs": (dims, vp / 1.75)}, coords=coords).to_netcdf(
        tmp_path / "04_build_model" / "starting_model.nc"
    )
    (tmp_path / "03_velest_analyse").mkdir()
    pd.DataFrame(
        {"sta": ["LONG1", "LONG2"], "alias": ["L001", "L002"], "ptcor": [0.05, -0.02],
         "stcor": [0.09, -0.04], "nobs": [30, 12]}
    ).to_csv(tmp_path / "03_velest_analyse" / "station_delays.csv", index=False)  # fmt: skip
    (tmp_path / "01_prep").mkdir()
    pd.DataFrame({"net": "XX", "sta": ["LONG1", "LONG2"], "alias": ["L001", "L002"]}).to_csv(
        tmp_path / "01_prep" / "aliases.csv", index=False
    )
    cfg = OSVConfig(
        project={"workdir": str(tmp_path)}, coordinates={"origin": {"lat": 14.1, "lon": 121.21}}
    )
    return cfg, xr.load_dataset(tmp_path / "06_invert" / "final_model.nc")


def test_export_grids_delays_and_summary(tmp_path) -> None:
    cfg, final = _project(tmp_path)
    res = export_nlloc(
        tmp_path, cfg, tmp_path / "out", root="m", half_width_km=8.0, z_range_km=(-0.5, 7.0),
        spacing_km=0.25,
    )  # fmt: skip
    p, s = read_nlloc_grid(res.roots["P"]), read_nlloc_grid(res.roots["S"])
    assert p.grid_type == "SLOW_LEN" and (p.lat0, p.lon0) == (14.1, 121.21)
    assert p.shape == (65, 65, 31) and p.orig == (-8.0, -8.0, -0.5)
    vp, vs = 0.25 / p.values, 0.25 / s.values
    x, y, z = p.axes()
    # the linear depth profile is reproduced by the B-spline, also outside the FMTOMO grid
    edge = 0  # x = -8 km lies outside the node grid: padded 1-D model
    np.testing.assert_allclose(vp[edge, 32, :], 3.0 + 0.3 * z, rtol=1e-5)
    np.testing.assert_allclose(vp[32, edge, 10:20], 3.0 + 0.3 * z[10:20], rtol=1e-5)
    # the anomaly survives at the centre (smoothed by the B-spline); Vs keeps Vp/Vs 1.75
    k = int(np.argmin(np.abs(z - 2.0)))
    assert vp[32, 32, k] > 3.0 + 0.3 * 2.0 + 0.05
    np.testing.assert_allclose(vp / vs, 1.75, rtol=1e-5)
    # station corrections with station codes (not VELEST aliases)
    d = pd.read_csv(res.delays_csv)
    assert set(d["station"]) == {"LONG1", "LONG2"} and set(d["phase"]) == {"P", "S"}
    assert d.set_index(["station", "phase"]).loc[("LONG1", "S"), "delay_s"] == pytest.approx(0.09)
    meta = json.loads((tmp_path / "out" / "m_export.json").read_text())
    assert meta["s_model"].startswith("final Vp") and "VELEST" in meta["station_corrections"]
    assert 0 < meta["fraction_inside_fmtomo_grid"] < 1
    # x/y of the grid nodes are NonLinLoc SIMPLE coordinates
    xs, ys = simple_latlon_to_xy(14.1, 121.21, 14.1, 121.21)
    assert (float(xs), float(ys)) == (0.0, 0.0) and x[32] == pytest.approx(0.0)


def test_export_needs_inverted_model(tmp_path) -> None:
    cfg = OSVConfig(project={"workdir": str(tmp_path)})
    with pytest.raises(NllocExportError, match="invert stage"):
        export_nlloc(tmp_path, cfg, tmp_path / "out")


def test_cli_export_nlloc(tmp_path) -> None:
    from typer.testing import CliRunner

    from osv.cli import app

    _project(tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"project: {{name: t, workdir: {tmp_path}}}\n"
        "coordinates: {origin: {lat: 14.1, lon: 121.21}}\n"
    )
    out = tmp_path / "nll"
    r = CliRunner().invoke(
        app,
        ["export-nlloc", "-c", str(cfg), "--out", str(out), "--root", "x", "--spacing-km", "0.5",
         "--half-width-km", "4", "--zmin-km", "0", "--zmax-km", "4"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert (out / "x.P.mod.hdr").exists() and (out / "x.S.mod.buf").exists()
    assert "station corrections" in r.output
