"""Tests for osv.model3d.sources (FMTOMO sources, receivers and observed times)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import GridConfig, Model3DConfig
from osv.coords import spherical_distance_km
from osv.io.fmtomo import grids as fg
from osv.io.fmtomo import sources as so
from osv.io.fmtomo import times
from osv.model3d.grid import define_grid
from osv.model3d.sources import (
    FmtomoDataError,
    build_fmtomo_data,
    load_velest_tables,
    write_fmtomo_data,
)

REPO = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("OSV_FMTOMO_BIN", REPO / "build" / "bin"))
RUNS = REPO / "projects" / "rk-nm-2017" / "runs"
RUN01, RUN03 = RUNS / "01_prep", RUNS / "03_velest_analyse"

T0 = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")


def _synth():
    ev = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "time": [T0, T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=2)],
            "lat": [-38.50, -38.51, -38.52],
            "lon": [176.20, 176.21, 176.19],
            "depth_km_bsl": [2.0, 3.0, 2.5],
            "split": ["calibration", "calibration", "validation"],
        }
    )
    st = pd.DataFrame(
        {
            "alias": ["A", "B", "C"],
            "lat": [-38.55, -38.45, -38.50],
            "lon": [176.15, 176.25, 176.30],
            "sensor_elevation_m": [400.0, 300.0, -100.0],  # C is a borehole sensor
        }
    )
    rows = []
    for e, (_, r) in zip(ev["event_id"], ev.iterrows(), strict=True):
        for s in "ABC":
            for ph, dt in (("P", 3.0), ("S", 5.0)):
                rows.append(
                    {
                        "event_id": e,
                        "pick_id": f"{e}-{s}-{ph}",
                        "phase": ph,
                        "time": r["time"] + pd.Timedelta(seconds=dt),
                        "alias": s,
                        "sigma_s": 0.02 if ph == "P" else 0.04,
                    }
                )
    pk = pd.DataFrame(rows)
    hyp = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "time": [
                str((T0 + pd.Timedelta(seconds=0.5)).tz_localize(None)),
                str((T0 + pd.Timedelta(hours=1, seconds=-0.25)).tz_localize(None)),
            ],
            "lat": [-38.505, -38.515],
            "lon": [176.205, 176.215],
            "depth_km": [2.2, 3.1],
        }
    )
    delays = pd.DataFrame({"alias": ["A", "B"], "ptcor": [0.1, -0.1], "stcor": [0.2, np.nan]})
    return pk, ev, st, hyp, delays


def test_velest_calibration_two_phases():
    pk, ev, st, hyp, _ = _synth()
    cfg = Model3DConfig()
    d = build_fmtomo_data(
        pk, ev, st, cfg, hypocentres="velest", phases=("P", "S"), hypocentre_table=hyp
    )
    assert d.n_sources == 2 and d.n_receivers == 3 and d.n_observations == 12
    assert [p.vtypes for p in d.sources[0].paths] == [(1,), (2,)]
    assert all(p.sequence == (0, 1) for p in d.sources[0].paths)
    assert d.sources[0].depth_km == 2.2 and d.sources[0].lat == -38.505
    # borehole receiver C is below the sensor elevation datum: depth = +0.1 km
    np.testing.assert_allclose(d.receivers.positions["depth_km"], [-0.4, -0.3, 0.1])
    # travel times relative to the RELOCATED origin time
    m = d.mapping
    p1 = m[(m.event_id == "e1") & (m.phase == "P")]
    np.testing.assert_allclose(p1["time"], 2.5)
    p2 = m[(m.event_id == "e2") & (m.phase == "S")]
    np.testing.assert_allclose(p2["time"], 5.25)
    # alignment with receivers.in ray order and unique keys
    assert (d.observed["receiver"].to_numpy() == d.receivers.rays["receiver"].to_numpy()).all()
    assert not m.duplicated(["source_idx", "receiver_idx", "path_idx"]).any()
    assert m["valid"].all()


def test_validation_subset_needs_catalog():
    pk, ev, st, hyp, _ = _synth()
    with pytest.raises(FmtomoDataError, match="no events left"):
        build_fmtomo_data(pk, ev, st, Model3DConfig(), subset="validation", hypocentre_table=hyp)
    d = build_fmtomo_data(pk, ev, st, Model3DConfig(), hypocentres="catalog", subset="validation")
    assert d.n_sources == 1 and d.n_observations == 3
    np.testing.assert_allclose(d.observed["time"], 3.0)
    d = build_fmtomo_data(
        pk,
        ev,
        st,
        Model3DConfig(),
        hypocentres="catalog",
        subset="all",
        phases=("S",),
        layout="separate",
    )
    assert d.n_sources == 3 and d.sources[0].paths[0].vtypes == (1,)


def test_dropped_event_findings():
    pk, ev, st, hyp, _ = _synth()
    d = build_fmtomo_data(pk, ev, st, Model3DConfig(), subset="all", hypocentre_table=hyp)
    assert d.n_sources == 2
    assert any("no VELEST relocation" in f for f in d.findings)


def test_station_delays_option():
    pk, ev, st, hyp, dl = _synth()
    off = Model3DConfig(apply_velest_station_delays=False)
    base = build_fmtomo_data(pk, ev, st, off, hypocentre_table=hyp, phases=("P", "S"))
    cfg = Model3DConfig(apply_velest_station_delays=True)
    d = build_fmtomo_data(
        pk, ev, st, cfg, hypocentre_table=hyp, phases=("P", "S"), station_delays=dl
    )
    diff = (base.mapping["time"] - d.mapping["time"]).to_numpy()
    exp = base.mapping.apply(
        lambda r: {("A", "P"): 0.1, ("B", "P"): -0.1, ("A", "S"): 0.2}.get(
            (r.station, r.phase), 0.0
        ),
        axis=1,
    )
    np.testing.assert_allclose(diff, exp)
    assert any("lack a VELEST station delay" in f for f in d.findings)
    nodl = build_fmtomo_data(pk, ev, st, cfg, hypocentre_table=hyp)
    assert any("no station delays were given" in f for f in nodl.findings)


def test_invalid_picks_dropped_or_kept():
    pk, ev, st, hyp, _ = _synth()
    pk.loc[0, "time"] = T0 - pd.Timedelta(seconds=10)
    d = build_fmtomo_data(pk, ev, st, Model3DConfig(), hypocentre_table=hyp)
    assert d.n_observations == 5 and any("dropped" in f for f in d.findings)
    d = build_fmtomo_data(pk, ev, st, Model3DConfig(), hypocentre_table=hyp, keep_invalid=True)
    assert d.n_observations == 6 and (~d.observed["valid"]).sum() == 1
    assert d.observed["time"].min() == -100.0


def test_duplicates_and_bad_arguments():
    pk, ev, st, hyp, _ = _synth()
    dup = pd.concat([pk, pk.iloc[[0]].assign(pick_id="x", sigma_s=0.5)])
    d = build_fmtomo_data(dup, ev, st, Model3DConfig(), hypocentre_table=hyp)
    assert d.n_observations == 6 and any("duplicate" in f for f in d.findings)
    for kw in ({"phases": ("X",)}, {"hypocentres": "foo"}, {"subset": "bar"}, {"layout": "z"}):
        with pytest.raises(FmtomoDataError):
            build_fmtomo_data(pk, ev, st, Model3DConfig(), hypocentre_table=hyp, **kw)
    with pytest.raises(FmtomoDataError, match="hypocentre_table"):
        build_fmtomo_data(pk, ev, st, Model3DConfig())
    with pytest.raises(FmtomoDataError, match="above the surface"):
        build_fmtomo_data(pk, ev, st, Model3DConfig(), hypocentre_table=hyp, surface_depth_km=2.5)
    build_fmtomo_data(
        pk,
        ev,
        st,
        Model3DConfig(),
        hypocentre_table=hyp,
        surface_depth_km=lambda la, lo: np.full(np.shape(la), 0.0),
    )


def test_grid_containment():
    pk, ev, st, hyp, _ = _synth()
    stg = st.assign(elevation_m=st["sensor_elevation_m"], used=True)
    cfg = Model3DConfig(
        grid=GridConfig(dx_km=2, dy_km=2, dz_km=2, pad_km=2, zmax_km=8), propgrid_refine=2
    )
    g = define_grid(stg, ev, cfg)
    d = build_fmtomo_data(pk, ev, st, cfg, grid=g, hypocentre_table=hyp)
    assert d.n_sources == 2
    far = st.assign(lon=st["lon"] + 5.0)
    with pytest.raises(FmtomoDataError, match="outside the propagation grid"):
        build_fmtomo_data(pk, ev, far, cfg, grid=g, hypocentre_table=hyp)


def test_write_roundtrip(tmp_path):
    pk, ev, st, hyp, _ = _synth()
    d = build_fmtomo_data(pk, ev, st, Model3DConfig(), hypocentre_table=hyp, phases=("P", "S"))
    files = write_fmtomo_data(d, tmp_path, source_uncertainty=(1.0, 0.01, 0.01))
    assert set(files) == {
        "sources.in",
        "sourcesref.in",
        "receivers.in",
        "otimes.dat",
        "otimes_mapping.csv",
    }
    srcs = so.read_sources(tmp_path / "sources.in")
    rec = so.read_receivers(tmp_path / "receivers.in")
    obs = times.read_observed(tmp_path / "otimes.dat")
    so.validate_receivers(rec, srcs)
    assert len(srcs) == 2 and len(rec.positions) == 3 and len(obs) == 12
    assert so.read_sources(tmp_path / "sourcesref.in")[0].uncertainty is not None
    mp = pd.read_csv(tmp_path / "otimes_mapping.csv")
    assert len(mp) == 12 and list(mp.columns)[:4] == [
        "source_idx",
        "receiver_idx",
        "path_idx",
        "ray_idx",
    ]
    np.testing.assert_allclose(obs["time"], d.observed["time"], atol=1e-6)


# ---------------------------------------------------------------------------------------
# real project
# ---------------------------------------------------------------------------------------


def _real():
    need = [
        RUN01 / "picks.parquet",
        RUN01 / "events.parquet",
        RUN01 / "stations.parquet",
        RUN03 / "final_hypocentres.parquet",
    ]
    if not all(p.exists() for p in need):
        pytest.skip("project runs 01_prep / 03_velest_analyse not available")
    return (
        pd.read_parquet(RUN01 / "picks.parquet"),
        pd.read_parquet(RUN01 / "events.parquet"),
        pd.read_parquet(RUN01 / "stations.parquet"),
        load_velest_tables(RUN03),
    )


def test_real_project_counts_and_consistency():
    pk, ev, st, tabs = _real()
    off = Model3DConfig(apply_velest_station_delays=False)
    d = build_fmtomo_data(pk, ev, st, off, phases=("P", "S"), **tabs)
    cal = ev[ev["split"] == "calibration"]
    hyp = tabs["hypocentre_table"]
    ids = set(cal["event_id"]) & set(hyp["event_id"])
    sub = pk[pk["event_id"].isin(ids) & pk["phase"].isin(["P", "S"])]
    assert d.n_sources == len(ids)
    assert d.n_receivers == sub["alias"].nunique()
    m = d.mapping
    assert (m["phase"] == "P").sum() == (sub["phase"] == "P").sum()
    assert (m["phase"] == "S").sum() == (sub["phase"] == "S").sum()
    assert d.n_observations == len(sub) and m["valid"].all()
    # travel time differs from Stage 1 only by the origin-time change (VELEST relocation)
    j = m.merge(pk[["pick_id", "travel_time_s"]], on="pick_id")
    j = j.merge(hyp[["event_id", "time"]].rename(columns={"time": "t_rel"}), on="event_id")
    j = j.merge(ev[["event_id", "time"]].rename(columns={"time": "t_cat"}), on="event_id")
    dt0 = (pd.to_datetime(j["t_rel"], utc=True) - j["t_cat"]).dt.total_seconds()
    np.testing.assert_allclose(j["time"], j["travel_time_s"] - dt0, atol=1e-6)
    assert np.abs(dt0).max() < 2.0
    assert (d.receivers.positions["depth_km"] > 0).sum() > 0  # borehole sensors below sea level
    tmp = d.observed.merge(
        m[["receiver_idx", "ray_idx"]],
        left_on=["receiver", "ray"],
        right_on=["receiver_idx", "ray_idx"],
    )
    assert len(tmp) == d.n_observations


# ---------------------------------------------------------------------------------------
# integration: fm3d on a constant-velocity model
# ---------------------------------------------------------------------------------------


@pytest.mark.integration
def test_fm3d_arrivals_map_back(tmp_path):
    from osv.tools import run_tool

    if not (BIN / "fm3d").exists():
        pytest.skip(f"{BIN / 'fm3d'} not built")
    pk, ev, st, tabs = _real()
    cfg = Model3DConfig(propgrid_refine=2)
    stg = st.assign(elevation_m=st["sensor_elevation_m"])
    g = define_grid(stg, ev, cfg)
    pick = pk[pk["event_id"].isin(ev.sort_values("event_id")["event_id"].iloc[:60])]
    d = build_fmtomo_data(pick, ev, st, cfg, grid=g, phases=("P", "S"), **tabs)
    # keep three sources and a handful of receivers
    keep_src = [1, d.n_sources // 2, d.n_sources]
    m = d.mapping[d.mapping["source_idx"].isin(keep_src)]
    sub = pick[pick["pick_id"].isin(m["pick_id"])]
    sel_ev = ev[ev["event_id"].isin(m["event_id"])]
    d = build_fmtomo_data(sub, sel_ev, st, cfg, grid=g, phases=("P", "S"), **tabs)
    assert 1 <= d.n_sources <= 3

    vp, vs = 5.0, 3.0
    vg = g.to_vgrids(vp, vs)
    itf = g.flat_interfaces()
    fg.write_vgrids(tmp_path / "vgrids.in", vg)
    fg.write_interfaces(tmp_path / "interfaces.in", itf)
    fg.write_propgrid(tmp_path / "propgrid.in", g.propgrid)
    write_fmtomo_data(d, tmp_path)
    from osv.io.fmtomo import controls as c

    c.ModeSet().write(tmp_path / "mode_set.in")
    c.FrechetSpec(False, [], [], []).write(tmp_path / "frechet.in")
    run_tool("fm3d", tmp_path, overrides={"fm3d": BIN / "fm3d"}, timeout=1800)
    arr = times.read_arrivals(tmp_path / "arrivals.dat")
    assert len(arr) == d.n_observations and arr["valid"].all()
    arr = times.attach_path_ids(arr, d.receivers).rename(columns={"time": "t_fm"})
    mp = pd.read_csv(tmp_path / "otimes_mapping.csv")
    j = arr.merge(
        mp,
        left_on=["receiver", "source", "ray"],
        right_on=["receiver_idx", "source_idx", "ray_idx"],
    )
    assert len(j) == len(arr)
    assert (j["path"] == j["path_idx"]).all()
    pos = d.receivers.positions
    for r in j.itertuples():
        s, p = d.sources[r.source - 1], pos.loc[r.receiver]
        epi = float(spherical_distance_km(s.lat, s.lon, p.lat, p.lon))
        dist = float(np.hypot(epi, s.depth_km - p.depth_km))
        v = vp if r.phase == "P" else vs
        assert r.t_fm == pytest.approx(dist / v, rel=0.02, abs=0.05)
