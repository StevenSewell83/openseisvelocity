"""Synthetic recovery test: 3-D relocation beats 1-D relocation when the truth is 3-D.

A small network (4 x 4 stations) and a linear east-west velocity gradient on top of a layered
1-D model.  Synthetic P times come from ``fm3d`` through the *true* 3-D model.  The events are
then relocated with VELEST (layered 1-D model without the gradient) and with FMTOMO (true
model, velocities fixed) from deliberately wrong catalogue hypocentres.  Needs the ``fm3d``,
``frechgen``, ``invert3d``, ``residuals``, ``scorrect`` and ``velest`` binaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osv.config.schema import OSVConfig
from osv.io.fmtomo import (
    PathSpec,
    Receivers,
    Source,
    write_interfaces,
    write_propgrid,
    write_vgrids,
)
from osv.io.velest import ModelBlock, VelestModel
from osv.model3d.extrude import extrude_1d
from osv.model3d.grid import define_grid
from osv.resolution.models import write_vgrids as grid_vgrids
from osv.resolution.synth import BaseInputs, forward_times
from osv.tools import ToolNotFoundError, which_tool
from osv.validation.relocate import (
    RelocateOptions,
    hypocentre_offsets_km,
    relocate_validation,
)

LAT0, LON0 = -38.55, 176.2
BINARIES = ("fm3d", "frechgen", "invert3d", "residuals", "scorrect", "velest")


def _have_tools() -> bool:
    try:
        for b in BINARIES:
            which_tool(b)
    except ToolNotFoundError:
        return False
    return True


LAYERS = pd.DataFrame(
    {
        "depth_top_km": [-0.5, 1.0, 2.5, 4.0],
        "vp": [2.6, 3.4, 4.4, 5.4],
        "vpvs": 1.73,
    }
)


@pytest.mark.integration
@pytest.mark.skipif(not _have_tools(), reason="fmtomo/velest binaries not built")
def test_3d_relocation_recovers_truth_better_than_1d(tmp_path) -> None:
    la, lo = np.meshgrid(np.linspace(0, 0.10, 4), np.linspace(0, 0.12, 4), indexing="ij")
    st = pd.DataFrame(
        {
            "alias": [f"S{i:02d}" for i in range(la.size)],
            "lat": LAT0 + la.ravel(),
            "lon": LON0 + lo.ravel(),
            "sensor_elevation_m": 0.0,
            "elevation_m": 0.0,
        }
    )
    cfg = OSVConfig.model_validate(
        {
            "model3d": {
                "grid": {"dx_km": 1.0, "dy_km": 1.0, "dz_km": 1.0, "pad_km": 2.0, "zmax_km": 8.0},
                "propgrid_refine": 2,
                "apply_velest_station_delays": False,
            },
            "fmtomo": {"relocate": "linear", "iterations": 4, "damping": 1.0},
            "coordinates": {"origin": {"lat": LAT0 + 0.05, "lon": LON0 + 0.06}},
            "data_selection": {"max_epi_dist_km": 60.0},
        }
    )
    grid = define_grid(st, None, cfg.model3d)

    # true model: layered background times a 15 % east-west gradient (west slow, east fast)
    bg = extrude_1d(LAYERS, grid)
    lon = bg["lon"].to_numpy()
    f = 1.0 + 0.15 * (lon - lon.mean()) / (0.5 * np.ptp(lon))
    true = bg.copy()
    true["vp"] = bg["vp"] * f[None, None, :]
    true["vs"] = bg["vs"] * f[None, None, :]

    # true events (inside the network, 2-4 km deep), observed everywhere
    rng = np.random.default_rng(3)
    n_ev = 8
    tsrc = [
        Source(
            float(rng.uniform(2.0, 4.0)),
            float(LAT0 + rng.uniform(0.02, 0.08)),
            float(LON0 + rng.uniform(0.02, 0.10)),
            [PathSpec.direct()],
        )
        for _ in range(n_ev)
    ]
    pos = pd.DataFrame(
        {"depth_km": 0.0, "lat": st["lat"].to_numpy(), "lon": st["lon"].to_numpy()},
        index=pd.RangeIndex(1, len(st) + 1, name="receiver"),
    )
    rays = pd.DataFrame(
        [(r, s, 1) for r in pos.index for s in range(1, n_ev + 1)],
        columns=["receiver", "source", "path"],
    )
    rec = Receivers(pos, rays)
    rr = rays.assign(ray=rays.groupby("receiver").cumcount() + 1)
    obs = pd.DataFrame(
        {
            "receiver": rr["receiver"],
            "source": rr["source"],
            "ray": rr["ray"],
            "subray": 0,
            "time": 1.0,
            "uncertainty": 0.03,
        }
    )
    base = BaseInputs(grid.flat_interfaces(), grid.propgrid, tsrc, rec, obs)
    arr = forward_times(true, grid, base, tmp_path / "truth", n_jobs=4).set_index(
        ["receiver", "source", "ray", "subray"]
    )

    t_origin = pd.Timestamp("2020-01-01 00:00:00", tz="UTC") + pd.to_timedelta(
        np.arange(n_ev) * 600, unit="s"
    )
    noise = rng.normal(0.0, 0.01, len(rr))
    rows = []
    for k, r in enumerate(rr.itertuples(index=False)):
        tt = float(arr.loc[(r.receiver, r.source, r.ray, 0), "time"]) + noise[k]
        rows.append(
            {
                "event_id": f"ev{r.source}",
                "pick_id": f"p{k}",
                "phase": "P",
                "alias": st["alias"].iloc[r.receiver - 1],
                "sta": st["alias"].iloc[r.receiver - 1],
                "time": t_origin[r.source - 1] + pd.to_timedelta(tt, unit="s"),
                "sigma_s": 0.03,
                "velest_weight": 0,
                "station_lat": st["lat"].iloc[r.receiver - 1],
                "station_lon": st["lon"].iloc[r.receiver - 1],
                "sensor_elevation_m": 0.0,
            }
        )
    picks = pd.DataFrame(rows)

    # catalogue start: shifted by ~1.2 km, origin time wrong by 0.3 s
    shift = rng.normal(0.0, 1.0, (n_ev, 3))
    events = pd.DataFrame(
        {
            "event_id": [f"ev{i + 1}" for i in range(n_ev)],
            "time": t_origin + pd.to_timedelta(0.3, unit="s"),
            "lat": [s.lat + shift[i, 0] / 111.2 for i, s in enumerate(tsrc)],
            "lon": [s.lon + shift[i, 1] / (111.2 * 0.78) for i, s in enumerate(tsrc)],
            "depth_km_bsl": [max(0.5, s.depth_km + 0.8 * shift[i, 2]) for i, s in enumerate(tsrc)],
            "split": "validation",
        }
    )

    # the 3-D model handed to the validation: the truth
    mdir = tmp_path / "model3d"
    mdir.mkdir()
    write_vgrids(mdir / "vgrids.in", grid_vgrids(grid, true, with_s=False))
    write_interfaces(mdir / "interfaces.in", grid.flat_interfaces())
    write_propgrid(mdir / "propgrid.in", grid.propgrid)

    # the 1-D model: constant-velocity layers without the gradient
    vm = VelestModel(
        title="synthetic 1-D",
        p=ModelBlock(
            layers=pd.DataFrame(
                {
                    "vel": LAYERS["vp"].to_numpy(),
                    "depth": LAYERS["depth_top_km"].to_numpy(),
                    "vdamp": 1.0,
                    "reflector": " ",
                }
            ),
            label="P",
        ),
    )
    res = relocate_validation(
        events,
        picks,
        vm,
        None,
        mdir,
        grid,
        cfg,
        tmp_path / "val",
        n_jobs=4,
        stations=st,
        options=RelocateOptions(n_iterations=4),
    )
    ev = res.events.set_index("event_id")
    truth = pd.DataFrame(
        {
            "lat": [s.lat for s in tsrc],
            "lon": [s.lon for s in tsrc],
            "depth": [s.depth_km for s in tsrc],
        },
        index=[f"ev{i + 1}" for i in range(n_ev)],
    ).loc[ev.index]
    truth.to_csv(tmp_path / "truth.csv")

    def err(tag: str) -> np.ndarray:
        d = hypocentre_offsets_km(
            truth["lat"], truth["lon"], truth["depth"], ev[f"lat_{tag}"], ev[f"lon_{tag}"],
            ev[f"depth_{tag}"],
        )  # fmt: skip
        return np.sqrt(d["dh_km"] ** 2 + d["dz_km"] ** 2).to_numpy()

    e1, e3 = err("1d"), err("3d")
    print("1-D error km", np.round(e1, 2), "median", np.median(e1))
    print("3-D error km", np.round(e3, 2), "median", np.median(e3))
    print("rms 1d", ev["rms_1d"].round(3).tolist(), "3d", ev["rms_3d"].round(3).tolist())
    assert len(ev) == n_ev
    assert np.median(e3) < 0.5 * np.median(e1)
    assert np.median(e3) < 0.5
    assert (res.events["rms_3d"] < res.events["rms_1d"]).mean() > 0.7
    assert res.stats["rms"]["median_3d"] < 0.04
    assert res.stats["rms"]["p_3d_better"] < 0.05
    assert not any(f.code == "validation_3d_worse" for f in res.findings)
