"""Tests for osv.data.plots (headless, synthetic inputs)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from osv.data import plots

ORIGIN = (-38.55, 176.194)
CUTOFFS = {
    "min_p": 6,
    "min_s": 0,
    "max_azgap": 180.0,
    "max_rms": 0.2,
    "depth_km": [-1.0, 8.0],
    "max_epi_dist_km": 25.0,
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n_ev, n_st = 12, 8
    events = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n_ev)],
            "lat": ORIGIN[0] + rng.normal(0, 0.02, n_ev),
            "lon": ORIGIN[1] + rng.normal(0, 0.02, n_ev),
            "depth_km_bsl": rng.uniform(1, 4, n_ev),
            "time": pd.date_range("2017-05-01", periods=n_ev, freq="7D", tz="UTC"),
            "rms": rng.uniform(0.05, 0.25, n_ev),
            "azgap_deg": rng.uniform(30, 200, n_ev),
            "n_P": rng.integers(3, 30, n_ev),
            "n_S": rng.integers(0, 10, n_ev),
            "nearest_station_km": rng.uniform(0.5, 5, n_ev),
            "mag": rng.uniform(0, 2, n_ev),
            "mag_valid": True,
            "selected": [True] * 10 + [False] * 2,
            "split": ["calibration"] * 8 + ["validation"] * 2 + [""] * 2,
        }
    )
    stations = pd.DataFrame(
        {
            "net": "NZ",
            "sta": [f"ST{i}" for i in range(n_st)],
            "alias": [f"ST{i}" for i in range(n_st)],
            "lat": ORIGIN[0] + rng.normal(0, 0.03, n_st),
            "lon": ORIGIN[1] + rng.normal(0, 0.03, n_st),
            "elevation_m": 300.0,
            "sensor_depth_m": [0.0] * 7 + [250.0],
            "sensor_elevation_m": [300.0] * 7 + [50.0],
        }
    )
    rows = []
    for i, e in events.iterrows():
        for _, s in stations.iterrows():
            for ph in ("P", "S"):
                d = rng.uniform(2, 15)
                rows.append(
                    {
                        "event_id": e.event_id,
                        "alias": s.alias,
                        "phase": ph,
                        "sigma_s": rng.choice([0.01, 0.03, 0.08]),
                        "hypocentral_km": d,
                        "travel_time_s": d / (4.0 if ph == "P" else 2.3),
                        "velest_weight": int(rng.integers(0, 3)),
                        "outlier": bool(rng.random() < 0.05),
                        "selected": bool(i < 10),
                        "station_lat": s.lat,
                        "station_lon": s.lon,
                        "sensor_elevation_m": s.sensor_elevation_m,
                    }
                )
    return events, stations, pd.DataFrame(rows)


def _ok(fig):
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    plt.close(fig)


def test_map(data):
    ev, st, pk = data
    _ok(plots.plot_station_event_map(ev, st, pk))


@pytest.mark.parametrize("kind", ["ew", "ns"])
def test_sections(data, kind):
    ev, st, pk = data
    _ok(plots.plot_depth_section(ev, st, ORIGIN, kind, picks=pk))


def test_profile_section(data):
    ev, st, pk = data
    prof = {
        "name": "A-A'",
        "start": [ORIGIN[0] - 0.05, ORIGIN[1]],
        "end": [ORIGIN[0] + 0.05, ORIGIN[1]],
    }
    _ok(plots.plot_depth_section(ev, st, ORIGIN, "profile", prof, pk, swath_km=50))
    with pytest.raises(ValueError):
        plots.plot_depth_section(ev, st, ORIGIN, "profile")


def test_histograms_magnitude_panel(data):
    ev, _, _ = data
    fig = plots.plot_histograms(ev, CUTOFFS)
    assert sum(a.axison for a in fig.axes) == 7
    _ok(fig)
    fig = plots.plot_histograms(ev.assign(mag_valid=False), CUTOFFS)
    assert sum(a.axison for a in fig.axes) == 6
    _ok(fig)


def test_timeline_and_picks(data):
    ev, _, pk = data
    _ok(plots.plot_event_timeline(ev))
    _ok(plots.plot_picks_per_station(pk, 5))
    _ok(plots.plot_uncertainty(pk, {0.02: 0, 0.05: 1, 0.1: 2}))


def test_travel_times_and_wadati(data):
    _, _, pk = data
    fits = {
        "P": {"coef": [0.0, 0.25, 0.0], "sigma_s": 0.1},
        "S": {"coef": [0.43, 0.0], "sigma_s": 0.2},
    }
    _ok(plots.plot_travel_times(pk, fits, 3.0))
    pairs = pd.DataFrame(
        {"tp_s": [1.0, 2.0, 3.0], "sp_s": [0.7, 1.5, 2.0], "outlier": [False, False, True]}
    )
    wad = {"vpvs": 1.72, "vpvs_ci": [1.70, 1.74], "event_vpvs_median": 1.75}
    _ok(plots.plot_wadati(pairs, wad, pd.DataFrame({"vpvs": [1.6, 1.7, 1.8, 1.9]})))
    _ok(plots.plot_wadati(pd.DataFrame(), {}))


@pytest.mark.parametrize("kind", ["ew", "ns"])
def test_rays(data, kind):
    ev, st, pk = data
    _ok(plots.plot_ray_map(pk, ev, st))
    _ok(plots.plot_ray_section(pk, ev, st, ORIGIN, kind))


def test_station_residuals_and_rejections(data):
    _, st, _ = data
    n = len(st)
    resid = pd.DataFrame(
        {
            "station": list(st["alias"]) * 2,
            "phase": ["P"] * n + ["S"] * n,
            "n": 10,
            "median_residual_s": np.linspace(-0.2, 0.3, 2 * n),
            "se_s": 0.02,
            "systematic": [True] + [False] * (2 * n - 1),
        }
    )
    _ok(plots.plot_station_residuals(resid))
    _ok(plots.plot_station_residual_map(resid, st))
    rej = pd.DataFrame(
        {
            "level": ["event", "pick", "pick"],
            "id": ["a", "b", "c"],
            "reason": ["min_p", "outlier", "outlier"],
        }
    )
    _ok(plots.plot_rejections(rej, 10, 100, 4))
    _ok(plots.plot_rejections(rej.iloc[:0], 10, 100))


def test_empty_inputs():
    _ok(plots.plot_histograms(pd.DataFrame(), CUTOFFS))
    _ok(plots.plot_picks_per_station(pd.DataFrame()))
    _ok(plots.plot_ray_map(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
