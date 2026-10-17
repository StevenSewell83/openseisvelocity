"""Tests for osv.data.outliers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from osv.data.outliers import analyse_outliers, fit_travel_time, fit_wadati, outlier_flags

REPO = Path(__file__).resolve().parents[2]
VPVS = 1.75


def _synthetic(n_ev=30, n_sta=8, seed=1, n_bad=6):
    rng = np.random.default_rng(seed)
    rows = []
    v0, g = 4.0, 0.15
    for e in range(n_ev):
        depth = rng.uniform(1, 6)
        for s in range(n_sta):
            epi = rng.uniform(2, 40)
            d = float(np.hypot(epi, depth))
            tp = (2 / g) * np.arcsinh(g * d / (2 * v0)) + rng.normal(0, 0.01)
            ts = VPVS * tp + rng.normal(0, 0.03)
            for ph, t in (("P", tp), ("S", ts)):
                rows.append(
                    dict(
                        event_id=f"e{e}",
                        net="NZ",
                        sta=f"S{s}",
                        alias=f"S{s}",
                        phase=ph,
                        travel_time_s=t,
                        hypocentral_km=d,
                    )
                )
    df = pd.DataFrame(rows)
    tp = df[df.phase == "P"].set_index(["event_id", "sta"]).travel_time_s
    key = pd.MultiIndex.from_frame(df[["event_id", "sta"]])
    df["ts_minus_tp_s"] = np.where(
        df.phase == "S", df.travel_time_s - tp.reindex(key).to_numpy(), np.nan
    )
    s_idx = df.index[df.phase == "S"]
    bad = rng.choice(s_idx, n_bad, replace=False)
    df.loc[bad, "travel_time_s"] += rng.choice([-1, 1], n_bad) * 2.0
    df.loc[bad, "ts_minus_tp_s"] = df.loc[bad, "travel_time_s"] - tp.reindex(key[bad]).to_numpy()
    return df, set(bad)


def test_wadati_recovers_vpvs_and_flags_outliers():
    df, bad = _synthetic()
    w = fit_wadati(df, 3.0, n_boot=200, seed=3)
    assert w.vpvs == pytest.approx(VPVS, abs=0.02)
    assert w.vpvs_ci[0] < VPVS < w.vpvs_ci[1] or abs(w.vpvs - VPVS) < 0.02
    assert w.vpvs_ci[0] <= w.vpvs <= w.vpvs_ci[1]
    flagged = set(w.pairs.index[w.pairs.outlier])
    assert bad <= flagged
    assert len(flagged - bad) <= 0.05 * w.n_pairs
    assert len(w.events) == 30
    assert w.event_vpvs_median == pytest.approx(VPVS, abs=0.05)


def test_wadati_is_seeded():
    df, _ = _synthetic()
    a = fit_wadati(df, n_boot=50, seed=5).vpvs_ci
    b = fit_wadati(df, n_boot=50, seed=5).vpvs_ci
    assert a == b


def test_wadati_too_few_pairs():
    df, _ = _synthetic(n_ev=1, n_sta=2, n_bad=0)
    with pytest.raises(ValueError, match="at least 3"):
        fit_wadati(df)


def test_tt_fit_constant_gradient():
    df, _ = _synthetic(n_ev=40, n_sta=10, n_bad=0)
    # inject outliers into P
    p_idx = df.index[df.phase == "P"][:5]
    df.loc[p_idx, "travel_time_s"] += 1.5
    f = fit_travel_time(df, "P", 3.0)
    true = (2 / 0.15) * np.arcsinh(0.15 * f.table.distance_km / 8.0)
    ok = ~f.table.index.isin(p_idx)
    clean = f.table[ok]
    assert np.abs(f.predict(clean.distance_km.to_numpy()) - true[ok]).max() < 0.1
    assert f.table.loc[p_idx, "outlier"].all()
    assert f.n_outliers <= 5 + 0.03 * f.n


def test_combined_flags_and_reason():
    df, bad = _synthetic()
    df.loc[df.index[0], "travel_time_s"] = -1.0  # invalid (P pick)
    res = analyse_outliers(df, 3.0, n_boot=50)
    flags = outlier_flags(df, 3.0, n_boot=50)
    assert flags.index.equals(df.index) and flags.dtype == bool
    assert flags.equals(res.flags.rename("outlier"))
    assert bad <= set(flags.index[flags])
    assert res.pick_table.loc[df.index[0], "reason"] == "tt_invalid"
    codes = {f.code for f in res.findings}
    assert {"wadati_vpvs", "outliers_P", "outliers_S"} <= codes
    assert res.vpvs == pytest.approx(VPVS, abs=0.03)


def test_station_systematic_residual_and_inconsistent_event():
    df, _ = _synthetic(n_bad=0)
    # station S0 is 0.4 s late on P
    m = (df.sta == "S0") & (df.phase == "P")
    df.loc[m, "travel_time_s"] += 0.4
    # event e0: Vp/Vs 1.45 (S times scaled)
    e = (df.event_id == "e0") & (df.phase == "S")
    df.loc[e, "travel_time_s"] *= 1.45 / VPVS
    tp = df[df.phase == "P"].set_index(["event_id", "sta"]).travel_time_s
    key = pd.MultiIndex.from_frame(df[["event_id", "sta"]])
    df["ts_minus_tp_s"] = np.where(
        df.phase == "S", df.travel_time_s - tp.reindex(key).to_numpy(), np.nan
    )
    res = analyse_outliers(df, 3.0, n_boot=20)
    sysm = res.stations[res.stations.systematic]
    assert "S0" in set(sysm.station)
    assert "e0" in set(res.wadati.events.event_id[res.wadati.events.inconsistent])


# --- integration -----------------------------------------------------------------

_DIR = Path(os.environ.get("OSV_STATIONXML_DIR", "/home/steve/seismic/Station_Files"))
_FILES = sorted(_DIR.glob("*_STATIONXML.xml")) if _DIR.is_dir() else []
_CAT = REPO / "manual_only_catalog.xml"


@pytest.mark.integration
@pytest.mark.skipif(not _FILES or not _CAT.exists(), reason="real data not available")
def test_real_data():
    from osv.data.associate import associate_picks
    from osv.data.quakeml import read_catalog
    from osv.data.stations import read_stations

    ap = associate_picks(read_catalog(_CAT), read_stations(_FILES))
    res = analyse_outliers(ap)
    print("vpvs", res.vpvs, res.wadati.vpvs_ci, res.counts())
    assert 1.5 < res.vpvs < 2.0
    assert res.flags.index.equals(ap.picks.index)
    assert res.counts()["P"]["n"] == 2231


def test_tt_curve_does_not_bend_over_at_sparse_far_distances():
    rng = np.random.default_rng(4)
    d = np.concatenate([rng.uniform(1, 20, 400), rng.uniform(25, 27, 8)])
    t = d / 4.6 + rng.normal(0, 0.05, len(d))
    df = pd.DataFrame({"phase": "P", "hypocentral_km": d, "travel_time_s": t})
    f = fit_travel_time(df, "P", 3.0)
    far = f.table[f.table.distance_km > 24]
    assert not far.outlier.any()
    assert (np.diff(f.curve_t) >= -1e-9).all()
    assert f.n_outliers <= 0.03 * f.n
    # too few picks: nothing assessed
    small = fit_travel_time(df.iloc[:10], "P", 3.0)
    assert small.n_outliers == 0 and small.n_not_assessed == 10
