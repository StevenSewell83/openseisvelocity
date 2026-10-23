"""Iterated station terms (fmtomo.station_terms: iterate) on synthetic run directories."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osv.fmtomo.invert import InversionError, StationTermSpec, update_station_terms
from osv.io.fmtomo import read_observed, write_arrivals, write_observed

STATICS = {0: 0.05, 1: -0.03, 2: 0.01, 3: -0.03}  # s, pick-weighted mean zero (equal counts)


def _run(tmp_path, n_ev=30, few_station=None, rng_seed=0):
    """otimes/arrivals pair: obs = calc + static(station) + origin(event) + noise."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in STATICS:
        for e in range(n_ev):
            if few_station is not None and s == few_station and e >= 3:
                continue
            rows.append((s, e))
    st = np.array([r[0] for r in rows])
    ev = np.array([r[1] for r in rows])
    calc = 1.0 + rng.uniform(0, 2, len(rows))
    origin = rng.normal(0, 0.2, n_ev)[ev]
    obs_t = calc + np.array([STATICS[s] for s in st]) + origin + rng.normal(0, 0.002, len(rows))
    keys = pd.DataFrame({"receiver": np.arange(1, len(rows) + 1), "source": st + 1, "ray": 1})
    obs = keys.assign(subray=0, time=obs_t, uncertainty=0.05)
    obs.loc[5, "time"] = -100.0  # an invalid pick stays untouched
    write_observed(tmp_path / "otimes.dat", obs)
    arr = keys.assign(subray=0, time=calc, diffracted=False, headwave=False)
    write_arrivals(tmp_path / "arrivals.dat", arr)
    return st, ev


def test_terms_recover_statics_despite_origin_times(tmp_path) -> None:
    st, ev = _run(tmp_path)
    spec = StationTermSpec(station=st, event=ev).fresh()
    before = read_observed(tmp_path / "otimes.dat")
    out = update_station_terms(tmp_path, spec, 1)
    got = out.set_index("station")["delta_s"]
    for s, v in STATICS.items():
        assert got[s] == pytest.approx(v, abs=0.003)
    after = read_observed(tmp_path / "otimes.dat")
    assert after.loc[5, "time"] == -100.0
    d = (before["time"] - after["time"]).to_numpy()
    np.testing.assert_allclose(np.delete(d, 5), np.delete([got[s] for s in st], 5), atol=1e-5)
    # a second update finds (almost) nothing left; the terms accumulate
    out2 = update_station_terms(tmp_path, spec, 2)
    assert out2["delta_s"].abs().max() < 0.002
    assert spec.terms[0] == pytest.approx(STATICS[0], abs=0.004)
    assert set(spec.table()["station"]) == set(STATICS)


def test_few_picks_no_term_and_relaxation(tmp_path) -> None:
    st, ev = _run(tmp_path, few_station=2)
    spec = StationTermSpec(station=st, event=ev, min_picks=5, relaxation=0.5).fresh()
    out = update_station_terms(tmp_path, spec, 1).set_index("station")
    assert out.loc[2, "delta_s"] == 0.0 and out.loc[2, "n"] == 3
    used = out.drop(index=2)
    assert np.average(used["delta_s"], weights=used["n"]) == pytest.approx(0.0, abs=1e-9)
    # half of the static (relative to the other used stations)
    assert used.loc[0, "delta_s"] - used.loc[1, "delta_s"] == pytest.approx(0.04, abs=0.003)


def test_schedule_and_template_reuse() -> None:
    spec = StationTermSpec(station=np.zeros(2), event=np.zeros(2), every=2, start_iteration=2)
    assert [spec.due(i) for i in range(1, 7)] == [False, True, False, True, False, True]
    a = spec.fresh()
    a.terms[0] = 1.0
    assert spec.terms == {} and spec.fresh().terms == {}


def test_mismatched_rows_raise(tmp_path) -> None:
    st, ev = _run(tmp_path)
    with pytest.raises(InversionError, match="one row per observation"):
        update_station_terms(tmp_path, StationTermSpec(station=st[:-1], event=ev[:-1]), 1)
