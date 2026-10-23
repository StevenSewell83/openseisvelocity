"""Coupled P+S and Vp/Vs inversions in Stage 4 (``osv.fmtomo.stage.invert_coupled``).

Fast tests monkeypatch the joint runners of :mod:`osv.fmtomo.joint`; the integration test runs the
real Extras programs on a tiny synthetic geometry with P and S synthetic data.
"""

from __future__ import annotations

import dataclasses
import importlib.util  # noqa: E402
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from osv.config.schema import FmtomoConfig, ResolutionConfig
from osv.fmtomo import stage as st
from osv.fmtomo.joint import VpVsInversionResult
from osv.fmtomo.rundir import RunInputs
from osv.io.fmtomo import read_observed, read_vgrids
from osv.resolution.models import checkerboard, write_vgrids
from osv.resolution.synth import make_synthetic_data
from osv.stages import StageError
from osv.tools import ToolNotFoundError, which_tool

_spec = importlib.util.spec_from_file_location(
    "_osv_test_stage4", Path(__file__).with_name("test_stage4.py")
)
_t4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t4)
_gaussian_model, _have_tools, _setup = _t4._gaussian_model, _t4._have_tools, _t4._setup

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "build" / "test-runs" / "osv_010"


@pytest.fixture
def workdir():
    WORK.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="run-", dir=WORK))
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        WORK.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------------------
# config mapping
# ---------------------------------------------------------------------------------------
def test_config_relocation_validation():
    assert FmtomoConfig().joint.relocation is None
    ok = FmtomoConfig(phases=["P", "S"], joint={"relocation": "joint"})
    assert ok.joint.relocation == "joint"
    with pytest.raises(ValueError, match="nonlinear"):
        FmtomoConfig(phases=["P", "S"], relocate="linear", joint={"relocation": "joint"})
    with pytest.raises(ValueError, match="nonlinear"):
        FmtomoConfig(phases=["P", "S"], relocate="none", joint={"relocation": "p_fixed_for_s"})
    FmtomoConfig(phases=["P", "S"], relocate="linear", joint={"relocation": "independent"})
    with pytest.raises(ValueError, match="independent"):
        FmtomoConfig(phases=["vpvs"], joint={"relocation": "independent"})


@pytest.mark.parametrize(
    ("kw", "have", "expect"),
    [
        ({"phases": ["P"]}, ["P"], None),
        ({"phases": ["P", "S"]}, ["P", "S"], None),  # default: separate runs
        ({"phases": ["P", "S"], "joint": {"relocation": "independent"}}, ["P", "S"], None),
        ({"phases": ["P", "S"], "joint": {"relocation": "joint"}}, ["P", "S"], "joint"),
        (
            {"phases": ["P", "S"], "joint": {"relocation": "p_fixed_for_s"}},
            ["P", "S"],
            "p_fixed_for_s",
        ),
        ({"phases": ["vpvs"]}, ["P", "S"], "joint"),  # [vpvs] defaults to joint relocation
        (
            {"phases": ["vpvs"], "joint": {"relocation": "p_fixed_for_s"}},
            ["P", "S"],
            "p_fixed_for_s",
        ),
    ],
)
def test_coupled_mode(kw, have, expect):
    assert st.coupled_mode(FmtomoConfig(**kw), have) == expect


def test_coupled_mode_errors():
    with pytest.raises(StageError, match="both phases"):
        st.coupled_mode(FmtomoConfig(phases=["vpvs"]), ["P"])
    with pytest.raises(StageError, match="nonlinear"):
        st.coupled_mode(FmtomoConfig(phases=["vpvs"], relocate="none"), ["P", "S"])


def test_station_ids():
    def pd_(names, n):
        m = pd.DataFrame({"source_idx": range(1, len(names) + 1), "station": names})
        return SimpleNamespace(mapping=m, inputs=SimpleNamespace(sources=[0] * n))

    ids = st.station_ids({"P": pd_(["A", "B", "C"], 3), "S": pd_(["C", "X", "A"], 3)})
    assert ids["P"] == [1, 2, 3]
    assert ids["S"] == [3, 3 + 2, 1]  # C -> P station 3, X (S only) above the P count, A -> 1


def test_common_event_numbers():
    p = pd.DataFrame({"event_idx": [1, 1, 2, 3], "event_id": ["a", "a", "b", "c"]})
    s = pd.DataFrame({"event_idx": [1, 2, 2], "event_id": ["c", "x", "x"]})  # c is also in P
    ids = st.common_event_numbers({"P": p, "S": s})
    assert list(ids["P"]) == [1, 1, 2, 3]
    assert list(ids["S"]) == [3, 4, 4]  # c keeps its P number, the S-only event comes after


# ---------------------------------------------------------------------------------------
# wiring with monkeypatched runners
# ---------------------------------------------------------------------------------------
def _datas(grid, base, bg, drop_s_event: int | None = None):
    """Nonlinear PhaseData of P and S on the toy geometry (S optionally lacks one event)."""
    datas = {}
    for ph in ("P", "S"):
        obs = base.observed.copy()
        if ph == "S" and drop_s_event is not None:
            obs = obs[obs["source"] != drop_s_event].reset_index(drop=True)
        inp = RunInputs(
            write_vgrids(grid, bg, with_s=False, uncertainty=0.5),
            base.interfaces,
            base.propgrid,
            base.sources,
            base.receivers,
            obs,
        )
        mp = st.default_mapping(obs, ph)
        mp["event_id"] = "ev" + mp["source_idx"].astype(str)
        inp, mp = st.to_nonlinear_geometry(inp, mp, ph)
        info = st.ModelInfo.from_model(inp.grids, sources=inp.sources, receivers=inp.receivers)
        datas[ph] = st.PhaseData(ph, inp, mp, info, True)
    return datas


def _stub_finish(monkeypatch, calls):
    def fake(run, res, pd_, damping, smoothing, thr, keep_rays, clean):
        calls.append((Path(run).name, pd_.phase, damping, smoothing, clean))
        (Path(run) / "marker").write_text("x")
        return st.PhaseOutcome(
            pd_.phase,
            Path(run),
            SimpleNamespace(run_dir=Path(run)),
            pd.DataFrame({"iter": [0]}),
            pd_.mapping,
            True,
            damping=damping,
            smoothing=smoothing,
        )

    monkeypatch.setattr(st, "_finish_phase", fake)


def _fake_inv(run_dir):
    return SimpleNamespace(run_dir=run_dir, iterations=pd.DataFrame({"iter": [0, 1]}))


@pytest.mark.parametrize("mode", ["joint", "p_fixed_for_s"])
def test_invert_coupled_wiring(tmp_path, monkeypatch, mode):
    grid, base = _setup(n_src=6)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    datas = _datas(grid, base, bg, drop_s_event=2)
    rec: dict = {}
    finish: list = []
    _stub_finish(monkeypatch, finish)

    def fake_joint(root, ctl, n_iter, n_jobs, keep, **kw):
        rec["joint"] = (Path(root), ctl, n_iter, n_jobs)
        rec["src_p"] = (Path(root) / "invert_p" / "src.dat").read_text()
        rec["src_s"] = (Path(root) / "invert_s" / "src.dat").read_text()
        return SimpleNamespace(
            p=_fake_inv(Path(root) / "invert_p"),
            s=_fake_inv(Path(root) / "invert_s"),
            flags=["note"],
        )

    def fake_ps(p_dir, s_dir, ctl, n_iter, n_jobs, keep, *, relocation, **kw):
        rec["ps"] = (Path(p_dir), Path(s_dir), relocation, n_iter)
        return SimpleNamespace(
            p=_fake_inv(Path(p_dir)), s=_fake_inv(Path(s_dir)), notes=["n"], success=True
        )

    monkeypatch.setattr(st, "run_joint_inversion", fake_joint)
    monkeypatch.setattr(st, "run_ps_inversions", fake_ps)
    fm = FmtomoConfig(phases=["P", "S"], joint={"relocation": mode}, iterations=3)
    reg = {
        "P": {"damping": 10.0, "smoothing": 20.0, "source": "x"},
        "S": {"damping": 30.0, "smoothing": 40.0, "source": "x"},
    }
    co = st.invert_coupled(tmp_path, datas, fm, reg, 2, mode=mode, vpvs=False)
    root = tmp_path / "run_joint"
    # coupling files and both phase directories were assembled
    for f in ("tomo3dj.in", "gridlocj.in", "jointimes.in"):
        assert (root / f).exists()
    if mode == "joint":
        assert rec["joint"][0] == root and rec["joint"][2] == 3 and rec["joint"][3] == 2
        ctl = rec["joint"][1]
        # per-phase regularisation reaches the controls of each run
        assert (
            ctl.runs["P"].invert3d.global_damping == 10.0
            and ctl.runs["S"].invert3d.global_damping == 30.0
        )
        assert (
            ctl.runs["P"].invert3d.global_smoothing == 20.0
            and ctl.runs["S"].invert3d.global_smoothing == 40.0
        )
        assert co.joint is not None
    else:
        assert rec["ps"] == (root / "invert_p", root / "invert_s", "p_fixed_for_s", 3)
    # src.dat ids: the same event has the same number in P and S although S lacks event 2
    ids_p = {int(r.split()[0]) for r in _records(tmp_path, "P", datas)}
    ids_s = {int(r.split()[0]) for r in _records(tmp_path, "S", datas)}
    assert ids_p == {1, 2, 3, 4, 5, 6} and ids_s == {1, 3, 4, 5, 6}
    # station-sources carry the Extras receiver id: same station, same id in P and S
    from osv.io.fmtomo import read_sources

    sp = read_sources(tmp_path / "run_P" / "sources.in")
    ss = read_sources(tmp_path / "run_S" / "sources.in")
    assert [x.extra_id for x in sp] == list(range(1, len(sp) + 1))
    assert [x.extra_id for x in ss] == list(range(1, len(ss) + 1))  # same stations
    # phase directories carry the names later stages use; the coupling root keeps the rest
    assert (tmp_path / "run_P" / "marker").exists() and (tmp_path / "run_S" / "marker").exists()
    assert (tmp_path / "run_P" / "mapping.csv").exists()
    assert not (root / "invert_p").exists()
    assert co.outcomes["P"].run_dir == tmp_path / "run_P"
    assert co.outcomes["S"].run_dir == tmp_path / "run_S"
    assert [c[1:4] for c in finish] == [("P", 10.0, 20.0), ("S", 30.0, 40.0)]
    assert co.vpvs is None and co.vpvs_model is None


def _records(tmp_path, ph, datas):
    """Event ids of the src.dat of phase ``ph`` (read from the moved run directory)."""
    toks = (tmp_path / f"run_{ph}" / "src.dat").read_text().split("\n")
    out = []
    for ln in toks[1:]:
        t = ln.split()
        if len(t) == 3:  # "id time uncertainty"
            out.append(ln.strip())
    return out


def test_invert_coupled_vpvs_wiring(tmp_path, monkeypatch):
    grid, base = _setup(n_src=4)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    datas = _datas(grid, base, bg)
    finish: list = []
    _stub_finish(monkeypatch, finish)
    rec: dict = {}

    def fake_joint(root, ctl, n_iter, n_jobs, keep, **kw):
        return SimpleNamespace(
            p=_fake_inv(Path(root) / "invert_p"), s=_fake_inv(Path(root) / "invert_s"), flags=[]
        )

    def fake_vpvs(root, ctl, n_iter, keep, *, forward, n_jobs, **kw):
        rec.update(root=Path(root), ctl=ctl, n_iter=n_iter, forward=forward, n_jobs=n_jobs)
        vd = Path(root) / "vpvs"
        vd.mkdir()
        from osv.io.fmtomo import write_vgrids as write_fm

        write_fm(vd / "vgrids.in", datas["P"].inputs.grids)
        (vd / "frechet.dat").write_text("big")
        return VpVsInversionResult(
            vpvs_dir=vd,
            iterations=pd.DataFrame({"iter": [0, 1], "rms_s": [0.1, 0.05]}),
            final_model=vd / "vgrids.in",
            ratio_of_vp_vs_model=None,
            residuals_path=vd / "vpvsresids.out",
        )

    monkeypatch.setattr(st, "run_joint_inversion", fake_joint)
    monkeypatch.setattr(st, "run_vpvs_inversion", fake_vpvs)
    fm = FmtomoConfig(phases=["vpvs"], iterations=2, save_rays=False)
    reg = {p: {"damping": 5.0, "smoothing": 6.0, "source": "x"} for p in ("P", "S")}
    co = st.invert_coupled(tmp_path, datas, fm, reg, 3, mode="joint", vpvs=True)
    assert rec["forward"] is True and rec["n_iter"] == 2 and rec["n_jobs"] == 3
    assert rec["ctl"].vpvs is not None and rec["ctl"].driver == "tomo3dps"
    # the S rays are needed by the Vp/Vs inversion even if the config does not keep them
    assert rec["ctl"].runs["S"] is not None
    assert co.vpvs is not None and co.vpvs_model is not None
    assert co.vpvs_model.dims == ("depth", "lat", "lon")
    assert not (tmp_path / "run_joint" / "vpvs" / "frechet.dat").exists()  # heavy files removed
    assert [c[1] for c in finish] == ["P", "S"]


def test_write_model_with_vpvs(tmp_path):
    grid, base = _setup(n_src=3)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)
    start = bg.copy()
    outcomes = {}
    for ph, v in (("P", "vp"), ("S", "vs")):
        run = tmp_path / f"run_{ph}"
        run.mkdir()
        from osv.io.fmtomo import write_vgrids as write_fm

        write_fm(run / "vgrids.in", write_vgrids(grid, bg.assign(vp=bg[v]), with_s=False))
        outcomes[ph] = st.PhaseOutcome(ph, run, None, pd.DataFrame(), pd.DataFrame(), True)
    vp = xr.DataArray(
        np.full(bg["vp"].transpose("depth", "lat", "lon").shape, 1.75),
        dims=("depth", "lat", "lon"),
        name="vpvs",
    )
    ds = st._write_model(tmp_path / "final_model.nc", start, outcomes, 10.0, vp)
    assert {"vp", "vs", "vp_start", "vs_start", "vpvs"} <= set(ds.data_vars)
    assert float(ds["vpvs"].mean()) == pytest.approx(1.75)
    # Stage 5 helpers: the model with vp, vs and the direct vpvs is read without error
    from osv.analysis.exports import model_variables

    mv = model_variables(ds)
    assert "vpvs" in mv and "dvs_pct" in mv
    bad = xr.DataArray(np.ones((2, 2, 2)), dims=("depth", "lat", "lon"))
    with pytest.raises(StageError, match="Vp/Vs grid"):
        st._write_model(tmp_path / "x.nc", start, outcomes, 10.0, bad)


# ---------------------------------------------------------------------------------------
# integration: joint P+S on a tiny synthetic geometry, real programs
# ---------------------------------------------------------------------------------------
def _have_joint_tools() -> bool:
    try:
        for t in ("gridloc", "gridlocj", "jointimes"):
            which_tool(t)
    except ToolNotFoundError:
        return False
    return _have_tools()


def _synthetic_ps(workdir):
    """P and S synthetic data (one anomaly, vs = vp / DEFAULT_VPVS) as nonlinear PhaseData."""
    grid, base = _setup(n_src=20)
    bg = checkerboard(grid, 5.0, 2.0, 0.0)  # homogeneous; vs = vp / DEFAULT_VPVS
    true = _gaussian_model(grid, bg, 8.0, 1.2)
    true["vs"] = true["vp"] / (bg["vp"] / bg["vs"])
    cfg = ResolutionConfig(noise={"scale": 0.1})
    datas = {}
    for ph, v in (("P", "vp"), ("S", "vs")):
        # the S model is a "Vp" model of the S velocities (one velocity type per run)
        tm = true.assign(vp=true[v]).drop_vars("vs")
        syn = make_synthetic_data(
            tm, grid, base, workdir / f"synth_{ph}", cfg, seed=5, n_jobs=4, phase=ph
        )
        start = bg.assign(vp=bg[v])
        inp = RunInputs(
            write_vgrids(grid, start, with_s=False, uncertainty=0.5),
            base.interfaces,
            base.propgrid,
            base.sources,
            base.receivers,
            read_observed(syn.otimes),
        )
        mp = st.default_mapping(inp.otimes, ph)
        mp["event_id"] = "ev" + mp["source_idx"].astype(str)
        inp, mp = st.to_nonlinear_geometry(inp, mp, ph)
        info = st.ModelInfo.from_model(inp.grids, sources=inp.sources, receivers=inp.receivers)
        datas[ph] = st.PhaseData(ph, inp, mp, info, True)
    return grid, base, bg, datas


@pytest.mark.integration
@pytest.mark.skipif(not _have_joint_tools(), reason="FMTOMO/Extras binaries not built")
def test_joint_ps_synthetic_one_iteration(workdir):
    """P and S synthetic data of one anomaly: coupled relocation and inversion, 1 iteration."""
    grid, base, bg, datas = _synthetic_ps(workdir)
    fm = FmtomoConfig(
        package="extras",
        relocate="nonlinear",
        phases=["P", "S"],
        joint={"relocation": "joint"},
        iterations=1,
        damping=1.0,
        smoothing=1.0,
        subspace_dimension=30,
        save_rays=False,
    )
    reg = {p: {"damping": 1.0, "smoothing": 1.0, "source": "config"} for p in ("P", "S")}
    co = st.invert_coupled(workdir / "out", datas, fm, reg, 4, mode="joint", dws_threshold=1.0)
    assert co.joint is not None and co.joint.success
    assert list(co.joint.iterations["iter"]) == [0, 1]
    for ph in ("P", "S"):
        oc = co.outcomes[ph]
        assert oc.run_dir == workdir / "out" / f"run_{ph}"
        it = oc.iterations
        assert it["rms_ms"].iloc[-1] < it["rms_ms"].iloc[0]
        assert oc.residuals is not None and len(oc.residuals) > 0
        assert oc.relocation is not None and len(oc.relocation) == len(base.sources)
        assert oc.dws is not None and bool(oc.mask.any())
        assert not (oc.run_dir / "frechet.dat").exists()
        # the final model is a valid FMTOMO grid of the right shape
        fin = read_vgrids(oc.run_dir / "vgrids.in").get(1, 1)["velocity"]
        assert fin.shape == tuple(datas[ph].inputs.grids.get(1, 1)["velocity"].shape)
    start = bg.copy()
    ds = st._write_model(workdir / "final_model.nc", start, co.outcomes, 1.0)
    assert {"vp", "vs", "dws", "dws_vs", "mask", "mask_vs"} <= set(ds.data_vars)


@pytest.mark.integration
@pytest.mark.skipif(not _have_joint_tools(), reason="FMTOMO/Extras binaries not built")
def test_p_fixed_for_s_and_vpvs_synthetic(workdir):
    """P relocates, S keeps the P hypocentres; then the direct Vp/Vs inversion (1 iteration)."""
    try:
        for t in ("mkvpvs", "frvpvs", "vpvsinv", "vpvsresids"):
            which_tool(t)
    except ToolNotFoundError:
        pytest.skip("Vp/Vs programs not built")
    grid, base, bg, datas = _synthetic_ps(workdir)
    fm = FmtomoConfig(
        package="extras",
        relocate="nonlinear",
        phases=["vpvs"],
        joint={"relocation": "p_fixed_for_s"},
        iterations=1,
        damping=1.0,
        smoothing=1.0,
        subspace_dimension=30,
        save_rays=False,
    )
    reg = {p: {"damping": 1.0, "smoothing": 1.0, "source": "config"} for p in ("P", "S")}
    co = st.invert_coupled(
        workdir / "out", datas, fm, reg, 4, mode="p_fixed_for_s", vpvs=True, dws_threshold=1.0
    )
    assert co.vpvs is not None and co.vpvs.success and co.vpvs_model is not None
    assert list(co.vpvs.iterations["iter"]) == [0, 1]
    assert np.isfinite(co.vpvs_model.to_numpy()).all()
    assert (co.outcomes["S"].iterations["mean_hypo_shift_km"] == 0).all()  # S hypocentres fixed
    ds = st._write_model(workdir / "final_model.nc", bg, co.outcomes, 1.0, co.vpvs_model)
    assert {"vp", "vs", "vpvs", "dws", "dws_vs"} <= set(ds.data_vars)


def test_read_otimes_wrapped_rows(tmp_path):
    """Rows written with 17 digits wrap over two lines; the reader is token based."""
    from osv.fmtomo.joint import _read_otimes_lines

    f = tmp_path / "otimesref.dat"
    f.write_text(
        "         2\n"
        "           1           1           1           0      1.5852503876567987\n"
        "                    0.01\n"
        "           2           1           1           0   1.14323282       0.01\n"
    )
    hdr, rows = _read_otimes_lines(f)
    assert rows == [
        ["1", "1", "1", "0", "1.5852503876567987", "0.01"],
        ["2", "1", "1", "0", "1.14323282", "0.01"],
    ]
    assert hdr.split() == ["2"]


@pytest.mark.integration
@pytest.mark.skipif(not _have_joint_tools(), reason="FMTOMO/Extras binaries not built")
def test_joint_ps_iterated_station_terms(workdir):
    """Separate P and S statics added to the data come back as P and S station terms."""
    grid, base, bg, datas = _synthetic_ps(workdir)
    rng = np.random.default_rng(2)
    n_st = len(datas["P"].inputs.sources)
    statics = {"P": rng.normal(0, 0.04, n_st), "S": rng.normal(0, 0.07, n_st)}
    for ph, sv in statics.items():
        sv -= sv.mean()
        o = datas[ph].inputs.otimes.copy()
        ok = o["time"] > 0
        o.loc[ok, "time"] += sv[o.loc[ok, "source"].to_numpy() - 1]  # stations = FMTOMO sources
        datas[ph] = dataclasses.replace(
            datas[ph], inputs=dataclasses.replace(datas[ph].inputs, otimes=o)
        )
    fm = FmtomoConfig(
        package="extras",
        relocate="nonlinear",
        phases=["P", "S"],
        joint={"relocation": "joint"},
        iterations=4,
        damping=50.0,
        smoothing=50.0,
        subspace_dimension=20,
        save_rays=False,
        station_terms="iterate",
    )
    reg = {p: {"damping": 50.0, "smoothing": 50.0, "source": "config"} for p in ("P", "S")}
    co = st.invert_coupled(workdir / "out", datas, fm, reg, 4, mode="joint", dws_threshold=1.0)
    assert co.joint is not None and co.joint.success
    for ph in ("P", "S"):
        t = co.outcomes[ph].result.station_terms
        assert t is not None and len(t) == n_st
        idx = t["station"].astype(str).str.extract(r"(\d+)$")[0].astype(int).to_numpy() - 1
        # S converges more slowly (its terms trade off with Vs and the shared relocation):
        # about 0.89 after 3 updates, 0.92 after 7
        assert np.corrcoef(t["term_s"], statics[ph][idx])[0, 1] > (0.9 if ph == "P" else 0.85), ph
        h = co.outcomes[ph].result.station_term_history
        steps = h.groupby("iter")["delta_s"].apply(lambda d: d.abs().max()).to_numpy()
        assert steps[-1] < steps[0] / 3, ph  # the updates shrink
    # P+S not allowed in the p_fixed_for_s mode
    with pytest.raises(st.StageError, match="iterate"):
        st.invert_coupled(workdir / "o2", datas, fm, reg, 4, mode="p_fixed_for_s")
