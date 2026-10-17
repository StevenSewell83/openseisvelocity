"""Compare the final model with the user geology / well model (task 6.5).

:func:`compare_with_user_model` evaluates three things:

* **Difference volumes** ``final - user`` (absolute and percent of the user value) for Vp (and Vs
  when both models have it) on the interior nodes of the final model.  A user model on another
  grid is resampled linearly onto the final grid.
* **Per-unit statistics**: mean / median / std of the final Vp in each geological unit
  (``unit_id``) next to the user value assigned to the unit (the velocity law evaluated at the
  nodes), with the bias and RMS difference between them.
* **Well profiles**: the final model (and the starting model, when given) sampled along each
  well trajectory at the model-node depths of the upscaled log and compared with it (bias, RMS,
  mean absolute error, correlation), for all depths and for the DWS-covered part only; plus a
  denser curve along the well for plotting.

:func:`make_comparison_figures` hands the well-profile panels, per-unit box plots and the
difference slices / sections (the :mod:`osv.analysis.plots_model` helpers with the user model in
the role of the reference) to a ``save(fig, name, caption, section)`` callable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from osv.analysis.plots_model import (
    ModelFigureOptions,
    _profile,
    _slug,
    model_limits,
    plot_section_comparison,
    plot_slice_matrix,
)
from osv.plotting.slices import (
    KM_PER_DEG,
    ModelFields,
    ModelMasks,
    crop_interior,
    default_profiles,
)
from osv.plotting.style import apply_style

log = logging.getLogger(__name__)

__all__ = [
    "ComparisonResult",
    "compare_with_user_model",
    "make_comparison_figures",
    "plot_difference_sections",
    "plot_difference_slices",
    "plot_unit_boxplots",
    "plot_well_profiles",
]

Saver = Callable[..., Any]
#: Largest number of samples kept per unit for the box plots.
MAX_UNIT_SAMPLES = 20000
#: Dense well curves have this many points per model node spacing.
CURVE_OVERSAMPLE = 4
SECTION = "compare_user"
DIMS = ("depth", "lat", "lon")


@dataclass
class ComparisonResult:
    """Output of :func:`compare_with_user_model`.

    Attributes
    ----------
    difference
        Interior-node dataset (dims ``depth, lat, lon``): ``user_vp`` (user model on the final
        grid), ``dvp`` (km/s), ``dvp_pct`` (% of user), the same for Vs when available, and
        ``unit_id``.
    unit_stats
        One row per unit: ``unit, name, n, final_mean, final_median, final_std, user_mean,
        user_median, user_std, bias, bias_pct, rmse, covered_n, final_mean_covered``.
    well_profiles
        Upscaled-log nodes: ``well, depth_km, lat, lon, log_vp`` (upscaled), ``start_vp``,
        ``final_vp``, ``covered``.
    well_curves
        Dense curves along each well (``well, depth_km, lat, lon, final_vp, start_vp, covered``).
    well_misfit
        Misfit per well and ``model`` (``final`` / ``starting``) and ``subset`` (``all`` /
        ``covered``): ``n, bias, rmse, mae, corr, rmse_pct`` (model minus upscaled log).
    logs
        Conditioned log samples (``well, depth_km, vp, ...``) for plotting.
    summary
        Scalar statistics (volume-wide RMS / mean difference, worst unit, ...).
    """

    difference: xr.Dataset
    unit_stats: pd.DataFrame
    well_profiles: pd.DataFrame
    well_curves: pd.DataFrame
    well_misfit: pd.DataFrame
    logs: pd.DataFrame
    summary: dict[str, Any]
    unit_names: list[str] = field(default_factory=list)
    fields: ModelFields | None = field(default=None, repr=False)
    masks: ModelMasks | None = field(default=None, repr=False)
    samples: dict[int, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def _as_dataset(obj: Any) -> xr.Dataset | None:
    if obj is None or isinstance(obj, xr.Dataset):
        return obj
    ds = getattr(obj, "dataset", None)
    if isinstance(ds, xr.Dataset):
        return ds
    raise TypeError(f"expected an xarray.Dataset (or an object with .dataset), got {type(obj)}")


def _on_grid(ds: xr.Dataset, like: xr.Dataset, name: str, method: str = "linear") -> np.ndarray:
    """Variable ``name`` of ``ds`` on the coordinates of ``like`` (dims depth, lat, lon)."""
    da = ds[name].transpose("depth", "lat", "lon")
    same = all(
        da[d].size == like[d].size and np.allclose(da[d].values, like[d].values)
        for d in ("depth", "lat", "lon")
    )
    if same:
        return np.asarray(da.values)
    out = da.interp(
        depth=like["depth"],
        lat=like["lat"],
        lon=like["lon"],
        method=method,  # type: ignore[arg-type]
    )
    return np.asarray(out.values)


def _unit_array(
    unit_id: Any, user: xr.Dataset, like: xr.Dataset, final: xr.Dataset
) -> np.ndarray | None:
    if unit_id is None:
        if "unit_id" not in user:
            return None
        return _on_grid(user, like, "unit_id", "nearest")
    if isinstance(unit_id, xr.DataArray):
        return _on_grid(xr.Dataset({"unit_id": unit_id}), like, "unit_id", "nearest")
    arr = np.asarray(unit_id)
    if arr.shape == tuple(like.sizes[d] for d in ("depth", "lat", "lon")):
        return arr
    if arr.shape == tuple(final.sizes[d] for d in ("depth", "lat", "lon")):
        return np.asarray(crop_interior(final.assign(unit_id=(DIMS, arr)))["unit_id"].values)
    full = tuple(user.sizes[d] for d in ("depth", "lat", "lon"))
    if arr.shape == full:
        return _on_grid(
            user.assign(unit_id=(("depth", "lat", "lon"), arr)), like, "unit_id", "nearest"
        )
    raise ValueError(f"unit_id: shape {arr.shape} matches neither the final nor the user grid")


def _sample(fields: ModelFields, key: str, d: np.ndarray, la: np.ndarray, lo: np.ndarray):
    """``fields.evaluate`` that returns NaN outside the interior instead of clipping."""
    v = np.array(fields.evaluate(key, d, la, lo), dtype=float)
    inside = (
        (d >= fields.depth[0])
        & (d <= fields.depth[-1])
        & (la >= fields.lat[0])
        & (la <= fields.lat[-1])
        & (lo >= fields.lon[0])
        & (lo <= fields.lon[-1])
    )
    v[~inside] = np.nan
    return v


def _stats(x: np.ndarray) -> tuple[float, float, float]:
    if x.size == 0:
        return math.nan, math.nan, math.nan
    return float(np.mean(x)), float(np.median(x)), float(np.std(x))


def _misfit(model: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(model) & np.isfinite(ref)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "bias": math.nan, "rmse": math.nan, "mae": math.nan,
                "corr": math.nan, "rmse_pct": math.nan}  # fmt: skip
    m, r = model[ok], ref[ok]
    e = m - r
    corr = float(np.corrcoef(m, r)[0, 1]) if n > 2 and m.std() > 0 and r.std() > 0 else math.nan
    return {
        "n": n,
        "bias": float(e.mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.abs(e).mean()),
        "corr": corr,
        "rmse_pct": float(100.0 * np.sqrt(np.mean(e**2)) / np.mean(r)),
    }


def _trajectory(w: Any, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latitude / longitude of a :class:`WellProfile` at the depths (wellhead + deviation)."""
    order = np.argsort(w.depth_km)
    z = w.depth_km[order]
    north = np.interp(depth, z, w.north_km[order])
    east = np.interp(depth, z, w.east_km[order])
    lat = w.lat + north / KM_PER_DEG
    lon = w.lon + east / (KM_PER_DEG * math.cos(math.radians(w.lat)))
    return lat, lon


# ----------------------------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------------------------


def compare_with_user_model(
    final: xr.Dataset,
    user_model_ds: xr.Dataset,
    wells: Any = None,
    unit_id: Any = None,
    *,
    starting: Any = None,
    coverage: Any = None,
    dws_threshold: float = 10.0,
    reference_1d: Any = None,
    unit_names: Sequence[str] | None = None,
) -> ComparisonResult:
    """Compare the final model with the user geology / wells model.

    Parameters
    ----------
    final
        Final model dataset (``vp`` and optionally ``vs``; ``attrs['n_cushion']`` honoured).
    user_model_ds
        The user model (geology and/or wells dataset) with ``vp`` (km/s), optionally ``vs`` and
        ``unit_id``.  Other
        grids are resampled linearly onto the interior of ``final``.
    wells
        :class:`~osv.model3d.wells.WellsResult` (needs ``profiles``, ``logs`` and ``wells``); omit
        to skip the well profiles.
    unit_id
        Unit index per node (``-1`` background; ``DataArray`` or array on the final or user
        grid); default ``user_model_ds['unit_id']``.  Names come from ``unit_names`` or
        ``attrs['unit_names']``.
    starting
        Starting model (dataset or ``StartingModel``) to include in the well comparison.
    coverage, dws_threshold
        DWS (see :meth:`ModelMasks.from_coverage`) and threshold; enables the covered-only
        statistics and the coverage shading in the well panels.
    reference_1d
        1-D reference passed to :class:`~osv.plotting.slices.ModelFields`.
    """
    user = _as_dataset(user_model_ds)
    start = _as_dataset(starting)
    assert user is not None
    fin = crop_interior(final)
    masks = ModelMasks.from_coverage(coverage, dws_threshold) if coverage is not None else None

    # --- difference volumes ---
    coords = {d: fin[d] for d in ("depth", "lat", "lon")}
    dims = ("depth", "lat", "lon")
    data: dict[str, Any] = {}
    uvp = _on_grid(user, fin, "vp")
    fvp = fin["vp"].transpose(*dims).values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        data["user_vp"] = (dims, uvp)
        data["dvp"] = (dims, fvp - uvp)
        data["dvp_pct"] = (dims, 100.0 * (fvp - uvp) / uvp)
        if "vs" in fin and "vs" in user:
            uvs = _on_grid(user, fin, "vs")
            fvs = fin["vs"].transpose(*dims).values.astype(float)
            data["user_vs"] = (dims, uvs)
            data["dvs"] = (dims, fvs - uvs)
            data["dvs_pct"] = (dims, 100.0 * (fvs - uvs) / uvs)
    units = _unit_array(unit_id, user, fin, final)
    if units is not None:
        data["unit_id"] = (dims, units.astype(np.int32))
    diff = xr.Dataset(data, coords=coords)
    diff["dvp"].attrs["units"] = "km/s"

    # --- per-unit statistics ---
    names = list(unit_names) if unit_names is not None else list(user.attrs.get("unit_names", []))
    covered_vp = None
    if masks is not None:
        dd, la, lo = np.meshgrid(fin["depth"], fin["lat"], fin["lon"], indexing="ij")
        covered_vp = masks.covered_at("p", dd, la, lo)
    rows: list[dict[str, Any]] = []
    samples: dict[int, dict[str, np.ndarray]] = {}
    if units is not None:
        rng = np.random.default_rng(0)
        for u in sorted(int(x) for x in np.unique(units)):
            sel = (units == u) & np.isfinite(fvp) & np.isfinite(uvp)
            if not sel.any():
                continue
            f, v = fvp[sel], uvp[sel]
            fm, fmed, fsd = _stats(f)
            um, umed, usd = _stats(v)
            cov = sel & covered_vp if covered_vp is not None else None
            name = names[u] if 0 <= u < len(names) else ("background" if u < 0 else f"unit {u}")
            rows.append(
                {
                    "unit": u,
                    "name": name,
                    "n": int(sel.sum()),
                    "final_mean": fm,
                    "final_median": fmed,
                    "final_std": fsd,
                    "user_mean": um,
                    "user_median": umed,
                    "user_std": usd,
                    "bias": float(np.mean(f - v)),
                    "bias_pct": float(100.0 * np.mean(f - v) / um),
                    "rmse": float(np.sqrt(np.mean((f - v) ** 2))),
                    "covered_n": int(cov.sum()) if cov is not None else int(sel.sum()),
                    "final_mean_covered": float(fvp[cov].mean())
                    if cov is not None and cov.any()
                    else (fm if cov is None else math.nan),
                }
            )
            k = min(MAX_UNIT_SAMPLES, f.size)
            pick = rng.choice(f.size, k, replace=False) if k < f.size else slice(None)
            samples[u] = {"final": f[pick], "user": v[pick], "pct": (100 * (f - v) / v)[pick]}
    unit_stats = pd.DataFrame(rows)

    # --- wells ---
    fields = ModelFields(final, reference_1d, start)
    cols_prof = ["well", "depth_km", "lat", "lon", "log_vp", "start_vp", "final_vp", "covered"]
    prof_rows: list[pd.DataFrame] = []
    curve_rows: list[pd.DataFrame] = []
    mis: list[dict[str, Any]] = []
    logs = pd.DataFrame()
    if wells is not None:
        logs = wells.logs
        step = float(np.median(np.diff(fields.depth))) if fields.depth.size > 1 else 0.1

        def sample_set(d: np.ndarray, la: np.ndarray, lo: np.ndarray) -> dict[str, np.ndarray]:
            out = {"final_vp": _sample(fields, "vp", d, la, lo)}
            out["start_vp"] = (
                _sample(fields, "start_vp", d, la, lo)
                if start is not None
                else np.full(d.shape, np.nan)
            )
            cov = masks.covered_at("p", d, la, lo) if masks is not None else None
            out["covered"] = np.ones(d.shape, bool) if cov is None else np.asarray(cov, bool)
            return out

        for _, g in wells.profiles.groupby("well", sort=False):
            d, la, lo = (g[c].to_numpy(float) for c in ("depth_km", "lat", "lon"))
            s = sample_set(d, la, lo)
            prof_rows.append(
                pd.DataFrame(
                    {
                        "well": g["well"].to_numpy(),
                        "depth_km": d,
                        "lat": la,
                        "lon": lo,
                        "log_vp": g["vp"].to_numpy(float),
                        **s,
                    }
                )  # fmt: skip
            )
        for w in wells.wells:
            z = np.arange(w.depth_km.min(), w.depth_km.max(), step / CURVE_OVERSAMPLE)
            if z.size < 2:
                continue
            la, lo = _trajectory(w, z)
            curve_rows.append(pd.DataFrame({"well": w.name, "depth_km": z, "lat": la, "lon": lo,
                                            **sample_set(z, la, lo)}))  # fmt: skip
    wp = pd.concat(prof_rows, ignore_index=True) if prof_rows else pd.DataFrame(columns=cols_prof)
    wc = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    for well, g in wp.groupby("well", sort=False):
        for model, col in (("final", "final_vp"), ("starting", "start_vp")):
            if model == "starting" and start is None:
                continue
            for subset in ("all", "covered"):
                gg = g if subset == "all" else g[g["covered"].astype(bool)]
                mis.append(
                    {"well": well, "model": model, "subset": subset,
                     **_misfit(gg[col].to_numpy(float), gg["log_vp"].to_numpy(float))}
                )  # fmt: skip
    well_misfit = pd.DataFrame(mis)

    ok = np.isfinite(fvp) & np.isfinite(uvp)
    summary: dict[str, Any] = {
        "n_nodes": int(ok.sum()),
        "mean_dvp_pct": float(np.nanmean(diff["dvp_pct"].values[ok])) if ok.any() else math.nan,
        "rms_dvp_pct": float(np.sqrt(np.nanmean(diff["dvp_pct"].values[ok] ** 2)))
        if ok.any()
        else math.nan,
        "rmse_dvp": float(np.sqrt(np.nanmean(diff["dvp"].values[ok] ** 2)))
        if ok.any()
        else math.nan,
        "n_wells": int(wp["well"].nunique()) if len(wp) else 0,
    }
    if len(unit_stats):
        worst = unit_stats.loc[unit_stats["bias_pct"].abs().idxmax()]
        summary["worst_unit"] = str(worst["name"])
        summary["worst_unit_bias_pct"] = float(worst["bias_pct"])
    if len(well_misfit):
        f_all = well_misfit[(well_misfit.model == "final") & (well_misfit.subset == "all")]
        summary["well_rmse_final_mean"] = float(f_all["rmse"].mean())
        if start is not None:
            s_all = well_misfit[(well_misfit.model == "starting") & (well_misfit.subset == "all")]
            summary["well_rmse_start_mean"] = float(s_all["rmse"].mean())
    log.info("user-model comparison: %s", summary)

    # fields for the difference figures: user model plays the role of the reference
    user_view = xr.Dataset(
        {
            "vp": (dims, _on_grid(user, final, "vp")),
            **({"vs": (dims, _on_grid(user, final, "vs"))} if "vs" in user else {}),
        },
        coords={d: final[d] for d in dims},
        attrs=final.attrs,
    )
    return ComparisonResult(
        difference=diff,
        unit_stats=unit_stats,
        well_profiles=wp,
        well_curves=wc,
        well_misfit=well_misfit,
        logs=logs,
        summary=summary,
        unit_names=names,
        fields=ModelFields(final, reference_1d, user_view),
        masks=masks,
        samples=samples,
    )


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------


def _uncovered_runs(z: np.ndarray, covered: np.ndarray) -> list[tuple[float, float]]:
    runs: list[tuple[float, float]] = []
    start = None
    for i, c in enumerate(covered):
        if not c and start is None:
            start = z[i]
        if c and start is not None:
            runs.append((float(start), float(z[i])))
            start = None
    if start is not None:
        runs.append((float(start), float(z[-1])))
    return runs


def plot_well_profiles(result: ComparisonResult, max_columns: int = 4) -> Figure:
    """One panel per well: log, upscaled log, starting and final Vp against depth.

    Depth ranges not covered by the DWS mask (when one was given) are shaded grey.
    """
    wp, wc = result.well_profiles, result.well_curves
    if wp.empty:
        raise ValueError("no well profiles in the comparison (give wells=...)")
    apply_style()
    names = list(dict.fromkeys(wp["well"]))
    nc = min(max_columns, len(names))
    nr = math.ceil(len(names) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(2.9 * nc + 0.6, 4.2 * nr), squeeze=False,
                            sharey=True, constrained_layout=True)  # fmt: skip
    mis = result.well_misfit
    for ax in axs.ravel():
        ax.set_visible(False)
    for ax, name in zip(axs.ravel(), names, strict=False):
        ax.set_visible(True)
        g = wp[wp["well"] == name]
        lg = result.logs[result.logs["well"] == name] if len(result.logs) else None
        c = wc[wc["well"] == name] if len(wc) else None
        if c is not None and len(c):
            for a, b in _uncovered_runs(c["depth_km"].to_numpy(), c["covered"].to_numpy(bool)):
                ax.axhspan(a, b, color="0.85", zorder=0, lw=0)
        if lg is not None and len(lg):
            ax.plot(lg["vp"], lg["depth_km"], color="0.6", lw=0.7, label="log", zorder=1)
        ax.step(
            g["log_vp"], g["depth_km"], where="mid", color="k", lw=1.2, label="upscaled", zorder=3
        )
        if c is not None and len(c):
            if c["start_vp"].notna().any():
                ax.plot(c["start_vp"], c["depth_km"], color="tab:blue", ls="--", lw=1.1,
                        label="starting", zorder=4)  # fmt: skip
            ax.plot(c["final_vp"], c["depth_km"], color="tab:red", lw=1.4, label="final", zorder=5)
        else:
            ax.plot(g["final_vp"], g["depth_km"], color="tab:red", lw=1.4, label="final", zorder=5)
        txt = ""
        if len(mis):
            m = mis[(mis.well == name) & (mis.subset == "all")].set_index("model")
            if "final" in m.index:
                txt = f"final RMSE {m.loc['final', 'rmse']:.2f} km/s"
                if "starting" in m.index:
                    txt += f"\nstart RMSE {m.loc['starting', 'rmse']:.2f} km/s"
        ax.set_title(f"{name}\n{txt}", fontsize=8)
        ax.set_xlabel("Vp (km/s)")
        ax.invert_yaxis() if not ax.yaxis_inverted() else None
    for ax in axs[:, 0]:
        ax.set_ylabel("Depth (km)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=len(lab), fontsize=7)
    shade = " Grey shading: not covered by the DWS mask." if result.masks is not None else ""
    fig.suptitle("Well profiles: log, upscaled log, starting and final model" + shade, fontsize=9)
    return fig


def plot_unit_boxplots(result: ComparisonResult) -> Figure:
    """Box plots per geological unit: final Vp (with the user value) and the difference in %."""
    if not result.samples:
        raise ValueError("no per-unit samples (the user model has no unit_id)")
    apply_style()
    st = result.unit_stats
    us = list(st["unit"])
    labels = [f"{n}\n(n={k})" for n, k in zip(st["name"], st["n"], strict=True)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(6.0, 1.3 * len(us) + 4), 4.2),
                                 constrained_layout=True)  # fmt: skip
    pos = np.arange(len(us)) + 1
    bp = a1.boxplot([result.samples[u]["final"] for u in us], positions=pos, widths=0.55,
                    patch_artist=True, showfliers=False)  # fmt: skip
    for b in bp["boxes"]:
        b.set(facecolor="tab:red", alpha=0.45)
    ub = a1.boxplot([result.samples[u]["user"] for u in us], positions=pos, widths=0.25,
                    patch_artist=True, showfliers=False)  # fmt: skip
    for b in ub["boxes"]:
        b.set(facecolor="0.3", alpha=0.8)
    a1.plot(pos, st["user_mean"], "k_", ms=14, mew=1.5)
    a1.set_xticks(pos, labels, fontsize=7)
    a1.set_ylabel("Vp (km/s)")
    a1.set_title("Final (red, wide) vs user-assigned (dark, narrow)", fontsize=8)
    a2.boxplot([result.samples[u]["pct"] for u in us], positions=pos, widths=0.5,
               patch_artist=True, showfliers=False,
               boxprops={"facecolor": "tab:blue", "alpha": 0.5})  # fmt: skip
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(pos, labels, fontsize=7)
    a2.set_ylabel("final - user (% of user Vp)")
    a2.set_title("Difference per unit", fontsize=8)
    fig.suptitle("Final Vp per geological unit", fontsize=9)
    return fig


_RELABEL = (
    ("Starting Vp", "User-model Vp"),
    ("Starting Vs", "User-model Vs"),
    ("dVp vs starting model", "dVp vs user model"),
    ("dVs vs starting model", "dVs vs user model"),
    ("dVp vs starting (%)", "dVp vs user model (%)"),
    ("dVs vs starting (%)", "dVs vs user model (%)"),
    ("Starting vs final", "User model vs final"),
    ("starting", "user model"),
)


def _relabel(fig: Figure) -> Figure:
    """Rename 'starting model' texts to 'user model' (the user model fills that role)."""

    def sub(s: str) -> str:
        for a, b in _RELABEL:
            s = s.replace(a, b)
        return s

    texts = [t for t in fig.texts]
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
    for t in texts:
        if t.get_text():
            t.set_text(sub(t.get_text()))
    if fig._suptitle is not None:
        fig._suptitle.set_text(sub(fig._suptitle.get_text()))
    return fig


def _need_fields(result: ComparisonResult) -> ModelFields:
    if result.fields is None:
        raise ValueError("comparison result carries no model fields")
    return result.fields


def plot_difference_slices(
    result: ComparisonResult,
    depths: Sequence[float],
    quantity: str = "vp",
    opts: ModelFigureOptions | None = None,
) -> Figure:
    """Rows of depth slices: user model, final model and their percentage difference."""
    fields = _need_fields(result)
    k = quantity[-2:]
    keys = [f"start_{k}", k, f"d{k}_start"]
    opts = opts or ModelFigureOptions()
    lm = model_limits(fields, keys, result.masks, opts)
    lo, hi = min(lm[keys[0]][0], lm[keys[1]][0]), max(lm[keys[0]][1], lm[keys[1]][1])
    lm[keys[0]] = lm[keys[1]] = (lo, hi)
    fig = plot_slice_matrix(fields, keys, depths, result.masks, limits=lm, opts=opts,
                            title=f"User model vs final {k.upper()}")  # fmt: skip
    return _relabel(fig)


def plot_difference_sections(
    result: ComparisonResult,
    profile: Any,
    quantity: str = "vp",
    opts: ModelFigureOptions | None = None,
) -> Figure:
    """Section along ``profile``: user model, final model and percentage difference."""
    fig = plot_section_comparison(_need_fields(result), profile, result.masks,
                                  quantity=quantity, opts=opts)  # fmt: skip
    return _relabel(fig)


def make_comparison_figures(
    save: Saver,
    result: ComparisonResult,
    cfg: Any = None,
    *,
    depths: Sequence[float] | None = None,
    profiles: Sequence[Any] | None = None,
    options: ModelFigureOptions | None = None,
) -> list[str]:
    """Build the user-model comparison figures; each goes to ``save(fig, name, caption, section)``.

    ``cfg`` (optional) supplies ``cfg.plotting.depth_slices_km`` and ``cfg.plotting.profiles``;
    otherwise four depths spread over the model and the default E-W / N-S profiles are used.
    Returns the names of the figures saved.
    """
    fields = _need_fields(result)
    opts = options or ModelFigureOptions()
    if depths is None:
        depths = list(getattr(getattr(cfg, "plotting", None), "depth_slices_km", []) or [])
    if not depths:
        depths = list(np.linspace(fields.depth[0], fields.depth[-1], 6)[1:-1])
    if profiles is None:
        profiles = list(getattr(getattr(cfg, "plotting", None), "profiles", []) or [])
    profs = [_profile(p) for p in profiles] or default_profiles(fields.lat, fields.lon)
    saved: list[str] = []

    def emit(fig: Figure, name: str, caption: str) -> None:
        save(fig, name, caption, SECTION)
        saved.append(name)

    if len(result.well_profiles):
        emit(
            plot_well_profiles(result),
            "user_well_profiles",
            "Final model sampled along each well compared with the log, the upscaled log and the "
            "starting model; grey = outside DWS coverage.",
        )
    if result.samples:
        emit(
            plot_unit_boxplots(result),
            "user_unit_boxplots",
            "Final Vp per geological unit against the user-assigned values, and the difference "
            "in percent.",
        )
    for q in ("vp", "vs"):
        if q == "vs" and "vs" not in fields.model:
            continue
        if q == "vs" and "vs" not in (fields.starting if fields.starting is not None else {}):
            continue
        emit(
            plot_difference_slices(result, depths, q, opts),
            f"user_diff_slices_{q}",
            f"{q.upper()} of the user model, the final model and their difference (% of user).",
        )
    for name, st, en in profs:
        emit(
            plot_difference_sections(result, (name, st, en), "vp", opts),
            f"user_diff_section_{_slug(name)}",
            f"Section {name}: user-model Vp, final Vp and difference.",
        )
    return saved
