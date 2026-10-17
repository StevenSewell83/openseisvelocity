"""Pydantic v2 schema for ``config.yaml``.

Every user-changeable setting lives here, with its default and a description. The
descriptions are used to generate the commented ``osv init`` template and the config
reference documentation. Unknown keys are rejected so typos give errors.

Field metadata (``json_schema_extra``):

``is_path``
    The value is a filesystem path (or list of paths); the loader resolves it relative to
    the config file's directory.
``glob``
    A list of paths whose entries may contain glob patterns that the loader expands.
``example``
    Extra text shown as commented example lines in the template.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


def F(  # noqa: N802
    default: Any,
    description: str,
    *,
    path: bool = False,
    glob: bool = False,
    example: str | None = None,
    **constraints: Any,
) -> Any:
    """Build a ``Field`` with a description and OSV metadata."""
    extra: dict[str, Any] = {}
    if path or glob:
        extra["is_path"] = True
    if glob:
        extra["glob"] = True
    if example:
        extra["example"] = example
    return Field(
        default=default,
        description=description,
        json_schema_extra=extra or None,
        **constraints,
    )


def FF(factory: Any, description: str) -> Any:  # noqa: N802
    """A ``Field`` for a nested section, built from its default factory."""
    return Field(default_factory=factory, description=description)


# ---------------------------------------------------------------------------
# Reusable validated types
# ---------------------------------------------------------------------------


def _check_lat(v: float) -> float:
    if not -90.0 <= v <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90], got {v}")
    return v


def _check_lon(v: float) -> float:
    if not -180.0 <= v <= 180.0:
        raise ValueError(f"longitude must be in [-180, 180], got {v}")
    return v


def _check_latlon(v: tuple[float, float]) -> tuple[float, float]:
    _check_lat(v[0])
    _check_lon(v[1])
    return v


def _check_lower_upper(v: tuple[float, float]) -> tuple[float, float]:
    if not v[0] < v[1]:
        raise ValueError(f"lower bound must be < upper (got {v[0]} and {v[1]})")
    return v


def _check_positive_list(v: list[float]) -> list[float]:
    if not v:
        raise ValueError("must not be empty")
    if any(x <= 0 for x in v):
        raise ValueError("all values must be > 0")
    return v


LatLon = Annotated[tuple[float, float], AfterValidator(_check_latlon)]
Bounds = Annotated[tuple[float, float], AfterValidator(_check_lower_upper)]
PositiveList = Annotated[list[float], AfterValidator(_check_positive_list)]

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


# ---------------------------------------------------------------------------
# project / inputs / coordinates
# ---------------------------------------------------------------------------


class ProjectConfig(_Base):
    """Project identity and run-wide settings."""

    name: str = F("my-project", "Short project name (letters, digits, '.', '_', '-').")
    workdir: Path = F(
        Path("runs"),
        "Directory under which numbered stage directories are created.",
        path=True,
    )
    n_jobs: int = F(-1, "Parallel worker processes; -1 uses all CPU cores.")
    random_seed: int = F(42, "Seed for all random number generators (reproducibility).")

    @model_validator(mode="after")
    def _check(self) -> ProjectConfig:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                "name: must start with a letter/digit and contain only letters, digits, "
                "'.', '_', '-'"
            )
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs: must be -1 (all cores) or >= 1")
        return self


class InputsConfig(_Base):
    """Input data files."""

    stationxml: list[Path] = F(
        [],
        "StationXML files; entries may be glob patterns (expanded by the loader).",
        glob=True,
    )
    quakeml: list[Path] = F([], "QuakeML catalogue files; entries may be glob patterns.", glob=True)
    origin_selection: str = F(
        "preferred",
        "Which origin to use per event: 'preferred', 'latest' or 'by_method:<method id>'.",
        pattern=r"^(preferred|latest|by_method:.+)$",
    )
    evaluation_mode: Literal["any", "manual", "automatic"] = F(
        "any", "Keep picks with this evaluation mode ('any' keeps all, including unset)."
    )
    use_arrivals: bool = F(
        False,
        "Build travel times from arrivals instead of picks. Keep false when the catalogue's "
        "arrivals may be corrupted or duplicated.",
    )
    weight_source: Literal["uncertainty", "comment", "both"] = F(
        "uncertainty",
        "Where pick weights come from: the pick time uncertainty, a 'wgt = x' comment, or both.",
    )
    station_aliases: Literal["auto"] | dict[str, str] | Path = F(
        "auto",
        "VELEST needs station codes of at most 4 characters: 'auto' generates aliases, or give "
        "an explicit {code: alias} mapping, or a path to a CSV with columns code,alias "
        "(or net,sta,alias).",
        path=True,
    )
    dem: Path | None = F(None, "Optional GeoTIFF DEM used for the surface interface.", path=True)


class OriginConfig(_Base):
    """Geographic origin of the local coordinate system."""

    lat: float = F(..., "Origin latitude, degrees north (negative = south).")
    lon: float = F(..., "Origin longitude, degrees east (negative = west).")

    @field_validator("lat")
    @classmethod
    def _lat(cls, v: float) -> float:
        return _check_lat(v)

    @field_validator("lon")
    @classmethod
    def _lon(cls, v: float) -> float:
        return _check_lon(v)


class CoordinatesConfig(_Base):
    """Coordinate system."""

    origin: OriginConfig | None = F(
        None, "Local-system origin; null uses the centroid of the stations."
    )
    projection: str = F(
        "local_tm",
        "'local_tm' (transverse Mercator about the origin) or an EPSG code such as 'epsg:2193'.",
        pattern=r"^(local_tm|epsg:\d+)$",
    )
    borehole_sensors: Literal["use_depth_from_stationxml", "ignore_depth"] = F(
        "use_depth_from_stationxml",
        "How to treat borehole sensors: use the sensor depth from StationXML, or ignore it.",
    )
    depth_datum: Literal["sea_level", "surface", "reference"] = F(
        "sea_level",
        "Datum of the catalogue (QuakeML/NLLoc) hypocentre depths. 'sea_level': depth is "
        "below mean sea level (NLLoc default when the grid origin is at sea level, even if "
        "the grid top is above it); 'surface': depth is below the local ground surface at "
        "the epicentre (needs surface elevations from stations or a DEM); 'reference': "
        "depth is below the fixed elevation reference_elevation_m.",
    )
    reference_elevation_m: float | None = F(
        None,
        "Elevation (m above sea level, positive up) of the depth datum; required when "
        "depth_datum is 'reference' (e.g. 600 for an NLLoc grid top at +0.6 km).",
    )

    @model_validator(mode="after")
    def _check_datum(self) -> CoordinatesConfig:
        if self.depth_datum == "reference" and self.reference_elevation_m is None:
            raise ValueError(
                "reference_elevation_m: required when depth_datum is 'reference' "
                "(elevation in metres above sea level of the depth datum)"
            )
        if self.depth_datum != "reference" and self.reference_elevation_m is not None:
            raise ValueError(
                "reference_elevation_m: only valid with depth_datum 'reference'; "
                "remove it or set depth_datum to 'reference'"
            )
        return self


# ---------------------------------------------------------------------------
# data selection
# ---------------------------------------------------------------------------


class MagnitudeSelection(_Base):
    """Magnitude filter."""

    min: float | None = F(None, "Minimum magnitude; null disables the filter.")
    ignore_if_invalid: bool = F(
        True, "Skip magnitude filtering/plots when the catalogue magnitudes are all invalid."
    )


class DataSelectionConfig(_Base):
    """Event and pick selection (Stage 1)."""

    region_polygon: list[LatLon] | None = F(
        None,
        "Polygon of [lat, lon] vertices; events outside are dropped. null keeps all.",
        min_length=3,
        example="[[-38.60, 176.15], [-38.60, 176.25], [-38.50, 176.25], [-38.50, 176.15]]",
    )
    region_geojson: Path | None = F(
        None,
        "GeoJSON file (Polygon/MultiPolygon, lon/lat, holes allowed, optionally inside a "
        "Feature/FeatureCollection). Used instead of region_polygon when set.",
        path=True,
    )
    depth_km: Bounds = F((-1.0, 8.0), "[min, max] event depth in km (positive down).")
    min_p: int = F(6, "Minimum P picks per event.", ge=0)
    min_s: int = F(0, "Minimum S picks per event.", ge=0)
    max_azgap: float = F(180.0, "Maximum azimuthal gap in degrees.", gt=0, le=360)
    max_rms: float = F(0.20, "Maximum catalogue RMS residual in seconds.", gt=0)
    max_epi_dist_km: float = F(25.0, "Maximum epicentral distance to a used station, km.", gt=0)
    magnitude: MagnitudeSelection = FF(MagnitudeSelection, "Magnitude filter.")
    pick_uncertainty_to_weight: dict[float, int] = F(
        {0.02: 0, 0.05: 1, 0.10: 2, 0.20: 3},
        "Map from pick uncertainty upper limit (s) to VELEST weight class (0 best, 4 reject).",
    )
    wadati_outlier_sigma: float = F(
        3.0, "Wadati-diagram outlier rejection threshold (sigma).", gt=0
    )
    validation_fraction: float = F(0.15, "Fraction of events held out for validation.", ge=0, lt=1)

    @model_validator(mode="after")
    def _check(self) -> DataSelectionConfig:
        m = self.pick_uncertainty_to_weight
        if not m:
            raise ValueError("pick_uncertainty_to_weight must not be empty")
        if any(k <= 0 for k in m):
            raise ValueError("pick_uncertainty_to_weight: uncertainties must be > 0")
        if any(not 0 <= w <= 4 for w in m.values()):
            raise ValueError("pick_uncertainty_to_weight: weights must be in 0..4")
        ws = [m[k] for k in sorted(m)]
        if ws != sorted(ws):
            raise ValueError(
                "pick_uncertainty_to_weight: weight must not decrease as uncertainty increases"
            )
        return self


# ---------------------------------------------------------------------------
# VELEST
# ---------------------------------------------------------------------------


class VpVsConfig(_Base):
    """Vp/Vs treatment in VELEST trial models."""

    mode: Literal["fixed", "sampled", "independent_s_model"] = F(
        "fixed",
        "'fixed' uses 'value' and 'sampled' draws one Vp/Vs per model from 'range'; both run "
        "VELEST with nsp=3 (S picks used with Vs = Vp/vpvs, only a P model inverted). "
        "'independent_s_model' samples a separate S velocity model (nsp=2).",
    )
    value: float | Literal["wadati"] = F(
        1.75,
        "Vp/Vs ratio for mode 'fixed'. 'wadati' uses the Stage 1 Wadati-diagram ratio "
        "(prep stage's wadati.json), resolved when Stage 2 starts "
        "(osv.velest.workflow.resolve_vpvs).",
    )
    range: Bounds = F((1.65, 1.85), "[min, max] Vp/Vs for mode 'sampled'.")
    s_bounds: list[Bounds] | None = F(
        None,
        "[lower, upper] Vs bounds in km/s per layer for mode 'independent_s_model' (same "
        "length as layers_top_km). Default: derived from vp_bounds and 'range' "
        "(lower/range[1], upper/range[0]).",
    )

    @field_validator("value")
    @classmethod
    def _check_value(cls, v: float | str) -> float | str:
        if not isinstance(v, str) and not v > 1.0:
            raise ValueError("Vp/Vs must be > 1")
        return v

    @model_validator(mode="after")
    def _check(self) -> VpVsConfig:
        if self.range[0] <= 1.0:
            raise ValueError("range: Vp/Vs must be > 1")
        if self.s_bounds is not None and any(lo <= 0 or hi < lo for lo, hi in self.s_bounds):
            raise ValueError("s_bounds: need 0 < lower <= upper for every layer")
        return self


class HypocentrePerturbation(_Base):
    """Starting-hypocentre perturbation tests."""

    xy: float = F(1.0, "Horizontal perturbation, km.", ge=0)
    z: float = F(1.0, "Vertical perturbation, km.", ge=0)
    n_tests: int = F(50, "Number of perturbed relocation tests.", ge=0)
    distribution: Literal["uniform", "gaussian"] = F(
        "uniform",
        "Offset distribution: 'uniform' in [-xy, xy] per horizontal axis and [-z, z] in depth; "
        "'gaussian' with standard deviations xy and z.",
    )


class StabilityConfig(_Base):
    """Stability tests of the final 1-D model (Stage 2, task 3.5)."""

    enabled: bool = F(
        True,
        "Run the stability tests (hypocentre perturbation, reference-station swap, LVZ). When "
        "false only the fixed-model run that gives the final station delays and hypocentres "
        "is made.",
    )

    fixed_vthet: float = F(
        1.0e5,
        "Velocity damping (vthet) that keeps velocities fixed in the fixed-model runs "
        "(scale = sqrt(othet/vthet), velest.f SETUNT). 1e5 holds velocities to < 0.01 km/s; "
        "values of 1e8 or more overflow VELEST's control-file reader.",
        gt=0, le=1.0e7,
    )  # fmt: skip
    reference_stations: list[str] = F(
        [], "Alternative reference stations (aliases); empty selects them automatically."
    )
    n_reference_stations: int = F(
        3, "Number of automatically chosen alternative reference stations.", ge=0
    )
    delay_swtfac: float | None = F(
        0.01,
        "With nsp = 3, S readings give no station-delay partials although their residual "
        "contains the delay (ptcor * vpvs), so inverting delays with S data weighted makes "
        "the RMS climb. Runs that invert delays (the fixed-model delay step, the "
        "reference-station test) therefore weight S by this factor; the other tests hold "
        "the delays of the fixed-model run fixed. null keeps the S weight and inverts "
        "delays in every run (the previous behaviour).",
        ge=0,
    )
    lvz_test: bool = F(True, "Run the low-velocity-layer on/off test.")
    horizontal_mislocation_warn_km: float = F(
        0.5, "Warn when the mean horizontal mislocation exceeds this (km).", gt=0
    )
    vertical_mislocation_warn_km: float = F(
        0.5, "Warn when the mean vertical mislocation exceeds this (km).", gt=0
    )
    velocity_spread_warn_km_s: float = F(
        0.2, "Warn when a layer's Vp spread over the perturbed free runs exceeds this.", gt=0
    )
    reference_swap_dv_warn_km_s: float = F(
        0.1, "Warn when a reference-station swap changes a layer's Vp by more than this.", gt=0
    )


class VelestDamping(_Base):
    """VELEST damping factors."""

    othet: float = F(0.01, "Origin-time damping.", ge=0)
    xythet: float = F(0.01, "Epicentre damping.", ge=0)
    zthet: float = F(0.01, "Depth damping.", ge=0)
    vthet: float = F(1.0, "Velocity damping.", ge=0)
    stathet: float = F(0.01, "Station-correction damping.", ge=0)


class VelestSelect(_Base):
    """How to pick the preferred 1-D model from the ensemble."""

    method: Literal["best_percent", "best_n", "rms_below"] = F(
        "best_percent",
        "'best_percent': best 'value' percent by RMS; 'best_n': best 'value' models; "
        "'rms_below': all models with RMS below 'value' s.",
    )
    value: float = F(5.0, "Percent, count or RMS threshold depending on 'method'.", gt=0)
    statistic: Literal["median", "mean", "best"] = F(
        "median", "Statistic over the selected models giving the final model."
    )
    final_model: Literal["median", "best", "closest", "iterated"] = F(
        "median",
        "Which model becomes the final model. 'median': the per-layer 'statistic' of the "
        "selected runs; 'best': the single lowest-RMS run; 'closest': the selected run nearest "
        "the per-layer statistic (a model VELEST actually produced); 'iterated': the statistic "
        "model iterated with VELEST (delays frozen) to a fixed point. The p05-p95 uncertainty "
        "is always taken from the selected ensemble.",
    )

    @model_validator(mode="after")
    def _check(self) -> VelestSelect:
        if self.method == "best_percent" and self.value > 100:
            raise ValueError("value: percent must be <= 100")
        if self.method == "best_n" and self.value != int(self.value):
            raise ValueError("value: must be a whole number for method 'best_n'")
        return self


class DampingStage(_Base):
    """One stage of the VELEST damping schedule (Kissling et al. 1994).

    Fields left ``None`` inherit the corresponding top-level ``velest`` setting
    (``damping.*``, ``ittmax``, ``invertratio``), so the last stage normally runs with the
    user's damping.
    """

    name: str = F("stage", "Stage name (used in logs and the history table).", min_length=1)
    othet: float | None = F(None, "Origin-time damping (None: velest.damping.othet).", ge=0)
    xythet: float | None = F(None, "Epicentre damping (None: velest.damping.xythet).", ge=0)
    zthet: float | None = F(None, "Depth damping (None: velest.damping.zthet).", ge=0)
    vthet: float | None = F(None, "Velocity damping (None: velest.damping.vthet).", ge=0)
    stathet: float | None = F(
        None, "Station-correction damping (None: velest.damping.stathet).", ge=0
    )
    ittmax: int | None = F(None, "Iterations of this stage (None: velest.ittmax).", ge=1)
    swtfac: float | None = F(
        None,
        "Weight factor of S readings in this stage (None: the control file value). With "
        "nsp = 3 a small value (e.g. 0.01) makes the stage a P-only inversion, which is "
        "the consistent way to estimate station delays.",
        ge=0,
    )
    invertratio: int | None = F(
        None, "VELEST 'invertratio' of this stage (None: velest.invertratio).", ge=0
    )
    use_previous_hypocentres: bool = F(
        True, "Start from the previous stage's relocated hypocentres (velout.cnv)."
    )
    use_previous_station_corrections: bool = F(
        True,
        "Start from the previous stage's station delays (written to the stage's station "
        "file; sets iusestacorr = 1 so VELEST reads them).",
    )
    invert_station_delays: bool | None = F(
        None,
        "Invert for station delays in this stage (VELEST nsinv). None: as the control "
        "default (invert). With nsp = 3 (fixed Vp/Vs) the S delay is ptcor * vpvs but S "
        "readings contribute no delay partials, so inverting delays with S data weighted "
        "makes the RMS erratic (see docs/conventions.md); False holds the delays fixed.",
    )


def _default_stages() -> list[DampingStage]:
    return [
        DampingStage(name="stage1_high_damping", vthet=10.0, stathet=1.0),
        DampingStage(name="stage2_final"),
    ]


class ReseedConfig(_Base):
    """Second-round Monte Carlo started from the best models of the previous round."""

    enabled: bool = F(False, "Run reseeded rounds after the first Monte Carlo round.")
    best_percent: float = F(
        10.0, "Percent of the previous round's converged models (lowest RMS) used as seeds.",
        gt=0, le=100,
    )  # fmt: skip
    n_rounds: int = F(1, "Number of additional (reseeded) rounds.", ge=1)
    perturb_vp_km_s: float = F(
        0.1, "Standard deviation (km/s) of the Gaussian Vp perturbation of each seed model "
        "(Vs follows with the seed's Vp/Vs); 0 reuses the seeds unchanged.", ge=0,
    )  # fmt: skip
    n_copies: int = F(
        0,
        "Perturbed copies per seed model; 0 chooses round(100 / best_percent) so the round "
        "has about as many models as the first.",
        ge=0,
    )
    inherit_hypocentres: bool = F(True, "Seeds start from their own relocated hypocentres.")
    inherit_station_corrections: bool = F(True, "Seeds start from their own station delays.")


class IterateUntilConfig(_Base):
    """Kissling repeat-until-stable loop on the final model."""

    max_cycles: int = F(5, "Maximum number of repeated VELEST runs (0 disables the loop).", ge=0)
    model_change_km_s: float = F(
        0.02, "Stop when the largest layer velocity change is below this (km/s).", gt=0
    )
    hypo_change_km: float = F(0.1, "Stop when the mean hypocentre shift is below this (km).", gt=0)
    drift_warn_km_s: float = F(
        0.2,
        "Warn when the repeat-until-stable loop moves a layer's Vp from the ensemble final "
        "model by more than this (km/s).",
        gt=0,
    )


class VpVsGridConfig(_Base):
    """Optional Vp/Vs grid search with the final model (consistency check of the Wadati ratio)."""

    enabled: bool = F(False, "Run VELEST once per ratio in 'values' and plot RMS against Vp/Vs.")
    values: list[float] = F(
        [1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80],
        "Vp/Vs ratios to test (each > 1).",
        min_length=3,
    )

    @field_validator("values")
    @classmethod
    def _check_values(cls, v: list[float]) -> list[float]:
        if any(not x > 1.0 for x in v):
            raise ValueError("values: every Vp/Vs must be > 1")
        return v


class VdampAutoConfig(_Base):
    """Factors of ``velest.vdamp: auto``."""

    near_surface: float = F(
        10.0,
        "vdamp of layers that start above the deepest station (their velocity trades off "
        "with the station delays).",
        ge=0, le=999,
    )  # fmt: skip
    below_events: float = F(
        3.0,
        "vdamp of layers that start below the deepest hypocentre (sampled by refracted rays "
        "only).",
        ge=0, le=999,
    )  # fmt: skip


class VelestConfig(_Base):
    """VELEST Monte Carlo 1-D inversion (Stage 2)."""

    layers_top_km: list[float] = F(
        [-0.6, 0.2, 0.7, 1.2, 1.7, 2.2, 3.0, 4.0],
        "Depth to the top of each layer, km (positive down, may be negative above sea level).",
        min_length=1,
    )
    vp_bounds: list[Bounds] = F(
        [
            (1.8, 3.5),
            (2.5, 4.0),
            (3.0, 4.5),
            (3.5, 5.0),
            (4.0, 5.5),
            (4.5, 6.0),
            (5.0, 6.2),
            (5.5, 6.5),
        ],  # fmt: skip
        "[lower, upper] Vp bounds in km/s for each layer (same length as layers_top_km).",
        min_length=1,
    )
    vpvs: VpVsConfig = FF(VpVsConfig, "Vp/Vs treatment.")
    vdamp: float | list[float] | Literal["auto"] = F(
        1.0,
        "VELEST per-layer velocity damping factor (the model file's vdamp; effective damping "
        "is damping.vthet * vdamp): a number for all layers, one value per layer, or 'auto' "
        "(heavier damping for layers above the deepest station and below the deepest event, "
        "see vdamp_auto).",
    )
    vdamp_auto: VdampAutoConfig = FF(VdampAutoConfig, "Factors used by vdamp: auto.")
    monotonic: bool = F(True, "Require Vp to be non-decreasing with depth in trial models.")
    n_models: int = F(1000, "Number of Monte Carlo trial models.", ge=1)
    include_reference_model: bool = F(
        True, "Also write the mid-bounds reference model as trial 'MC_0000'."
    )
    sampler: Literal["latin_hypercube", "sobol", "uniform"] = F(
        "latin_hypercube", "Sampling scheme for trial velocity models."
    )
    hypocentre_perturbation_km: HypocentrePerturbation = FF(
        HypocentrePerturbation, "Starting hypocentre perturbation tests."
    )
    damping: VelestDamping = FF(VelestDamping, "VELEST damping factors.")
    ittmax: int = F(
        30,
        "Maximum VELEST iterations per run (also the default of every stage). Real data "
        "rarely meets VELEST's own stop rule: 0/40 runs stopped before the limit at 9, 30 "
        "or 60, but the median final RMS fell from 0.0945 (9) to 0.0738 (30) and stayed "
        "flat beyond.",
        ge=1,
    )
    invertratio: int = F(
        1, "VELEST 'invertratio': invert for velocities every N-th iteration.", ge=0
    )
    select: VelestSelect = FF(VelestSelect, "Selection of the preferred model from the ensemble.")
    stages: list[DampingStage] = Field(
        default_factory=_default_stages,
        description="Damping schedule run sequentially for every trial model; each stage "
        "starts from the previous stage's model (and optionally hypocentres and station "
        "delays). An empty list runs a single stage with the top-level damping.",
    )
    stability: StabilityConfig = FF(StabilityConfig, "Stability tests of the final model.")
    reseed: ReseedConfig = FF(ReseedConfig, "Reseeded rounds from the best models.")
    iterate_until: IterateUntilConfig = FF(
        IterateUntilConfig, "Repeat VELEST on the final model until changes are negligible."
    )
    vpvs_grid: VpVsGridConfig = FF(VpVsGridConfig, "Optional Vp/Vs grid search (final model).")
    compact_runs: bool = F(
        True,
        "Delete scratch files of successful VELEST runs, keeping the control file, model, main "
        "outputs, logs and status (saves disk for large ensembles).",
    )

    @model_validator(mode="after")
    def _check(self) -> VelestConfig:
        names = [st.name for st in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stages: stage names must be unique")
        tops = self.layers_top_km
        if any(b <= a for a, b in zip(tops, tops[1:], strict=False)):
            raise ValueError("layers_top_km: must be strictly increasing")
        if len(self.vp_bounds) != len(tops):
            raise ValueError(
                f"vp_bounds: has {len(self.vp_bounds)} entries but layers_top_km has {len(tops)}"
            )
        if any(lo <= 0 for lo, _ in self.vp_bounds):
            raise ValueError("vp_bounds: velocities must be > 0")
        if isinstance(self.vdamp, list):
            if len(self.vdamp) != len(tops):
                raise ValueError(
                    f"vdamp: has {len(self.vdamp)} entries but layers_top_km has {len(tops)}"
                )
            if any(not 0 <= x <= 999 for x in self.vdamp):
                raise ValueError("vdamp: values must be between 0 and 999")
        elif not isinstance(self.vdamp, str) and not 0 <= self.vdamp <= 999:
            raise ValueError("vdamp: must be between 0 and 999")
        if self.vpvs.s_bounds is not None and len(self.vpvs.s_bounds) != len(tops):
            raise ValueError(
                f"vpvs.s_bounds: has {len(self.vpvs.s_bounds)} entries but layers_top_km "
                f"has {len(tops)}"
            )
        if self.monotonic:
            for i in range(1, len(self.vp_bounds)):
                if self.vp_bounds[i][1] < self.vp_bounds[i - 1][0]:
                    raise ValueError(
                        f"vp_bounds[{i}]: upper bound is below the lower bound of the layer "
                        "above, so a monotonic model is impossible"
                    )
        return self


# ---------------------------------------------------------------------------
# 3-D model
# ---------------------------------------------------------------------------


class GridConfig(_Base):
    """FMTOMO inversion grid."""

    dx_km: float = F(0.5, "Velocity-node spacing in x (east), km.", gt=0)
    dy_km: float = F(0.5, "Velocity-node spacing in y (north), km.", gt=0)
    dz_km: float = F(0.5, "Velocity-node spacing in depth, km.", gt=0)
    pad_km: float = F(3.0, "Padding around the stations/events, km.", ge=0)
    zmax_km: float = F(8.0, "Maximum model depth, km.", gt=0)
    top_margin_km: float = F(
        0.2, "Height of the model top above the highest station elevation, km.", ge=0.05
    )
    extent_from: Literal["both", "stations", "events"] = F(
        "both", "Which positions define the horizontal extent (before padding)."
    )
    max_memory_gb: float = F(
        16.0, "Warn when the estimated fm3d memory for the propagation grid exceeds this.", gt=0
    )


class ConstantLaw(_Base):
    """Constant velocity within a unit."""

    law: Literal["constant"]
    v0: float = F(..., "Velocity, km/s.", gt=0)


class GradientLaw(_Base):
    """Linear velocity gradient within a unit: v = v0 + k * (depth below the reference)."""

    law: Literal["gradient"]
    v0: float = F(..., "Velocity at the reference depth, km/s.", gt=0)
    k: float = F(..., "Gradient, km/s per km of depth.")
    reference: Literal["unit_top", "surface", "sea_level"] = F(
        "unit_top", "Depth origin of the law: top of the unit, ground surface or sea level."
    )


class ExponentialLaw(_Base):
    """Compaction law: v = vmax - (vmax - v0) * exp(-h / lambda_km)."""

    law: Literal["exponential"]
    v0: float = F(..., "Velocity at the reference depth, km/s.", gt=0)
    vmax: float = F(..., "Asymptotic velocity at great depth, km/s.", gt=0)
    lambda_km: float = F(..., "E-folding depth, km.", gt=0)
    reference: Literal["unit_top", "surface", "sea_level"] = F(
        "unit_top", "Depth origin of the law: top of the unit, ground surface or sea level."
    )


VelocityLaw = Annotated[ConstantLaw | GradientLaw | ExponentialLaw, Field(discriminator="law")]


class Horizon(_Base):
    """A geological unit bounded below by a surface."""

    name: str = F(..., "Unit name.", min_length=1)
    surface: Path = F(..., "Base surface file (XYZ, GeoTIFF or GOCAD TSurf).", path=True)
    vp: VelocityLaw = F(..., "P-velocity law within the unit.")
    vs: VelocityLaw | None = F(None, "Optional S-velocity law; default uses the Vp/Vs ratio.")
    vpvs: float = F(1.75, "Vp/Vs ratio used when no S-velocity law is given.", gt=1.0)
    crs: str | None = F(
        None, "CRS of the surface file (e.g. 'EPSG:2193'); null = file's own CRS or lon/lat."
    )
    z_convention: Literal["depth", "elevation"] = F(
        "depth", "Surface z values are depth (positive down) or elevation (positive up)."
    )
    units: Literal["m", "km"] = F("m", "Unit of the surface z values.")


class VariogramConfig(_Base):
    """Variogram for ordinary kriging of well velocities."""

    model: Literal["spherical", "exponential", "gaussian"] = F("spherical", "Variogram model.")
    range_km: float = F(10.0, "Variogram range (practical range for exponential), km.", gt=0)
    sill: float = F(1.0, "Sill (kriging weights only depend on nugget/sill ratio).", gt=0)
    nugget: float = F(0.0, "Nugget (same units as the sill).", ge=0)


class WellsConfig(_Base):
    """Well-log (sonic) constraints."""

    las_dir: Path | None = F(None, "Directory of LAS well logs; null disables wells.", path=True)
    table: Path | None = F(
        None,
        "Wells table CSV: columns name, las_file, x/y (or lon/lat), elevation_m (elevation of "
        "the depth datum above sea level) and optional deviation_file (MD, inclination, azimuth). "
        "Null = take locations from the LAS headers.",
        path=True,
    )
    crs: str | None = F(None, "CRS of the table x/y (e.g. 'EPSG:2193'); null = lon/lat.")
    curve: str = F("DT", "Log curve mnemonic to convert (sonic slowness).", min_length=1)
    shear_curve: str | None = F("DTS", "Shear sonic slowness curve mnemonic; null = none.")
    density_curve: str | None = F(
        "RHOB", "Bulk density curve for Backus averaging; absent = Gardner's relation."
    )
    depth_reference: Literal["KB", "GL", "MSL"] = F(
        "KB",
        "Datum of the log depths: kelly bushing, ground level (both at 'elevation_m' above sea "
        "level) or mean sea level.",
    )
    units: Literal["auto", "us/ft", "us/m"] = F("auto", "Units of the sonic slowness curves.")
    smoothing_m: float = F(50.0, "Vertical smoothing length applied to logs, m.", ge=0)
    despike_window_m: float = F(
        2.0, "Median-filter window used to remove spikes, m (0 = off).", ge=0
    )
    despike_threshold: float = F(5.0, "Spike threshold in median absolute deviations.", gt=0)
    upscale: Literal["backus", "average", "none"] = F(
        "backus", "Upscaling to the model dz: Backus, slowness average or point sampling."
    )
    vpvs: float = F(1.75, "Vp/Vs used where no shear log exists.", gt=1.0)
    interp: Literal["kriging", "idw", "nearest", "linear"] = F(
        "kriging", "Interpolation of well velocities between wells ('linear' is an alias of idw)."
    )
    idw_power: float = F(2.0, "Inverse-distance weighting power.", gt=0)
    variogram: VariogramConfig = FF(VariogramConfig, "Variogram used by kriging.")
    max_distance_km: float = F(
        10.0, "Distance from the nearest well beyond which the background is used.", gt=0
    )
    taper_km: float = F(2.0, "Width of the taper to the background inside max_distance_km.", ge=0)


class UserModelConfig(_Base):
    """Geology/well-based user-supplied velocity model inputs."""

    grid_file: Path | None = F(
        None, "NetCDF/CSV velocity grid with x,y,z,vp[,vs] columns.", path=True
    )
    grid_crs: str | None = F(
        None, "CRS of the grid_file x/y (e.g. 'EPSG:2193'); null = lon/lat or the file's attribute."
    )
    z_convention: Literal["depth", "elevation"] = F(
        "depth", "grid_file z is depth (positive down) or elevation (positive up)."
    )
    units: Literal["m", "km"] = F("km", "Unit of the grid_file z coordinate.")
    grid_columns: dict[str, str] = F(
        {}, "Column/variable names in grid_file, keys x, y, z, vp, vs (defaults: the same names)."
    )
    horizons: list[Horizon] = F(
        [],
        "Geological units, ordered top to bottom.",
        example=(
            "- {name: Huka, surface: data/huka_base.xyz, vp: {law: gradient, v0: 2.2, k: 0.6}}\n"
            "- {name: Waiora, surface: data/waiora_base.tsurf, vp: {law: constant, v0: 3.4}}"
        ),
    )
    wells: WellsConfig = FF(WellsConfig, "Well-log constraints.")
    background: Literal["velest_1d", "constant", "none"] = F(
        "velest_1d",
        "Background model outside the user model's coverage: the extruded VELEST 1-D model, a "
        "constant (background_vp_km_s) or none (NaN outside coverage).",
    )
    background_vp_km_s: float | None = F(None, "Vp of the 'constant' background, km/s.", gt=0)
    background_vpvs: float = F(1.75, "Vp/Vs of the 'constant' background.", gt=1.0)
    blend_taper_km: float = F(
        1.0, "Width of the taper blending user model into background, km.", ge=0
    )
    smoothing_km: float = F(0.5, "Smoothing length applied to the combined model, km.", ge=0)
    smoothing_target: Literal["blend", "user"] = F(
        "blend",
        "'blend': smooth the blended model; 'user': smooth only the user model before blending.",
    )


class ExtrusionUncertaintyConfig(_Base):
    """A priori model uncertainty (second column of ``vgrids.in``, std in km/s)."""

    mode: Literal["constant", "spread"] = F(
        "constant",
        "'constant': the same value everywhere; 'spread': half the p05-p95 spread of each "
        "1-D layer (needs *_p05/*_p95 columns), bounded by min/max_km_s.",
    )
    value_km_s: float = F(0.3, "Uncertainty for mode 'constant', km/s.", gt=0)
    min_km_s: float = F(0.1, "Lower bound for mode 'spread', km/s.", gt=0)
    max_km_s: float = F(1.0, "Upper bound for mode 'spread', km/s.", gt=0)


class ExtrusionConfig(_Base):
    """Extension of the 1-D VELEST model to 3-D (``model3d.source = velest_1d``)."""

    mode: Literal["gradient", "layered"] = F(
        "gradient",
        "'layered': constant-velocity steps at the layer tops; 'gradient': linear "
        "interpolation between layer mid-depths.",
    )
    smoothing_km: float = F(
        0.0, "Gaussian smoothing length (sigma) along depth for mode 'layered', km.", ge=0
    )
    uncertainty: ExtrusionUncertaintyConfig = FF(
        ExtrusionUncertaintyConfig, "A priori uncertainty written to the vgrids."
    )


class ModelChecksConfig(_Base):
    """Thresholds of the physical checks on the starting model."""

    vp_min_km_s: float = F(1.0, "Vp below this is flagged, km/s.", gt=0)
    vp_max_km_s: float = F(8.5, "Vp above this is flagged, km/s.", gt=0)
    vpvs_min: float = F(1.4, "Vp/Vs below this is flagged.", gt=0)
    vpvs_max: float = F(2.2, "Vp/Vs above this is flagged.", gt=0)
    max_gradient_km_s_per_km: float = F(
        3.0, "Velocity gradient magnitude above this is flagged, (km/s)/km.", gt=0
    )


class Model3DConfig(_Base):
    """3-D starting model (Stage 3)."""

    source: Literal["velest_1d", "user_grid", "geology", "wells", "geology+wells"] = F(
        "velest_1d", "Origin of the 3-D starting model."
    )
    grid: GridConfig = FF(GridConfig, "Inversion grid geometry.")
    propgrid_refine: int = F(
        5, "Forward-grid refinement factor relative to the inversion grid.", ge=1
    )
    source_refine_factor: int = F(
        5, "fm3d refinement of the grid around each source (propgrid.in line 4).", ge=1
    )
    source_refine_cells: int = F(
        5, "Extent of the refined grid around each source, in propagation cells.", ge=1
    )
    surface: Literal["dem", "flat", "station_elevations"] = F(
        "station_elevations", "Source of the top (surface) interface."
    )
    surface_elevation_m: float = F(
        0.0, "Elevation of the flat surface (surface = 'flat'), m above sea level."
    )
    surface_smoothing_km: float = F(
        1.0,
        "Gaussian smoothing length of the station-elevation surface (and of the DEM after "
        "resampling), km.",
        ge=0,
    )
    extrusion: ExtrusionConfig = FF(ExtrusionConfig, "1-D to 3-D extension of the velocity model.")
    station_delays: Literal["diagnostic", "near_surface_layer"] = F(
        "diagnostic",
        "'diagnostic': only report station delays; 'near_surface_layer': model them.",
    )
    apply_velest_station_delays: bool = F(
        False,
        "Subtract the Stage 2 VELEST station delays (ptcor/stcor) from the observed travel "
        "times written for FMTOMO. Off (default): the 3-D model must explain the delays itself "
        "(they are the signal of near-surface structure, but the 1-D-derived delays also absorb "
        "hypocentre and model errors); on: the data are made consistent with the 1-D model so "
        "the residual 3-D signal is smaller, at the risk of removing real structure.",
    )
    user_model: UserModelConfig = FF(UserModelConfig, "Inputs for the user-supplied model.")
    uncertainty_from_weight: bool = F(
        False,
        "Increase the a priori uncertainty where the user model's weight is low "
        "(factor uncertainty_low_weight_factor at weight 0, capped by extrusion max_km_s).",
    )
    uncertainty_low_weight_factor: float = F(
        2.0, "Uncertainty multiplier at zero user-model weight (uncertainty_from_weight).", ge=1
    )
    checks: ModelChecksConfig = FF(ModelChecksConfig, "Physical checks of the starting model.")

    @model_validator(mode="after")
    def _check(self) -> Model3DConfig:
        um, src = self.user_model, self.source
        if src == "user_grid" and um.grid_file is None:
            raise ValueError("user_model.grid_file: required when source is 'user_grid'")
        if src in ("geology", "geology+wells") and not um.horizons:
            raise ValueError(
                f"user_model.horizons: at least one entry required when source is '{src}'"
            )
        if src in ("wells", "geology+wells") and um.wells.las_dir is None:
            raise ValueError(f"user_model.wells.las_dir: required when source is '{src}'")
        return self


# ---------------------------------------------------------------------------
# FMTOMO
# ---------------------------------------------------------------------------


class TradeoffConfig(_Base):
    """Damping/smoothing trade-off sweeps."""

    damping: PositiveList = F([1, 5, 10, 20, 50, 100, 200, 500], "Damping values to test.")
    smoothing: PositiveList = F([1, 10, 50, 100, 200, 500, 1000], "Smoothing values to test.")
    mode: Literal["sequential", "full_grid"] = F(
        "sequential",
        "'sequential': sweep damping then smoothing; 'full_grid': all combinations.",
    )
    pick: Literal["max_curvature", "manual"] = F(
        "max_curvature", "How to choose the preferred values from the L-curve."
    )


class GridlocConfig(_Base):
    """Extras ``gridloc``: grid search over the whole propagation grid.

    ``gridloc`` evaluates every node of the propagation grid (``arrtimes.dat``), so its
    search spacing is the ``propgrid.in`` spacing and there is no window around the event;
    the sub-cell refinement is the only resolution control.
    """

    min_stations: int = F(5, "Do not relocate events with fewer picks than this.", ge=1)
    max_origin_time_shift_s: float = F(5.0, "Maximum origin-time perturbation, s.", gt=0)
    refine: bool = F(True, "Refine the minimum with a trilinear sub-cell search.")
    subcell_dicing: int = F(10, "Sub-cell dicing of the refined search (must be > 1).", ge=2)


class JointVpVsConfig(_Base):
    """Experimental joint P/S (gridlocj) and Vp/Vs (vpvsinv) settings."""

    s_weight: float = F(3.0, "gridlocj: S weight (higher = S times count less).", gt=0)
    sp_weight: float = F(100.0, "gridlocj: weight of S-P residuals.", gt=0)
    relocation_based_on: Literal[1, 2, 3, 4] = F(
        1, "gridlocj: 1 P only, 2 P + unique S, 3 P + S-P + unique S, 4 P and S."
    )
    vpvs_max_depth_km: float = F(50.0, "mkvpvs: maximum depth of the Vp/Vs model, km.", gt=0)
    sparse_fraction: float = F(0.1, "vpvsinv: fraction of max G size for the sparse matrix.", gt=0)


class FmtomoConfig(_Base):
    """FMTOMO inversion (Stage 4)."""

    package: Literal["base", "extras"] = F(
        "extras", "'base' FMTOMO or the patched 'extras' variant."
    )
    phases: list[Literal["P", "S", "vpvs"]] = F(
        ["P"],
        "Phases to invert: [P], [P, S] (separate), or [vpvs] (joint Vp/Vs).",
        min_length=1,
    )
    relocate: Literal["none", "linear", "nonlinear"] = F(
        "nonlinear", "Event relocation during inversion ('nonlinear' uses gridloc)."
    )
    iterations: int = F(6, "Non-linear inversion iterations.", ge=1)
    damping: float = F(50.0, "Damping (used when not running the trade-off).", ge=0)
    smoothing: float = F(100.0, "Smoothing (used when not running the trade-off).", ge=0)
    tradeoff: TradeoffConfig = FF(TradeoffConfig, "Trade-off sweep settings.")
    subspace_dimension: int = F(
        20,
        "Subspace dimension of invert3d (raised to the number of parameter classes).",
        ge=1,
        le=50,
    )
    invert_velocity: bool = F(True, "Invert for velocity (False: relocation only).")
    invert_interfaces: list[int] = F(
        [], "1-based ids of interfaces whose depth is inverted (empty: none)."
    )
    source_damping_position: float = F(
        1.0, "Relative damping of linear source position updates (invert3d damp1).", ge=0
    )
    source_damping_time: float = F(
        1.0, "Relative damping of linear origin-time updates (invert3d damp2).", ge=0
    )
    min_velocity_km_s: float = F(0.5, "Minimum velocity permitted by invert3d, km/s.", gt=0)
    min_interface_distance_km: float = F(
        0.07, "Minimum distance between interfaces kept by invert3d, km.", ge=0
    )
    save_rays: bool = F(True, "Save ray paths (rays.dat) in the forward runs.")
    station_terms: Literal["none", "preapply"] = F(
        "none",
        "FMTOMO has no station terms. 'preapply' subtracts per-station mean residuals "
        "(supplied to assemble_run_dir) from the observed times before inversion.",
    )
    gridloc: GridlocConfig = FF(GridlocConfig, "Nonlinear relocation (Extras gridloc) settings.")
    joint: JointVpVsConfig = FF(
        JointVpVsConfig, "Joint P+S / Vp/Vs (experimental, Extras tomo3dj and tomo3dps)."
    )

    @model_validator(mode="after")
    def _check(self) -> FmtomoConfig:
        ph = self.phases
        if len(set(ph)) != len(ph):
            raise ValueError("phases: duplicate entries")
        if "vpvs" in ph and len(ph) > 1:
            raise ValueError("phases: 'vpvs' (joint) cannot be combined with P or S")
        if len(set(self.invert_interfaces)) != len(self.invert_interfaces) or any(
            i < 1 for i in self.invert_interfaces
        ):
            raise ValueError("invert_interfaces: need distinct ids >= 1")
        if self.package == "base" and (self.relocate == "nonlinear" or "vpvs" in ph):
            raise ValueError(
                "package: 'base' FMTOMO has no nonlinear relocation or Vp/Vs inversion "
                "(use package: extras, or relocate: linear/none and phases P/S)"
            )
        return self


# ---------------------------------------------------------------------------
# resolution tests / plotting
# ---------------------------------------------------------------------------


class CheckerboardConfig(_Base):
    """Checkerboard resolution tests."""

    cell_km: PositiveList = F([1.5, 2.0, 3.0], "Cell sizes to test, km.")
    amplitude_pct: float = F(5.0, "Velocity perturbation amplitude, percent.", gt=0, lt=100)
    offset_half_cell: bool = F(True, "Also run a copy offset by half a cell.")


class SpikeConfig(_Base):
    """Spike (point-anomaly) resolution test."""

    x_km: float = F(..., "East offset from origin, km.")
    y_km: float = F(..., "North offset from origin, km.")
    z_km: float = F(..., "Depth, km.")
    radius_km: float = F(..., "Spike radius, km.", gt=0)
    pct: float = F(..., "Velocity perturbation, percent.")


class NoiseConfig(_Base):
    """Synthetic data noise."""

    from_pick_uncertainty: bool = F(True, "Scale noise to each pick's uncertainty.")
    scale: float = F(1.0, "Noise scale factor.", ge=0)


class ResolutionConfig(_Base):
    """Resolution tests (Stage 6)."""

    checkerboard: CheckerboardConfig = FF(CheckerboardConfig, "Checkerboard test settings.")
    spikes: list[SpikeConfig] = F(
        [],
        "Spike tests.",
        example="- {x_km: 0, y_km: 0, z_km: 2, radius_km: 0.75, pct: -8}",
    )
    input_models: list[Literal["final_3d", "user_model", "starting"]] = F(
        ["final_3d", "user_model"], "Models used as the test background."
    )
    noise: NoiseConfig = FF(NoiseConfig, "Noise added to synthetic data.")


class ProfileConfig(_Base):
    """A vertical cross-section."""

    name: str = F(..., 'Profile label, e.g. "A-A\'".', min_length=1)
    start: LatLon = F(..., "[lat, lon] of the profile start.")
    end: LatLon = F(..., "[lat, lon] of the profile end.")

    @model_validator(mode="after")
    def _check(self) -> ProfileConfig:
        if self.start == self.end:
            raise ValueError("end: must differ from start")
        return self


class PlottingConfig(_Base):
    """Figure settings."""

    backend: Literal["matplotlib", "pygmt"] = F("matplotlib", "Plotting backend.")
    format: list[Literal["png", "pdf", "svg"]] = F(
        ["png", "pdf"], "Figure file formats.", min_length=1
    )
    dpi: int = F(200, "Raster resolution.", ge=50, le=1200)
    profiles: list[ProfileConfig] = F(
        [],
        "Cross-section profiles.",
        example='- {name: "A-A\'", start: [-38.62, 176.19], end: [-38.46, 176.21]}',
    )
    depth_slices_km: list[float] = F(
        [0, 0.5, 1, 1.5, 2, 3, 4, 5], "Depths of horizontal slices, km."
    )
    dws_mask_threshold: float = F(
        10.0, "Mask model where the derivative weight sum is below this.", ge=0
    )


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


class OSVConfig(_Base):
    """Top-level openSeisVelocity configuration (the contents of ``config.yaml``)."""

    project: ProjectConfig = FF(ProjectConfig, "Project identity and run-wide settings.")
    inputs: InputsConfig = FF(InputsConfig, "Input data files and how to read them.")
    coordinates: CoordinatesConfig = FF(CoordinatesConfig, "Coordinate system.")
    data_selection: DataSelectionConfig = FF(
        DataSelectionConfig, "Event and pick selection (Stage 1)."
    )
    velest: VelestConfig = FF(VelestConfig, "VELEST Monte Carlo 1-D inversion (Stage 2).")
    model3d: Model3DConfig = FF(Model3DConfig, "3-D starting model (Stage 3).")
    fmtomo: FmtomoConfig = FF(FmtomoConfig, "FMTOMO inversion (Stage 4).")
    resolution: ResolutionConfig = FF(ResolutionConfig, "Resolution tests (Stage 6).")
    plotting: PlottingConfig = FF(PlottingConfig, "Figure settings.")
