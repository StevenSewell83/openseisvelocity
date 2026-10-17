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
    station_aliases: Literal["auto"] | dict[str, str] = F(
        "auto",
        "VELEST needs station codes of at most 4 characters: 'auto' generates aliases, or give "
        "an explicit {code: alias} mapping.",
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
        "'fixed' uses 'value'; 'sampled' draws Vp/Vs uniformly from 'range'; "
        "'independent_s_model' samples a separate S velocity model.",
    )
    value: float = F(1.75, "Vp/Vs ratio for mode 'fixed'.", gt=1.0)
    range: Bounds = F((1.65, 1.85), "[min, max] Vp/Vs for mode 'sampled'.")
    s_bounds: list[Bounds] | None = F(
        None,
        "[lower, upper] Vs bounds in km/s per layer for mode 'independent_s_model' (same "
        "length as layers_top_km). Default: derived from vp_bounds and 'range' "
        "(lower/range[1], upper/range[0]).",
    )

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

    @model_validator(mode="after")
    def _check(self) -> VelestSelect:
        if self.method == "best_percent" and self.value > 100:
            raise ValueError("value: percent must be <= 100")
        if self.method == "best_n" and self.value != int(self.value):
            raise ValueError("value: must be a whole number for method 'best_n'")
        return self


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
    ittmax: int = F(9, "Maximum VELEST iterations per run.", ge=1)
    invertratio: int = F(
        1, "VELEST 'invertratio': invert for velocities every N-th iteration.", ge=0
    )
    select: VelestSelect = FF(VelestSelect, "Selection of the preferred model from the ensemble.")

    @model_validator(mode="after")
    def _check(self) -> VelestConfig:
        tops = self.layers_top_km
        if any(b <= a for a, b in zip(tops, tops[1:], strict=False)):
            raise ValueError("layers_top_km: must be strictly increasing")
        if len(self.vp_bounds) != len(tops):
            raise ValueError(
                f"vp_bounds: has {len(self.vp_bounds)} entries but layers_top_km has {len(tops)}"
            )
        if any(lo <= 0 for lo, _ in self.vp_bounds):
            raise ValueError("vp_bounds: velocities must be > 0")
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


class ConstantLaw(_Base):
    """Constant velocity within a unit."""

    law: Literal["constant"]
    v0: float = F(..., "Velocity, km/s.", gt=0)


class GradientLaw(_Base):
    """Linear velocity gradient within a unit: v = v0 + k * (depth below unit top)."""

    law: Literal["gradient"]
    v0: float = F(..., "Velocity at the top of the unit, km/s.", gt=0)
    k: float = F(..., "Gradient, km/s per km of depth.")


VelocityLaw = Annotated[ConstantLaw | GradientLaw, Field(discriminator="law")]


class Horizon(_Base):
    """A geological unit bounded below by a surface."""

    name: str = F(..., "Unit name.", min_length=1)
    surface: Path = F(..., "Base surface file (XYZ, GeoTIFF or GOCAD TSurf).", path=True)
    vp: VelocityLaw = F(..., "P-velocity law within the unit.")
    vs: VelocityLaw | None = F(None, "Optional S-velocity law; default uses the Vp/Vs ratio.")


class WellsConfig(_Base):
    """Well-log (sonic) constraints."""

    las_dir: Path | None = F(None, "Directory of LAS well logs; null disables wells.", path=True)
    curve: str = F("DT", "Log curve mnemonic to convert (sonic slowness).", min_length=1)
    smoothing_m: float = F(50.0, "Vertical smoothing length applied to logs, m.", ge=0)
    interp: Literal["kriging", "idw", "linear"] = F(
        "kriging", "Interpolation of well velocities between wells."
    )


class UserModelConfig(_Base):
    """Geology/well-based user-supplied velocity model inputs."""

    grid_file: Path | None = F(
        None, "NetCDF/CSV velocity grid with x,y,z,vp[,vs] columns.", path=True
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
    background: Literal["velest_1d", "none"] = F(
        "velest_1d", "Background model outside the user model's coverage."
    )
    blend_taper_km: float = F(
        1.0, "Width of the taper blending user model into background, km.", ge=0
    )
    smoothing_km: float = F(0.5, "Smoothing length applied to the combined model, km.", ge=0)


class Model3DConfig(_Base):
    """3-D starting model (Stage 3)."""

    source: Literal["velest_1d", "user_grid", "geology", "wells", "geology+wells"] = F(
        "velest_1d", "Origin of the 3-D starting model."
    )
    grid: GridConfig = FF(GridConfig, "Inversion grid geometry.")
    propgrid_refine: int = F(
        5, "Forward-grid refinement factor relative to the inversion grid.", ge=1
    )
    surface: Literal["dem", "flat", "station_elevations"] = F(
        "station_elevations", "Source of the top (surface) interface."
    )
    station_delays: Literal["diagnostic", "near_surface_layer"] = F(
        "diagnostic",
        "'diagnostic': only report station delays; 'near_surface_layer': model them.",
    )
    user_model: UserModelConfig = FF(UserModelConfig, "Inputs for the user-supplied model.")

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

    @model_validator(mode="after")
    def _check(self) -> FmtomoConfig:
        ph = self.phases
        if len(set(ph)) != len(ph):
            raise ValueError("phases: duplicate entries")
        if "vpvs" in ph and len(ph) > 1:
            raise ValueError("phases: 'vpvs' (joint) cannot be combined with P or S")
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
