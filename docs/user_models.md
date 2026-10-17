# Bringing your own model: grids and geology

Stage 3 can start the 3-D inversion from your own knowledge instead of the VELEST 1-D model.
Two inputs are supported here: a complete **3-D velocity grid** (`source: user_grid`) and a
**layer-cake geology model** built from mapped surfaces (`source: geology`). Both are set in
`model3d.user_model` of the project YAML.

## Conventions (read this first)

* Coordinates inside osv are latitude/longitude (WGS84, north and east positive).
* **Depth** is in km, positive down, measured from sea level. Elevation is metres, positive up.
* Tell osv how *your* files store z: `z_convention: depth` (positive down) or
  `elevation` (positive up), and `units: m` or `km`. Getting this wrong flips or scales your
  model, so check the log line reporting coverage and the Stage 3 figures.
* Horizontal coordinates in projected systems (e.g. NZTM, `EPSG:2193`) are given with `crs`.
  Without a CRS, x/y are longitude/latitude.
* Velocities are km/s. A grid whose median velocity is above 100 is assumed to be m/s and
  converted (with a warning).
* The model grid includes one cushion node on every side; the imported models are
  produced on the full grid, with the cushions filled by the same rules.

## 3-D velocity grid (`user_grid`)

```yaml
model3d:
  source: user_grid
  user_model:
    grid_file: data/ohaaki_vp.nc      # or a .csv
    grid_crs: EPSG:2193               # x/y CRS; null = lon/lat (NetCDF may carry a 'crs' attribute)
    z_convention: elevation           # depth | elevation
    units: m                          # unit of the z coordinate
    grid_columns: {x: E, y: N, z: RL, vp: Vp, vs: Vs}   # only needed if names differ
```

**NetCDF**: a 3-D variable `vp` (optional `vs`) on regular axes. Axes are named `lon`, `lat`,
`depth`, or `x`, `y`, `z` (with a CRS from `grid_crs` or a `crs` attribute). Axes may be
ascending or descending and need not be uniformly spaced. Interpolation is trilinear.

**CSV**: one row per point with columns `x, y, z, vp[, vs]` (names configurable; comma, space
or semicolon separated). A complete regular lattice is detected and interpolated trilinearly.
Scattered points (e.g. from a geostatistical model) are interpolated with Delaunay linear
interpolation, and nodes farther than three times the median point spacing from any point are
left undefined.

In both cases nodes outside the data are **NaN**, and a boolean *coverage mask* says where the
file provided data. Stage 4.7 later blends the user model into the background using that
mask, so partial coverage is fine. A file that does not overlap the model at all is an error
(usually a wrong CRS, unit or z convention).

## Geology horizons (`geology`)

```yaml
model3d:
  source: geology
  user_model:
    horizons:                          # ordered TOP to BOTTOM; each surface is the BASE of its unit
      - name: Huka
        surface: data/huka_base.xyz
        crs: EPSG:2193
        z_convention: elevation        # RL in metres
        units: m
        vp: {law: gradient, v0: 2.2, k: 0.6}      # v = v0 + k * (depth below top of unit)
        vpvs: 1.8                                  # Vs = Vp / vpvs   (default 1.75)
      - name: Waiora
        surface: data/waiora_base.ts   # GOCAD TSurf
        crs: EPSG:2193
        vp: {law: exponential, v0: 3.0, vmax: 5.0, lambda_km: 1.5}
        vs: {law: constant, v0: 1.9}                # explicit Vs law instead of vpvs
    background: velest_1d              # below the last surface and where no surface exists
```

### Surface formats

| Format | Suffix | Notes |
| --- | --- | --- |
| XYZ ASCII | `.xyz .txt .csv .dat` | three columns `x y z`; `#` comments and a text header line are skipped; scattered points are triangulated, NaN outside their outline |
| GeoTIFF | `.tif .tiff` | needs `rasterio` (`pip install openseisvelocity[geo]`); the file's CRS is used unless `crs` is given; nodata cells are undefined |
| GOCAD TSurf | `.ts .tsurf` | `VRTX`/`PVRTX` + `TRGL`; triangles interpolated linearly. The `ZPOSITIVE` header is ignored in favour of your `z_convention` (a warning is logged on mismatch) |

Each surface is evaluated at the model's lateral nodes, giving depth (km bsl); outside the
surface the value is undefined (NaN).

### Stacking rules

* Unit *i* lies between the base of unit *i-1* and its own base; the first unit extends from the
  top of the model.
* Below the last surface the **background** is used (the VELEST 1-D model, a constant, or
  undefined if `background: none`).
* Where any surface is undefined, the whole lateral column falls back to the background and
  the count is reported.
* **Crossing surfaces** (a deeper horizon above a shallower one) would give negative
  thickness. osv clips the deeper surface to the one above (zero-thickness unit) and reports
  how many nodes were affected. If that number is large, fix the surfaces instead.
* A unit-ID volume (`unit_id`, `-1` = background) is produced alongside Vp and Vs.

### Velocity laws

`h` is the depth below the law's reference (never negative).

| `law` | Formula | Parameters |
| --- | --- | --- |
| `constant` | `v0` | `v0` |
| `gradient` | `v0 + k h` | `v0`, `k` (km/s per km), `reference` |
| `exponential` (compaction) | `vmax - (vmax - v0) exp(-h / lambda_km)` | `v0`, `vmax`, `lambda_km`, `reference` |

`reference` is `unit_top` (default: the top of the unit, or the ground surface for the first
unit), `surface` (the ground surface) or `sea_level`. Choose `sea_level` for laws calibrated
as velocity versus absolute depth.

## From Python

```python
from osv.model3d.user_grid import load_user_grid
from osv.model3d.geology import build_geology_model

ds, coverage = load_user_grid("ohaaki_vp.csv", grid, crs="EPSG:2193", units="m",
                              z_convention="elevation")
geo, findings = build_geology_model(cfg.model3d.user_model.horizons, grid,
                                    background=lambda z: 5.0 + 0 * z)
for f in findings:
    print(f.level, f.code, f.message)
```

## Well logs (`wells`, `geology+wells`)

Sonic logs in LAS files constrain the model directly (`source: wells`) or set the velocity
of each geological unit (`source: geology+wells`, which also needs `horizons`). Reading LAS
needs `lasio` (`pip install lasio`, part of the `geo` extra).

```yaml
model3d:
  source: wells                    # or geology+wells
  user_model:
    wells:
      table: data/wells/wells.csv  # name, las_file, x, y (or lon, lat), elevation_m, deviation_file
      las_dir: data/wells/
      crs: EPSG:2193               # CRS of x/y in the table; null = lon/lat
      curve: DT                    # P sonic;  shear_curve: DTS;  density_curve: RHOB
      depth_reference: KB          # KB | GL | MSL  (datum of the log depths)
      units: auto                  # auto | us/ft | us/m
      smoothing_m: 50              # running mean of slowness along the well
      despike_window_m: 2          # median-filter window; despike_threshold in MADs
      upscale: backus              # backus | average | none
      interp: kriging              # kriging | idw | nearest
      variogram: {model: spherical, range_km: 10, sill: 1, nugget: 0}
      max_distance_km: 10          # beyond this the background is used ...
      taper_km: 2                  # ... with a linear taper over this width
```

* **Wells table.** One row per well: `name`, `las_file` (relative to `las_dir`), wellhead
  `x`,`y` in `crs` (or `lon`,`lat`), `elevation_m` and optionally `deviation_file`. Without a
  table the locations are read from the LAS header (`XCOORD/YCOORD` or `LAT/LON`, `EKB/KB/ELEV`).
* **Depth.** LAS depth is measured depth (MD) from the datum. `elevation_m` is the height of
  that datum above sea level (kelly bushing for `KB`, ground for `GL`); model depth is
  `(TVD - elevation_m)/1000` km. With `MSL` the log depth is already below sea level and the
  elevation is ignored. A deviation survey (CSV with `md`, `incl`/`inclination`,
  `azi`/`azimuth`, degrees, azimuth clockwise from the grid north) is converted to TVD and
  easting/northing offsets with the minimum-curvature method; without a survey the well is
  vertical. Beyond the last survey station the last direction is continued (warning).
  In the lateral interpolation a deviated well is located where it actually is at each depth.
* **Velocity.** Slowness in us/ft or us/m (`units: auto` uses the curve unit, else assumes
  us/m if the median exceeds 200) becomes km/s. Nulls (-999.25), non-physical values (Vp
  outside 0.3-9 km/s) and spikes are removed. Without a shear log Vs = Vp/`vpvs`.
* **Upscaling** to the model node spacing: `backus` is the Backus average (harmonic mean of the
  moduli `rho vp^2` and `rho vs^2`, arithmetic density); without a density log
  **Gardner's relation** `rho = 0.31 (1000 vp)^0.25` g/cc supplies it. `average` is the
  travel-time average `1/<1/v>`; `none` samples the log at the node.
  A node is covered by a well only if at least half of its block holds log data.
* **Lateral interpolation** per depth slice, using the wells that have data at that depth:
  ordinary kriging (spherical, exponential or gaussian variogram), IDW (`idw_power`) or the
  nearest well. The output has `weight` (1 near a well, tapering to 0 at `max_distance_km`) and
  a boolean `mask` (weight > 0); the background fills the rest (NaN if none is given).
* **geology+wells.** For each unit the parameters of its velocity law (the *type* is the one
  in the horizon entry: `constant` mean, `gradient` `v0`,`k` or `exponential`) are fitted by
  weighted least squares to the log samples inside the unit (between its top and base
  surfaces at the well position), and `vpvs` follows the shear log. The configured
  parameters are used for units without (at least 10) log samples. Fit quality (rms, R^2) is
  reported as findings.

```python
from osv.model3d.wells import load_wells, build_wells_model, build_geology_wells_model

wells, findings = load_wells(cfg.model3d.user_model.wells, grid, base_dir=project_dir)
res = build_wells_model(cfg.model3d.user_model.wells, grid, background=bg, wells=wells)
res.dataset          # vp, vs, weight, mask
res.profiles         # upscaled per-well profile at the model nodes (for plotting)
res.logs             # conditioned log samples
geo, fits, findings = build_geology_wells_model(
    cfg.model3d.user_model.horizons, grid, cfg.model3d.user_model.wells, background=bg)
```

Findings report wells outside the grid, short logs, missing curves, unit-fit quality and
the number of nodes not constrained by any well.
