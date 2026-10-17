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
