# openSeisVelocity — Development Plan

Earthquake-based velocity inversion package for microseismic datasets:
**Monte Carlo VELEST** (1D minimum model) → **3D starting model** → **FMTOMO** 3D inversion (with trade-off curves) → **analysis** → **resolution testing** (checkerboard, spike, input-model).

- VELEST source: `/home/steve/seismic/velest`
- FMTOMO v1.2 + FMTOMO Extras v1.1: `/home/steve/seismic/FMTOMO`
- Test dataset: `manual_only_catalog.xml` (QuakeML) + `/home/steve/seismic/Station_Files/*_STATIONXML.xml`

_Last updated: 2026-10-17_

---

## 1. Requirements

1. An installer that sets up the compilers, builds the Fortran executables and installs the Python packages.
2. Every stage produces diagnostic figures for checking its outputs and finding problems.
3. All user settings in one `config.yaml`.
4. Inputs are **StationXML** (stations) and a **QuakeML** catalogue (events and picks).
5. Option to supply a **user-defined 3D starting model** built from geology (horizons) and/or well data.
6. Robust and reproducible: restartable stages, provenance records and tests.

---

## 2. Architecture

A Python package (`openseisvelocity`, command `osv`) that runs the existing Fortran programs, with the Fortran source kept essentially as distributed. Python handles setup, file formats, parallel runs, analysis and plotting.

```
openSeisVelocity/
├── install.sh / install.py   # one-command installer
├── environment.yml           # conda-forge environment
├── Dockerfile / apptainer.def
├── fortran/                  # vendored VELEST + FMTOMO + Extras, CMake build, small patches
├── src/osv/
│   ├── config.py             # pydantic schema for config.yaml
│   ├── io/                   # cnv, sta, mod, VELEST .OUT, FMTOMO *.in/*.dat readers/writers
│   ├── data/                 # Stage 1: QuakeML/StationXML prep and QC
│   ├── velest/               # Stage 2: MC generator, parallel runner, ensemble analysis
│   ├── model3d/              # Stage 3: 1D / user grid / geology / wells → FMTOMO grids
│   ├── fmtomo/               # Stage 4: inversion runner, trade-off sweeps
│   ├── analysis/             # Stage 5: slices, ray coverage, residuals, relocations
│   ├── resolution/           # Stage 6: checkerboard, spike, input-model recovery
│   ├── validation/           # Stage 7: 1D vs 3D relocation, bootstrap
│   ├── plotting/             # matplotlib (default), optional PyGMT
│   └── report/               # per-stage report.html + project report
├── tests/                    # format round-trips, FMTOMO example1 check, synthetic end-to-end
└── projects/<name>/          # one numbered folder per stage, each with manifest.json
```

**Design rules**
- **Each run gets its own folder.** The current `velestMC.sh` edits one shared `velest.cmn` in place with `sed`, so runs can't happen in parallel and nothing records what was run. Instead, every VELEST or FMTOMO run gets its own folder with its own control files.
- **Stages are restartable.** Each stage reads the previous stage's results and writes a `manifest.json` (config hash, inputs, binary versions, timings).
- **Every file format gets a typed reader and writer**, with a round-trip unit test.
- **Coordinates are stored as true signed lat/lon.** Conversion to VELEST's sign convention happens only when writing VELEST files. The older `.cnv` files used `176.19W` for NZ, which is consistent inside VELEST but mirrored.

---

## 3. Installer and environment

```bash
./install.sh            # or: python install.py --prefix ~/osv --no-gmt --jobs 8
osv doctor              # checks the whole toolchain and runs the smoke tests
```

**What `install.sh` does:**
1. **Checks the system:** OS, architecture, disk space, and whether conda, mamba or micromamba is available. If none is, it installs micromamba locally, without needing sudo.
2. **Creates the `osv` environment** from `environment.yml` (conda-forge):
   - `gfortran` (conda-forge compilers), `make`, `cmake`.
   - `python=3.11`, `obspy`, `numpy`, `scipy`, `pandas`, `xarray`, `netcdf4`, `matplotlib`, `cartopy`, `pyproj`, `pydantic`, `pyyaml`, `typer`, `joblib`, `lasio`, `rasterio`, `pyvista`.
   - Optional: `gmt`, `pygmt`.
3. **Compiles the Fortran** with one CMake/Makefile build. This replaces FMTOMO's ksh `compileall` scripts and their hard-coded compiler settings.
   - Small patches with a short reason for each: configurable VELEST array limits (`ieq`, stations, layers), gfortran ≥ 11 fixes (`-fallow-argument-mismatch`, legacy flags), optional `-O2 -march=native`.
   - Executables go into `$CONDA_PREFIX/bin/osv-tools/`, so they never clash with an existing `velest` on PATH.
4. **Installs the package:** `pip install -e .`
5. **Runs smoke tests:**
   - VELEST on a small synthetic `.cnv`.
   - FMTOMO `example1`, compared with the reference output.
   - A tiny synthetic run through all stages.

**Other routes:** Docker/Apptainer images for HPC clusters, and `--system-compilers` for machines that already have gfortran.

**`osv doctor`** reports executable versions, the compiler and flags used, the VELEST array limits, CPU count and missing optional packages. Each problem comes with the fix.

---

## 4. Configuration (`config.yaml`)

Every user setting lives in this file. It is checked by a pydantic schema when loaded, so a mistake gives an error naming the bad key. `osv init <project>` writes a fully commented template, and `osv config show` prints the settings actually in use. Each stage saves a copy of the config it ran with in its manifest.

```yaml
project:
  name: rk-nm-2017
  workdir: ./runs/rk-nm-2017
  n_jobs: 24
  random_seed: 42

inputs:
  stationxml: [/home/steve/seismic/Station_Files/*_STATIONXML.xml]
  quakeml:    [./manual_only_catalog.xml]
  origin_selection: preferred          # preferred | latest | by_method:<id>
  evaluation_mode: any                 # any | manual | automatic
  use_arrivals: false                  # build from picks; arrivals may be unreliable
  weight_source: uncertainty           # uncertainty | comment | both
  station_aliases: auto                # VELEST needs ≤4-char codes
  dem: null                            # optional GeoTIFF for the surface interface

coordinates:
  origin: {lat: -38.55, lon: 176.194}
  projection: local_tm                 # or epsg:2193
  borehole_sensors: use_depth_from_stationxml

data_selection:
  region_polygon: null
  depth_km: [-1.0, 8.0]
  min_p: 6
  min_s: 0
  max_azgap: 180
  max_rms: 0.20
  max_epi_dist_km: 25
  magnitude: {min: null, ignore_if_invalid: true}
  pick_uncertainty_to_weight: {0.02: 0, 0.05: 1, 0.10: 2, 0.20: 3}
  wadati_outlier_sigma: 3
  validation_fraction: 0.15

velest:
  layers_top_km: [-0.6, 0.2, 0.7, 1.2, 1.7, 2.2, 3.0, 4.0]
  vp_bounds:     [[1.8,3.5],[2.5,4.0],[3.0,4.5],[3.5,5.0],[4.0,5.5],[4.5,6.0],[5.0,6.2],[5.5,6.5]]
  vpvs: {mode: fixed, value: 1.75}     # fixed | sampled:[1.65,1.85] | independent_s_model
  monotonic: true
  n_models: 1000
  sampler: latin_hypercube
  hypocentre_perturbation_km: {xy: 1.0, z: 1.0, n_tests: 50}
  damping: {othet: 0.01, xythet: 0.01, zthet: 0.01, vthet: 1.0, stathet: 0.01}
  ittmax: 9
  invertratio: 1
  select: {method: best_percent, value: 5, statistic: median}

model3d:
  source: velest_1d                    # velest_1d | user_grid | geology | wells | geology+wells
  grid: {dx_km: 0.5, dy_km: 0.5, dz_km: 0.5, pad_km: 3, zmax_km: 8}
  propgrid_refine: 3
  surface: station_elevations          # dem | flat | station_elevations
  station_delays: diagnostic           # diagnostic | near_surface_layer
  user_model:
    grid_file: null                    # NetCDF/CSV x,y,z,vp[,vs]
    horizons:
      - {name: Huka,   surface: data/huka_base.xyz,    vp: {law: gradient, v0: 2.2, k: 0.6}}
      - {name: Waiora, surface: data/waiora_base.tsurf, vp: {law: constant, v0: 3.4}}
    wells: {las_dir: data/wells/, curve: DT, smoothing_m: 50, interp: kriging}
    background: velest_1d
    blend_taper_km: 1.0
    smoothing_km: 0.5

fmtomo:
  package: extras                      # base | extras
  phases: [P]                          # P | P,S | vpvs (joint)
  relocate: nonlinear                  # none | linear | nonlinear (gridloc)
  iterations: 6
  damping: 50.0
  smoothing: 100.0
  tradeoff:
    damping:   [1, 5, 10, 20, 50, 100, 200, 500]
    smoothing: [1, 10, 50, 100, 200, 500, 1000]
    mode: sequential                   # sequential | full_grid
    pick: max_curvature                # max_curvature | manual

resolution:
  checkerboard: {cell_km: [1.5, 2.0, 3.0], amplitude_pct: 5, offset_half_cell: true}
  spikes: [{x_km: 0, y_km: 0, z_km: 2, radius_km: 0.75, pct: -8}]
  input_models: [final_3d, user_model]
  noise: {from_pick_uncertainty: true, scale: 1.0}

plotting:
  backend: matplotlib                  # matplotlib | pygmt
  format: [png, pdf]
  dpi: 200
  profiles: [{name: "A-A'", start: [-38.62, 176.19], end: [-38.46, 176.21]}]
  depth_slices_km: [0, 0.5, 1, 1.5, 2, 3, 4, 5]
  dws_mask_threshold: 10
```

---

## 5. Stages, outputs and diagnostic figures

Each stage writes a `figures/` folder and a `report.html` that collects them with a short summary table. Every stage also runs automatic checks, and failed checks appear as warnings at the top of the report.

### Stage 0: Install and doctor
- **Output:** a text report only.

### Stage 1: Data preparation (StationXML + QuakeML)
- **Read:** ObsPy `read_inventory` / `read_events`.
- **Stations:** lat/lon/elevation plus borehole channel `depth`. Station epochs are matched to pick times.
- **Picks:**
  - Linked to stations by `waveform_id`.
  - Phase from `phase_hint`, or the arrival's phase when `use_arrivals` is set.
  - Pick uncertainty (symmetric or lower/upper bounds) or weight comment mapped to a VELEST weight class.
- **Event selection:** cut-offs from the config. Duplicate picks are removed. Optional calibration/validation split.
- **Station codes:** aliases saved to `aliases.csv` where needed.
- **Outputs:** clean event and pick tables (Parquet), plus the VELEST and FMTOMO input files.
- **Figures:**
  - Map of stations and events, before and after filtering.
  - Depth cross-sections.
  - Histograms (picks per event, azimuthal gap, RMS, depth, magnitude if valid) with the cut-offs marked.
  - Events over time.
  - Picks per station (P and S).
  - Travel time vs distance (P and S) coloured by weight, with outliers marked.
  - Wadati diagram with the fitted Vp/Vs and outliers.
  - Ray-path coverage map.
  - Pick-uncertainty distributions.
  - Table of why each event or pick was rejected.
- **Checks:** missing StationXML, picks outside station epochs, corrupted or duplicate arrivals, invalid magnitudes, stations with few picks.

### Stage 2: Monte Carlo VELEST and analysis
- **Generating trial models:**
  - Configurable layer boundaries, Vp ranges per layer, and a Vp/Vs range or a separate S model.
  - Monotonic constraint on or off; sampling by Latin hypercube or uniform random.
- **Running:**
  - Run trials in parallel, each in its own folder.
  - Damping in stages (Kissling approach), then a second round seeded from the best models.
  - Optionally randomise starting hypocentres to test for hypocentre bias.
- **Analysis:**
  - Parse `.OUT` files: RMS per iteration, final model, station delays, hypocentre shifts, convergence.
  - Choose the final 1D model by minimum RMS, the median of the best N%, or minimum layer variance. Report spread as uncertainty.
- **Stability tests:** shift hypocentres ±x km with the final model fixed, swap the reference station, and run with low-velocity zones on and off.
- **Figures:**
  - All trial models (start and final) coloured by final RMS, with the selected model in bold.
  - Density of the best N% of models.
  - Per-layer velocity histograms.
  - RMS vs each layer's velocity.
  - RMS vs iteration for every run.
  - Map of runs that failed or diverged.
  - Station delays on a map, and delay vs elevation.
  - Mean and spread of hypocentre shifts.
  - Hypocentre-perturbation test results.
  - Residuals per station and per event.
  - Vp/Vs results.
- **Checks:** convergence rate, best models on a bound of the search range, layers whose velocity never moves from the start (unresolved), spread of the reference-station delay.

### Stage 3: 3D starting model builder
There are four ways to build it, and the 1D background always fills any area the user model doesn't cover:

| `model3d.source` | Description |
|---|---|
| `velest_1d` | 1D model extended to 3D, layered or smoothed into gradients |
| `user_grid` | NetCDF/CSV x, y, z, Vp[, Vs] in any CRS, resampled with pyproj/xarray |
| `geology` | Horizons (XYZ, GeoTIFF, GOCAD TSurf). Each unit gets a velocity rule: constant, gradient or compaction curve. Units stacked by stratigraphic order, with a check for negative thickness |
| `wells` | LAS sonic (DT/DTS) logs or CSV velocity vs depth. Converted to velocity, smoothed, upscaled with Backus averaging, then kriging or inverse-distance between wells |
| `geology+wells` | Well logs set the velocity of each geological unit |

- Blended into the background over `blend_taper_km`, then optionally smoothed.
- **FMTOMO files written:** `vgrids.in`, `interfaces.in` (surface from DEM or station elevations; sources must be below it), `propgrid.in`, `sources.in`, `receivers.in`, `otimes.dat`.
- The same tool builds the "input velocity model" used in resolution tests.
- **Figures:**
  - Depth slices and profiles of the starting model.
  - Difference from the 1D model.
  - 1D profiles at each well with logs overlaid.
  - Horizon surfaces and the unit-ID volume.
  - Map of where the user model applies and where the taper is.
  - Histogram of velocities.
  - Gradient-magnitude map (sharp jumps break FMTOMO's eikonal solver).
  - Forward-grid refinement and surface interface vs stations and events.
  - Starting residuals for the 1D model vs the user model.
- **Checks:** unphysical velocities, events above the surface interface, large velocity jumps, user model fitting the data worse than the 1D model.

### Stage 4: FMTOMO 3D inversion and trade-off curves
- Run `tomo3d`, or with Extras run `tomo3di` / `gridloc` for nonlinear hypocentre relocation.
- P only, P plus S, or a joint Vp/Vs inversion (`vpvsinv`).
- **Trade-off curves:**
  - Parallel sweep over damping ε and smoothing η.
  - Plot data variance against model variance (damping) and model roughness (smoothing).
  - Pick the L-curve corner automatically by maximum curvature, with manual override. Sweep ε then η, or the full grid.
- **Figures:**
  - L-curves with the chosen point marked, plus a heatmap for a full grid.
  - Misfit vs iteration for every sweep run.
  - Small slices from runs either side of the chosen point.
  - Residual histograms per iteration.
  - Per-station mean residual maps.
  - Hypocentre shifts per iteration.
  - Number of rays and sources lost per iteration.
- **Checks:** misfit stops improving, misfit goes up, L-curve corner hard to find, velocities clipped at limits.

### Stage 5: Analysing the outputs
- **Figures:**
  - Depth slices and profiles: absolute velocity, percentage change from 1D and from the starting model, Vp/Vs, all masked by ray coverage.
  - Seismicity before and after relocation.
  - Ray-hit and derivative-weight-sum (DWS) maps.
  - Ray paths for selected events.
  - Final vs initial residuals (by station, event, phase, and against distance and azimuth).
  - Variance-reduction summary.
  - Comparison of the final model with the user geology/well model at each well.
- **Exports:** NetCDF, VTK (ParaView), CSV.

### Stage 6: Resolution tests
- **Checkerboard:** several cell sizes and amplitudes, offset by half a cell.
- **Spike tests.**
- **Input-model tests:** recover the final 3D model (restoring test), the user geology/well model, or custom anomalies.
- **Method:** real source-receiver geometry, Gaussian noise matched to pick uncertainty, and the same ε, η, relocation and stopping settings as the real inversion. P and S tested separately.
- **Figures:**
  - Input and recovered models side by side.
  - Recovery maps: correlation and amplitude ratio.
  - Smearing direction.
  - Synthetic L-curve vs the real one.
  - Noise realisations.
  - Combined "resolvable" mask, which is sent back to mask the Stage 5 figures.

### Stage 7: Validation and reporting
- Validation events relocated with 1D vs 3D: RMS, hypocentre changes, clustering.
- Optional bootstrap or jackknife over events, with standard-deviation slices.
- One project report linking all stage reports.

---

## 6. Command line

```
osv init | doctor | prep | velest-mc | velest-analyse | build-model |
    tradeoff | invert | analyse | resolution | report | run --all
```
- `--from` / `--to` / `--force` re-run stages.
- `osv plot <stage>` regenerates figures without re-running the stage.
- `osv config show` prints the settings in use.

---

## 7. Milestones

| Milestone | Delivers | Checked against |
|---|---|---|
| M1 | Installer, `osv doctor`, CMake build of all Fortran, config schema and template, cnv/sta/mod I/O | Round-trip tests; FMTOMO example1 |
| M2 | StationXML/QuakeML prep, QC and figures | Test dataset (below) |
| M3 | Parallel MC VELEST, analysis and figures | Compare with previous Ngatamariki MC runs |
| M4 | 3D builder: 1D extrusion, user grid, geology horizons, wells; figures | Synthetic 1D gives zero residuals |
| M5 | FMTOMO I/O, inversion, trade-off sweeps, figures | Synthetic known-model recovery |
| M6 | Resolution suite and coverage masks | Checkerboard on test geometry |
| M7 | Analysis, validation, HTML reports, Docker image, docs and example project | Full end-to-end run |

---

## 8. Test dataset

**Catalogue:** `manual_only_catalog.xml`, Rotokawa–Ngatamariki, Apr–Dec 2017.

| | |
|---|---|
| Events | 79, one NLLoc origin each (`RK-NM_2017-18_nlloc_time`) |
| Picks | 2231 P, 709 S. Per event: 7–53 P (median 27) and 0–23 S (median 8) |
| Event locations | Depth 0.2–5.6 km, azimuthal gap 32–158°, RMS 0.05–0.19 s |
| Stations | 63, all `NZ` network codes of 4 characters |
| StationXML | All present in `/home/steve/seismic/Station_Files/`; every pick falls inside its station's epochs |
| Borehole sensors | NS12 at 514 m depth, NS13 and NS14 at 250 m, THQ2 at 80 m |
| Station elevations | 250–590 m |

**Problems Stage 1 must handle:**
1. **Corrupted arrivals.** In every origin, one S arrival is repeated about 37 times (2880 duplicates), and 2861 picks aren't referenced by any arrival. Build travel times from the picks (`use_arrivals: false`).
2. **Invalid magnitudes.** All are ML −6.19, so skip magnitude filtering and plots automatically.
3. **`evaluation_mode` unset.** Default to keeping all picks.
4. **Uncertainties stored as lower/upper bounds** (P mostly 0.01–0.05 s, S 0.03–0.10 s; 546 asymmetric), plus a `wgt = 1` / `wgt = 0.8` comment. Handled by `weight_source`.
5. **S picks on many channel names** (EHT, EH1, EHE, EHN, EHR, EHQ, EHL, EH2). Accept them by phase label.

**What the dataset means for the inversion:**
- About 2900 picks for roughly 316 hypocentre parameters plus velocity nodes. Use coarse nodes (about 0.5–1 km) and checkerboard cells of 1.5–3 km, and rely heavily on resolution tests.
- S coverage is thinner. Default to P first, and add S or Vp/Vs after checking the Stage 1 Wadati and S coverage figures.

---

## 9. Defaults assumed unless changed
- **Fortran edits:** small patches are acceptable, each with a short reason.
- **S waves:** P first by default, with P plus S and joint Vp/Vs as options.
- **Plotting:** matplotlib by default, PyGMT optional.
- **Geology formats:** XYZ, GeoTIFF and GOCAD TSurf. Others (Leapfrog, Petrel exports) can be added.
