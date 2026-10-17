# Troubleshooting and FAQ

Start with `osv doctor --config config.yaml`: it checks the tools, the smoke tests, free disk
space, write permission on the work directory, and whether your data fit the VELEST array limits.
Then open `<workdir>/index.html` (the project report), which lists every QC failure and warning
by stage. A stage that fails writes the traceback to its `manifest.json` (`error`) and to
`<stage dir>/logs/`.

## Installation and environment

**`velest` or `fm3d` not found.** Activate the environment (`conda activate osv`) or set
`OSV_TOOLS_DIR=<env>/bin/osv-tools`; rebuild with `./install.sh --clean`. See
[installation](install.md#troubleshooting).

**Disk space.** The installer needs about 6 GB. A project is dominated by the FMTOMO runs:
the propagation grid of the rk-nm-2017 example has 2.3 M nodes (about 0.5 GB of memory per
`fm3d` process), and the trade-off sweep and resolution tests each run many inversions in their
own directories. Expect several GB to tens of GB for a project with a fine grid. `osv doctor
--config config.yaml` reports free space on the work directory and `TMPDIR`; set
`project.workdir` and `TMPDIR` to a large disk, and lower `project.n_jobs` if memory runs out
(each parallel FMTOMO job holds its own propagation grid).

**`PROJ_DATA` / `proj.db` errors (EPSG lookups fail, "no database context").** An older PROJ
installation (conda, system or GDAL) shadows the one bundled with pyproj. `osv` overrides a
stale `PROJ_DATA` automatically unless `OSV_KEEP_PROJ_DATA=1` is set. If you still see errors,
`unset PROJ_DATA PROJ_LIB`, or point `PROJ_DATA` at the directory printed by `osv doctor`.

## Data preparation

**"duplicated arrivals", "picks not referenced by any arrival", "no evaluation_mode".** Common
catalogue quirks. They are handled (duplicates dropped, picks used directly because
`inputs.use_arrivals` is false). They only need action if you expected the arrivals to be used.

**Borehole stations.** Channels with a sensor depth greater than 0 get their elevation from the
surface elevation minus the depth. `coordinates.borehole_sensors` chooses between using the
depth from StationXML (default) and ignoring it. Warnings such as "channels have coordinates
inconsistent with their station" mean the channel and station coordinates in the StationXML
disagree by more than 10 m; fix the StationXML if the sensor depth matters. VELEST receives the
elevation as an integer in metres and, as the reference station for the delays, prefers a
well-recorded station (not a borehole one).

**Station codes longer than 4 characters.** VELEST needs at most 4. `inputs.station_aliases:
auto` generates the aliases (`aliases.csv` in `01_prep`); supply a mapping or CSV to control them.

**Hypocentre depth datum.** If the catalogue depths are below sea level but your stations are at
300-600 m elevation, check `coordinates.depth_datum` (`sea_level`, `surface`, `reference`). A wrong
datum shifts all events vertically and shows up as large systematic travel-time residuals.

## VELEST (Stage 2)

**"all N VELEST runs failed".** Open `02_velest_mc/runs/<run>/` and read the `*.log` and
`status.json`. Typical causes: the `velest` binary cannot start (run `osv doctor`), the data
exceed the compiled array limits (`osv doctor --config ...` shows `project:ieq/ist/maxobs`,
rebuild with `./install.sh --velest-ieq ...`), or invalid input files. Fix the cause and run
`osv velest-mc --force`.

**VELEST divergence (`0% of runs converged`, `1 of 201 runs diverged`).** VELEST runs that stop
after "four backups" (RMS rose) are marked diverged. Occasional divergence from random starts is
normal; the analysis only ranks runs that did not fail. If most diverge: raise the damping
(`velest.vdamp`, `velest.stages`), narrow `velest.vp_bounds`, check the layering and the pick
outliers, and read [Stage 2](stage2_velest.md#why-runs-diverged-80-of-201). A final-model
warning "repeating VELEST on the final model moved a layer's Vp by ..." means the model is not at a
fixed point; increase `velest.iterate_until.max_cycles` or accept the ensemble spread.

**Layer 0 ends at the lower bound.** The search range is too narrow for the near-surface layer.
Lower the first entry of `velest.vp_bounds`, or accept it if the physical velocity is below the
range. **Low-velocity zone warnings** are informational unless `lowveloclay` matters for your
geology (see the Stage 2 doc).

## Vp/Vs

**Which Vp/Vs is used?** Stage 1 measures Vp/Vs from S-P Wadati pairs (rk-nm-2017: 1.716, 95% CI
1.703-1.729) and Stage 2 uses it (`velest.vpvs`). The VELEST Vp/Vs grid search
(`velest.vpvs_grid`) may give a slightly different RMS minimum (1.68 there); the S times are not
inverted in FMTOMO unless `fmtomo.phases` includes `S` or `vpvs`. "Per-event Vp/Vs inconsistent
with the median" usually flags events with few S picks or mis-picked S phases; remove them with
`data_selection.min_s` or tighter outlier thresholds. If the Wadati residual sigma is large (above
about 0.3 s), review the S picks before trusting S results.

## 3-D model and inversion

**Node spacing exceeds one third of the station spacing.** The data cannot constrain a grid that
fine. Increase `model3d.grid.dx_km`/`dy_km`/`dz_km` or accept that the model is mostly smoothing-controlled.

**Vp outside bounds in the starting model.** The near-surface 1-D layer is slower than
`model3d.checks.vp_min_km_s` (1.0 km/s; the example's shallowest layer is 0.7 km/s). Check the VELEST model, or lower the limit if the slow layer is real.

**L-curve corner at the edge of the sweep / chi-squared far from 1 / unconverged sweep.** Widen
the swept range, raise `fmtomo.tradeoff.iterations`, or set `damping`/`smoothing` manually. A
reduced chi-squared of about 3 (as in the example) means the stated pick uncertainties are too small
or the 3-D model cannot explain the picks; interpret the model as a smooth correction to the 1-D
model.

### Borehole stations and station delays

**Why are the VELEST station delays subtracted before the 3-D inversion, and should the
borehole stations be excluded?** FMTOMO cannot invert for station terms. The default
`model3d.apply_velest_station_delays: true` subtracts the Stage 2 delays from the observed
times, so the station-scale near-surface signal, which a 0.5 km grid cannot resolve, does not
have to be explained by the model. On rk-nm-2017 the station mean residuals correlate with the
VELEST delays at r = 0.92. The stage 3 forward check may still warn about a mean residual and a
weak correlation (r = 0.15) of the starting model's residuals with the delays; that compares
different quantities and does not mean the delays should be dropped.

The example was run three times (before the origin-time fix of osv-jac.13, so the absolute numbers differ from the
[tutorial](tutorial.md), which was rerun afterwards: P RMS 56.3 to 32.6 ms, variance reduction 66.7%; the
comparison between the columns is what matters) with the same Stage 1-2 and 5 results (the scripts and logs are
in `build/ab/`; `A` is the default):

| | A: delays applied | B: delays off | C: `velest_station_delays_exclude: borehole` |
|---|---|---|---|
| Final P RMS (start 131-136 ms) | 56.7 ms | 49.5 ms | 53.9 ms |
| Variance reduction | 81.9% | 86.7% | 83.2% |
| dVp std in covered nodes | 2.7% | 8.3% | 3.1% |
| Largest |dVp| | 14.5% | 49.9% | 15.4% |
| Std of station mean residuals | 29.9 ms | 40.9 ms | 29.8 ms |
| Validation median RMS 1-D / 3-D | 20.9 / 15.7 ms | 20.9 / 16.1 ms | 20.9 / 16.4 ms |
| Validation events better in 3-D | 83% | 58% | 83% |

Without the delays (B) the data are fitted better, but only because the model absorbs the
delays into the shallowest nodes: perturbations of up to 50% appear around single stations
(for example -11% at NS14 in the top layer) and the validation gain over the 1-D model shrinks.
That is a better fit by the wrong mechanism, so B is not recommended. Excluding the borehole
stations' delays (C) lowered their residuals a little but moved them into +4 to +10% near-surface
anomalies (NS12, NS13) without improving validation (3-D median 16.4 ms against 15.7 ms). The
delays are relative to the reference station and may partly double-count the sensor depth, but
removing them does not fix that; it only moves the signal into the model.
Keep the delays applied and, if a borehole station looks odd, check its StationXML depth and
elevation (see [borehole stations](#data-preparation)) rather than switching the delay off. To repeat the
comparison, set `model3d.apply_velest_station_delays` or `model3d.velest_station_delays_exclude`
in a copy of the config with a separate `project.workdir`, link the Stage 1-2 and 5 directories and run
`osv build-model`, `osv invert`, `osv analyse` and `osv validation`.

## Poor resolution

"Only 2.6% of the volume has DWS >= 0.1" and "50% of the covered volume is resolved at some tested scale" (with only 26% at 2 km) (the
example) indicate sparse ray coverage. Do not interpret features outside the resolvable mask,
which the analysis figures draw once `osv resolution` has run. To improve: use more events or
stations, relax `data_selection` thresholds (carefully), use a coarser grid
(`model3d.grid.dx_km`, `dy_km`, `dz_km`), stronger smoothing, or test larger checkerboard sizes
(`resolution.checkerboard.cell_km`). Compare the result with the 1-D relocation in Stage 7: if the
3-D model does not reduce the validation RMS (`fraction_3d_better`), the data do not support the
added 3-D structure.

## Re-running and stale stages

Edit `config.yaml` and run `osv run`: stages whose config sections, inputs, upstream runs or tool
binaries changed are re-run, all others are skipped. To force a stage use `--force`; to redo
figures only use `osv plot STAGE`. Each executable run lives in its own directory, so you can
inspect the exact control files and logs of any run.
