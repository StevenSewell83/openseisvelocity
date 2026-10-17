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

**VELEST divergence (`0% of runs converged`, `80 of 201 runs diverged`).** VELEST runs that stop
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
reduced chi-squared of 13-16 (as in the example) means the stated pick uncertainties are too small
or the 3-D model cannot explain the picks; interpret the model as a smooth correction to the 1-D
model.

## Poor resolution

"Only 2.6% of the volume has DWS >= 0.1" and "10% of the covered volume is resolved" (the
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
