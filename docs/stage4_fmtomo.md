# Stage 4: trade-off sweeps and the FMTOMO inversion (`osv tradeoff`, `osv invert`)

Chooses the damping and smoothing with L-curves, then runs the final 3-D inversion with `fm3d`,
`frechgen`, `invert3d` and `residuals` (`osv.fmtomo`). Needs the `fm3d` toolchain (`osv doctor`) and a
completed Stage 3 (`osv build-model`).

```bash
osv tradeoff -c config.yaml            # -> <workdir>/05_tradeoff/
osv invert   -c config.yaml            # -> <workdir>/06_invert/
osv run --to invert -c config.yaml
osv plot tradeoff -c config.yaml       # regenerate figures from the written outputs
```

Open `report.html` of each stage first.

## Config keys

- `fmtomo.phases`: `[P]`, or `[P, S]` (separate runs, one velocity type per FMTOMO run; Stage 3 must have
  written S data).
- `fmtomo.relocate`: `none`, `linear` (hypocentres inverted with the velocities, needs
  `fmtomo.source_uncertainty_km`, the a priori hypocentre uncertainty written to `sources.in`) or
  `nonlinear` (Extras `gridloc`; stations become the FMTOMO sources and events the receivers, see below).
- `fmtomo.iterations`, `damping`, `smoothing` (used when the trade-off is manual or skipped),
  `subspace_dimension`, `min_velocity_km_s`, `save_rays`.
- `fmtomo.tradeoff.damping`, `smoothing` (swept values), `mode` (`sequential`: damping at the configured
  smoothing, then smoothing at the chosen damping; `full_grid`), `pick` (`max_curvature`, `manual`),
  `iterations` (per sweep run, fewer than the final inversion), `max_concurrent` (sweep runs at the same
  time; each gets `project.n_jobs / max_concurrent` `fm3d` processes).
- `plotting.dws_mask_threshold`: absolute derivative-weight-sum threshold of the coverage mask;
  `plotting.depth_slices_km`, `plotting.profiles` for the model figures.
- `project.n_jobs`: parallel `fm3d` processes.

## Steps

`tradeoff`: for each phase run the sweeps (`osv.fmtomo.tradeoff.run_tradeoff`) on the Stage 3 inputs, pick the
corners (maximum Menger curvature in log-log space), write the results, figures and checks. Run directories
are `05_tradeoff/<phase>/d<damping>_s<smoothing>/`; a finished run (with `summary.json`) is reused when the
stage is repeated, so an interrupted sweep resumes. Heavy files (`frechet.dat`, `rays.dat`, arrival tables)
are deleted after the metrics are extracted. Sweep runs do not save rays.

`invert`: run the iteration loop with the chosen values, then (before the heavy files are deleted)
compute the derivative weight sum (DWS) of the final model from the `frechet.dat` of the last forward run, build
the residual and relocation tables, export the model and make the figures.

Geometry: `none` and `linear` use the Stage 3 files unchanged (events are sources, stations receivers). For
`nonlinear` the roles swap (`osv.fmtomo.stage.to_nonlinear_geometry`): stations are sources placed just below the
surface interface (one direct path each), events are receivers; the `gridloc` search writes the time fields of all
stations over the whole propagation grid (`arrtimes.dat`), which is large for fine grids: check the disk first.

The disk guard (`osv.fmtomo.stage.MIN_FREE_GB`, 3 GB) stops the stages cleanly before a run if too little
space is left; finished runs are kept.

## Outputs of `invert` (names used by Stages 5 and 6)

| file | content |
| --- | --- |
| `final_model.nc` | dataset on the **full** FMTOMO grid (cushion nodes included, `attrs["n_cushion"]`): `vp` (`vs` if S was inverted), `vp_start`, `vp_bg` (and the `vs` equivalents), `dws` (`dws_vs`), `mask` (`mask_vs`; int8, 1 where `dws >= plotting.dws_mask_threshold`). DWS is 0 in the cushion |
| `final_vgrids.in`, `final_sources.in` | final model and hypocentres in FMTOMO format (`_S` suffix for S). For `relocate: nonlinear` the hypocentres are the FMTOMO *receivers* |
| `residuals_final.parquet` | per-pick residuals (observed - calculated) of the final model with event, station, distance, azimuth, `phase` |
| `relocation.parquet` | hypocentre shifts against the Stage 3 positions (`dx_km`, `dy_km`, `dz_km`, `d3_km`, `dt_s`), `phase` |
| `inversion_iterations.csv` | per phase and iteration: `rms_ms`, `variance_s2`, `chi2` (reduced), `model_variance`, roughness, `n_valid`, `n_lost`, `vel_min`, `vel_max`, `frac_at_min_velocity`, `mean_hypo_shift_km`, ... |
| `chosen_regularisation.json` | damping, smoothing and where they came from |
| `exports/` | NetCDF, VTK and CSV of the model with masks and the relocated hypocentres |
| `run_<phase>/` | the run directory without heavy files (`iterations/NN/` keeps model, sources, calculated times and logs of every iteration) |

Outputs of `tradeoff`: `tradeoff.json` (chosen values per phase, full results), `tradeoff.csv` (all runs).

## Figures

Trade-off: damping and smoothing L-curves (all points labelled with the parameter value, the chosen point
highlighted and the manual one if different, curvature in a second panel), heatmaps of the grid for
`mode: full_grid`, misfit against iteration for every sweep run, depth slices of the model change of the runs
either side of the chosen damping and smoothing.

Inversion: misfit against iteration, residual histograms of every iteration, station mean residual map
(starting against final), hypocentre shifts per iteration, picks lost per iteration, velocity minimum / maximum
and the fraction of nodes at the minimum velocity, plus the standard residual set
(`osv.analysis.plots_residuals`) and model figures masked by the DWS (`osv.analysis.plots_model`).

## QC checks and what to do

- `lcurve_corner`: no convex L-curve corner; the configured value is used. Widen or refine the sweep, or set
  `tradeoff.pick: manual` and choose by eye.
- `lcurve_at_edge`: the corner is the second or second-last swept value; the real corner may lie outside.
- `sweep_failed`, `no_convergence`: failed runs, or runs whose misfit still changed by more than 2 % in the last
  iteration (the L-curve is of unconverged models; raise `tradeoff.iterations`).
- `misfit_not_decreasing` (fail when the final RMS is not below the starting RMS): damping too strong, or data
  and geometry inconsistent.
- `misfit_increasing`: the RMS rose after iteration k; the linearised step overshoots, raise damping or
  smoothing.
- `min_velocity_clipped`: nodes sit at `min_velocity_km_s`; the model is poorly constrained there.
- `rays_lost`: more than 5 % of the picks lost their ray between the first and last iteration (rays leave the
  model or sources moved out of it).
- `chi2_far_from_one`: reduced chi-squared outside 0.5..2. Above 2 the uncertainties are too small (or the
  model cannot explain the data); below 0.5 they are too large. Model uncertainties derived with the pick
  uncertainties should be rescaled by `sqrt(chi2)`.
