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

- `fmtomo.phases`: `[P]`; `[P, S]` (one velocity type per FMTOMO run; Stage 3 must have written S data;
  how the two runs share hypocentres is set by `fmtomo.joint.relocation`); or `[vpvs]` (P and S inversions
  followed by the direct Vp/Vs inversion). See "Joint P+S and Vp/Vs" below.
- `fmtomo.joint.relocation` (`joint`, `p_fixed_for_s`, `independent`; default `independent` for `[P, S]`,
  `joint` for `[vpvs]`), `fmtomo.joint.s_weight`, `sp_weight`, `relocation_based_on` (`gridlocj`),
  `vpvs_max_depth_km`, `sparse_fraction` (Vp/Vs inversion).
- `fmtomo.relocate`: `none`, `linear` (hypocentres inverted with the velocities, needs
  `fmtomo.source_uncertainty_km`, the a priori hypocentre uncertainty written to `sources.in`) or
  `nonlinear` (Extras `gridloc`; stations become the FMTOMO sources and events the receivers, see below).
- `fmtomo.iterations`, `damping`, `smoothing` (used when the trade-off is manual or skipped),
  `subspace_dimension`, `min_velocity_km_s`, `save_rays`.
- `fmtomo.tradeoff.damping`, `smoothing` (swept values), `mode` (`sequential`: damping at the configured
  smoothing, then smoothing at the chosen damping; `full_grid`), `pick` (`max_curvature`, `manual`),
  `iterations` (per sweep run, fewer than the final inversion), `max_concurrent` (sweep runs at the same
  time; each gets `project.n_jobs / max_concurrent` `fm3d` processes).
- `fmtomo.station_terms`: `none`; `preapply` (fixed terms supplied to `assemble_run_dir`); `iterate`
  (P and S station terms estimated in the 3-D model between the iterations, see below), with
  `fmtomo.station_term_iteration.every`, `start_iteration`, `relaxation`, `min_picks`.
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

## Joint P+S and Vp/Vs (experimental, `package: extras`)

Needs `relocate: nonlinear` and Stage 3 data of both phases (`osv build-model` writes S when `fmtomo.phases`
contains `S` or `vpvs`). The Extras programs `jointimes`, `gridlocj`, `mkvpvs`, `frvpvs`, `vpvsinv` and
`vpvsresids` must be built.

| `phases` / `joint.relocation` | what runs |
| --- | --- |
| `[P]` | one P inversion (`run_P/`) |
| `[P, S]`, `independent` | two separate inversions; each relocates the events on its own, so the P and S hypocentres of one event differ (`run_P/`, `run_S/`) |
| `[P, S]`, `p_fixed_for_s` | P relocates; the P hypocentres and origin times are copied to the S run (S picks of events with a P pick), which keeps them fixed |
| `[P, S]`, `joint` | `tomo3dj` loop (`osv.fmtomo.joint.run_joint_inversion`): `gridlocj` relocates every event with the P and S time fields together, then both velocity inversions run, each iteration |
| `[vpvs]` | the P and S inversions of `joint.relocation` (default `joint`), then the linear Vp/Vs inversion from S-P times (`run_vpvs_inversion`: `mkvpvs`, `frvpvs`, `vpvsinv`, `vpvsresids`) on the final models, `fmtomo.iterations` iterations, `fmtomo.damping` and `smoothing` as regularisation |

The coupled runs live in `06_invert/run_joint/` (`src.dat`, `jointimes.in`, `gridlocj.in`, `vpvs/`); when
they are done `invert_p/` and `invert_s/` are moved to `run_P/` and `run_S/` so the outputs below, the plots
and Stages 5-7 treat them like separate runs. Details of the wiring (`osv.fmtomo.stage.invert_coupled`):

* P and S events are matched by `event_id` (Stage 3 numbers the events of each phase separately); the
  fourth column of the station `sources.in` is the Extras receiver id (the P station index; S stations of
  the same name share it, S-only stations get a number above the P count).
* The sweep (`osv tradeoff`) is not coupled: it sweeps damping and smoothing of P and of S separately
  (independent runs, no joint relocation) and the final run uses each phase's own pick. Damping is a
  property of the velocity inversion, the coupling mainly changes the hypocentres. For `[vpvs]` the Vp/Vs
  step is not swept.
* Both phases must have the same propagation grid (Stage 3 guarantees it), at most 300 stations, 5000
  events and 30 characters in file names (limits of the Extras programs).
* `vpvs` needs `rays.dat` of the S run, so `save_rays` is forced on for it (kept files are still pruned by
  the usual size rule).
* Extra outputs: `joint_iterations.csv` (P and S misfit per iteration, also after relocation),
  `vpvs_iterations.csv` (S-P misfit and Vp/Vs range), `final_vgrids_vpvs.in` (direct Vp/Vs grid),
  `final_vgridvpvs.in` (node-wise Vp / Vs of the two solutions, for comparison), `coupled_mode` and the
  Vp/Vs damping in `chosen_regularisation.json`. `final_model.nc` gains `vpvs` (the directly inverted ratio,
  same grid as `vp`); Stage 5 figures and exports show `vp / vs` of the two models, as for separate runs.
* QC: `vpvs_failed`, `vpvs_misfit_not_decreasing`, `vpvs_note` besides the usual per-phase checks.

The disk guard (`osv.fmtomo.stage.MIN_FREE_GB`, 3 GB) stops the stages cleanly before a run if too little
space is left; finished runs are kept.

## Iterated station terms (`fmtomo.station_terms: iterate`, experimental)

FMTOMO has no station parameters. With `iterate`, after each iteration (from `start_iteration`,
every `every`-th, never after the last) each station's mean residual is computed per phase, after
removing each event's mean (the origin-time trade-off); it is multiplied by `relaxation`, centred
(pick-weighted mean zero, since a common constant is an origin time) and subtracted from that
station's observed times in `otimes.dat` and, for the nonlinear geometry, in the `src.dat` from
which `gridloc`/`gridlocj` rebuild `otimes.dat` at every relocation. Stations with fewer than
`min_picks` picks of a phase get no term. The trade-off sweep, the resolution tests (same procedure
on the synthetic data) and the validation (final P terms subtracted from the held-out events) use
the same terms. Combine it with `model3d.apply_velest_station_delays: false` and strongly damped
shallow nodes (`model3d.extrusion.uncertainty.mode: by_depth`, e.g. `[[-1.0, 0.05], [0.5, 0.3]]`) so
that the terms, not the near-surface velocities, take the station statics. Supported for single-phase
runs and `joint.relocation: joint`. S terms converge more slowly than P (they trade off with Vs and
the shared relocation); check `station_terms_last_step_ms_S` (QC `station_terms_not_converged`).

**Caveat (MacBan 2018-2019, 16 stations, compact cluster):** on real data the undamped alternation
does not converge. A station constant trades off with the velocities along that station's ray
bundle, the hypocentres and the origin times, and `invert3d` damps the velocities toward the
reference model at every iteration, so the terms keep taking over station-shaped signal: over 30
iterations the P terms drifted by about 4.5 ms per iteration (from 55 % to 98 % of the VELEST delays,
still growing), S terms at stations with 10-24 picks reached 0.3-0.5 s, and the misfit rose after
iteration ~12. A prior on the terms cannot fix this (the station means are determined to a few ms
by hundreds of picks). The result therefore depends on the number of iterations; terms estimated
with the velocities fixed are the robust alternative.

## Outputs of `invert` (names used by Stages 5 and 6)

| file | content |
| --- | --- |
| `final_model.nc` | dataset on the **full** FMTOMO grid (cushion nodes included, `attrs["n_cushion"]`): `vp` (`vs` if S was inverted, `vpvs` for `phases: [vpvs]`), `vp_start`, `vp_bg` (and the `vs` equivalents), `dws` (`dws_vs`), `mask` (`mask_vs`; int8, 1 where `dws >= plotting.dws_mask_threshold`). DWS is 0 in the cushion |
| `final_vgrids.in`, `final_sources.in` | final model and hypocentres in FMTOMO format (`_S` suffix for S). For `relocate: nonlinear` the hypocentres are the FMTOMO *receivers* |
| `residuals_final.parquet` | per-pick residuals (observed - calculated) of the final model with event, station, distance, azimuth, `phase` |
| `relocation.parquet` | hypocentre shifts against the Stage 3 positions (`dx_km`, `dy_km`, `dz_km`, `d3_km`, `dt_s`), `phase` |
| `inversion_iterations.csv` | per phase and iteration: `rms_ms`, `variance_s2`, `chi2` (reduced), `model_variance`, roughness, `n_valid`, `n_lost`, `vel_min`, `vel_max`, `frac_at_min_velocity`, `mean_hypo_shift_km`, ... |
| `chosen_regularisation.json` | damping, smoothing and where they came from |
| `station_terms.csv`, `station_terms_history.csv` | with `fmtomo.station_terms: iterate`: final term per `phase` and `station` (`term_s`, subtracted from the observed times; `n` picks) and every update (`iter`, `delta_s`) |
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
