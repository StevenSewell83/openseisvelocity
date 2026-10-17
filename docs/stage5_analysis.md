# Stage 5: analysis of the inversion (`osv analyse`)

Turns the result of the final inversion into figures, exports, summary numbers and QC checks.
Needs completed `invert`, `build_model`, `prep` and `velest_analyse`; no binaries are run.

```bash
osv analyse -c config.yaml        # -> <workdir>/07_analyse/
osv plot analyse -c config.yaml   # regenerate figures and exports without re-running
```

Open `report.html` first.

## Using the resolution stage (Stage 6)

`resolution` comes after `analyse` in the pipeline and is **not** required. If
`<workdir>/08_resolution/` is complete when `analyse` runs, its `resolvable_mask.nc` is used. If
not, the figures are masked by ray coverage only (check `resolution_not_applied`). Run
`osv resolution` and then `osv plot analyse` to add the `*_resolvable` figures (masked by DWS AND
resolvable) and the `resolvable_*` masks in `exports/`. A failed or incomplete resolution stage is
ignored.

## Config keys

- `plotting.dws_mask_threshold`: coverage mask is `dws >= threshold` (P: `dws`, S: `dws_vs`).
- `plotting.depth_slices_km`, `plotting.profiles`, `plotting.format`, `plotting.dpi`.
- `model3d.source` / `model3d.user_model`: any source other than `velest_1d` adds the comparison with
  the user model. `fmtomo.relocate` tells the residual figures which geometry the run used.

QC thresholds (variance reduction, "moved" distance, ...) are the `Thresholds` dataclass in
`osv.analysis.stage` (module defaults, not yet config keys).

## Inputs read

From `06_invert/`: `final_model.nc` (needed), and optional `relocation.parquet`,
`residuals_final.parquet`, `inversion_iterations.csv`, `run_<phase>/` (with `iterations/NN` and
`mapping.csv`, for the residual figures). From `04_build_model/`: `starting_model.nc`,
`fmtomo/stations.csv`, wells (re-read for the well comparison). From `03_velest_analyse/`:
`final_model.csv` (1-D reference when the model has no `vp_bg`). Missing optional files are logged
and become `info`/`warn` checks; the stage still completes.

## Outputs (`07_analyse/`)

- `summary.json`: all numbers below and the checks. `perturbation_stats.csv`.
- `figures/`: model slices/sections (absolute, % from 1-D, % from start, Vp/Vs, masked),
  starting vs final comparisons, residual histograms and against distance/azimuth, station map,
  per-event RMS, variance reduction, hypocentre shifts and seismicity before/after, and
  `user_*` figures (well profiles, unit boxplots, difference slices and sections) for a user model.
  The user model is the Stage 3 starting model (with `unit_id`) and, for well sources, the wells.
- `exports/`: `model.nc`, `model.vtk`/`.vts`, `model.csv`, hypocentres and stations (VTK, CSV);
  the model carries `covered_vp/vs`, `dws`, `dws_vs` and, if applied, `resolvable_vp/vs`.

## Summary numbers

`variance_reduction_pct_<phase>`, `rms_start_ms_<phase>`, `rms_final_ms_<phase>` (FMTOMO
iterations), `final_rms_ms_<phase>`, `n_picks_<phase>` (final residuals), `covered_fraction_vp/vs`
(volume fraction, nodes weighted by cos latitude), `resolvable_fraction_*`, `perturbation`
(mean, std, 5th/95th percentile, max change in % inside the coverage, against the start and the 1-D
background), `relocation_<phase>` (median horizontal, depth and 3-D shifts, fraction moved more
than 1 km).

## QC checks and what to do

- `variance_increased` (fail), `poor_variance_reduction` (warn, below 10%): the final model fits no
  better than the start. Check regularisation (`osv tradeoff`), data weights and the starting model.
- `change_outside_coverage`: more than 5% of the uncovered nodes changed by over 10%: the damping or
  smoothing is too weak where no rays constrain the model; raise it or mask these nodes.
- `small_coverage`: under 5% of the volume has DWS above the threshold: lower
  `plotting.dws_mask_threshold` or check the ray geometry.
- `many_events_moved`: over 25% of the events moved by more than 1 km: check `fmtomo.source_damping_*`.
- `no_coverage`, `no_relocation`, `no_misfit`, `no_residual_figures`, `no_starting_model`,
  `figures_failed`, `user_comparison_failed`, `export_failed`: missing or failing optional parts.
- `resolution_not_applied` (info): see above.
