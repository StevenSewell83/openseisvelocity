# Stage 2: Monte Carlo VELEST (`osv velest-mc`, `osv velest-analyse`)

Finds a minimum 1-D velocity model by running VELEST from many random starting models, selects the
best, and tests its stability. Needs the `velest` binary (`osv doctor`) and a completed Stage 1.

```bash
osv velest-mc -c config.yaml        # Monte Carlo runs -> <workdir>/02_velest_mc/
osv velest-analyse -c config.yaml   # analysis, final model, stability -> <workdir>/03_velest_analyse/
osv run --to velest_analyse -c config.yaml   # prep + both, skipping up-to-date stages
osv plot velest_analyse -c config.yaml       # regenerate figures from the written outputs
```

Open `report.html` in each stage directory first.

## Config keys (`velest.*`)

- Model: `layers_top_km`, `vp_bounds`, `monotonic`, `sampler`, `n_models`, `include_reference_model`.
- `vpvs.mode` (`fixed`, `sampled`, `independent_s_model`), `vpvs.value` (a number or `wadati`, the
  Stage 1 Wadati ratio), `vpvs.range`.
- Damping: `damping.*`, `stages` (damping schedule), `ittmax`, `invertratio`, `reseed.*`.
- Analysis: `select` (`method`, `value`, `statistic`), `iterate_until.*` (including
  `drift_warn_km_s`), `hypocentre_perturbation_km.*`, `stability.*` (`enabled: false` keeps only the
  fixed-model run), `vpvs_grid` (`enabled`, `values`), `compact_runs` (delete scratch files of
  successful runs).
- `project.n_jobs`, `project.random_seed`.

## Steps

`velest_mc`: resolve Vp/Vs, generate trial models, run the damping schedule (and reseeded rounds)
for every model in parallel, store the run table.

`velest_analyse`:
1. Parse the runs, select the best models, take the per-layer statistic as the final model with
   p05/p95 uncertainty, and run the ensemble QC checks.
2. Re-run VELEST with that model fixed (and the stability tests) for station delays and hypocentres
   consistent with it.
3. Repeat VELEST from its own outputs (`iterate_until`, delays frozen for fixed Vp/Vs) as a drift check.
4. Optional Vp/Vs grid search; figures, checks, summary.

The final model is the ensemble statistic, not the iterated model; the iteration only reports
whether it is a fixed point.

## Outputs

`02_velest_mc/`: `trial_models.parquet`, `runs.parquet` (paths relative to the stage dir),
`workflow_history.csv`, `mc_summary.json`, `runs/` (one directory per run).

`03_velest_analyse/` (names used by Stage 3):
- `final_model.mod`, `final_model.csv` (`layer, depth_top_km, vp, vs, vpvs`, `*_std`, `*_p05`, `*_p95`, `hits`).
- `station_delays.csv` (`sta, alias, lat, lon, elevation, ptcor, stcor, nobs`), from the fixed-model run.
- `final_hypocentres.parquet` (`event_id` of Stage 1, `cnv_id`, `lat`, `lon`, `depth_km`, `rms`, shifts).
- `summary.json`; ensemble tables (`runs.parquet`, `final_models.parquet`, `station_delays_all.parquet`,
  `hypocentres.parquet`, `station_delays_ensemble.csv`); `stability/` tables, `stability_runs/`,
  `final/iterate/`, `iterate_convergence.csv`, `vpvs_grid.csv`.

## Figures

Spaghetti plot of all models, velocity density, layer histograms, RMS against layer velocity and
iteration, stage RMS, convergence summary, station-delay map and delay vs elevation, hypocentre
shifts and final hypocentres, stability tests, residuals, Vp/Vs grid search.

## QC checks and what to do

- `convergence_rate` / `failures`: few runs stop by VELEST's own rule or many diverge. Raise `ittmax`,
  strengthen the first-stage damping, or check Stage 1 data quality.
- `at_bound`: selected runs sit on a Vp bound; widen `vp_bounds`. A top layer far below its bounds
  means it is not constrained (few rays above the stations): merge or remove it.
- `rms_improvement`, `delay_spread`, `reference_station`: how much the inversion improved the fit
  and how well station delays are determined.
- `reference_swap_velocity_change`, `stability_runs_stopped_early`, `lvz_switch_changes_model`,
  hypocentre-perturbation checks: the model depends on the reference station, starting hypocentres
  or the low-velocity option; treat the affected layers as poorly resolved.
- `iterate_model_drift`: repeating VELEST moves the model away from the ensemble median; the
  inversion is not well conditioned for these layers.
- `vpvs_grid_best`: compare with the Wadati ratio; the RMS curve is usually flat.
