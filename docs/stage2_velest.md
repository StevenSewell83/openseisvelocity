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
- Layer damping: `vdamp` (a number, a list with one value per layer, or `auto`) and `vdamp_auto.*`.
- `vpvs.mode` (`fixed`, `sampled`, `independent_s_model`), `vpvs.value` (a number or `wadati`, the
  Stage 1 Wadati ratio), `vpvs.range`.
- Damping: `damping.*`, `stages` (damping schedule; each stage can set `swtfac` and `invert_station_delays`), `ittmax`, `invertratio`, `reseed.*`.
- Analysis: `select` (`method`, `value`, `statistic`, `final_model`: `median`, `best`, `closest` or `iterated`), `iterate_until.*` (including
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

By default (`select.final_model: median`) the final model is the ensemble statistic, not the iterated
model; the iteration only reports how far VELEST moves it (see "Choosing layers and damping").

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

- `convergence_rate` / `failures`: few runs stop by VELEST's own rule or many diverge. Real data rarely
  meets VELEST's stop rule (0 % converged is normal there). Many diverged runs with `nsp = 3` mean
  a stage inverts station delays with S data weighted: keep the default `stages` (see below). Otherwise
  raise `ittmax`, strengthen the first-stage damping, or check Stage 1 data quality.
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

## Choosing layers and damping

Evidence from the Rotokawau calibration set (`projects/rk-nm-2017`: 65 events at 1.1-4.6 km (Stage 1 depths), 63
stations at -0.59 to +0.16 km depth, fixed Vp/Vs 1.716, 2417 picks). Every experiment used
200 random models plus the mid-bounds reference model, VELEST with `ittmax: 30`, and then the
median model of the best 5 % of the runs was re-run: fixed-model run (canonical RMS), repeat-until-stable
loop with frozen delays (5 cycles; "drift" is the largest layer change from the median model) and the
reference-station swap (largest layer change over three alternative reference stations). Raw numbers:
`build/s310/results/*.csv` of the evaluation (not committed).

### Why runs "diverged" (80 of 201)

No run failed, none had a NaN or an enormous starting residual (the starting RMS of the first stage is
0.20-0.40 s, after it 0.055-0.099 s). A run counts as diverged when the final RMS of its last stage is
above its start; here the RMS rose by 0.0002-0.012 s (median +0.006 s) with a four-iteration saw-tooth
over up to 30 iterations. The cause is the delay treatment, not the events: with `nsp = 3` VELEST applies the
S delay `ptcor * vpvs` but gives S readings no delay partials, so a stage that inverts delays
with S weighted is inconsistent and the RMS creeps up. Extra damping of the near-surface layer does not help
under the old schedule (with `vdamp` auto, 50 or 200, 201 of 201 runs diverged), and one weak event
is no problem (largest event RMS 0.15 s at 4.9 km; median 0.04 s; 1 of 65 events above 3 times the median).

The default `stages` therefore estimate delays from P only and hold them fixed once S is added:

1. `p_high_damping`: `vthet: 10`, `stathet: 1`, `swtfac: 0.01` (P only; delays and hypocentres settle);
2. `p_free`: user damping, `swtfac: 0.01` (P only, delays inverted);
3. `full_frozen_delays`: all picks, `invert_station_delays: false`.

(With `nsp = 2` the early stages only down-weight S; set `stages` yourself if that matters.)

| schedule (8 layers, `vdamp` 1) | diverged | RMS of runs, median (min-max) s | top-layer Vp of the median, km/s | median model re-run RMS, s | drift, km/s | ref swap, km/s |
|---|---|---|---|---|---|---|
| old: high damping, then free with delays inverted | 80 | 0.0587 (0.0516-0.1060) | 0.71 | 0.0450 | 0.97 | 0.66 |
| last stage delays frozen | 0 | 0.0589 (0.0482-0.0925) | 1.91 | 0.0429 | 0.78 | 0.21 |
| high damping P only, then all picks frozen (2 stages) | 1 | 0.0434 (0.0423-0.0578) | 2.27 | 0.0425 | 0.87 | 0.20 |
| **P-only damped, P-only free, all picks frozen (3 stages, default)** | 1 | **0.0419 (0.0414-0.0533)** | 1.17 | **0.0417** | **0.33** | 0.47 |

(A P-only last stage gives RMS 0.032 s, not comparable; its median model fits as 0.0436 s.)
The 0.70 km/s top layer of the old run came from this schedule, not from the layer itself: with the
frozen-delay schedules the top layer no longer collapses (median 1.2-2.3 km/s, none of the runs below 1.5
for the 7-layer model below). Eight layers are still not identifiable: models with top layers from 0.7 to
2.3 km/s and layers 3-4 from 2.5 to 4.5 km/s fit equally well (0.042-0.045 s) and drift 0.3-1.4 km/s on
repetition, hence the layering below.

### Layering

Only 6 of the 65 events lie above 2.2 km and none above 1.06 km, so the layers between the stations and
2 km are crossed by steeply travelling rays only: thin layers there trade off against each other and
against the station delays. Variants (3-stage schedule, `vdamp` 1, two random seeds: 42 / 7):

| layer tops, km | layers | diverged | median model RMS, s | AIC | drift, km/s | ref swap, km/s |
|---|---|---|---|---|---|---|
| -0.6 0.2 0.7 1.2 1.7 2.2 3.0 4.0 (old) | 8 | 1 / 6 | 0.0416 / 0.0417 | -14703 / -14697 | 0.34 / 0.74 | 0.47 / 0.10 |
| **-0.6 0.5 1.0 1.5 2.0 2.5 3.5 (default)** | 7 | 1 / 2 | 0.0418 / 0.0417 | -14689 / -14699 | **0.15 / 0.28** | **0.017 / 0.029** |
| -0.6 0.2 0.7 1.2 1.7 2.2 3.0 | 7 | 4 / 7 | 0.0415 / 0.0415 | -14726 / -14723 | 0.31 / 0.75 | 0.11 / 0.22 |
| -0.6 1.0 2.0 3.0 4.0 | 5 | 5 / 4 | 0.0416 / 0.0418 | -14720 / -14692 | 0.29 / 0.22 | 0.044 / 0.032 |

AIC uses the weighted RMS (`n log(RSS/n) + 2k`, n = 2417, k = layers + 63 delays + 4 x 65 hypocentre
parameters). Its differences (up to 36) change sign between the seeds and the RMS differ by 0.0003 s,
so the fit does not choose between the layerings; an F-test of the 8-layer against the 7-layer model with
the same top layers gives a negative RSS gain, so the extra layer buys nothing. The stability tests do:
the merged-top 7-layer model has the smallest drift and reference-swap changes in both seeds (0.017-0.029
km/s against 0.10-0.47 for the old layering); the 5-layer model is as stable but too coarse for the 0.5 km
FMTOMO grid. The merged-top layers are the default (`vp_bounds` interpolate the old bounds at the layer
mid-depths). Layers above 2 km still have an ensemble IQR of 0.5-0.9 km/s: the `unresolved_layer` check
flags them, and Stage 3 should treat them with their p05-p95 range.

The first layer top (-0.6 km, above the highest station at -0.59 km) can stay: starting at the lowest
station elevation instead would put most stations above the model.

### `vdamp`

7 layers, 3-stage schedule, seed 42 (damping factor of layer 0 given; other layers 1):

| `vdamp` | diverged | top-layer Vp median (p05-p95 of the 200 runs), km/s | runs below the lower bound 1.96 | median model RMS, s | drift | ref swap |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.80 (1.72-3.25) | 20 % | 0.0418 | 0.15 | 0.017 |
| `auto` (10 for layers above the deepest station) | 1 | 2.02 (2.02-3.48) | 2.5 % | 0.0418 | 0.17 | 0.028 |
| 50 | 1 | 2.19 (2.06-3.53) | 0 % | 0.0418 | 0.60 | 0.032 |
| 200 | 1 | 2.19 (2.06-3.53) | 0 % | 0.0418 | 0.26 | 0.033 |
| 20, 10, 3, 1, ... | 1 | 2.04 (2.05-3.51) | 0.5 % | 0.0418 | 0.24 | 0.023 |

Damping the top layer changes neither the fit nor the stability: it only keeps the layer near the sampled
starting range (2.0-2.2 km/s) instead of the lower bound (1.8 km/s, which the data prefer weakly:
`at_bound`). With the schedule above the layer does not collapse even undamped, so `vdamp` is a
guard-rail; `auto` is the default (it damps the one layer that contains the stations and costs nothing).
The layer's Vp stays unresolved (IQR 0.8 km/s); do not interpret it. Note the damping reaches every stage and
the repeat-until-stable loop since this evaluation (it was lost after stage 1 before: VELEST's output model
carries no `vdamp`).

### Final-model policy

7 layers, `vdamp: auto`, 3-stage schedule, seeds 42 / 7. Each model is re-run with fixed velocities (RMS), then repeated
(drift) and with swapped reference stations; "inside" asks if it lies in the p05-p95 band of the selected runs.

| `select.final_model` | re-run RMS, s | drift, km/s | ref swap, km/s | distance from median, km/s | inside p05-p95 |
|---|---|---|---|---|---|
| **median (default)** | 0.0418 / 0.0417 | 0.17 / 0.20 | 0.028 / 0.053 | 0 | yes / yes |
| best | 0.0418 / 0.0418 | 0.15 / 0.17 | 0.039 / 0.030 | 0.20 / 0.13 | no / no |
| closest | 0.0418 / 0.0417 | 0.27 / 0.19 | 0.034 / 0.034 | 0.14 / 0.07 | yes / yes |
| iterated | 0.0419 / 0.0419 | 0.16 / 0.15 | 0.029 / 0.032 | 0.17 / 0.20 | no / no |

All four fit identically (differences of 0.0001 s are far below the pick noise); none is a fixed point of
VELEST with default damping. Repeating VELEST from the median model with frozen delays moves it by
0.17, 0.12, 0.10, 0.07, 0.04, 0.03, 0.02 km/s per cycle and converges (change < 0.02 km/s) only in cycle 8,
at 0.47 km/s from the median, with the RMS unchanged (0.0420 to 0.0421): the model drifts along a
valley that the data do not constrain, and the "iterated" model is not a fixed point either when
re-iterated (drift 0.15-0.16 over 5 cycles). Only more velocity damping stops it: `damping.vthet: 5` still
drifts 0.27 km/s in 10 cycles, `20` moves 0.05 km/s in 2 cycles (damping, not information).
The best run lies outside the ensemble band and `iterated` leaves it, so the median stays the default: representative,
inside its own uncertainty and as stable as the alternatives. A drift of 0.15-0.3 km/s in the repeat loop is
what this data gives, so `iterate_until.drift_warn_km_s` (0.2) is borderline: read `iterate_model_drift`
as "the deep layers are not resolved to better than ~0.3 km/s".
