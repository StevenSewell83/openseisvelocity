# Tutorial: the rk-nm-2017 example, end to end

This walk-through follows a real run on the project `projects/rk-nm-2017`: 79 manually located
microseismic events and 63 stations at a geothermal field in New Zealand. All numbers below
come from the `manifest.json` summaries of that run (single workstation, 20 jobs). Your numbers
will differ, but the sequence, the files and the way to read them are the same. Figures are
copied from the run's `figures/` directories.

You need a working installation ([installation](install.md); `osv doctor` must not report FAIL)
and a project directory with a `config.yaml` (see `projects/example/README.md` in the repository
for how to create one). The example project's config points at a QuakeML catalogue
(`manual_only_catalog.xml`) and StationXML files.

```bash
conda activate osv
cd projects/rk-nm-2017
osv doctor --config config.yaml        # tools, smoke tests, disk space, VELEST array limits
osv config show -c config.yaml         # check the settings in use
osv run -c config.yaml                 # everything, a couple of hours with 20 jobs
```

You can also run the stages one at a time (`osv prep`, `osv velest-mc`, ...), which is what the
sections below do. Every stage:

* reads its inputs from upstream stage directories (it refuses to run if they are missing);
* writes `<workdir>/NN_name/` with `manifest.json`, `figures/`, `logs/` and `report.html`;
* is skipped on the next `osv run` unless its config section, inputs, upstream runs or tools changed.

Open `<workdir>/index.html` at any time for the project report; it shows what exists so far.

## How to read a stage report

Each `report.html` has the same layout: the stage status and run time, **QC checks** (pass, warn,
fail, each with a message saying what to do), summary numbers, the figures with captions, and the
files written. Failures stop the pipeline; warnings do not, but each one deserves a look. The
`manifest.json` holds the same information in machine-readable form (`status`, `wall_time_s`,
`checks`, `summary`, `outputs`, `error`).

## Stage 1: prepare the data (`osv prep`, 7 s)

```bash
osv prep -c config.yaml
```

Reads the catalogue and StationXML, applies the depth datum, associates picks with station
epochs, flags outliers, applies the selection in `data_selection`, and writes the tables and the
VELEST input files. In the example:

| | Number |
|---|---|
| Events in / selected | 79 / 77 (2 rejected: `min_p`, `max_azgap`) |
| Calibration / validation events | 65 / 12 (`data_selection.validation_fraction: 0.15`) |
| P / S picks selected | 2171 / 674 (of 2231 / 709) |
| Stations with picks | 63 |
| Vp/Vs from Wadati diagrams | 1.716, 95% CI 1.703-1.729 |
| Median azimuthal gap | 51.5 degrees |
| VELEST reference station | NS14 |

![Map before and after selection](img/prep_map_before_after.png)

![Wadati diagram](img/prep_wadati.png)

Look at: `map_before_after` (did the selection remove what you intended), `wadati` (the points
should follow one straight line; its slope minus one is Vp/Vs), `travel_times`, `rejections`,
`station_residuals` and `picks_per_station`. Details: [Stage 1](stage1_prep.md).

Typical warnings and what to do:

* *"2880 duplicated arrival(s)", "2861 pick(s) are not referenced by any arrival"* - catalogue
  quirks, handled automatically; `inputs.use_arrivals` stays false.
* *"all 79 magnitudes are identical"* - the catalogue has no usable magnitudes; they are marked
  invalid and not used.
* *"4 station(s) have borehole sensors"*, *"10 channel(s) have coordinates inconsistent with their
  station"* - check the StationXML; see [borehole stations](troubleshooting.md#data-preparation).
* *"14 stations have a median travel-time residual exceeding the threshold"* - station delays or
  1-D misfit; Stage 2 estimates station delays.
* *"11 events have a per-event Vp/Vs inconsistent with the median"* - usually few S picks.

## Stage 2: Monte Carlo VELEST (`osv velest-mc`, `osv velest-analyse`)

```bash
osv velest-mc -c config.yaml        # 201 VELEST runs in parallel
osv velest-analyse -c config.yaml   # select, stability tests, delays, Vp/Vs grid, final model, 35 s
```

`velest-mc` draws 200 random starting models (Latin hypercube within `velest.vp_bounds`) plus the
reference model, runs VELEST on each through three damping stages, and writes `runs.parquet`.
It needs the compiled `velest` binary. `velest-analyse` ranks the runs, selects the best 5%
(`velest.select`), runs perturbation and reference-station stability tests, repeats VELEST on the
final model until it stops changing (`velest.iterate_until`) and writes `final_model.csv`,
`final_model.mod`, station delays and the Vp/Vs grid search.

| | Result |
|---|---|
| Runs eligible / selected | 121 eligible, 7 selected (closest: MC_0111, RMS 0.0519 s) |
| Final model RMS | 0.0863 s (ensemble 0.052-0.087 s) |
| Final Vp (km/s) | 0.70 (-0.6 km), 3.00 (0.2), 3.85 (0.7), 3.73 (1.2), 4.49 (1.7), 4.98 (2.2), 5.53 (3.0), 5.95 (4.0) |
| Station delays | 63 stations, largest spread 0.12 s (HRRZ) |
| Vp/Vs grid search minimum | 1.68 (Wadati 1.716) |

![Velocity spaghetti plot](img/velest_spaghetti.png)

Look at: `velocity_spaghetti` and `velocity_density` (the ensemble spread: narrow where the data
constrain the model), `layer_histograms`, `rms_vs_layer_velocity`, `stability_*` (does the answer
change with the starting hypocentres or the reference station), `station_delay_map` and
`vpvs`. Details: [Stage 2](stage2_velest.md), including how to choose layers and damping.

Typical warnings:

* *"0% of 201 runs converged" and "80 diverged"* - VELEST stops after four RMS backups;
  expected for part of the ensemble, see [VELEST divergence](troubleshooting.md#velest-stage-2).
* *"layer 0: 100% of the selected runs end within 2% of the lower bound"* - widen the first entry
  of `velest.vp_bounds` or accept a slow near-surface layer.
* *"low-velocity zone: layer 3 is 0.12 km/s slower than layer 2"* - informational; with
  `velest.monotonic: true` the trial models cannot create one, so it comes from the inversion.
* *"reference station swapped from NS14 ... layer 7 Vp changes by 0.28 km/s (limit 0.1)"* - the
  deepest layer depends on which station's delay is fixed; treat it as less certain.
* *"repeating VELEST on the final model moved a layer's Vp by 1.09 km/s"* - the model is not at a
  fixed point (shallow layers trade off); the tutorial run still used it, with this caveat in mind.

## Stage 3: build the 3-D starting model (`osv build-model`, 66 s)

```bash
osv build-model -c config.yaml
```

Extrudes the final 1-D model into a 3-D FMTOMO grid (`model3d.grid`), optionally blends in your own
grids, horizons or wells ([user models](user_models.md)), writes every FMTOMO input and traces the
observed rays with `fm3d` as a forward check.

| | Result |
|---|---|
| Source | `velest_1d` |
| Velocity grid | 19 x 75 x 62 = 103,488 nodes at 0.5 km |
| Propagation grid | 55 x 223 x 184 (2.26 M nodes, 0.167 km) |
| Sources / receivers / P observations | 65 / 62 / 1847 |
| Vp range | 0.705-5.945 km/s |
| Forward RMS of the starting model | 0.0933 s (0.0778 s after removing the mean) |

![Vp slices of the starting model](img/build_vp_slices.png)

Look at: `grid_geometry`, `vp_slices`, `section_*`, and above all the forward-check figures
`residual_histograms`, `residual_distance` and `station_residuals`. The starting model should
fit the picks about as well as VELEST does (RMS 0.086 s here); a much worse fit means a problem
with coordinates, the depth datum or station delays. Details: [Stage 3](stage3_model.md).

Typical warnings:

* *"node spacing 0.50 km exceeds one third of the median nearest-station spacing (1.19 km)"* -
  the data cannot constrain such a fine grid; the regularisation will do most of the work.
* *"Vp outside [1, 8.5] km/s at 14784 nodes"* - the 0.7 km/s near-surface layer; check or relax
  `model3d.checks`.
* *"21 receiver(s) were above the interpolated surface; surface lifted locally"* - station
  elevations versus the smoothed surface (`model3d.surface_smoothing_km`).
* *"32 P residuals beyond 5 robust standard deviations"* - look at the stations involved.

## Stage 4: damping, smoothing and the inversion (`osv tradeoff`, `osv invert`)

```bash
osv tradeoff -c config.yaml     # 14 short inversions, about 42 min
osv invert -c config.yaml       # final inversion with relocation, about 6.5 min
```

`tradeoff` sweeps the damping and smoothing, builds L-curves of data misfit against model norm
and picks the corner by maximum curvature (`fmtomo.tradeoff`). `invert` runs the full
inversion with the chosen values (or the ones you fix in `fmtomo`) and relocates the events.

| | Result |
|---|---|
| Chosen damping / smoothing (P) | 0.2 / 10 (maximum curvature, 14 sweep runs) |
| RMS start to final | 93.3 to 63.9 ms (31.5% misfit reduction, 53.0% variance reduction) |
| Reduced chi-squared | 13.4 |
| Median hypocentre shift | 0.255 km |
| Covered fraction of nodes (DWS) | 2.6% |

![L-curve for damping](img/tradeoff_lcurve_damping.png)

![Final Vp slices](img/invert_slices_vp.png)

Look at: `lcurve_damping` and `lcurve_smoothing` (is the chosen point at a clear corner),
`misfit_vs_iteration` (has it converged), `slices_vp`, `slices_dvp_1d` (perturbation relative to
the 1-D model), `residual_histograms` (centred and narrower than before),
`hypocentre_shifts` and `seismicity_before_after`. Details: [Stage 4](stage4_fmtomo.md).

Typical warnings:

* *"the corner (10) is next to the edge of the swept range 1..1000"* - widen the sweep or set
  the smoothing yourself.
* *"14 of 14 sweep runs still changed their misfit by more than 2%"* - the sweep iterations are few
  (`fmtomo.tradeoff.iterations`); the L-curve is indicative.
* *"reduced chi-squared is 13.38, not near 1"* - the stated pick uncertainties are smaller than the
  achievable fit; the inversion is an RMS reduction, not a statistical fit.
* *"nodes sit at the minimum velocity (0.5 km/s): the inversion was clipped"* - poorly
  constrained or too weakly regularised region (about 0.03% here).

## Stage 5: analyse the inversion (`osv analyse`, 11 s)

```bash
osv analyse -c config.yaml
```

Computes perturbation statistics, produces the figures and the exports (NetCDF, CSV, VTK) and masks everything the data cannot see (`plotting.dws_mask_threshold`).

Look at: `slices_vp`, `section_a_a`, `compare_start_final_vp`, `variance_reduction`,
`residual_vs_azimuth` and `station_residual_map`. Only 2.6% of the Vp volume has DWS of at least
0.1 here, so most of the model is masked; that is the data, not a bug. After running Stage 6, run
`osv plot analyse -c config.yaml` to add the `_resolvable` figures that show only what the
resolution tests recovered. Details: [Stage 5](stage5_analysis.md).

![Start versus final model, resolvable volume only](img/analyse_compare_resolvable.png)

## Stage 6: resolution tests (`osv resolution`, about 49 min)

```bash
osv resolution -c config.yaml
```

Inverts synthetic data (checkerboards of 1.5, 2 and 3 km cells, spikes, and your own anomaly
bodies) through the real ray geometry with noise matched to the picks, using the same
regularisation. Six tests ran, all succeeded.

| | Result |
|---|---|
| Covered nodes (P) | 2315 |
| Resolved fraction | 0.9% at 2 km, 10.4% at 3 km |
| Median resolved scale | 3 km (smallest anomaly of interest 1.5 km) |

![Resolved scale](img/resolution_resolved_scale.png)

Look at: `true_vs_recovered_slices_*` (does the checkerboard come back where you want to
interpret), `recovery_amp_P` and `recovery_corr_P` (amplitude and correlation recovery),
`resolved_scale_P`, `smearing_spikes_P`. The `resolvable_mask.nc` file marks nodes whose recovery
passes `resolution.mask`. Warning *"median resolved scale 3 km is coarser than the smallest
anomaly of interest"* tells you that features smaller than that must not be interpreted.
Details: [Stage 6](stage6_resolution.md).

## Stage 7: validation (`osv validation`, 109 s)

```bash
osv validation -c config.yaml
```

Relocates the 12 held-out events with VELEST in the final 1-D model and with FMTOMO in the final 3-D
model, using the same picks.

| | Result |
|---|---|
| Median RMS, 1-D / 3-D | 56.5 / 22.1 ms (improvement 28.1 ms) |
| Events better / worse in 3-D | 83% / 8% |
| Wilcoxon p-value | 0.0049 |

![1-D versus 3-D RMS](img/validation_rms_scatter.png)

Points below the 1:1 line in `validation_rms_scatter` are events that fit better in 3-D. A low
fraction of better events, or a high p-value, means the 3-D model does not generalise. Details:
[Stage 7](stage7_validation.md).

## The project report (`osv report`)

```bash
osv report -c config.yaml
open runs/index.html
```

Combines everything: pipeline status (9 of 9 stages complete here), all failures and warnings
grouped by stage (0 failures and 33 warnings in the example), headline numbers and key
figures. `10_report/summary.md` is a compact Markdown version to paste into notes.
See [reporting](reporting.md).

## Iterating

* Change a setting in `config.yaml` and run `osv run`: only the affected stages and everything
  downstream re-run.
* To tweak figures only, edit the `plotting` section and run `osv plot STAGE`.
* To redo one stage use `osv STAGE --force`.
* To start over, delete the stage directories (or the whole `workdir`).
* Use [troubleshooting](troubleshooting.md) when a stage fails or a warning is unclear.
