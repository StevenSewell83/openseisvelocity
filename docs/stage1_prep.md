# Stage 1: data preparation (`osv prep`)

Turns a QuakeML catalogue and StationXML files into clean, selected event, pick and station
tables, ready-to-run VELEST inputs, diagnostic figures and an HTML report.

```bash
osv prep -c config.yaml            # run (skipped if inputs/config are unchanged)
osv prep -c config.yaml --force    # re-run
osv run --to prep -c config.yaml   # same, via the pipeline
osv plot prep -c config.yaml       # regenerate figures only
```

Results are written to `<project.workdir>/01_prep/`; open `report.html` first.

## Inputs and config keys

- `inputs.quakeml`, `inputs.stationxml` (files or globs), `inputs.origin_selection`,
  `inputs.evaluation_mode`, `inputs.use_arrivals`, `inputs.weight_source`,
  `inputs.station_aliases`.
- `coordinates.origin`, `coordinates.borehole_sensors`, `coordinates.depth_datum`,
  `coordinates.reference_elevation_m`.
- `data_selection.*`: `region_polygon`/`region_geojson`, `depth_km`, `min_p`, `min_s`,
  `max_azgap`, `max_rms`, `max_epi_dist_km`, `magnitude`, `pick_uncertainty_to_weight`,
  `wadati_outlier_sigma`, `validation_fraction`.
- `plotting.profiles`, `plotting.format`, `plotting.dpi`; `project.random_seed`.

## Processing steps

1. Read the catalogue (one origin per event, picks with uncertainties) and the stations (epochs,
   channels, sensor depths); generate VELEST station aliases (max 4 characters).
2. Apply the depth datum so event depths are below sea level.
3. Associate picks with station epochs and compute distances, azimuths and gaps.
4. Analyse outliers: pooled Wadati Vp/Vs fit, robust travel-time fits, per-station residuals.
5. Apply the pick-level and event-level filters, assign VELEST weight classes, and split the
   selected events into calibration and validation sets.
6. Write the tables, the VELEST inputs, QC findings, figures and report.

## Outputs (`01_prep/`)

- `events.parquet`, `picks.parquet`: selected events (with `split`) and picks (with `velest_weight`).
- `all_events.parquet`, `all_picks.parquet`: everything, with `selected`, `reject_reasons` and outlier columns.
- `stations.parquet`, `channels.parquet`, `aliases.csv`: station epochs (with `used`, `n_P`, `n_S`), channels, aliases.
- `rejections.csv`, `split.csv`: one row per failed filter; calibration/validation assignment.
- `wadati.json`, `wadati_pairs.parquet`, `wadati_events.csv`, `tt_fits.json`: Vp/Vs and travel-time fits.
- `outliers.csv`, `station_residuals.csv`: flagged picks; median residual per station.
- `velest/`: `events.cnv`, `stations.sta`, `velest_inputs_summary.json`, `event_map.csv` (CNV id to event id), and
  `velest_control.json`, the prefilled control (load with `osv.data.stage.load_velest_control`; no model file yet).
- `velest_validation/`: the same for validation events (only if there are any).
- `manifest.json`, `report.html`, `figures/`.

## Figures

Maps (stations and events before/after selection, depth sections, profiles, ray maps),
distributions (histograms, timeline, pick uncertainty), travel times, Wadati diagram, stations
(picks per station, residuals, residual map) and selection (rejection reasons).

## QC checks and what to do

- `duplicate_arrivals`: the catalogue repeats arrivals within an origin. Duplicates are dropped
  and picks are used (`use_arrivals: false`); no action unless counts look wrong.
- `invalid_magnitudes`: magnitudes are missing or constant/implausible. Magnitude filters and plots
  are skipped (`magnitude.ignore_if_invalid`); VELEST does not use them. Fix the catalogue if
  you need magnitude-based selection.
- Missing uncertainties: picks without uncertainty get the worst usable weight or are rejected
  depending on `pick_uncertainty_to_weight`; check the pick-uncertainty figure and consider
  `weight_source: comment`.
- Borehole sensors: sensor depths come from StationXML (`borehole_sensors`). Check they appear
  as marked in the map and sections; use `ignore_depth` to treat them as surface stations.
- Systematic station residuals (`station_systematic_residuals`): stations are consistently early
  or late versus the 1-D curve. Often real near-surface delays; check coordinates and
  elevation, and expect Stage 2 station corrections to absorb them.
- Outliers (`outliers_P`/`outliers_S`, `wadati_inconsistent_events`): flagged picks are rejected
  from selection. Review `outliers.csv`; tighten or relax `wadati_outlier_sigma`, and re-pick
  events with inconsistent Vp/Vs.
