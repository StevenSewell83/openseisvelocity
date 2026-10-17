# Stage 7: validation (`osv validation`)

Relocates the events held out in Stage 1 (`events.split == "validation"`, fraction
`data_selection.validation_fraction`) twice with the same picks: with VELEST in the final 1-D model
(with its station delays) and with FMTOMO in the final 3-D model (velocities fixed, linear
relocation). The per-event RMS residuals and hypocentres are then compared. Because these events
did not take part in the inversion, a better fit with the 3-D model is evidence that it is real
and not over-fitted.

```bash
osv validation -c config.yaml        # -> <workdir>/09_validation/
osv run --to validation -c config.yaml
osv plot validation -c config.yaml   # regenerate figures from the written outputs
```

Needs `prep`, `velest_analyse` and `build_model`; `invert` is used when it has completed. Its
`final_vgrids.in` is then the 3-D model. Without it the Stage 3 starting model
(`04_build_model/fmtomo/vgrids.in`) is used and a `validation_starting_model` warning is recorded.
`interfaces.in` and `propgrid.in` always come from `04_build_model/fmtomo/`. The stage is marked
stale when `final_vgrids.in` changes. Needs the VELEST and FMTOMO binaries (`osv doctor`).

## Config keys (`validation.*`)

- `enabled` (default true): when false the stage completes with empty outputs.
- `max_iterations`: Gauss-Newton iterations of the 3-D relocation.
- `n_jobs`: parallel `fm3d` processes (default `project.n_jobs`, capped at 8).
- `use_s`: ask for S as well (default: when `fmtomo.phases` has S/vpvs). Both relocations
  currently use P only so they are comparable; a note says so.
- `source_sigma_km`, `velest_ittmax`: damping of the source update and VELEST iterations.
- `worse_fraction_warn`, `min_events_stat`, `improve_tol_ms`, `max_hypo_shift_warn_km`: QC
  thresholds.
- Also used: `model3d.apply_velest_station_delays` (whether 3-D times get the VELEST delays; if
  off, an info check says the comparison favours the 1-D model), `fmtomo.*`, `velest.*`.

## Outputs (`09_validation/`)

- `validation_events.parquet`: per event `lat0/lon0/depth0` (catalogue), `*_1d`, `*_3d`
  hypocentres, `rms_1d`, `rms_3d`, `rms_change` (1-D minus 3-D, s; positive means 3-D is better),
  `dx_km`, `dy_km`, `dh_km`, `dz_km`, `n_picks`.
- `stats.json`: paired statistics (`stats.rms`: medians, improvement, fractions better/worse/tied,
  Wilcoxon p), hypocentre differences, clustering, phases, finding codes, `model_source`
  (`invert` or `build_model`).
- `work/`: `velest_1d/`, `fmtomo_3d/` and the intermediate tables.

## Figures and checks

`validation_rms_scatter`, `validation_rms_change`, `validation_locations`,
`validation_depth_change`. Checks include 3-D worse than 1-D for too many events, large
hypocentre shifts, dropped events, missing station delays in 3-D, and the starting-model fallback.
