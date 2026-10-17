# Stage 3: starting model and forward check (`osv build-model`)

Builds the 3-D starting model and every FMTOMO input for the inversion, then traces the observed
rays through the model with `fm3d` to check how well it fits the picks. Needs the `fm3d` binary
(`osv doctor`), a completed Stage 1 and the Stage 2 analysis (`velest_analyse`).

```bash
osv build-model -c config.yaml       # -> <workdir>/04_build_model/
osv run --to build_model -c config.yaml
osv plot build_model -c config.yaml  # regenerate figures from the written outputs
```

Open `report.html` first. A user model (grid, geology, wells; see `user_models.md`) is chosen with
`model3d.source`; `velest_1d` extends the Stage 2 model alone.

## Config keys

- `model3d.source`, `model3d.user_model.*` (blend taper, smoothing, background), `model3d.extrusion.*`.
- `model3d.grid.*` (`dx_km`, `dy_km`, `dz_km`, `pad_km`, `zmax_km`, `top_margin_km`, `max_memory_gb`),
  `propgrid_refine`, `source_refine_factor`, `source_refine_cells`.
- `model3d.surface` (`dem`, `flat`, `station_elevations`), `surface_smoothing_km`, `inputs.dem`.
- `model3d.apply_velest_station_delays`: subtract the Stage 2 delays from the observed times.
- `model3d.velest_station_delays_exclude`: stations (list of codes/aliases, or `borehole`) whose delays are not subtracted.
- `model3d.checks.*`: thresholds for Vp, Vp/Vs and the velocity gradient.
- `fmtomo.phases`: S data are written when it contains `S` or `vpvs`.
- `plotting.profiles`, `plotting.depth_slices_km`, `project.n_jobs` (parallel `fm3d` processes).

## Steps

1. Define the velocity and propagation grids from the stations and the relocated events.
2. Build the starting model (user model blended into the 1-D background).
3. Build the surface and base interfaces; move sources that lie above the surface just below it.
4. Write sources, receivers and observed travel times of the **calibration** events at their VELEST
   hypocentres (origin time of the relocation, so a travel time is pick time minus that origin).
5. Forward check: trace all picks through the starting model, and through the 1-D background if the
   user model differs from it, in parallel (`osv.fmtomo.forward`).
6. Write outputs, figures, checks and report.

## Outputs (`04_build_model/`, names used by Stage 4)

`fmtomo/`:
- `vgrids.in` (Vp, type 1, with a-priori uncertainty), `interfaces.in`, `propgrid.in`.
- P: `sources.in`, `sourcesref.in`, `receivers.in`, `otimes_P.dat`, `otimes_P_mapping.csv`
  (`otimes` row to event, station, phase, pick), `events.csv`, `stations.csv`.
- S (if requested): `vgrids_S.in` (Vs as type 1), `sources_S.in`, `sourcesref_S.in`,
  `receivers_S.in`, `otimes_S.dat`, `otimes_S_mapping.csv`, `events_S.csv`.

Also `starting_model.nc` (xarray; full grid with cushions: `vp`, `vs`, uncertainties, `weight`,
`vp_bg`, `vs_bg`, `unit_id` for geology), `grid.json`, `forward_check.json` and `forward_check/`
(`residuals.parquet`, `by_station.csv`, `by_event.csv`, `arrivals.dat` of each run).

## Forward check

`residual = observed - calculated`. Besides RMS the summary has the RMS after removing each event's
mean (`rms_demeaned_s`; the origin times are those of the VELEST model, so another model leaves an
origin-time offset) and the median absolute deviation. For a user model the same picks are traced
through the background; a change of RMS above 2% is called `better` or `worse`. The VELEST RMS
(all phases, with station delays) is listed for orientation: the 3-D check applies no delays by
default, so it is expected to be larger. The mean P residual per station should follow the VELEST
delays (correlation in the report).

### Origin times and the 1-D extrusion (osv-jac.13)

- `final_hypocentres.parquet` (Stage 2) must carry VELEST's **relocated** origin time in `time`
  (the catalogue time is `time0`). It used to carry the catalogue time, so every travel time was
  too long by the relocation's origin shift (mean 0.075 s on rk-nm-2017), a +0.075 s bias in the
  forward check. A Stage 2 output from before the fix must be regenerated (or `time` replaced by the
  `velout.cnv` origin times of the fixed-model run).
- The VELEST layers are constant-velocity slabs, so the default extrusion is `layered`
  (`model3d.extrusion.mode`). `gradient` (linear between layer mid-depths) does not preserve the
  layer average over the thick slow top layer and is about 0.02 s faster in P time for the
  near-station rays. FMTOMO's cubic B-spline of the 0.5 km nodes smears each velocity step over about
  one node, which makes the model slightly faster than VELEST's exact layers: on rk-nm-2017 the same
  picks give a mean P residual of +0.043 s (`dz_km` 0.5) against +0.010 s (`dz_km` 0.1), a
  constant offset that the relocation of the origin times absorbs (event-demeaned RMS 0.035 s).
- With the station delays applied (default) the per-station mean residuals are what is left after
  the delays, so the reported correlation with the VELEST delays is expected to be low; it only
  tests the delays when `apply_velest_station_delays` is off.

## Figures

Vp and dVp depth slices, E-W, N-S and configured cross sections (with stations, events and surface),
well profiles with logs and horizon maps / unit sections (if present), user-model weight map, Vp and
Vp/Vs histograms, gradient-magnitude map, grid and propagation grid against stations and events,
residual histograms (1-D against starting model), residual against distance, per-station residual map.

## QC checks and what to do

- `grid_warning`, `grid_size`: node spacing finer than the station spacing can resolve, or fm3d memory
  above `grid.max_memory_gb`. Coarsen the grid or lower `propgrid_refine`.
- `vp_range`, `vpvs_range`, `large_gradient`: nodes outside the physical limits or with sharp
  gradients (they upset the eikonal solver). A very slow top layer usually means the VELEST layer is
  unconstrained: merge it in Stage 2.
- `interfaces`: surface lifted around stations, sources moved below the surface or outside the grid.
- `fm3d_invalid_arrivals`: rays that fm3d could not trace; the picks are left out of the residuals.
- `residual_bias`: mean residual large against the RMS (model too fast or slow, or delays not applied).
- `user_model_fits_worse`: the starting model fits the picks worse than the 1-D background it
  replaces; check units, z convention and the blend before inverting.
- `velest_delay_agreement`: weak correlation with the VELEST station delays.
