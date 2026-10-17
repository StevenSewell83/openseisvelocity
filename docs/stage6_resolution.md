# Stage 6: resolution tests (`osv resolution`)

Inverts synthetic data for known models with the real source-receiver geometry, noise matched to
the pick uncertainties and the same regularisation, iterations and relocation as the real
inversion, then measures how well each model is recovered. Needs `build_model` and `tradeoff`
(damping and smoothing per phase from `tradeoff.json`); `invert` is optional but recommended.

```bash
osv resolution -c config.yaml        # -> <workdir>/08_resolution/
osv run --to resolution -c config.yaml
osv plot resolution -c config.yaml   # regenerate figures from the written outputs
```

Needs the FMTOMO binaries (`osv doctor`) and at least 3 GB of free disk; it stops early with a
message otherwise. Tests are resumable: a finished test with unchanged settings is skipped.

## Config keys (`resolution.*`, `fmtomo.*`)

- `checkerboard.cell_km`, `amplitude_pct`, `offset_half_cell` (extra copy shifted half a cell),
  `shape`, `depth_cells`.
- `spikes`, `anomalies` (`block`, `cylinder`, `slab`) and `input_models`: `final_3d` (restoring
  test, uses `invert`'s `final_model.nc`; skipped with a warning if `invert` has not completed),
  `file` (`input_model_file`), `anomaly`. `user_model` and `starting` are reported as skipped
  (their perturbation against the starting model is zero).
- `noise.*`, `hypocentre_noise_km.*`, `mask.min_corr`, `mask.min_amplitude`, `mask.window_km`,
  `mask.combine_with_dws`.
- `fmtomo.iterations`, `fmtomo.relocate`, `fmtomo.phases` (P and S are tested separately, each
  with its own chosen damping and smoothing), `project.n_jobs`, `project.random_seed`.
- `plotting.depth_slices_km`, `plotting.profiles` (sections; default W-E and S-N through the grid
  centre).

## Coverage

With a completed `invert` stage the coverage is its DWS mask (`mask`, `mask_vs`). Otherwise a
straight-ray proxy of the Stage 3 geometry is used: a node is covered when at least 3 source-receiver
segments cross its cell (`MIN_PROXY_RAYS`). The source is recorded in `resolution_runs.json`
(`coverage_source`). Coverage restricts the global metrics and, with `mask.combine_with_dws`, the
resolvable mask.

## Outputs (`08_resolution/`)

- `resolvable_mask.nc` (stable name; read by Stage 5): bool `vp` (`vs`), `resolvable`, float
  `resolved_scale_km_vp` (`_vs`), `resolved_scale_km` on the full grid. A node is resolvable if for at
  least one checkerboard cell size the local correlation and amplitude recovery pass the
  thresholds; the resolved scale is the smallest such cell size.
- `resolution_runs.json`: chosen regularisation per phase, iterations, coverage source, one row
  per test (success, misfit before and after, directories, flags).
- `metrics/`: `metrics.nc` (local correlation, amplitude, rms ratio, sign agreement per test),
  `summary.csv` / `summary.json` (global correlation, semblance, amplitude ratio, spike
  smearing, checkerboard anisotropy).
- `resolved_fractions.csv`: fraction of the covered volume resolved at each cell size, in three
  equal depth ranges and overall.
- `runs/<test>/`: `true_model.nc`, `recovered_model.nc`, `iterations.csv`, `summary.json`; heavy
  FMTOMO files are deleted.

## Figures

`true_vs_recovered_slices_<test>` and `..._sections_<test>` (common symmetric scale, at most six
tests: non-checkerboard first, then the largest cells), `checkerboard_matrix_<phase>` (rows cell
size, columns depth; zero crossings of the truth overlaid), `recovery_corr_<phase>`,
`recovery_amp_<phase>`, `resolved_scale_<phase>` (slices and sections), `smearing_<test>` (spike
centroid offsets and principal axes on a map and section), `checkerboard_anisotropy_<phase>`,
`noise_realisation_<phase>` (noise against pick uncertainties), `summary_metrics`, and
`lcurve_synthetic_vs_real_*` when `synthetic_tradeoff.json` (a `{label: picks}` map in the stage
directory) is present.

## Checks

- `tests_failed` (warn; fail if all failed).
- `no_resolvable_node` (fail): no node passes at any scale.
- `resolved_fraction` (warn if more than half of the covered volume is unresolved).
- `resolution_coarser_than_anomaly` (warn): median resolved scale exceeds the smallest configured
  spike or anomaly body.
- `final_3d_skipped`, `user_model_skipped`, `starting_skipped` (info or warn).

Interpret model features only where `resolvable` is true; amplitudes are reliable only where the
amplitude recovery is near 1.
