# Exporting the model for NonLinLoc (`osv export-nlloc`)

```bash
osv export-nlloc -c config.yaml --out nlloc_model --root macban3d --origin 14.1 121.21
```

Writes, from a project whose `invert` stage has completed:

| file | content |
| --- | --- |
| `<root>.P.mod.hdr/.buf`, `<root>.S.mod.hdr/.buf` | NonLinLoc 3-D model grids: `SLOW_LEN` (slowness x spacing), 4-byte floats, cubic cells, `TRANSFORM SIMPLE LatOrig <lat> LongOrig <lon> RotCW 0` |
| `<root>_station_delays.csv` | `station` (station code), `phase`, `delay_s`, `n`: the corrections the model was inverted with, for `LOCDELAY` |
| `<root>_export.json` | provenance: model, S model, source of the corrections, grid, velocity ranges |

- **Model**: evaluated as `fm3d` evaluates it (tensor-product cubic B-spline of the nodes). Where the
  NonLinLoc grid reaches beyond the FMTOMO grid the node grid is padded with its edge nodes (the
  unchanged 1-D starting model), so the export continues the 1-D model laterally, its top layer above
  and its half-space below. A P-only inversion's S grid keeps the starting model's Vp/Vs.
- **Station corrections**: the model only applies together with the corrections subtracted from the
  observed times before the inversion: Stage 3 terms (`model3d.estimate_station_terms`), iterated terms
  (`fmtomo.station_terms: iterate`) or the VELEST delays (`model3d.apply_velest_station_delays`).
  NonLinLoc's `LOCDELAY` subtracts the delay from the observed time, the same sign. VELEST aliases are
  mapped back to the station codes (`01_prep/aliases.csv`).
- **Grid**: `--origin` must equal NonLinLoc's `TRANS SIMPLE` origin. The extent (`--half-width-km`,
  `--zmin-km`, `--zmax-km`; z positive down from sea level) must contain NonLinLoc's location grid
  (OpenMEQ checks this: with `nlloc.grid.size_km: [20, 20, 10]` the location grid spans x, y +-10 km and
  z from -2 km, so the defaults +-10.5 km and -2.5..10.5 km fit).

## Use in OpenMEQ

```yaml
nlloc:
  velocity_model:
    type: 3d
    file: /path/to/nlloc_model/macban3d          # root of <root>.{P,S}.mod.{hdr,buf}
  grid:
    origin: [14.100, 121.210]                    # = --origin
  station_delays_csv: /path/to/nlloc_model/macban3d_station_delays.csv
```

## Accuracy (MacBan, `runs_ps_fixedterms`)

`Grid2Time` (Podvin-Lecomte finite differences) on the exported grids was compared with the `fm3d`
times of all 1884 P and 1584 S picks at the final hypocentres. It is exact in a homogeneous model but
overestimates times near a station in the slow, steep-gradient top layer, roughly in proportion to the
spacing (vertical rays below MB04: +19 ms at 0.1 km, +8 ms at 0.05 km, +4 ms at 0.025 km). At the
default 0.05 km the times are 1.2-1.3 % longer than `fm3d` (P +16 ms, S +27 ms median), almost all of it
common to an event: after removing each event's mean (its origin time) the differences are 6 ms (P) and
10 ms (S) RMS, and the per-station offsets are within +-8 ms (P) and +-13 ms (S), well below the pick
uncertainties (20 and 50 ms). Each 0.05 km grid is ~185 MB, and so is every station's travel-time grid.
