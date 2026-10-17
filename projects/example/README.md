# Setting up a new project

This directory documents the layout of an openSeisVelocity project; it holds no data and no
`config.yaml`. A project is any directory containing a `config.yaml`. A complete worked example is
`projects/rk-nm-2017`, explained step by step in the
[tutorial](../../docs/tutorial.md).

## 1. Create the project

```bash
conda activate osv
osv init projects/mysite            # or: osv init /data/mysite --name mysite
```

This writes `projects/mysite/config.yaml`: every setting of the schema with its default, allowed
values and a description as comments. (The same keys are listed in the
[configuration reference](../../docs/config_reference.md).) `osv init` refuses to overwrite an
existing config unless you pass `--force`.

## 2. Lay out your data

```
projects/mysite/
  config.yaml
  catalogue.xml            # QuakeML: events, origins, picks (with time uncertainties)
  stations/                # StationXML: one or more files, with channel coordinates and depths
    network1.xml
    network2.xml
  dem.tif                  # optional GeoTIFF elevation model (inputs.dem)
  geology/, wells/         # optional, for user models (see docs/user_models.md)
  runs/                    # created by osv: one numbered directory per stage
```

The files can live anywhere: relative paths in `config.yaml` are resolved against the directory
containing the config, `inputs.stationxml` and `inputs.quakeml` accept glob patterns, and
`${VAR}` / `${VAR:-default}` are expanded from the environment (handy for data shared between
projects).

Requirements on the data:

* QuakeML: every event has an origin with a time, latitude, longitude and depth, and picks with
  phase hints (`P`/`S`), waveform IDs matching the StationXML, and ideally time uncertainties
  (they set the pick weights, `inputs.weight_source`).
* StationXML: coordinates and elevation for each station epoch; borehole sensors need a channel
  depth.

## 3. Edit the config

The settings you almost always need to touch:

```yaml
project:
  name: mysite
  n_jobs: 8                       # parallel processes; -1 uses all cores
inputs:
  stationxml: [./stations/*.xml]
  quakeml: [./catalogue.xml]
coordinates:
  origin: {lat: -38.55, lon: 176.19}   # local-system origin; null = station centroid
  depth_datum: sea_level               # how the catalogue depths are referenced
data_selection:
  depth_km: [-1.0, 8.0]
  min_p: 6
  max_epi_dist_km: 25.0
velest:
  layers_top_km: [-0.6, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]
  vp_bounds: [[1.9, 3.6], [2.8, 4.3], ...]   # one [lower, upper] pair per layer
```

Everything else has defaults. Check what will be used with `osv config show -c config.yaml`;
validation errors name the offending key. Choose the layers using the guidance in
[Stage 2](../../docs/stage2_velest.md#choosing-layers-and-damping), and read
[conventions](../../docs/conventions.md) for the coordinate and depth conventions.

## 4. Check and run

```bash
cd projects/mysite
osv doctor --config config.yaml     # tools, disk space, VELEST array limits for your data
osv prep                            # start with Stage 1 and read runs/01_prep/report.html
osv run                             # then the rest; re-running skips up-to-date stages
```

Run the early stages one at a time on a new dataset and read their reports before continuing;
the later stages (trade-off sweeps, resolution tests) take the longest. When something fails or a
warning is unclear see [troubleshooting](../../docs/troubleshooting.md). The project report is
`runs/index.html`.

Add `runs/` to `.gitignore`: it is large and fully reproducible from `config.yaml` and the data
(the config hash and random seed are recorded in every manifest).
