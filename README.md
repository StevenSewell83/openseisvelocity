# openSeisVelocity

Earthquake-based velocity inversion for microseismic data: Monte Carlo VELEST 1D models, 3D starting models (incl. geology/well data), FMTOMO 3D inversion with trade-off curves, and resolution testing.

See `PLAN.md` for the design.

## Install

```bash
./install.sh            # conda environment "osv" with compilers; builds VELEST/FMTOMO
conda activate osv      # (the exact command is printed at the end)
osv doctor
```

No conda? The installer downloads micromamba locally (no sudo). Machines with gfortran and
CMake can use `./install.sh --venv`. Options (`--prefix`, `--env-name`, `--no-gmt`,
`--system-compilers`, `--jobs`, `--dev`, `--dry-run`, ...) and troubleshooting are in
[docs/install.md](docs/install.md). Docker and Apptainer images: [docs/containers.md](docs/containers.md).

## Documentation

User guide, tutorial, stage guides and the generated configuration reference are in `docs/`.
Build the site with `uv pip install -e ".[docs]" && mkdocs build --strict` (or `mkdocs serve`).
New project? See [projects/example/README.md](projects/example/README.md).
