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
[docs/install.md](docs/install.md).
