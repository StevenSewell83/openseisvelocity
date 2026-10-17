# Installing openSeisVelocity

## Requirements

- Linux or macOS (Windows: use WSL2). x86-64 or arm64.
- About 6 GB free disk for the conda route (8-9 GB with GMT), 2 GB for `--venv`.
- `bash`, `curl` or `wget` (to fetch micromamba), and an internet connection (or see
  "No internet" below).
- Nothing else: with the default route, `install.sh` supplies the compilers, CMake, Python
  and every dependency through conda-forge. No sudo is needed.

## Quick start

```bash
git clone <repo> openSeisVelocity && cd openSeisVelocity
./install.sh                 # creates the conda environment "osv"
conda activate osv           # the command is printed at the end of the install
osv doctor
```

`install.sh` does the following, and is safe to re-run (existing environments are updated,
the Fortran build is incremental):

1. Checks OS, architecture, free disk space and (with `--system-compilers`) the compilers.
2. Finds conda, mamba or micromamba. If none exists it downloads micromamba into
   `~/.local/osv-micromamba` (no sudo, nothing else on the machine is touched).
3. Creates or updates the environment from `environment.yml` (and `environment-gmt.yml`).
4. Builds VELEST, FMTOMO and the FMTOMO Extras with `scripts/build_fortran.sh` using the
   environment's compilers. Executables go to `<env>/bin/osv-tools/`, so they never clash
   with an unrelated `velest` on `PATH`.
5. `pip install` of the package (editable with `--dev`).
6. Runs `osv doctor` (or, until it is available, a minimal smoke check that imports `osv` and
   locates `velest` and `fm3d`). With `--dev` it also runs `pytest -q`.

The environment is never activated or registered in `~/.bashrc`; your base conda
environment is not modified.

## Options

| Option | Meaning |
|---|---|
| `--env-name NAME` | conda environment name (default `osv`) |
| `--prefix PATH` | environment location (conda `-p`, or the venv directory) instead of a name |
| `--no-gmt` | skip GMT/PyGMT (`environment-gmt.yml`); large download |
| `--system-compilers` | use system `gfortran`, `cmake`, `make` and drop them from the conda environment |
| `--venv` | no conda: create a venv (uv if installed, else `python -m venv`) and pip install; implies `--system-compilers`; default prefix `~/.local/share/osv/venv`; no GMT |
| `--conda-exe PATH` | use a specific conda/mamba/micromamba (also `OSV_CONDA_EXE`) |
| `--jobs N` | parallel build jobs (default: all cores) |
| `--dev` | editable install with dev dependencies, and run the test suite at the end |
| `--skip-tests` | skip the final doctor/smoke checks |
| `--velest-ieq N` (also `--velest-ist`, `--velest-inltot`, `--velest-itotmodels`, `--velest-inshot`, `--velest-maxobs`) | VELEST array limits, passed to `scripts/build_fortran.sh` |
| `--no-velest`, `--no-fmtomo`, `--no-extras`, `--extras-prefix P`, `--clean` | other build options |
| `--dry-run` | print the steps without doing anything |
| `--force` | continue despite a failed disk-space check |

Environment variables: `OSV_MICROMAMBA_URL` (micromamba download URL, `{PLATFORM}` is replaced
by `linux-64`, `linux-aarch64`, `osx-64`...), `OSV_MICROMAMBA_HOME` (default
`~/.local/osv-micromamba`).

Examples:

```bash
./install.sh --env-name osv-dev --dev --jobs 8
./install.sh --prefix /data/envs/osv --no-gmt
./install.sh --venv --prefix ~/osv-venv          # uses system gfortran 9+ and cmake
./install.sh --velest-ieq 20000 --velest-ist 500 # larger VELEST arrays
./install.sh --dry-run                           # see what would happen
```

Using the tools without activating the environment: `osv` finds executables in
`$OSV_TOOLS_DIR`, `$CONDA_PREFIX/bin/osv-tools`, `<sys.prefix>/bin/osv-tools` (venvs and
`<env>/bin/osv`), the repo's `build/bin`, and finally `PATH`.

Docker and Apptainer images are described in [containers.md](containers.md).

## Troubleshooting

**No conda.** Nothing to do: the installer bootstraps micromamba. To use your own, put it
on `PATH` or pass `--conda-exe`. On a machine without conda and without access to
micro.mamba.pm, use `--venv`.

**No internet / behind a proxy.** Set `http_proxy`/`https_proxy` (conda, curl and pip honour
them). For offline machines: download the micromamba tarball elsewhere and point
`OSV_MICROMAMBA_URL` at a `file:///path/to/tarball` URL or a local mirror, mirror conda-forge
and set `CONDA_CHANNELS`/`.condarc`, or build a Docker/Apptainer image on a connected
machine (see [containers.md](containers.md)).

**Not enough disk space.** The conda package cache and the environment together need about
6 GB. Use `--prefix` on a larger disk, set `CONDA_PKGS_DIRS` to a larger disk, or run
`conda clean -a`. `--force` bypasses the check.

**HPC clusters.** Load a compiler and CMake from modules and avoid conda on the login node if
policy forbids it:
`module load gcc cmake python/3.11; ./install.sh --venv --prefix $HOME/osv-venv`.
gfortran must be 9 or newer. Run the install on a login node with internet; run `osv`
on compute nodes after `source $HOME/osv-venv/bin/activate`. An Apptainer image is the
other option.

**macOS.** The conda route works on Intel and Apple silicon (`osx-64`, `osx-arm64`). With
`--venv` or `--system-compilers`, install the toolchain with Homebrew
(`brew install gcc cmake`); `gfortran` comes with `gcc`. The Xcode command-line tools
must be installed. If a package is missing for your platform, `--venv` skips `geo` extras
automatically and says so.

**The solve is slow.** Install mamba or micromamba first (`conda install -n base -c conda-forge mamba`)
or let the installer fetch micromamba by running it on a `PATH` without conda.

**`velest` or `fm3d` not found.** Run `osv doctor`. Make sure you activated the environment,
or set `OSV_TOOLS_DIR=<env>/bin/osv-tools`. To rebuild: `./install.sh --clean`.

**Mixing gfortran versions.** Do not combine a conda environment with `--system-compilers`
paths in the same build directory; the installer uses a separate CMake build directory per
environment (`build/install-*`), so re-running is safe.

**Reading the `osv doctor` report.** `osv doctor` prints each check as OK, WARN or FAIL with a
fix for every problem; the exit code is 1 if anything failed. Options: `--quick` skips the smoke
tests, `--json` gives machine-readable output, `--config config.yaml` also checks the project
workdir (write permission, free space, `n_jobs`) and compares the VELEST array limits with the
number of events, stations and picks per event from Stage 1, and `--gridloc` adds a ~35 s
FMTOMO Extras test. Common results:

- *`tool:... found on PATH`* (WARN): an old or unrelated binary may shadow the osv build. Run
  `./install.sh` or set `OSV_TOOLS_DIR`.
- *`tool:... not found`* (FAIL): the executable was not built or was renamed or moved. Rebuild
  with `./install.sh --clean`. Missing FMTOMO Extras are only a warning.
- *`needs missing shared libraries`* (FAIL): activate the environment used for the build
  (`conda activate osv`) so `libgfortran` is found.
- *`smoke:velest` / `smoke:fm3d` FAIL*: the built program ran but its results differ from the
  stored reference (miscompiled, e.g. with `-ffast-math`), or it crashed. Rebuild with
  `./install.sh --clean`; the message names the run directory with the logs.
- *`project:ieq` / `project:ist` / `project:maxobs` FAIL*: your data exceed the VELEST array
  limits. Rebuild with e.g. `./install.sh --velest-ieq 20000`.
- *`disk:tmp` WARN* (less than 5 GB free): point `TMPDIR` at a larger disk.
- *`build:compiler` WARN*: no `build_info.json` in `share/osv-tools`; rebuild to record the
  compiler, flags and array limits.

## Uninstall

```bash
conda env remove -n osv              # or: conda env remove -p /path/to/prefix
rm -rf ~/.local/osv-micromamba       # only if the installer bootstrapped micromamba
rm -rf ~/.local/share/osv/venv       # venv route (or your --prefix)
rm -rf build                         # CMake build directories inside the checkout
```

Nothing is written outside the environment, `~/.local/osv-micromamba` and the repository's
`build/` directory.
