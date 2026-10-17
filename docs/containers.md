# Containers (Docker and Apptainer)

The images contain the Python package, VELEST, FMTOMO and the FMTOMO Extras, built with
`scripts/build_fortran.sh` on Debian 12 (gfortran 12, default VELEST array limits). The
executables live in `/opt/osv/bin/osv-tools` and `OSV_TOOLS_DIR` points there. The optional
geo extras (`cartopy`, `rasterio`, `pyvista`, `lasio`) and GMT are not included; use the
conda install if you need them.

## Docker

```bash
docker build -t osv .
docker run --rm osv doctor --quick          # health check
docker run --rm osv doctor                  # with the VELEST/fm3d smoke tests
docker run --rm osv --help
```

The entrypoint is `osv`, so arguments are `osv` subcommands. To run on a project, mount its
directory at `/work` (the working directory) and run as your own user so output files are
owned by you:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/projects/myproj:/work" osv doctor --config config.yaml
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD/projects/myproj:/work" osv <subcommand> --config config.yaml
```

Paths inside `config.yaml` must be valid inside the container (relative to `/work`, or under
another `-v host:container` mount, e.g. `-v /data/StationXML:/data/StationXML:ro`).
Use `--cpus`/`--memory` to limit resources; `project.n_jobs` should not exceed the CPUs given.
For an interactive shell: `docker run --rm -it --entrypoint bash -v "$PWD:/work" osv`.

The image has a `HEALTHCHECK` running `osv doctor --quick`. The VELEST limits are the build
defaults; to change them add `--velest-ieq N ...` to the `scripts/build_fortran.sh` line in the
`fortran` stage of the `Dockerfile`.

## Apptainer / Singularity (HPC)

Build on a machine with Apptainer (and internet), then copy the `.sif` to the cluster:

```bash
apptainer build osv.sif apptainer.def              # self-contained, from the repo root
# or convert the Docker image:
apptainer build osv.sif docker-daemon://osv:latest
```

```bash
apptainer run osv.sif doctor --quick
apptainer run --bind /scratch/myproj:/work --pwd /work osv.sif doctor --config config.yaml
apptainer exec osv.sif ls /opt/osv/bin/osv-tools
```

Apptainer maps your user and `$HOME` automatically, so no `-u` is needed. On clusters set
`APPTAINER_TMPDIR` to a large scratch disk when building, and bind the data directories you
need with `--bind`.

## Continuous integration

`.github/workflows/ci.yml` builds the Docker image and runs `docker run osv:ci doctor --quick`
on every push and pull request.
