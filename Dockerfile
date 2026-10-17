# syntax=docker/dockerfile:1
# openSeisVelocity container image.
#   docker build -t osv .
#   docker run --rm osv doctor --quick
# See docs/containers.md.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# ---- stage 1: build VELEST, FMTOMO and FMTOMO Extras ------------------------------------
FROM ${PYTHON_IMAGE} AS fortran
RUN apt-get update \
 && apt-get install -y --no-install-recommends gfortran cmake make \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY scripts/build_fortran.sh scripts/build_fortran.sh
COPY fortran fortran
RUN scripts/build_fortran.sh --prefix /opt/osv --build-dir /tmp/fbuild --jobs "$(nproc)" \
 && rm -rf /tmp/fbuild

# ---- stage 2: Python environment (wheels only, no compilers needed) ---------------------
FROM ${PYTHON_IMAGE} AS pyenv
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src src
RUN /opt/venv/bin/pip install .

# ---- stage 3: runtime --------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgfortran5 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=fortran /opt/osv /opt/osv
COPY --from=pyenv /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    OSV_TOOLS_DIR=/opt/osv/bin/osv-tools \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib
# Not root by default; override with `docker run -u $(id -u):$(id -g)` for bind mounts.
RUN useradd --create-home --uid 1000 --home-dir /home/osv osv \
 && mkdir -p /work && chown osv /work
USER osv
ENV HOME=/home/osv
WORKDIR /work
HEALTHCHECK --interval=5m --timeout=60s --start-period=10s --retries=1 \
  CMD ["osv", "doctor", "--quick"]
ENTRYPOINT ["osv"]
CMD ["--help"]
