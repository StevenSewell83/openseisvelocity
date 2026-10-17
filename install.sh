#!/usr/bin/env bash
# openSeisVelocity one-command installer.
#
# Usage: ./install.sh [options]
#
# Python environment:
#   --env-name NAME      conda environment name (default: osv)
#   --prefix PATH        environment location (conda -p / venv directory) instead of a name
#   --no-gmt             skip the optional GMT/PyGMT add-on (environment-gmt.yml)
#   --system-compilers   use the system gfortran/cmake/make instead of conda-forge compilers
#   --venv               create a Python venv (uv if available, else python -m venv) and pip
#                        install, instead of a conda environment; implies --system-compilers
#                        (default prefix: ~/.local/share/osv/venv). GMT is not installed.
#   --conda-exe PATH     use this conda/mamba/micromamba executable (env OSV_CONDA_EXE)
# Build / install:
#   --jobs N             parallel build jobs (default: nproc)
#   --dev                editable install (pip install -e) with dev dependencies
#   --skip-tests         skip the final doctor/smoke checks (and pytest with --dev)
#   --velest-ieq N  --velest-ist N  --velest-inltot N  --velest-itotmodels N
#   --velest-inshot N  --velest-maxobs N   VELEST array limits (see scripts/build_fortran.sh)
#   --no-velest | --no-fmtomo | --no-extras | --extras-prefix P | --clean   (build passthrough)
# Other:
#   --dry-run            print the steps without doing anything
#   --force              continue despite failed system checks (e.g. low disk space)
#   -h, --help           show this help
#
# Environment variables: OSV_MICROMAMBA_URL (download URL of the micromamba tarball),
# OSV_MICROMAMBA_HOME (default ~/.local/osv-micromamba), OSV_CONDA_EXE.
# Re-running is safe: existing environments are updated, the build is incremental.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="osv"; PREFIX=""; GMT=1; SYSTEM_COMPILERS=0; VENV=0; CONDA_EXE_ARG="${OSV_CONDA_EXE:-}"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
DEV=0; SKIP_TESTS=0; DRY_RUN=0; FORCE=0
BUILD_ARGS=()
MICROMAMBA_HOME="${OSV_MICROMAMBA_HOME:-$HOME/.local/osv-micromamba}"
# micromamba tarball URL; {PLATFORM} is replaced by linux-64, linux-aarch64, osx-64 or osx-arm64.
MICROMAMBA_URL="${OSV_MICROMAMBA_URL:-https://micro.mamba.pm/api/micromamba/{PLATFORM}/latest}"
MIN_DISK_GB_CONDA=6      # conda env with compilers + geo stack + package cache
MIN_DISK_GB_VENV=2
MIN_DISK_GB_GMT=3        # extra when GMT is installed
MGR=""; MGR_KIND=""

log()  { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

# run CMD...: execute, or only print under --dry-run
run() {
  if [[ $DRY_RUN -eq 1 ]]; then printf '[dry-run] %s\n' "$*"; return 0; fi
  "$@"
}

need_val() { [[ $# -ge 2 && -n "$2" ]] || die "option $1 needs a value (see --help)"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) need_val "$@"; ENV_NAME="$2"; shift 2 ;;
    --prefix) need_val "$@"; PREFIX="$2"; shift 2 ;;
    --no-gmt) GMT=0; shift ;;
    --system-compilers) SYSTEM_COMPILERS=1; shift ;;
    --venv) VENV=1; SYSTEM_COMPILERS=1; shift ;;
    --conda-exe) need_val "$@"; CONDA_EXE_ARG="$2"; shift 2 ;;
    --jobs|-j) need_val "$@"; JOBS="$2"; shift 2 ;;
    --dev) DEV=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --velest-ieq|--velest-ist|--velest-inltot|--velest-itotmodels|--velest-inshot|--velest-maxobs|--extras-prefix)
      need_val "$@"; BUILD_ARGS+=("$1" "$2"); shift 2 ;;
    --no-velest|--no-fmtomo|--no-extras|--clean) BUILD_ARGS+=("$1"); shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
done
[[ "$JOBS" =~ ^[0-9]+$ && "$JOBS" -ge 1 ]] || die "--jobs must be a positive integer"
[[ -f "$ROOT/environment.yml" && -f "$ROOT/scripts/build_fortran.sh" ]] \
  || die "run install.sh from a complete openSeisVelocity checkout ($ROOT lacks environment.yml)"

if [[ $VENV -eq 1 ]]; then
  [[ -n "$PREFIX" ]] || PREFIX="$HOME/.local/share/osv/venv"
  if [[ $GMT -eq 1 ]]; then GMT=0; log "venv mode: GMT/PyGMT not installed (needs the GMT C library; use the conda route)"; fi
fi
[[ -z "$PREFIX" ]] || { mkdir -p "$(dirname "$PREFIX")" 2>/dev/null || true; PREFIX="$(cd "$(dirname "$PREFIX")" 2>/dev/null && pwd)/$(basename "$PREFIX")"; }

# ---------------------------------------------------------------- system checks
check_system() {
  log "Checking the system"
  local os arch; os="$(uname -s)"; arch="$(uname -m)"
  case "$os" in Linux|Darwin) ;; *) die "unsupported OS '$os' (Linux and macOS only; on Windows use WSL2)" ;; esac
  echo "    OS: $os  arch: $arch  jobs: $JOBS"
  command -v git >/dev/null || warn "git not found (not required to install, needed for development)"

  local need=$MIN_DISK_GB_CONDA target avail_kb
  [[ $VENV -eq 1 ]] && need=$MIN_DISK_GB_VENV
  [[ $GMT -eq 1 ]] && need=$((need + MIN_DISK_GB_GMT))
  target="${PREFIX:-$HOME}"
  while [[ ! -d "$target" && "$target" != "/" ]]; do target="$(dirname "$target")"; done
  avail_kb="$(df -Pk "$target" | awk 'NR==2{print $4}')"
  echo "    free disk at $target: $((avail_kb / 1048576)) GB (need about $need GB)"
  if [[ "$avail_kb" -lt $((need * 1048576)) ]]; then
    if [[ $FORCE -eq 1 || $DRY_RUN -eq 1 ]]; then warn "low disk space (continuing: --force or --dry-run)"
    else die "not enough free disk space at $target (need about $need GB); free space, choose --prefix on a larger disk, or pass --force"; fi
  fi

  if [[ $SYSTEM_COMPILERS -eq 1 ]]; then
    local missing=()
    for t in gfortran cmake make; do command -v "$t" >/dev/null || missing+=("$t"); done
    if [[ ${#missing[@]} -gt 0 ]]; then
      if [[ $DRY_RUN -eq 1 ]]; then warn "missing system tools: ${missing[*]} (a real run would stop here)"
      else die "--system-compilers needs ${missing[*]} on PATH (e.g. 'sudo apt install gfortran cmake make' or 'brew install gcc cmake'; on HPC try 'module load gcc cmake')"; fi
    else
      echo "    gfortran: $(gfortran -dumpfullversion 2>/dev/null || gfortran --version | head -1)  cmake: $(cmake --version | head -1)"
    fi
  fi
}

# ---------------------------------------------------------------- conda manager
platform() {
  local os arch; os="$(uname -s)"; arch="$(uname -m)"
  case "$os-$arch" in
    Linux-x86_64) echo linux-64 ;;
    Linux-aarch64|Linux-arm64) echo linux-aarch64 ;;
    Darwin-x86_64) echo osx-64 ;;
    Darwin-arm64) echo osx-arm64 ;;
    *) die "no micromamba build for $os-$arch; install conda manually or use --venv" ;;
  esac
}

classify() { # sets MGR and MGR_KIND from an executable path
  MGR="$1"
  case "$(basename "$MGR")" in micromamba*) MGR_KIND=micromamba ;; mamba*) MGR_KIND=mamba ;; *) MGR_KIND=conda ;; esac
}

find_manager() {
  if [[ -n "$CONDA_EXE_ARG" ]]; then
    [[ -x "$CONDA_EXE_ARG" ]] || die "--conda-exe '$CONDA_EXE_ARG' is not executable"
    classify "$CONDA_EXE_ARG"; return
  fi
  local c d
  for c in micromamba mamba conda; do
    if command -v "$c" >/dev/null 2>&1; then classify "$(command -v "$c")"; return; fi
  done
  for d in "$MICROMAMBA_HOME/bin/micromamba" "$HOME/micromamba/bin/micromamba" \
           "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" \
           "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
    if [[ -x "$d" ]]; then classify "$d"; return; fi
  done
  return 1
}

bootstrap_micromamba() {
  local plat url dest="$MICROMAMBA_HOME"
  plat="$(platform)"; url="${MICROMAMBA_URL//\{PLATFORM\}/$plat}"
  log "No conda/mamba/micromamba found: installing micromamba into $dest (no sudo needed)"
  echo "    from $url"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] download $url and extract bin/micromamba to $dest/bin"
    MGR="$dest/bin/micromamba"; MGR_KIND=micromamba; return
  fi
  mkdir -p "$dest/bin"
  local tmp; tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" RETURN
  if command -v curl >/dev/null; then curl -fsSL "$url" -o "$tmp/mm.tar.bz2" || die "download failed: $url (no internet? see docs/install.md, or use --venv / --conda-exe)"
  elif command -v wget >/dev/null; then wget -q "$url" -O "$tmp/mm.tar.bz2" || die "download failed: $url"
  else die "need curl or wget to download micromamba"; fi
  tar -xjf "$tmp/mm.tar.bz2" -C "$tmp" bin/micromamba || die "could not extract micromamba (is bzip2 installed?)"
  install -m 755 "$tmp/bin/micromamba" "$dest/bin/micromamba"
  classify "$dest/bin/micromamba"
}

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$MICROMAMBA_HOME/root}"

mgr_env_exists() {
  if [[ -n "$PREFIX" ]]; then [[ -d "$PREFIX/conda-meta" ]]; return; fi
  "$MGR" env list 2>/dev/null | grep -Eq "(^|[[:space:]])${ENV_NAME}([[:space:]]|\$)"
}

target_args() { if [[ -n "$PREFIX" ]]; then printf '%s\n' -p "$PREFIX"; else printf '%s\n' -n "$ENV_NAME"; fi; }

env_run() { # run a command inside the (activated) environment
  local t; mapfile -t t < <(target_args)
  if [[ $MGR_KIND == conda ]]; then "$MGR" run --no-capture-output "${t[@]}" "$@"
  else "$MGR" run "${t[@]}" "$@"; fi
}

conda_file() { # echo the environment file to use (filtered when using system compilers)
  local f="$1"
  if [[ $SYSTEM_COMPILERS -eq 1 && "$f" == "$ROOT/environment.yml" ]]; then
    local out="${TMPDIR:-/tmp}/osv-environment-$$.yml"
    grep -Ev '^[[:space:]]*-[[:space:]]*(fortran-compiler|gfortran|cmake|make)([<>=[:space:]]|$)' "$f" > "$out"
    echo "$out"
  else echo "$f"; fi
}

apply_env_file() { # create or update the environment from a yml file
  local f; f="$(conda_file "$1")"
  local t; mapfile -t t < <(target_args)
  if mgr_env_exists; then
    log "Updating existing environment from $(basename "$1")"
    case $MGR_KIND in
      micromamba) run "$MGR" install -y "${t[@]}" -f "$f" ;;
      *) run "$MGR" env update "${t[@]}" -f "$f" --prune ;;
    esac
  else
    log "Creating environment from $(basename "$1") (the solve can take several minutes)"
    case $MGR_KIND in
      micromamba) run "$MGR" create -y "${t[@]}" -f "$f" ;;
      *) run "$MGR" env create "${t[@]}" -f "$f" ;;
    esac
  fi
  [[ "$f" == "$1" ]] || rm -f "$f"
}

resolve_prefix() {
  [[ -z "$PREFIX" ]] || return 0
  if [[ $DRY_RUN -eq 1 ]]; then PREFIX="<prefix of conda env '$ENV_NAME'>"; return; fi
  PREFIX="$(env_run printenv CONDA_PREFIX | tail -n1)"
  [[ -d "$PREFIX" ]] || die "could not determine the prefix of environment '$ENV_NAME'"
}

# ---------------------------------------------------------------- python environments
setup_conda() {
  if ! find_manager; then bootstrap_micromamba; fi
  log "Using $MGR_KIND: $MGR"
  apply_env_file "$ROOT/environment.yml"
  if [[ $GMT -eq 1 ]]; then apply_env_file "$ROOT/environment-gmt.yml"; fi
  resolve_prefix
}

setup_venv() {
  if [[ -x "$PREFIX/bin/python" ]]; then
    log "Reusing existing venv at $PREFIX"
  elif command -v uv >/dev/null 2>&1; then
    log "Creating venv with uv (Python 3.11) at $PREFIX"
    run uv venv --python 3.11 "$PREFIX"
  else
    local py=""
    for c in python3.12 python3.11 python3; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then py="$c"; break; fi
    done
    [[ -n "$py" ]] || die "need Python >= 3.11 (or uv) for --venv; see docs/install.md"
    log "Creating venv with $py at $PREFIX"
    run "$py" -m venv "$PREFIX"
  fi
}

pip_install() { # pip_install ARGS... : install into the target environment
  if [[ $VENV -eq 1 ]]; then
    if command -v uv >/dev/null 2>&1; then run uv pip install --python "$PREFIX/bin/python" "$@"
    else run "$PREFIX/bin/python" -m pip install "$@"; fi
  else
    run env_run python -m pip install "$@"
  fi
}

install_package() {
  local extras="" spec
  [[ $DEV -eq 1 ]] && extras="dev"
  if [[ $VENV -eq 1 ]]; then extras="${extras:+$extras,}geo"; fi
  spec="${extras:+[$extras]}"
  local mode=(); [[ $DEV -eq 1 ]] && mode=(-e)
  log "Installing the openseisvelocity package ${mode[*]:+(editable) }${spec:+with extras $extras}"
  if [[ $VENV -eq 1 ]]; then
    # Geo extras (rasterio, cartopy, pyvista) can lack wheels on some platforms: fall back without them.
    pip_install "${mode[@]}" "$ROOT$spec" || {
      warn "installing the 'geo' extras failed; retrying without them (geology/well inputs will be unavailable)"
      spec="${DEV:+[dev]}"; [[ $DEV -eq 1 ]] || spec=""
      pip_install "${mode[@]}" "$ROOT$spec"
    }
  else
    pip_install "${mode[@]}" "$ROOT$spec"
  fi
}

# ---------------------------------------------------------------- Fortran
build_fortran() {
  log "Building the Fortran executables (VELEST, FMTOMO, Extras)"
  local tag bdir
  tag="$(printf '%s' "$PREFIX" | cksum | cut -d' ' -f1)"
  bdir="$ROOT/build/install-$tag"
  if [[ $SYSTEM_COMPILERS -eq 1 ]]; then
    run bash "$ROOT/scripts/build_fortran.sh" --prefix "$PREFIX" --jobs "$JOBS" --build-dir "$bdir" \
      --fc "${FC:-gfortran}" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}
  else
    # inside the activated conda env $FC is the conda-forge gfortran
    run env_run bash -c 'exec "$0" --fc "${FC:-gfortran}" "$@"' "$ROOT/scripts/build_fortran.sh" \
      --prefix "$PREFIX" --jobs "$JOBS" --build-dir "$bdir" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}
  fi
}

# ---------------------------------------------------------------- checks
py_in_env() { if [[ $VENV -eq 1 ]]; then "$PREFIX/bin/python" "$@"; else env_run python "$@"; fi; }
osv_in_env() { if [[ $VENV -eq 1 ]]; then "$PREFIX/bin/osv" "$@"; else env_run osv "$@"; fi; }

smoke_checks() {
  log "Running checks"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] osv doctor  (fallback: import osv; which_tool velest/fm3d)"
    [[ $DEV -eq 1 ]] && echo "[dry-run] pytest -q"
    return 0
  fi
  local out rc=0
  out="$(osv_in_env doctor 2>&1)" || rc=$?
  if [[ $rc -eq 0 ]]; then echo "$out"
  elif grep -q 'not implemented' <<<"$out"; then
    echo "    'osv doctor' is not implemented yet; running the minimal smoke check"
    py_in_env -c '
import osv
from osv.tools import which_tool
for name in ("velest", "fm3d"):
    print("   ", name, "->", which_tool(name))
print("    osv", osv.__version__, "imported from", osv.__file__)
' || die "smoke check failed: the osv package or the Fortran tools are not usable"
  else
    echo "$out" >&2; die "'osv doctor' reported problems (see above)"
  fi
  if [[ $DEV -eq 1 ]]; then
    log "Running the test suite"
    (cd "$ROOT" && py_in_env -m pytest -q) || { TESTS_FAILED=1; warn "pytest reported failures (the installation itself completed)"; }
  fi
}

summary() {
  echo
  log "openSeisVelocity installed"
  echo "    environment: $PREFIX"
  echo "    tools:       $PREFIX/bin/osv-tools"
  if [[ $DRY_RUN -eq 1 ]]; then echo "    (dry run: nothing was changed)"; fi
  echo "    activate with:"
  if [[ $VENV -eq 1 ]]; then echo "        source $PREFIX/bin/activate"
  elif [[ $MGR_KIND == micromamba ]]; then
    echo "        eval \"\$($MGR shell hook -s bash)\" && micromamba activate ${PREFIX}"
  else echo "        conda activate ${PREFIX}"; fi
  echo "    then check with:  osv doctor"
}

TESTS_FAILED=0
log "openSeisVelocity installer$([[ $DRY_RUN -eq 1 ]] && echo ' (dry run)')"
check_system
if [[ $VENV -eq 1 ]]; then setup_venv; else setup_conda; fi
build_fortran
install_package
[[ $SKIP_TESTS -eq 1 ]] || smoke_checks
summary
exit "$((TESTS_FAILED * 3))"
