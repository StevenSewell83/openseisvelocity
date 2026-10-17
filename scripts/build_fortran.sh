#!/usr/bin/env bash
# Build and install VELEST, FMTOMO and FMTOMO Extras (CMake, out of tree).
#
# Usage: scripts/build_fortran.sh [options]
#   --prefix DIR        install into DIR/bin/osv-tools (data in DIR/share/osv-tools).
#                       Default (no --prefix): <repo>/build/bin  (searched by osv.tools)
#   --bindir DIR        install executables directly into DIR (overrides the above)
#   --build-dir DIR     CMake build directory (default <repo>/build/fortran)
#   --jobs N            parallel jobs (default: nproc)
#   --fc COMPILER       Fortran compiler (default: $FC, else gfortran)
#   --extras-prefix P   prefix for FMTOMO Extras file names (e.g. "x_"; set OSV_EXTRAS_PREFIX to match)
#   --no-velest | --no-fmtomo | --no-extras   skip a component
#   --velest-ieq N --velest-ist N --velest-inltot N --velest-itotmodels N
#   --velest-inshot N --velest-maxobs N       VELEST array limits
#                       (defaults 5000 / 300 / 20 / 2 / 0 / 180)
#   --clean             remove the build directory first
#   -h, --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX=""; BINDIR=""; BUILD_DIR="$ROOT/build/fortran"; JOBS="$(nproc 2>/dev/null || echo 2)"
FC_ARG="${FC:-gfortran}"; CLEAN=0
CM=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --bindir) BINDIR="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --jobs|-j) JOBS="$2"; shift 2 ;;
    --fc) FC_ARG="$2"; shift 2 ;;
    --extras-prefix) CM+=("-DOSV_EXTRAS_PREFIX=$2"); shift 2 ;;
    --no-velest) CM+=("-DOSV_BUILD_VELEST=OFF"); shift ;;
    --no-fmtomo) CM+=("-DOSV_BUILD_FMTOMO=OFF"); shift ;;
    --no-extras) CM+=("-DOSV_BUILD_EXTRAS=OFF"); shift ;;
    --velest-ieq) CM+=("-DOSV_VELEST_IEQ=$2"); shift 2 ;;
    --velest-ist) CM+=("-DOSV_VELEST_IST=$2"); shift 2 ;;
    --velest-inltot) CM+=("-DOSV_VELEST_INLTOT=$2"); shift 2 ;;
    --velest-itotmodels) CM+=("-DOSV_VELEST_ITOTMODELS=$2"); shift 2 ;;
    --velest-inshot) CM+=("-DOSV_VELEST_INSHOT=$2"); shift 2 ;;
    --velest-maxobs) CM+=("-DOSV_VELEST_MAXOBS=$2"); shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "build_fortran.sh: unknown option '$1' (see --help)" >&2; exit 2 ;;
  esac
done

command -v cmake >/dev/null || { echo "cmake not found: install it (e.g. conda install cmake)" >&2; exit 1; }
command -v "$FC_ARG" >/dev/null || { echo "Fortran compiler '$FC_ARG' not found: install gfortran or pass --fc" >&2; exit 1; }

if [[ -n "$BINDIR" ]]; then
  BIN="$BINDIR"; DATA="$(dirname "$BINDIR")/share/osv-tools"
elif [[ -n "$PREFIX" ]]; then
  BIN="$PREFIX/bin/osv-tools"; DATA="$PREFIX/share/osv-tools"
else
  BIN="$ROOT/build/bin"; DATA="$ROOT/build/share/osv-tools"
fi

[[ $CLEAN -eq 1 ]] && rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
FC="$FC_ARG" cmake -S "$ROOT/fortran" -B "$BUILD_DIR" -DCMAKE_Fortran_COMPILER="$FC_ARG" \
  -DOSV_BIN_DIR="$BIN" -DOSV_DATA_DIR="$DATA" "${CM[@]}"
cmake --build "$BUILD_DIR" -j "$JOBS"
cmake --install "$BUILD_DIR" >/dev/null
echo "Installed executables into $BIN (data files in $DATA)"
