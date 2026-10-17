# Patches to vendored Fortran

**No vendored source file has been edited.** All gfortran 11 compatibility is handled by
compiler flags and by build-time transformations in `fortran/CMakeLists.txt`.
Each change to the vendored sources must be listed in the table below.

| File | Change | Reason | Task |
|---|---|---|---|

## Build-time adaptations (no source edits)

| Item | Adaptation | Reason | Task |
|---|---|---|---|
| all programs | `-fallow-argument-mismatch` (when the compiler supports it) | gfortran >= 10 rejects legacy rank/type mismatches (`nn_subsf.f`, `libtau.f`, ...) | osv-8vx.5 |
| VELEST | `-std=legacy -ffixed-line-length-none -O2` (+ `-mcmodel=medium` if static arrays exceed 2 GB) | F77 code with long lines | osv-8vx.4 |
| VELEST | `fortran/velest/velest_common.inc.in` is a template of the vendored `velest_common.inc`: only the six values `ieq`, `inshot`, `itotmodels`, `inltot`, `ist`, `maxobsperevent` are replaced by CMake variables (defaults identical to the vendored include). `velest.f` is copied next to the generated include in the build directory so the generated file is the one `include`d. The vendored `velest_common.inc` stays as the reference for the defaults and is not used by the build. | configurable array limits | osv-8vx.4 |
| FMTOMO `fm3d` | Compiled from the `fmcode` source list of `fmcode/Makefile` in one CMake target (module dependency order resolved by CMake) | replaces `make fm3d` | osv-8vx.5 |
| `ak135.tbl`, `ak135.hed` | Regenerated at build time with `remodl` then `setbrn` (from `fmcode/aktimes`) and installed to `<prefix>/share/osv-tools/`. The tables shipped in the tarball were written with a different Fortran direct-access record-length convention (file is 4x larger) and are not used. fm3d only reads them (from the working directory) for teleseismic sources, never for local-earthquake runs. | gfortran-compatible tables | osv-8vx.5 |
| `tomo3d`, `tomo3di`, `tomo3dj`, `tomo3dps`, `cp_script` | Installed with the shebang `#!/usr/bin/ksh` replaced by `#!/usr/bin/env bash` plus a one-line `print()` shim implementing ksh's `print -u<fd>`; nothing else in the scripts is ksh-specific. | no ksh dependency | osv-8vx.5/.6 |
