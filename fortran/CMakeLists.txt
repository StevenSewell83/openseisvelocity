# Build VELEST, FMTOMO v1.2 and FMTOMO Extras v1.1 for openSeisVelocity.
# Normally driven by scripts/build_fortran.sh; usable directly:
#   cmake -S fortran -B build/fortran -DOSV_BIN_DIR=$PWD/build/bin && cmake --build build/fortran -j && cmake --install build/fortran
cmake_minimum_required(VERSION 3.16)
project(osv_fortran LANGUAGES Fortran)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(OSV_OPT_FLAGS "-O2" CACHE STRING "Optimisation flags for all programs")
set(CMAKE_Fortran_FLAGS_RELEASE "${OSV_OPT_FLAGS}")
# gfortran >= 10 rejects the legacy argument mismatches in the vendored F77 code
# (nn_subsf.f, libtau.f, ...) unless this flag is given.
include(CheckFortranCompilerFlag)
check_fortran_compiler_flag("-fallow-argument-mismatch" OSV_HAS_ALLOW_ARG_MISMATCH)
if(OSV_HAS_ALLOW_ARG_MISMATCH)
  string(APPEND CMAKE_Fortran_FLAGS " -fallow-argument-mismatch")
endif()

set(OSV_BIN_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Directory executables are installed into")
set(OSV_DATA_DIR "" CACHE PATH "Directory for data files (ak135.tbl/.hed); default <bin>/../share/osv-tools")
set(OSV_EXTRAS_PREFIX "" CACHE STRING "Prefix prepended to installed FMTOMO Extras file names")
option(OSV_BUILD_VELEST "Build VELEST" ON)
option(OSV_BUILD_FMTOMO "Build FMTOMO (fm3d, aktimes, base programs)" ON)
option(OSV_BUILD_EXTRAS "Build FMTOMO Extras" ON)

# VELEST array limits (see velest/velest_common.inc.in); defaults = vendored values.
set(OSV_VELEST_IEQ 5000 CACHE STRING "VELEST: max number of earthquakes")
set(OSV_VELEST_IST 300 CACHE STRING "VELEST: max number of stations")
set(OSV_VELEST_INLTOT 20 CACHE STRING "VELEST: max layers per velocity model")
set(OSV_VELEST_ITOTMODELS 2 CACHE STRING "VELEST: max number of velocity models")
set(OSV_VELEST_INSHOT 0 CACHE STRING "VELEST: max number of shots (simultaneous mode)")
set(OSV_VELEST_MAXOBS 180 CACHE STRING "VELEST: max observations per event")

if(NOT OSV_DATA_DIR)
  set(OSV_DATA_DIR "${OSV_BIN_DIR}/../share/osv-tools")
endif()

set(FMT "${CMAKE_CURRENT_SOURCE_DIR}/fmtomo")
set(EXT "${CMAKE_CURRENT_SOURCE_DIR}/fmtomo_extras")

# ---------------------------------------------------------------- VELEST
if(OSV_BUILD_VELEST)
  foreach(v IEQ IST INLTOT ITOTMODELS MAXOBS)
    if(NOT OSV_VELEST_${v} MATCHES "^[1-9][0-9]*$")
      message(FATAL_ERROR "OSV_VELEST_${v} must be a positive integer (got '${OSV_VELEST_${v}}')")
    endif()
  endforeach()
  if(NOT OSV_VELEST_INSHOT MATCHES "^[0-9]+$")
    message(FATAL_ERROR "OSV_VELEST_INSHOT must be a non-negative integer")
  endif()
  # Static memory estimate (bytes) of the dominant arrays: g (ikva*2 reals),
  # gcopy (60 x inva*2), and ~13 station x event arrays.
  math(EXPR _ilip "4*${OSV_VELEST_IEQ}+${OSV_VELEST_INSHOT}+${OSV_VELEST_INLTOT}+${OSV_VELEST_IST}")
  math(EXPR _inva "${_ilip}-1")
  math(EXPR _ikva "${_inva}*${_ilip}/2")
  math(EXPR _bytes "4*(2*${_ikva}+120*${_inva}+13*${OSV_VELEST_IST}*${OSV_VELEST_IEQ})")
  set(_vflags -std=legacy -ffixed-line-length-none ${OSV_OPT_FLAGS})
  if(_bytes GREATER 2000000000)
    list(APPEND _vflags -mcmodel=medium)
  endif()
  message(STATUS "VELEST ieq=${OSV_VELEST_IEQ} ist=${OSV_VELEST_IST} inltot=${OSV_VELEST_INLTOT} "
                 "itotmodels=${OSV_VELEST_ITOTMODELS} inshot=${OSV_VELEST_INSHOT} "
                 "maxobs=${OSV_VELEST_MAXOBS}: ~${_bytes} bytes static arrays (flags: ${_vflags})")
  set(_vdir "${CMAKE_BINARY_DIR}/velest_src")
  # velest.f is copied next to the generated include so that the generated
  # velest_common.inc (not the vendored default) is the one found.
  configure_file(velest/velest_common.inc.in "${_vdir}/velest_common.inc" @ONLY)
  configure_file(velest/velest.f "${_vdir}/velest.f" COPYONLY)
  add_executable(velest "${_vdir}/velest.f")
  target_compile_options(velest PRIVATE ${_vflags})
  target_link_options(velest PRIVATE ${_vflags})
  install(TARGETS velest RUNTIME DESTINATION "${OSV_BIN_DIR}")
endif()

# ---------------------------------------------------------------- driver scripts
# The tomo3d* drivers are ksh scripts. The only ksh-specific construct they use
# that bash lacks is `print -u<fd> text`, so they are installed with a bash
# shebang plus a small `print` shim (vendored files are left untouched).
set(OSV_PRINT_SHIM "#!/usr/bin/env bash
# Installed by openSeisVelocity: bash port of a ksh script (adds print -u<fd> shim).
print() { if [[ \"$1\" == -u* ]]; then local fd=\"\${1#-u}\"; shift; printf '%s\\n' \"$*\" >&\"$fd\"; else printf '%s\\n' \"$*\"; fi; }
")
function(osv_install_script src outname)
  file(READ "${src}" _txt)
  string(REGEX REPLACE "^#![^\n]*\n" "" _txt "${_txt}")
  set(_out "${CMAKE_BINARY_DIR}/scripts/${outname}")
  file(WRITE "${_out}" "${OSV_PRINT_SHIM}${_txt}")
  install(PROGRAMS "${_out}" DESTINATION "${OSV_BIN_DIR}")
endfunction()

# ---------------------------------------------------------------- FMTOMO
function(osv_f90_program name srcdir outname)
  add_executable(${name} "${srcdir}/${name}.f90")
  set_target_properties(${name} PROPERTIES
    OUTPUT_NAME ${outname}
    Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/mod_${name}")
  install(TARGETS ${name} RUNTIME DESTINATION "${OSV_BIN_DIR}")
endfunction()

if(OSV_BUILD_FMTOMO)
  foreach(p arraygen frechgen gmtslice grid3dg invert3d moddata obsdata residuals synthdata scorrect)
    osv_f90_program(${p} "${FMT}/source" ${p})
  endforeach()

  # fm3d (fmcode/Makefile): module mod_3dfm first, then the rest.
  set(_fm "${FMT}/fmcode")
  add_executable(fm3d
    "${_fm}/mod_3dfm.f90" "${_fm}/3dfmlib.f90" "${_fm}/propagate.f90" "${_fm}/rays.f90"
    "${_fm}/frechet.f90" "${_fm}/visual.f90" "${_fm}/3dfm_main.f90" "${_fm}/matchref.f90"
    "${_fm}/teleseismic.f90" "${_fm}/svdlib.f90" "${_fm}/nn_subsf.f" "${_fm}/stack.f90"
    "${_fm}/libsun.f" "${_fm}/libtau.f" "${_fm}/ellip.f" "${_fm}/sphdist.f")
  set_target_properties(fm3d PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/mod_fm3d")
  target_include_directories(fm3d PRIVATE "${_fm}")  # ttlim.inc
  install(TARGETS fm3d RUNTIME DESTINATION "${OSV_BIN_DIR}")

  # aktimes: remodl + setbrn, then generate ak135.tbl / ak135.hed (only needed by
  # fm3d for teleseismic sources; harmless for local earthquake runs).
  set(_ak "${_fm}/aktimes")
  add_executable(remodl "${_ak}/remodl.f" "${_ak}/emtvelin.f" "${_ak}/libtau.f" "${_ak}/libsun.f")
  add_executable(setbrn "${_ak}/setbrn.f" "${_ak}/libtau.f" "${_ak}/libsun.f")
  foreach(t remodl setbrn)
    target_include_directories(${t} PRIVATE "${_ak}")
  endforeach()
  set(_aktmp "${CMAKE_BINARY_DIR}/aktimes_run")
  file(MAKE_DIRECTORY "${_aktmp}")
  add_custom_command(
    OUTPUT "${_aktmp}/ak135.tbl" "${_aktmp}/ak135.hed"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_aktmp}"
    COMMAND ${CMAKE_COMMAND} -E copy "${_ak}/ak135.tvel" "${_aktmp}/ak135.tvel"
    COMMAND "$<TARGET_FILE:remodl>" > "${_aktmp}/remodl.log" 2>&1
    COMMAND "$<TARGET_FILE:setbrn>" > "${_aktmp}/setbrn.log" 2>&1
    WORKING_DIRECTORY "${_aktmp}"
    DEPENDS remodl setbrn
    COMMENT "Generating ak135 travel-time tables")
  add_custom_target(ak135_tables ALL DEPENDS "${_aktmp}/ak135.tbl" "${_aktmp}/ak135.hed")
  install(FILES "${_aktmp}/ak135.tbl" "${_aktmp}/ak135.hed" DESTINATION "${OSV_DATA_DIR}")

  osv_install_script("${FMT}/bin/tomo3d" tomo3d)
  install(FILES "${FMT}/fmcode/mode_set.in" DESTINATION "${OSV_DATA_DIR}")
endif()

# ---------------------------------------------------------------- Extras
if(OSV_BUILD_EXTRAS)
  foreach(p frvpvs gridloc gridlocj jointimes mkvpvs obsdatai vpvsinv vpvsresids grid3dgi
            gmtslicei synthdatai)
    osv_f90_program(${p} "${EXT}/source" "${OSV_EXTRAS_PREFIX}${p}")
  endforeach()
  foreach(s tomo3di tomo3dj tomo3dps cp_script)
    osv_install_script("${EXT}/source/${s}" "${OSV_EXTRAS_PREFIX}${s}")
  endforeach()
endif()

# ---------------------------------------------------------------- build_info.json
# Record how this build was made (read by `osv doctor`).
execute_process(COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short HEAD
                OUTPUT_VARIABLE _osv_git OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT _osv_git)
  set(_osv_git "unknown")
endif()
string(TIMESTAMP _osv_time "%Y-%m-%dT%H:%M:%SZ" UTC)
string(REPLACE ";" " " _vflags_str "${_vflags}")
string(STRIP "${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_RELEASE}" _osv_flags)
if(OSV_BUILD_VELEST)
  set(_osv_limits "\"ieq\": ${OSV_VELEST_IEQ}, \"ist\": ${OSV_VELEST_IST}, \"inltot\": ${OSV_VELEST_INLTOT}, \"itotmodels\": ${OSV_VELEST_ITOTMODELS}, \"inshot\": ${OSV_VELEST_INSHOT}, \"maxobs\": ${OSV_VELEST_MAXOBS}")
else()
  set(_osv_limits "")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/build_info.json"
"{\"compiler\": \"${CMAKE_Fortran_COMPILER}\", \"compiler_id\": \"${CMAKE_Fortran_COMPILER_ID}\", \"compiler_version\": \"${CMAKE_Fortran_COMPILER_VERSION}\", \"flags\": \"${_osv_flags}\", \"velest_flags\": \"${_vflags_str}\", \"build_type\": \"${CMAKE_BUILD_TYPE}\", \"velest_limits\": {${_osv_limits}}, \"extras_prefix\": \"${OSV_EXTRAS_PREFIX}\", \"git_commit\": \"${_osv_git}\", \"built_at\": \"${_osv_time}\"}\n")
install(FILES "${CMAKE_BINARY_DIR}/build_info.json" DESTINATION "${OSV_DATA_DIR}")
