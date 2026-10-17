# Coordinate and unit conventions

osv stores **true signed coordinates** everywhere: latitude/longitude in degrees (north and
east positive, WGS84), depth in km positive down, elevation in m positive up. Only the
`osv.io.*` writers convert to program conventions, using `osv.coords`.

All references below were verified in the vendored sources / manuals:
`velest.f` = `/home/steve/seismic/velest/src/velest.f`, `UG` = `VELEST-Usersguide1995.pdf`,
`FM` = `FMTOMO/docs/FMTOMO_Instructions.pdf`, `fmcode` = `fortran/fmtomo/fmcode`.

## Summary table

| Item | osv canonical | VELEST | FMTOMO |
|---|---|---|---|
| Latitude | degrees, N+ | File: `f7.4` + letter `N`/`S` (CNV `velest.f:3459`, `.sta` `velest.f:1531-1538`). Control `olat`: N positive, S negative (`velest.f:1248-1251`) | degrees, N+, S- (FM p. 15-16; `3dfm_main.f90:418-421`). Spherical angle, not colatitude. `vgrids.in`/`interfaces.in` headers in radians (`grid3dg.f90:164-167`) |
| Longitude | degrees, E+ | File: `f8.4` + letter `E`/`W`; letters are the truth (E is negated internally, `velest.f:3460`, `1539`). Control `olon`: **W positive**, E negative (`velest.f:1248-1251`, `1259-1263`, `10155-10156`; UG p. 12) | degrees, **E positive**, may exceed 180 (115W = 245E, FM p. 16 and p. 44) |
| Local x/y | `LocalProjection`: x east, y north, km | **x positive WEST**, y north, z down, km; axes rotated clockwise by `rotate` degrees (`velest.f:1293-1294`, `10017-10051`; UG p. 12-13) | not used (spherical) |
| Depth | km positive down (below sea level) | z positive down; model zero at sea level; hypocentre depth in CNV is `depth + zshift` (`velest.f:1865`) | `depth` column of sources/receivers: km, positive down, **negative above surface** (FM p. 39, 44) |
| Station elevation | m positive up | `.sta` field `i4`, integer metres, positive up (UG p. 20); `z = -ielev/1000` km (`velest.f:1551`); with `iuseelev=0` z is forced to 0 (`velest.f:1561-1565`) | receiver depth = `-elevation_m/1000` km (above surface is negative); needs model top above the highest receiver (FM p. 15) |
| Borehole sensor | elevation = surface elevation - sensor depth | use that elevation in the `.sta` (can be negative: `i4` allows -999..9999) | depth = `-(elevation)/1000` (positive below sea level) |
| Radius | n/a | n/a | `r = 6371 - depth` km (`3dfm_main.f90:209,222,420`; `mod_3dfm.f90:709`; `make_vgrids.f90:36`; FM p. 15) |
| Ellipsoid | WGS84 for geodesics | short-distance conversion on WGS72 (`REARTH=6378.135`, `1/f=298.26`, `velest.f:10168-10169`) | sphere, radius 6371 km, geographic latitude used as spherical angle |

## VELEST details

* **Short distance conversion** (`icoordsystem = 0`): `SETORG` / `DIST` / `REDIST` /
  `SDC` (`velest.f:10017-10240`). `osv.coords.VelestShortDistance` is an equivalent
  double-precision implementation; tests compare against a gfortran build of those
  routines (single precision: agreement better than 2 m at 15000 km, ~1e-4 km nearby).
  `icoordsystem = 1` ("longitude East", UG p. 12, "not fully tested") has **no
  implementation** in the source (only `= 2` is special-cased), so it behaves as 0: use 0.
  `icoordsystem = 2` is the Swiss grid and must not be used.
* **olat / olon**: write `olat = lat` (S negative) and `olon = -lon_east` (e.g. Ngatamariki
  `-38.55`, `-176.194`). The program header prints `olon <= 0` as `E`.
* **rotate**: the axes are rotated *clockwise* by `rotate` degrees (UG p. 12;
  `velest.f:1269-1271`, `10042-10048`); `0` is the normal choice.
* **CNV hypocentre** line `(3i2,1x,2i2,1x,f5.2,1x,f7.4,a1,1x,f8.4,a1,1x,f7.2,2x,f5.2)`
  (`velest.f:3447-3451`, read statement): the `N/S/E/W` test is **case-sensitive upper case**
  (`velest.f:3459-3460`); write upper case. VELEST writes the same format back
  (`velest.f:3280-3284`).
* **Station file**: first line is the Fortran format; typical
  `(a4,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)` (UG p. 20,
  `nmstats.sta`). Letters are upper-cased on read (`CASEFOLD`, `velest.f:1536-1537`).
  Elevation is an **integer** in metres.
* **zshift**: `depth_internal = depth_in_file + zshift` (`velest.f:1865`, UG p. 13;
  example: depths relative to an average station elevation of 900 m need
  `zshift = 0.9`). VELEST writes the internal depth (zshift included) to the output
  CNV, so subtract it again when reading results. Negative depths are allowed (above sea
  level); with `itopo > 0` a fixed depth `<= 0` is reset to -3 km before being set to the
  surface (`velest.f:1859`). Station z is `-elev/1000`; the top of model layer 2 must not
  coincide with a station depth (UG p. 20: put it at `+0.05` km if elevations are not used).

* **nsp / vpvs / swtfac / nmod** (`velest.f:1118-1128`, line numbers of `fortran/velest/velest.f`):
  * `nsp = 1`: P model and **P data only**. S readings (`S`, `s`) and S-P (`-`) are skipped
    when the phases are read (`velest.f:1892-1894`), so S picks are *not* used.
  * `nsp = 2`: independent S model. VELEST forces `nmod = 2` and reads two blocks from the
    model file; S travel times use the S model and the S station delays `stcor`
    (`velest.f:2766`, `2119-2150`).
  * `nsp = 3`: **S picks with a constant Vp/Vs**. Only the P model is read and inverted
    (`nmod` is forced to 1); every S ray is traced in the P layers with `Vs = Vp/vpvs`
    (`velest.f:4357-4360`), the S station delay is `ptcor * vpvs` (`velest.f:2767`,
    `4517`) and S readings give no station-delay partials (`velest.f:4100`). `vpvs` is read
    list-directed from the control line. Velocity changes of the P model are therefore
    constrained by P and S together, with the ratio held fixed.
  * `swtfac` multiplies the weight of S (and S-P) observations
    (`w = swtfac / 2**(2*ipwt)`, `velest.f:2077`, `2106`, `5782`); it is used for `nsp = 2`
    and `3` alike and does nothing when S picks are dropped (`nsp = 1`).
  * osv maps `velest.vpvs.mode` `fixed` and `sampled` to `nsp = 3` (the first with the
    configured `value`, the second with the model's own ratio per job) and
    `independent_s_model` to `nsp = 2`. `nsp = 1` is never used. Model files hold only the
    P block when `nmod = 1`. In ensemble outputs `vs = vp / vpvs` with each run's ratio
    for `nsp = 3` runs (`final_model.mod` still carries an informational S block).

## Depth datum

Catalogue depths (QuakeML/NLLoc) may be relative to sea level, to the local ground
surface, or to a fixed reference elevation (e.g. an NLLoc grid top above sea level).
`coordinates.depth_datum` (`sea_level` default | `surface` | `reference` with
`coordinates.reference_elevation_m`) states which one the catalogue uses. Stage 1 converts
with `osv.coords.to_sea_level_depth` / `osv.data.datum.apply_depth_datum`, which add the
column **`depth_km_bsl`** (km below sea level, positive down, negative above sea level; the
source datum is recorded in `df.attrs["depth_datum"]`). `depth_km_bsl = depth_km -
elevation_of_datum_m / 1000`. **All later stages use `depth_km_bsl` only**; the raw
`depth_km` is kept for provenance.

Consistent consumption (writers must follow this):

* **VELEST**: write `depth_km_bsl` as the CNV depth with `zshift = 0` and model layer
  depths measured from sea level; write station elevations (integer m, positive up) in the
  `.sta` with `iuseelev = 1` so station z = `-elev/1000` is on the same datum. If the user
  prefers `iuseelev = 0` (all stations at z = 0) instead, use
  `osv.coords.velest_zshift_for_reference_elevation` so depths and the model share a
  reference elevation, and record it in the manifest. Read results back by subtracting
  `zshift` (VELEST writes the internal depth).
* **FMTOMO**: `sources.in` depth = `depth_km_bsl`; receivers depth = `-elevation_m/1000`
  (`osv.coords.fmtomo_receiver_depth_km`). Radius is `6371 - depth`, so with all depths
  below the 6371 km sphere the sphere is sea level; the model top (interfaces/propgrid)
  must lie above the highest receiver/event (negative depth).
* Do not mix datums: a catalogue with `surface` datum needs surface elevations at the
  epicentres (station interpolation or DEM) before it can be used.

Dataset note (RK-NM): the NLLoc control files in `/home/steve/seismic/nlloc` use
`TRANS LAMBERT WGS-84 ...  0.0` and `VGGRID 200 200 100 0.0 -50.0 -0.6 0.25 ...`, i.e. the
grid z-origin is -0.6 km (grid top 0.6 km **above sea level**, depth axis referenced to sea
level, no vertical shift), `LOCGRID` has the same origin, and `GTSRCE` elevations are km
above sea level. The catalogue depths therefore are below sea level: default `sea_level`.

## Legacy NZ files with `W` longitudes: a consistent mirror

The legacy Ngatamariki files (`velest/MonteCarlo/nmstats.sta`, `*.cnv`, `velest.cmn`) contain
`WPRZ38.5196N 176.1624W ...`, `olat = 38.5500`, `olon = 176.194`. The real site is
38.55 S, 176.19 E, so **both** latitude and longitude are mirrored (point reflection,
`lat -> -lat`, `lon -> -lon`), and consistently so in the stations, the CNV hypocentres
and the control-file origin. VELEST only works with differences to the origin and its
short-distance formulas are symmetric under the reflection, so inter-station distances
and all velocity-inversion quantities are unchanged; the local frame is rotated by 180
degrees (`x' = -x`, `y' = -y`), with ~1e-5 relative asymmetry from the one-arcminute finite
difference used for km-per-degree of latitude (tested in
`tests/test_coords.py::test_legacy_nz_files_are_a_consistent_point_mirror`).
`osv.coords.legacy_mirror(lat, lon)` converts signed values parsed from such files to true
coordinates. The files are **not** consistent with true geography if mixed with
unmirrored data (e.g. StationXML or QuakeML positions): such data would be placed at
the wrong hemisphere relative to the origin.

## FMTOMO details

* **sources.in / receivers.in** (read by `fm3d`): per receiver/source the position line is
  `depth lat lon` (`read(2,*) rec%r,rec%lat,rec%long`, `3dfm_main.f90:418`; sources
  `3dfm_main.f90:198-222`; written by `moddata.f90:153,213`; example1 `receivers.in`:
  `-1.000000 -41.80000 140.2000`). **Caution:** the manual prose (p. 39, 44) says
  "latitude, longitude and depth", but the code, the generated files and the example
  all use depth first; the *input* pick/source files given to `obsdata`/`moddata` are
  lat, lon, depth (`obsdata.f90:222`). `r = 6371 - depth` is formed internally.
* **vgrids.in / interfaces.in**: radius in km (6371 + height; example1 interface at
  `6372.3` = 1.3 km above the surface), latitude/longitude in radians, longitude east
  (FM p. 18-24, `make_vgrids.f90`, `grid3dg.f90:164-167`). Grid extents in `grid3dg.in` are
  `radius range` (given as heights/depths relative to the surface: `1.5 -80.1`), lat range
  (N positive) and lon range (east, 0-360 for the western hemisphere).
* **propgrid.in**: origin radius is formed as `6371 + prmin - (nr-1)*dr` from a depth-like
  value (`make_vgrids.f90:65`), lat/lon in degrees converted to radians.
* **Distances**: FMTOMO is a sphere of 6371 km with geographic latitude treated as a
  spherical angle, so its distances differ from WGS84 geodesics by up to ~0.3 percent.
  Use `osv.coords.spherical_distance_km` when comparing to FMTOMO geometry.
