"""StationXML reader producing station-epoch and channel tables plus VELEST aliases.

``read_stations`` turns one or more StationXML files into

* ``stations``: one row per station epoch (a ``<Station>`` element, de-duplicated across
  overlapping files),
* ``channels``: one row per channel epoch,
* ``aliases``: ``(net, sta, alias)`` with VELEST-safe codes (at most 4 characters, unique),

and QC findings (:class:`~osv.report.checks.QCFinding`).  Conventions follow
``docs/conventions.md``: latitude/longitude in degrees (north/east positive), elevation and
depth in metres (elevation positive up, depth positive down below the surface).  Sensor
elevations come from :func:`osv.coords.borehole_sensor_elevation_m`.
"""

from __future__ import annotations

import json
import logging
import string
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from osv.config.schema import CoordinatesConfig, InputsConfig
from osv.coords import borehole_sensor_elevation_m, epicentral_distance_km
from osv.report.checks import QCFinding

logger = logging.getLogger(__name__)

__all__ = [
    "ALIAS_COLUMNS",
    "CHANNEL_COLUMNS",
    "STATION_COLUMNS",
    "StationTables",
    "build_aliases",
    "read_stations",
    "station_at",
    "station_epoch_index",
]

#: Coordinates differing by more than this (metres) count as "moved" / inconsistent.
MOVE_TOLERANCE_M = 10.0
#: Elevation differences (metres) above this between channel and station are reported.
ELEVATION_TOLERANCE_M = 1.0
#: VELEST station code length limit.
MAX_CODE_LEN = 4

STATION_COLUMNS = [
    "net",
    "sta",
    "alias",
    "loc",
    "start",
    "end",
    "lat",
    "lon",
    "elevation_m",
    "sensor_depth_m",
    "sensor_elevation_m",
    "channels",
    "sample_rates",
]
CHANNEL_COLUMNS = [
    "net",
    "sta",
    "loc",
    "cha",
    "start",
    "end",
    "epoch_start",
    "lat",
    "lon",
    "elevation_m",
    "depth_m",
    "sensor_elevation_m",
    "azimuth",
    "dip",
    "sample_rate",
]
ALIAS_COLUMNS = ["net", "sta", "alias", "source"]

_ALPHABET = string.digits + string.ascii_uppercase


# ----------------------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------------------


@dataclass
class StationTables:
    """Result of :func:`read_stations`."""

    stations: pd.DataFrame
    channels: pd.DataFrame
    aliases: pd.DataFrame
    findings: list[QCFinding] = field(default_factory=list)

    def finding(self, code: str) -> QCFinding | None:
        """Return the first finding with ``code`` or ``None``."""
        return next((f for f in self.findings if f.code == code), None)

    def alias_map(self) -> dict[tuple[str, str], str]:
        """Mapping ``(net, sta) -> alias``."""
        a = self.aliases
        return {(n, s): al for n, s, al in zip(a["net"], a["sta"], a["alias"], strict=True)}

    def station_at(self, net: Sequence[str] | None, sta: Sequence[str], time: Any) -> pd.DataFrame:
        """Active epoch row for each ``(net, sta, time)``; see :func:`station_at`."""
        return station_at(self.stations, net, sta, time)

    def write_aliases_csv(self, path: str | Path) -> Path:
        """Write ``net,sta,alias`` (plus ``source``) to ``path`` and return it."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.aliases.to_csv(p, index=False)
        return p

    def write(self, outdir: str | Path) -> dict[str, Path]:
        """Write ``stations.parquet``, ``channels.parquet``, ``aliases.csv``, findings JSON."""
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "stations": out / "stations.parquet",
            "channels": out / "channels.parquet",
            "aliases": out / "aliases.csv",
            "findings": out / "station_findings.json",
        }
        st = self.stations.copy()
        for col in ("channels", "sample_rates"):
            st[col] = st[col].map(list)
        st.to_parquet(paths["stations"])
        self.channels.to_parquet(paths["channels"])
        self.write_aliases_csv(paths["aliases"])
        paths["findings"].write_text(json.dumps([f.to_dict() for f in self.findings], indent=2))
        return paths


# ----------------------------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------------------------


def _ts(t: Any) -> pd.Timestamp | pd.NaT:
    if t is None:
        return pd.NaT
    return pd.Timestamp(int(t.ns), unit="ns", tz="UTC")


def _f(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _loc(code: str | None) -> str:
    return (code or "").strip()


def read_stations(
    paths: Sequence[str | Path] | str | Path | None = None,
    inputs_cfg: InputsConfig | None = None,
    coords_cfg: CoordinatesConfig | None = None,
    aliases: str | Path | Mapping[str, str] | None = None,
) -> StationTables:
    """Read StationXML file(s) into station-epoch / channel tables, aliases and findings.

    Parameters
    ----------
    paths
        StationXML file(s) (already glob-expanded). ``None`` uses ``inputs_cfg.stationxml``.
    inputs_cfg
        ``inputs`` config section; ``station_aliases`` is used unless ``aliases`` is given.
    coords_cfg
        ``coordinates`` config section; ``borehole_sensors`` selects whether sensor
        elevation is ``elevation - depth`` (``use_depth_from_stationxml``) or the station
        elevation (``ignore_depth``).
    aliases
        Overrides ``inputs_cfg.station_aliases``: ``"auto"``, a path to a CSV with columns
        ``code,alias`` (optionally ``net``; ``sta`` is accepted for ``code``) or a mapping
        ``{"CODE" or "NET.CODE": alias}``.

    Returns
    -------
    StationTables
    """
    from obspy import read_inventory

    icfg = inputs_cfg or InputsConfig()
    ccfg = coords_cfg or CoordinatesConfig()
    if paths is None:
        paths = icfg.stationxml
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ValueError("No StationXML files given; set inputs.stationxml in config.yaml")
    use_depth = ccfg.borehole_sensors == "use_depth_from_stationxml"
    if aliases is None:
        aliases = icfg.station_aliases

    st_rows: list[dict[str, Any]] = []
    ch_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in paths:
        inv = read_inventory(str(path))
        for net in inv:
            for sta in net:
                key = (net.code, sta.code, _ts(sta.start_date), _ts(sta.end_date))
                if key in seen:  # same epoch repeated by overlapping files
                    continue
                seen.add(key)
                srow, crows = _station_rows(net.code, sta, use_depth)
                st_rows.append(srow)
                ch_rows.extend(crows)

    stations = pd.DataFrame(st_rows).reindex(columns=STATION_COLUMNS)
    channels = pd.DataFrame(ch_rows).reindex(columns=CHANNEL_COLUMNS)
    for df, cols in ((stations, ["start", "end"]), (channels, ["start", "end", "epoch_start"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True)

    findings = _station_findings(stations, channels)
    alias_df = build_aliases(stations[["net", "sta"]].drop_duplicates(), aliases)
    amap = {
        (n, s): a
        for n, s, a in zip(alias_df["net"], alias_df["sta"], alias_df["alias"], strict=True)
    }
    stations["alias"] = [
        amap[(n, s)] for n, s in zip(stations["net"], stations["sta"], strict=True)
    ]
    stations = stations.sort_values(["net", "sta", "start"], kind="stable").reset_index(drop=True)
    channels = channels.sort_values(["net", "sta", "loc", "cha", "start"], kind="stable")
    channels = channels.reset_index(drop=True)
    findings.extend(_alias_findings(alias_df))
    return StationTables(stations, channels, alias_df, findings)


def _station_rows(
    net: str, sta: Any, use_depth: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    crows = []
    for ch in sta.channels:
        depth = _f(ch.depth)
        depth = 0.0 if np.isnan(depth) else depth
        elev = _f(ch.elevation)
        crows.append(
            {
                "net": net,
                "sta": sta.code,
                "loc": _loc(ch.location_code),
                "cha": ch.code,
                "start": _ts(ch.start_date),
                "end": _ts(ch.end_date),
                "epoch_start": _ts(sta.start_date),
                "lat": _f(ch.latitude),
                "lon": _f(ch.longitude),
                "elevation_m": elev,
                "depth_m": depth,
                "sensor_elevation_m": float(
                    borehole_sensor_elevation_m(elev, depth if use_depth else 0.0)
                ),
                "azimuth": _f(ch.azimuth),
                "dip": _f(ch.dip),
                "sample_rate": _f(ch.sample_rate),
            }
        )
    elev = _f(sta.elevation)
    depths = [c["depth_m"] for c in crows]
    depth = float(max(depths)) if depths else 0.0
    srow = {
        "net": net,
        "sta": sta.code,
        "loc": sorted({c["loc"] for c in crows}),
        "start": _ts(sta.start_date),
        "end": _ts(sta.end_date),
        "lat": _f(sta.latitude),
        "lon": _f(sta.longitude),
        "elevation_m": elev,
        "sensor_depth_m": depth,
        "sensor_elevation_m": float(borehole_sensor_elevation_m(elev, depth if use_depth else 0.0)),
        "channels": sorted({c["cha"] for c in crows}),
        "sample_rates": sorted({c["sample_rate"] for c in crows if not np.isnan(c["sample_rate"])}),
    }
    # keep the station's own channel set stable; ``loc`` as sorted list of location codes
    return srow, crows


# ----------------------------------------------------------------------------------------
# Findings
# ----------------------------------------------------------------------------------------


def _codes(df: pd.DataFrame) -> list[str]:
    return sorted({f"{n}.{s}" for n, s in zip(df["net"], df["sta"], strict=True)})


def _station_findings(stations: pd.DataFrame, channels: pd.DataFrame) -> list[QCFinding]:
    out: list[QCFinding] = []
    pairs = stations[["net", "sta"]].drop_duplicates()

    # duplicate station codes across networks
    nets_per_code = pairs.groupby("sta")["net"].apply(sorted)
    dup = {c: n for c, n in nets_per_code.items() if len(n) > 1}
    if dup:
        out.append(
            QCFinding(
                "duplicate_station_codes",
                "warn",
                f"{len(dup)} station code(s) used by more than one network "
                f"({', '.join(sorted(dup))}); VELEST sees codes only, so they are aliased.",
                len(dup),
                {"codes": {c: n for c, n in dup.items()}},
            )
        )

    # moved epochs
    moved: list[dict[str, Any]] = []
    for (net, sta), g in stations.groupby(["net", "sta"], sort=True):
        if len(g) < 2:
            continue
        g = g.sort_values("start")
        ref = g.iloc[0]
        for _, row in g.iloc[1:].iterrows():
            d = float(epicentral_distance_km(ref.lat, ref.lon, row.lat, row.lon)) * 1000.0
            dz = abs(row.elevation_m - ref.elevation_m)
            if d > MOVE_TOLERANCE_M:
                moved.append(
                    {
                        "station": f"{net}.{sta}",
                        "epoch_start": str(row.start),
                        "moved_m": round(d, 1),
                        "elevation_change_m": round(float(dz), 1),
                    }
                )
    if moved:
        out.append(
            QCFinding(
                "station_moved",
                "warn",
                f"{len({m['station'] for m in moved})} station(s) have epochs more than "
                f"{MOVE_TOLERANCE_M:.0f} m apart; pick association must use the epoch active "
                "at the pick time.",
                len(moved),
                {"epochs": moved},
            )
        )

    # channel coordinates inconsistent with the station epoch
    bad: list[dict[str, Any]] = []
    for row in channels.itertuples():
        ep = stations[
            (stations.net == row.net)
            & (stations.sta == row.sta)
            & (
                (stations.start == row.epoch_start)
                | (stations.start.isna() & pd.isna(row.epoch_start))
            )
        ]
        srow = ep.iloc[0]
        d = float(epicentral_distance_km(srow.lat, srow.lon, row.lat, row.lon)) * 1000.0
        dz = float(row.elevation_m - srow.elevation_m)
        ok_z = (
            np.isnan(dz)
            or abs(dz) <= ELEVATION_TOLERANCE_M
            or abs(dz + row.depth_m) <= ELEVATION_TOLERANCE_M
        )
        if d > MOVE_TOLERANCE_M or not ok_z:
            bad.append(
                {
                    "channel": f"{row.net}.{row.sta}.{row.loc}.{row.cha}",
                    "horizontal_offset_m": round(d, 1),
                    "elevation_offset_m": round(dz, 1),
                }
            )
    if bad:
        out.append(
            QCFinding(
                "channel_coordinates_inconsistent",
                "warn",
                f"{len(bad)} channel(s) have coordinates inconsistent with their station "
                f"(> {MOVE_TOLERANCE_M:.0f} m horizontally or elevation not matching the "
                "station elevation or station elevation minus depth).",
                len(bad),
                {"channels": bad[:50]},
            )
        )

    # channel depths differing within an epoch
    cd = channels.groupby(["net", "sta", "epoch_start"], dropna=False)["depth_m"].agg(
        ["min", "max"]
    )
    cd = cd[(cd["max"] - cd["min"]) > 0.5]
    if len(cd):
        differ = [f"{n}.{s}" for n, s, _ in cd.index]
        out.append(
            QCFinding(
                "channel_depths_differ",
                "info",
                f"{len(cd)} station epoch(s) have channels at different depths; the maximum "
                "depth is used as the sensor depth.",
                len(cd),
                {"stations": sorted(set(differ))},
            )
        )

    # zero / missing elevation
    z = stations[stations.elevation_m.isna() | (stations.elevation_m == 0)]
    if len(z):
        out.append(
            QCFinding(
                "elevation_missing_or_zero",
                "warn",
                f"{len(z)} station epoch(s) have zero or missing elevation.",
                len(z),
                {"stations": _codes(z)},
            )
        )

    # borehole sensors
    bh = stations[stations.sensor_depth_m > 0]
    if len(bh):
        out.append(
            QCFinding(
                "borehole_sensors",
                "info",
                f"{bh[['net', 'sta']].drop_duplicates().shape[0]} station(s) have borehole "
                "sensors (channel depth > 0).",
                int(bh[["net", "sta"]].drop_duplicates().shape[0]),
                {
                    "depths_m": {
                        f"{n}.{s}": float(d)
                        for n, s, d in zip(bh.net, bh.sta, bh.sensor_depth_m, strict=True)
                    }
                },
            )
        )

    # long codes and truncation clashes
    long = sorted({c for c in pairs["sta"] if len(c) > MAX_CODE_LEN})
    if long:
        out.append(
            QCFinding(
                "station_code_too_long",
                "warn",
                f"{len(long)} station code(s) exceed {MAX_CODE_LEN} characters and are aliased.",
                len(long),
                {"codes": long},
            )
        )
    trunc: dict[str, set[str]] = {}
    for c in pairs["sta"].unique():
        trunc.setdefault(c[:MAX_CODE_LEN].upper(), set()).add(c)
    clash = {k: sorted(v) for k, v in trunc.items() if len(v) > 1}
    if clash:
        out.append(
            QCFinding(
                "station_code_truncation_clash",
                "warn",
                f"{len(clash)} group(s) of station codes are identical after truncation to "
                f"{MAX_CODE_LEN} characters; they receive distinct aliases.",
                len(clash),
                {"groups": clash},
            )
        )
    return out


def _alias_findings(aliases: pd.DataFrame) -> list[QCFinding]:
    changed = aliases[aliases.sta.str.upper() != aliases.alias]
    changed = changed[changed.sta != changed.alias]
    if not len(changed):
        return []
    return [
        QCFinding(
            "station_aliases",
            "info",
            f"{len(changed)} station(s) need an alias for VELEST.",
            len(changed),
            {
                "aliases": {
                    f"{n}.{s}": a
                    for n, s, a in zip(changed.net, changed.sta, changed.alias, strict=True)
                }
            },
        )
    ]


# ----------------------------------------------------------------------------------------
# Aliases
# ----------------------------------------------------------------------------------------


def _load_user_aliases(spec: str | Path | Mapping[str, str]) -> dict[str, str]:
    """Normalise a user alias spec to ``{"CODE" or "NET.CODE": alias}``."""
    if isinstance(spec, Mapping):
        return {str(k): str(v).strip() for k, v in spec.items()}
    p = Path(spec)
    if not p.is_file():
        raise FileNotFoundError(f"station alias file not found: {p}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    code_col = "code" if "code" in df.columns else "sta" if "sta" in df.columns else None
    if code_col is None or "alias" not in df.columns:
        raise ValueError(f"{p}: alias CSV needs columns 'code' (or 'sta') and 'alias'")
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        key = (
            f"{r['net'].strip()}.{r[code_col].strip()}"
            if "net" in df.columns and r["net"].strip()
            else r[code_col].strip()
        )
        out[key] = r["alias"].strip()
    return out


def _candidates(code: str) -> Iterable[str]:
    """Deterministic 4-character candidates derived from ``code``."""
    base = "".join(ch for ch in code.upper() if ch.isalnum()) or "STA"
    base = base.ljust(MAX_CODE_LEN, "X")
    yield base[:MAX_CODE_LEN]
    for keep in (3, 2, 1):
        stem = base[:keep]
        width = MAX_CODE_LEN - keep
        n = len(_ALPHABET) ** width
        for i in range(n):
            digits = []
            v = i
            for _ in range(width):
                v, r = divmod(v, len(_ALPHABET))
                digits.append(_ALPHABET[r])
            yield stem + "".join(reversed(digits))
    raise RuntimeError("alias space exhausted")  # pragma: no cover


def build_aliases(
    pairs: pd.DataFrame, spec: str | Path | Mapping[str, str] = "auto"
) -> pd.DataFrame:
    """Build the ``net, sta, alias, source`` table for unique ``(net, sta)`` pairs.

    ``"auto"`` keeps a code of at most 4 characters that is unique across networks
    (case-insensitively); other stations get deterministic 4-character aliases (first four
    alphanumerics, then ``abc0``-style suffixes) that avoid every kept code and every other
    alias.  Stations are processed in sorted ``(net, sta)`` order, so a code shared by
    several networks gives its own spelling to the first network and suffixed aliases to
    the others.  A user mapping/CSV overrides the automatic choice; user aliases must be
    at most 4 characters and unique.
    """
    pairs = pairs.drop_duplicates().sort_values(["net", "sta"], kind="stable")
    keys = list(zip(pairs["net"], pairs["sta"], strict=True))
    user: dict[str, str] = (
        {} if (isinstance(spec, str) and spec == "auto") else _load_user_aliases(spec)
    )

    assigned: dict[tuple[str, str], tuple[str, str]] = {}
    for net, sta in keys:
        alias = user.get(f"{net}.{sta}", user.get(sta))
        if alias is None:
            continue
        if not alias or len(alias) > MAX_CODE_LEN or not alias.isascii() or " " in alias:
            raise ValueError(
                f"alias {alias!r} for {net}.{sta} must be 1-{MAX_CODE_LEN} ASCII characters "
                "without spaces"
            )
        assigned[(net, sta)] = (alias, "user")
    unknown = [
        k
        for k in user
        if (k not in {f"{n}.{s}" for n, s in keys} and k not in {s for _, s in keys})
    ]
    if unknown:
        logger.warning("station alias entries match no station: %s", ", ".join(sorted(unknown)))

    count: dict[str, int] = {}
    for _, sta in keys:
        count[sta.upper()] = count.get(sta.upper(), 0) + 1
    used = {a.upper() for a, _ in assigned.values()}
    # codes that can be kept as they are
    for net, sta in keys:
        if (net, sta) in assigned:
            continue
        if len(sta) <= MAX_CODE_LEN and count[sta.upper()] == 1 and sta.upper() not in used:
            assigned[(net, sta)] = (sta, "kept")
            used.add(sta.upper())
    for net, sta in keys:
        if (net, sta) in assigned:
            continue
        # prefer the original spelling for the first of several networks sharing a code
        for cand in _candidates(sta):
            if cand not in used:
                assigned[(net, sta)] = (cand, "generated")
                used.add(cand)
                break
    dups = pd.Series([a.upper() for a, _ in assigned.values()])
    if dups.duplicated().any():
        bad = sorted(set(dups[dups.duplicated()]))
        raise ValueError(f"station aliases are not unique (case-insensitive): {', '.join(bad)}")
    rows = [
        {"net": n, "sta": s, "alias": assigned[(n, s)][0], "source": assigned[(n, s)][1]}
        for n, s in keys
    ]
    return pd.DataFrame(rows, columns=ALIAS_COLUMNS)


# ----------------------------------------------------------------------------------------
# Epoch lookup
# ----------------------------------------------------------------------------------------


def station_epoch_index(
    stations: pd.DataFrame,
    net: Sequence[str] | None,
    sta: Sequence[str],
    time: Any,
) -> np.ndarray:
    """Positional index into ``stations`` of the epoch active at each query time.

    Parameters
    ----------
    stations
        The ``stations`` table (``net``, ``sta``, ``start``, ``end``).
    net
        Network code per query; ``None`` (or an empty string entry) matches any network.
    sta
        Station code per query.
    time
        Query times (anything ``pandas.to_datetime(..., utc=True)`` accepts).

    Returns
    -------
    numpy.ndarray of int
        Row position, or ``-1`` when no epoch of the station covers the time.  Start and
        end are inclusive; a missing end means open-ended.  With overlapping epochs the
        one with the latest start wins.
    """
    sta_a = np.asarray(sta, dtype=object)
    n = len(sta_a)
    t = pd.to_datetime(pd.Series(np.asarray(time)), utc=True).reset_index(drop=True)
    if len(t) != n:
        raise ValueError("net, sta and time must have the same length")
    q = pd.DataFrame({"qid": np.arange(n), "sta": sta_a, "time": t})
    st = stations[["net", "sta", "start", "end"]].copy()
    st["pos"] = np.arange(len(st))
    if net is None:
        q["net"] = ""
    else:
        net_a = np.asarray(net, dtype=object)
        if len(net_a) != n:
            raise ValueError("net, sta and time must have the same length")
        q["net"] = ["" if x is None or (isinstance(x, float) and np.isnan(x)) else x for x in net_a]
    with_net = q[q["net"] != ""]
    without = q[q["net"] == ""].drop(columns="net")
    parts = [
        with_net.merge(st, on=["net", "sta"], how="inner"),
        without.merge(st, on="sta", how="inner"),
    ]
    m = pd.concat(parts, ignore_index=True)
    ok = (m["start"].isna() | (m["start"] <= m["time"])) & (
        m["end"].isna() | (m["time"] <= m["end"])
    )
    m = m[ok].copy()
    m["start_key"] = m["start"].fillna(pd.Timestamp.min.tz_localize("UTC"))
    m = m.sort_values(["qid", "start_key", "pos"]).drop_duplicates("qid", keep="last")
    out = np.full(n, -1, dtype=np.int64)
    out[m["qid"].to_numpy()] = m["pos"].to_numpy()
    return out


def station_at(
    stations: pd.DataFrame,
    net: Sequence[str] | None,
    sta: Sequence[str],
    time: Any,
) -> pd.DataFrame:
    """Station-epoch row active at each ``(net, sta, time)`` (NaN row where none matches).

    The returned frame has the same length and order as the query (index ``0..n-1``) and
    an extra boolean column ``found``.  See :func:`station_epoch_index` for the matching
    rules.
    """
    idx = station_epoch_index(stations, net, sta, time)
    found = idx >= 0
    out = stations.reset_index(drop=True).iloc[np.where(found, idx, 0)].reset_index(drop=True)
    out.loc[~found, :] = np.nan
    out["found"] = found
    return out
