"""Low-level helpers shared by the FMTOMO file readers and writers.

FMTOMO reads nearly every file with Fortran *list-directed* input (``read(unit,*) ...``).
The behaviour that matters for parsing, and that :class:`RecordReader` reproduces, is:

* every ``read`` statement starts a **new record** (line), even if the previous statement
  did not use up all the values on its line (the rest of that line is discarded - this is how
  the trailing ``c: comment`` on control-file lines is ignored);
* if a statement needs more values than the record holds, it continues on the following
  records (this is why ``otimes.dat`` can hold one observation on two lines and why the
  source/path lists can wrap over several lines);
* blank records are skipped while a statement still needs values;
* ``read(unit,*)`` with an empty list consumes exactly one record (used to skip banners).

Fixed-width reads (``read(unit,'(a26)') name``) take the first *N* characters of the record
(:meth:`RecordReader.a`).  Writers use the Fortran formats of the original programs where
the programs use explicit formats (e.g. ``(2f12.8)`` for velocity nodes) and shortest
round-trip decimal text otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "FmtomoFormatError",
    "RecordReader",
    "fint",
    "fints",
    "fnum",
    "fline",
    "read_text",
    "write_text",
]

_SPLIT = re.compile(r"[,\s]+")
_MAX_PER_LINE = 6  # list-directed integer lists wrap at 6 x i12 = 72 columns (gfortran, 80)


class FmtomoFormatError(ValueError):
    """A FMTOMO file (or a data structure to be written) violates the file format."""


def read_text(path: str | Path) -> str:
    """Read a text file (latin-1 tolerant; FMTOMO files are plain ASCII)."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` (creating parent directories) and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="ascii", errors="replace")
    return p


def _to_float(tok: str) -> float:
    return float(tok.replace("d", "e").replace("D", "e"))


class RecordReader:
    """Record-oriented reader reproducing Fortran list-directed input semantics."""

    def __init__(self, text: str, source: str | Path | None = None) -> None:
        self._lines = text.splitlines()
        self._i = 0
        self.source = str(source) if source is not None else "<text>"

    # -- diagnostics ---------------------------------------------------------------------
    @property
    def lineno(self) -> int:
        """Number of records consumed so far (the last consumed record has this 1-based number)."""
        return self._i

    def error(self, msg: str) -> FmtomoFormatError:
        return FmtomoFormatError(f"{self.source}:{self._i}: {msg}")

    # -- raw records ---------------------------------------------------------------------
    def next_record(self) -> str:
        if self._i >= len(self._lines):
            raise self.error("unexpected end of file")
        line = self._lines[self._i]
        self._i += 1
        return line

    def skip(self, n: int = 1) -> None:
        """``read(unit,*)`` with an empty list, ``n`` times."""
        for _ in range(n):
            self.next_record()

    def at_eof(self) -> bool:
        """True if only blank records remain."""
        return all(not ln.strip() for ln in self._lines[self._i :])

    def remaining(self) -> int:
        """Number of non-blank records not yet consumed."""
        return sum(1 for ln in self._lines[self._i :] if ln.strip())

    def expect_eof(self, what: str) -> None:
        """Raise if non-blank records remain (a node/row count in the header is too small)."""
        for k in range(self._i, len(self._lines)):
            if self._lines[k].strip():
                self._i = k + 1
                raise self.error(
                    f"unexpected extra data after the end of {what}: "
                    f"{self._lines[k].strip()[:40]!r}"
                )

    # -- list-directed reads -------------------------------------------------------------
    def tokens(self, n: int) -> list[str]:
        """Collect ``n`` value tokens over as many records as needed (rest of the last dropped)."""
        if n <= 0:
            self.next_record()
            return []
        out: list[str] = []
        while len(out) < n:
            line = self.next_record()
            if not line.strip():
                continue
            for tok in _SPLIT.split(line.strip()):
                if tok:
                    out.append(tok)
                    if len(out) == n:
                        break
        return out

    def record_tokens(self, minimum: int) -> list[str]:
        """All tokens of the record holding the ``minimum``-th value (extra columns are kept)."""
        out: list[str] = []
        while len(out) < minimum:
            line = self.next_record()
            if not line.strip():
                continue
            out.extend(t for t in _SPLIT.split(line.strip()) if t)
        return out

    def ints(self, n: int) -> list[int]:
        toks = self.tokens(n)
        try:
            return [int(t) for t in toks]
        except ValueError:
            raise self.error(f"expected {n} integer value(s), got {toks!r}") from None

    def floats(self, n: int) -> list[float]:
        toks = self.tokens(n)
        try:
            return [_to_float(t) for t in toks]
        except ValueError:
            raise self.error(f"expected {n} numeric value(s), got {toks!r}") from None

    def int(self) -> int:
        return self.ints(1)[0]

    def float(self) -> float:
        return self.floats(1)[0]

    def word(self) -> str:
        """A list-directed character value (stops at blank, comma or slash)."""
        tok = self.tokens(1)[0]
        return tok.strip("'\"")

    def logical(self) -> bool:
        tok = self.tokens(1)[0].lstrip(".").upper()
        if tok[:1] == "T":
            return True
        if tok[:1] == "F":
            return False
        raise self.error(f"expected a logical value (T/F), got {tok!r}")

    # -- formatted reads -----------------------------------------------------------------
    def a(self, width: int) -> str:
        """``read(unit,'(a<width>)') s``: first ``width`` characters, right-trimmed."""
        return self.next_record()[:width].strip()


# ---- writers ---------------------------------------------------------------------------


def fint(v: int) -> str:
    """One list-directed integer (``i12``)."""
    return f"{int(v):12d}"


def fints(vals: list[int] | tuple[int, ...], per_line: int = _MAX_PER_LINE) -> list[str]:
    """Integer list as list-directed output lines (wrapped like gfortran)."""
    vals = [int(v) for v in vals]
    return [
        "".join(f"{v:12d}" for v in vals[i : i + per_line]) for i in range(0, len(vals), per_line)
    ]


def fnum(x: float) -> str:
    """Shortest decimal text that reads back as the same double."""
    return repr(float(x))


def fline(*vals: float | int) -> str:
    """One record with list-directed style columns (ints ``i12``, floats right-aligned)."""
    parts = []
    for v in vals:
        if isinstance(v, int) and not isinstance(v, bool):
            parts.append(f"{v:12d}")
        else:
            parts.append(f"{fnum(v):>24}")
    return "".join(parts)
