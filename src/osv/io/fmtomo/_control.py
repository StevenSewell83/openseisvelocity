"""Declarative engine for FMTOMO control files (value lines followed by ``c: comments``).

FMTOMO control files are read positionally, one value per statement, by the Fortran
programs (see the per-class docstrings for the exact ``read`` statements).  Banner/comment
lines are skipped by empty ``read(10,*)`` statements, so their *number* matters but not their
content.  A layout is a sequence of :class:`L` (a literal line that is skipped on reading and
written verbatim) and :class:`F` (a value line).  The writer pads the value to ``pad``
columns and appends ``c: <comment>``, reproducing the layout of the files shipped in
``fortran/fmtomo/source/inputfiles``.

Character values come in two flavours, both critical:

* ``a<width>`` (``kind="a"``): ``read(10,'(a26)') name`` takes the **first width characters**
  of the record, so the file name must fit and the comment must start after that column;
* ``w`` (list-directed ``read(10,*) name``): the value ends at a blank, comma **or slash**, so a
  path containing ``/`` is silently truncated (verified with gfortran).  The writer refuses
  such names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ._fortran import FmtomoFormatError, RecordReader, read_text, write_text

__all__ = ["ControlFile", "F", "L", "cfloat"]


@dataclass(frozen=True)
class L:
    """A literal banner line (skipped when reading)."""

    text: str = ""


@dataclass(frozen=True)
class F:
    """A value line.

    ``kinds`` is one letter per value: ``i`` int, ``f`` float, ``b`` logical (``T``/``F``),
    ``w`` list-directed word, ``a`` fixed-width character (``width`` characters).  A single
    letter gives a scalar, several letters a tuple.  ``joins`` are the separators between
    values (default two blanks).  ``sep`` precedes the comment (``"c: "``; ``""`` for logicals
    followed by a name, as in ``mode_set.in``).
    """

    name: str
    kinds: str
    comment: str = ""
    width: int = 0
    joins: tuple[str, ...] = ()
    pad: int | None = None
    sep: str = "c: "


def cfloat(x: float) -> str:
    """Compact float text for control files (``1.0``, ``0.07``, ``6371.0``)."""
    return repr(float(x))


def _fmt_one(kind: str, v: Any, f: F, owner: str) -> str:
    if kind == "i":
        if isinstance(v, bool) or int(v) != v:
            raise FmtomoFormatError(f"{owner}.{f.name}: expected an integer, got {v!r}")
        return str(int(v))
    if kind == "f":
        return cfloat(v)
    if kind == "b":
        return "T" if v else "F"
    s = str(v)
    if kind == "a":
        if len(s) > f.width:
            raise FmtomoFormatError(
                f"{owner}.{f.name}: {s!r} is {len(s)} characters; the Fortran reader "
                f"(a{f.width}) only takes the first {f.width}"
            )
        return s
    if kind == "w":
        if not s or any(c.isspace() or c in ",/;'\"" for c in s):
            raise FmtomoFormatError(
                f"{owner}.{f.name}: {s!r} cannot be read by list-directed input (blank, comma "
                "or slash ends the value); use a plain file name in the working directory"
            )
        return s
    raise AssertionError(kind)


class ControlFile:
    """Base class: subclasses are dataclasses and define ``_layout`` (and optionally ``_tail``)."""

    _layout: ClassVar[tuple[L | F, ...]] = ()
    _pad: ClassVar[int] = 28
    _tail: ClassVar[str] = ""

    # -- parsing -------------------------------------------------------------------------
    @classmethod
    def parse(cls, text: str, source: str | Path | None = None):
        rd = RecordReader(text, source)
        kwargs: dict[str, Any] = {}
        last = max(i for i, it in enumerate(cls._layout) if isinstance(it, F))
        for item in cls._layout[: last + 1]:  # trailing literal lines are never read
            if isinstance(item, L):
                rd.skip()
                continue
            kwargs[item.name] = cls._read_line(rd, item)
        cls._parse_extra(rd, kwargs)
        return cls(**kwargs)

    @classmethod
    def _parse_extra(cls, rd: RecordReader, kwargs: dict[str, Any]) -> None:
        """Hook for variable-length sections following the fixed layout."""

    @staticmethod
    def _read_line(rd: RecordReader, item: F) -> Any:
        if item.kinds == "a":
            return rd.a(item.width)  # formatted read: first ``width`` characters of the record
        toks = rd.tokens(len(item.kinds))
        vals: list[Any] = []
        for k, t in zip(item.kinds, toks, strict=True):
            try:
                if k == "i":
                    vals.append(int(t))
                elif k == "f":
                    vals.append(float(t.replace("d", "e").replace("D", "e")))
                elif k == "b":
                    u = t.lstrip(".").upper()[:1]
                    if u not in ("T", "F"):
                        raise ValueError(t)
                    vals.append(u == "T")
                elif k == "w":
                    vals.append(t.strip("'\""))
                else:
                    raise AssertionError(k)
            except ValueError:
                raise rd.error(f"{item.name}: cannot read {t!r} as kind {k!r}") from None
        return vals[0] if len(vals) == 1 else tuple(vals)

    @classmethod
    def read(cls, path: str | Path):
        return cls.parse(read_text(path), path)

    # -- formatting ----------------------------------------------------------------------
    def format(self) -> str:
        owner = type(self).__name__
        lines: list[str] = []
        for item in self._layout:
            if isinstance(item, L):
                lines.append(item.text)
                continue
            v = getattr(self, item.name)
            vals = (v,) if len(item.kinds) == 1 else tuple(v)
            if len(vals) != len(item.kinds):
                raise FmtomoFormatError(
                    f"{owner}.{item.name}: expected {len(item.kinds)} values, got {len(vals)}"
                )
            parts = [_fmt_one(k, x, item, owner) for k, x in zip(item.kinds, vals, strict=True)]
            joins = item.joins or ("  ",) * (len(parts) - 1)
            text = parts[0]
            for j, p in zip(joins, parts[1:], strict=False):
                text += j + p
            lines.append(self._with_comment(text, item))
        lines.extend(self._extra_lines())
        return "\n".join(lines) + "\n" + self._tail

    def _extra_lines(self) -> list[str]:
        """Hook: lines of the variable-length section following the fixed layout."""
        return []

    def _with_comment(self, text: str, item: F) -> str:
        if not item.comment:
            return text
        pad = item.pad if item.pad is not None else self._pad
        return f"{text:<{pad}}{item.sep}{item.comment}"

    def write(self, path: str | Path) -> Path:
        return write_text(path, self.format())
