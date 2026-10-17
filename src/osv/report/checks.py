"""Generic QC finding record, shared by all stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Severity = Literal["info", "warn", "fail"]
SEVERITIES: tuple[str, ...] = ("info", "warn", "fail")


@dataclass(frozen=True)
class QCFinding:
    """One quality-control observation.

    Parameters
    ----------
    code
        Short machine-readable identifier, e.g. ``"duplicate_arrivals"``.
    severity
        ``"info"``, ``"warn"`` or ``"fail"``.
    message
        Human-readable description (what was found and what was done about it).
    count
        Number of items affected (picks, events, ...); 0 when not applicable.
    details
        Free-form JSON-serialisable extra information.
    """

    code: str
    severity: Severity
    message: str
    count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict (JSON-friendly) representation."""
        return asdict(self)

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


def worst_severity(findings: list[QCFinding]) -> Severity:
    """Return the most severe level among ``findings`` (``"info"`` if empty)."""
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    worst = "info"
    for f in findings:
        if rank[f.severity] > rank[worst]:
            worst = f.severity
    return worst  # type: ignore[return-value]
