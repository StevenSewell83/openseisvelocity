"""Tests for install.sh / environment.yml (the optional full install is marked integration)."""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "install.sh"


def sh(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALL), *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def test_syntax() -> None:
    subprocess.run(["bash", "-n", str(INSTALL)], check=True)


def test_executable() -> None:
    assert INSTALL.stat().st_mode & 0o111


def test_help() -> None:
    r = sh("--help")
    assert r.returncode == 0
    for flag in (
        "--env-name",
        "--prefix",
        "--no-gmt",
        "--system-compilers",
        "--venv",
        "--jobs",
        "--dev",
        "--skip-tests",
        "--dry-run",
        "--velest-ieq",
    ):
        assert flag in r.stdout


def test_unknown_option() -> None:
    r = sh("--bogus")
    assert r.returncode != 0
    assert "unknown option" in r.stderr


def test_option_needs_value() -> None:
    assert sh("--prefix").returncode != 0


@pytest.mark.skipif(
    not all(shutil.which(t) for t in ("gfortran", "cmake", "make")), reason="no system compilers"
)
def test_dry_run_venv(tmp_path: Path) -> None:
    prefix = tmp_path / "venv"
    r = sh(
        "--dry-run",
        "--venv",
        "--dev",
        "--force",
        "--prefix",
        str(prefix),
        "--jobs",
        "3",
        "--velest-ieq",
        "9000",
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "build_fortran.sh" in out and "--velest-ieq 9000" in out and "--jobs 3" in out
    assert str(prefix) in out and "source" in out
    assert not prefix.exists(), "dry run must not create anything"


def test_dry_run_conda_filters_compilers(tmp_path: Path) -> None:
    r = sh(
        "--dry-run",
        "--force",
        "--system-compilers",
        "--no-gmt",
        "--prefix",
        str(tmp_path / "env"),
        "--conda-exe",
        "/bin/true",
    )
    # the dry run only echoes; it must mention the env file and never create the prefix
    assert r.returncode == 0, r.stderr
    assert "environment.yml" in r.stdout and "environment-gmt" not in r.stdout
    assert not (tmp_path / "env").exists()


def test_dry_run_gmt_default(tmp_path: Path) -> None:
    r = sh("--dry-run", "--force", "--prefix", str(tmp_path / "env"), "--conda-exe", "/bin/true")
    assert "environment-gmt.yml" in r.stdout


def _bootstrap(tmp_path: Path, urls: list[str]) -> subprocess.CompletedProcess[str]:
    """Run only bootstrap_micromamba() from install.sh with the given download URLs."""
    text = INSTALL.read_text()
    funcs = [line for line in text.splitlines() if line.startswith(("log()", "warn()", "die()"))]
    for name in ("platform", "classify", "bootstrap_micromamba"):
        start = text.index(f"\n{name}() {{") + 1
        funcs.append(text[start : text.index("\n}\n", start) + 2])
    quoted = " ".join(f"'{u}'" for u in urls)
    script = "\n".join(
        [
            "set -euo pipefail",
            *funcs,
            f"DRY_RUN=0; MICROMAMBA_HOME='{tmp_path / 'mm'}'; MICROMAMBA_URLS=({quoted})",
            'bootstrap_micromamba; "$MGR" --version',
        ]
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False
    )


@pytest.mark.skipif(not shutil.which("curl"), reason="no curl")
def test_bootstrap_micromamba_falls_back(tmp_path: Path) -> None:
    fake = tmp_path / "pkg" / "bin" / "micromamba"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\necho 9.9.9\n")
    fake.chmod(0o755)
    tarball = tmp_path / "mm.tar.bz2"
    subprocess.run(
        ["tar", "-cjf", str(tarball), "-C", str(fake.parents[1]), "bin/micromamba"], check=True
    )
    junk = tmp_path / "junk"
    junk.write_text("not a program\n")
    missing = tmp_path / "missing"

    # unreachable URL, then a conda package tarball
    r = _bootstrap(tmp_path, [f"file://{missing}", f"file://{tarball}"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("9.9.9")
    # a file that is not micromamba, then a bare executable (GitHub release asset)
    shutil.rmtree(tmp_path / "mm")
    r = _bootstrap(tmp_path, [f"file://{junk}", f"file://{fake}"])
    assert r.returncode == 0, r.stderr
    assert "not a working micromamba" in r.stderr
    # nothing reachable: a clear error
    shutil.rmtree(tmp_path / "mm")
    r = _bootstrap(tmp_path, [f"file://{missing}"])
    assert r.returncode != 0
    assert "could not download micromamba" in r.stderr


def test_default_micromamba_urls_include_github_fallback() -> None:
    text = INSTALL.read_text()
    assert "micro.mamba.pm/api/micromamba/{PLATFORM}/latest" in text
    assert "github.com/mamba-org/micromamba-releases/releases/latest/download" in text


def test_environment_files_have_required_packages() -> None:
    text = (ROOT / "environment.yml").read_text()
    for pkg in (
        "python=3.11",
        "obspy",
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "xarray",
        "netcdf4",
        "matplotlib",
        "pyproj",
        "pydantic>=2",
        "pyyaml",
        "typer",
        "joblib",
        "jinja2",
        "lasio",
        "rasterio",
        "pyvista",
        "cartopy",
        "cmake",
        "pytest",
        "ruff",
        "fortran-compiler",
    ):
        assert f"- {pkg}" in text, pkg
    assert "conda-forge" in text
    gmt = (ROOT / "environment-gmt.yml").read_text()
    assert "- gmt" in gmt and "- pygmt" in gmt


def _online() -> bool:
    try:
        socket.create_connection(("pypi.org", 443), timeout=3).close()
    except OSError:
        return False
    return True


@pytest.mark.integration
def test_full_venv_install(tmp_path: Path) -> None:
    """Real ``--system-compilers --venv`` install (needs gfortran, cmake, network)."""
    if not all(shutil.which(t) for t in ("gfortran", "cmake", "make")):
        pytest.skip("system gfortran/cmake/make not available")
    if not _online():
        pytest.skip("no network for pip")
    prefix = tmp_path / "venv"
    r = sh(
        "--system-compilers",
        "--venv",
        "--prefix",
        str(prefix),
        "--skip-tests",
        "--force",
        timeout=1800,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (prefix / "bin" / "osv-tools" / "velest").is_file()
    assert (prefix / "bin" / "osv-tools" / "fm3d").is_file()
    again = sh(
        "--system-compilers",
        "--venv",
        "--prefix",
        str(prefix),
        "--skip-tests",
        "--force",
        timeout=1800,
    )
    assert again.returncode == 0, "re-running must be idempotent"
    v = subprocess.run([str(prefix / "bin" / "osv"), "--version"], capture_output=True, text=True)
    assert v.returncode == 0
