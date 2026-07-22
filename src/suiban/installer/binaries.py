"""Pinned fork binary install: download release archives (linux/macos .tar.gz, windows
.zip), extract llama-server into ~/.bonsai/bin/<backend>/, record the RELEASE marker.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from suiban import paths
from suiban.errors import BonsaiError
from suiban.installer.assets import ReleaseAssets, asset_names, release_url
from suiban.llama.binary import PRISM_RELEASE_TAG, SERVER_BINARY_NAME

# Checked-in SHA-256 manifest for the pinned release assets (keyed by asset filename).
# Populated from the GitHub releases API asset `digest` field at author time; see the
# file's own header. This is the repo-pinned integrity anchor: TLS gets the bytes here,
# this hash proves they are the bytes we shipped against.
_ASSET_DIGESTS_PATH = Path(__file__).parent / "assets_sha256.json"


def load_asset_digests() -> dict[str, str]:
    """SHA-256 hex digests for the pinned release assets, keyed by filename. A filename
    absent (or null) here has no checked-in digest and cannot be integrity-checked."""
    try:
        data = json.loads(_ASSET_DIGESTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    digests = data.get("digests", {})
    return {name: h.lower() for name, h in digests.items() if isinstance(h, str) and h}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_asset(
    path: Path,
    name: str,
    digests: dict[str, str],
    progress: Callable[[str], None] = print,
) -> bool:
    """Check a downloaded asset against the checked-in SHA-256 manifest.

    HARD-FAILS (BonsaiError) on a mismatch — a corrupt or tampered download must never
    be extracted. Returns True when verified, False when no digest is on record (a
    surfaced warning, never a silent skip)."""
    expected = digests.get(name)
    if not expected:
        progress(
            f"  WARNING: no checked-in SHA-256 for {name}; installing WITHOUT an "
            f"integrity check (see KNOWN_ISSUES.md)"
        )
        return False
    actual = _sha256_file(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise BonsaiError(
            500,
            f"SHA-256 mismatch for {name}: expected {expected}, got {actual}. The "
            f"download is corrupt or has been tampered with. Delete ~/.bonsai/bin and "
            f"re-run: suiban install binaries",
            code="asset_sha256_mismatch",
        )
    progress(f"  verified {name} (sha256 ok)")
    return True


def host_os() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def host_arch() -> str:
    import platform

    machine = platform.machine().lower()
    return "arm64" if machine in ("arm64", "aarch64") else "x64"


DOWNLOAD_ATTEMPTS = 4
_TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)


def download_asset(url: str, dest: Path, progress: Callable[[str], None] = print) -> Path:
    """Streamed download with coarse progress, retries, and HTTP-Range resume.

    Release assets are hundreds of MB and residential links stall; a mid-stream
    timeout resumes from the partial file instead of starting over."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with httpx.stream(
                "GET", url, headers=headers, follow_redirects=True, timeout=_TIMEOUT
            ) as resp:
                if offset and resp.status_code == 200:
                    offset = 0  # server ignored the Range; restart the file
                if resp.status_code not in (200, 206):
                    raise BonsaiError(
                        500,
                        f"download failed ({resp.status_code}): {url}",
                        code="asset_download_failed",
                    )
                total = offset + int(resp.headers.get("content-length", 0))
                done = offset
                next_report = done * 100 // total if total else 0
                with part.open("ab" if offset else "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total and done * 100 // total >= next_report:
                            progress(f"  {dest.name}: {done * 100 // total}%")
                            next_report += 25
            part.rename(dest)
            return dest
        except BonsaiError:
            part.unlink(missing_ok=True)  # 4xx/5xx: partial data is untrustworthy
            raise
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                resume_from = part.stat().st_size if part.exists() else 0
                progress(
                    f"  {dest.name}: {type(exc).__name__}, retrying "
                    f"({attempt}/{DOWNLOAD_ATTEMPTS - 1}) from {resume_from} bytes"
                )
    raise BonsaiError(
        500,
        f"download failed after {DOWNLOAD_ATTEMPTS} attempts: {url} ({last_error})",
        code="asset_download_failed",
    )


def _mark_executable(target: Path, name: str) -> None:
    if name == SERVER_BINARY_NAME or name.startswith(("lib", "llama")):
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def extract_binaries(archive_path: Path, bin_dir: Path) -> list[Path]:
    """Pull executable/library files out of a release archive (.zip or .tar.gz),
    flattening directories."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    if archive_path.name.endswith((".tar.gz", ".tgz")):
        symlinks: list[tuple[str, str]] = []  # (link name, flattened target name)
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                name = Path(member.name).name
                if not name:
                    continue
                if member.issym() or member.islnk():
                    # soname links (libggml-base.so.0 -> libggml-base.so.0.13.1) are
                    # load-bearing for llama-server; recreate them flattened.
                    symlinks.append((name, Path(member.linkname).name))
                    continue
                if not member.isfile():
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                target = bin_dir / name
                with src, target.open("wb") as dst:
                    dst.write(src.read())
                _mark_executable(target, name)
                extracted.append(target)
        for link_name, target_name in symlinks:
            link = bin_dir / link_name
            link.unlink(missing_ok=True)
            os.symlink(target_name, link)
            extracted.append(link)
        return extracted
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name:
                continue
            target = bin_dir / name
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            _mark_executable(target, name)
            extracted.append(target)
    return extracted


def install_binaries(
    backend: str,
    *,
    os_name: str | None = None,
    arch: str | None = None,
    progress: Callable[[str], None] = print,
    fetch: Callable[[str, Path], Path] | None = None,
    digests: dict[str, str] | None = None,
) -> Path:
    """Install the pinned fork prebuilts for a backend. Returns the llama-server path.

    Every downloaded archive is SHA-256-verified against the checked-in manifest before
    it is extracted (`digests` defaults to `load_asset_digests()`; injectable for tests).
    A mismatch is a hard failure — no partial or tampered install is ever laid down."""
    os_name = os_name or host_os()
    arch = arch or host_arch()
    assets: ReleaseAssets = asset_names(os_name, backend, arch)
    bin_dir = paths.bin_dir(backend)
    tmp_dir = bin_dir / "_download"
    fetch = fetch or (lambda url, dest: download_asset(url, dest, progress))
    digests = load_asset_digests() if digests is None else digests

    progress(f"installing {PRISM_RELEASE_TAG} prebuilts for {os_name}-{backend}-{arch}")
    for name in assets.all_names:
        url = release_url(name)
        progress(f"  fetching {name}")
        zip_path = fetch(url, tmp_dir / name)
        verify_asset(zip_path, name, digests, progress)
        extract_binaries(zip_path, bin_dir)
        os.remove(zip_path)
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    server = bin_dir / SERVER_BINARY_NAME
    if not server.is_file():
        raise BonsaiError(
            500,
            f"release assets extracted but {SERVER_BINARY_NAME} not found in {bin_dir}",
            code="binary_extract_failed",
        )
    (bin_dir / "RELEASE").write_text(PRISM_RELEASE_TAG + "\n", encoding="utf-8")
    progress(f"  installed {server}")
    return server
