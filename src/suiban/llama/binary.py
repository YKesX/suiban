"""Resolution of the pinned PrismML llama.cpp fork binaries under ~/.bonsai/bin/.

Fork: PrismML-Eng/llama.cpp, branch `prism`, pinned release tag prism-b9596-9fcaed7.
The installer writes a RELEASE marker next to the binary; a TURBOQUANT marker is written
only by `suiban install turboquant` (source build with the vendored patchset).
"""

from __future__ import annotations

import sys
from pathlib import Path

from suiban import paths
from suiban.errors import BonsaiError

PRISM_BUILD = "b9596"
PRISM_SHA = "9fcaed7"
PRISM_RELEASE_TAG = f"prism-{PRISM_BUILD}-{PRISM_SHA}"
PRISM_REPO = "PrismML-Eng/llama.cpp"

SERVER_BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"


class BinaryMissing(BonsaiError):
    def __init__(self, path: Path) -> None:
        super().__init__(
            500,
            f"llama-server binary not found at {path}. Run: suiban install binaries",
            code="binary_missing",
        )
        self.path = path


def server_binary_path(backend: str) -> Path:
    return paths.bin_dir(backend) / SERVER_BINARY_NAME


def resolve_server_binary(backend: str) -> Path:
    """Path to the pinned fork's llama-server for a backend, or BinaryMissing."""
    path = server_binary_path(backend)
    if not path.is_file():
        raise BinaryMissing(path)
    return path


def installed_release(backend: str) -> str | None:
    """Tag recorded by the installer, or None if never installed / unknown."""
    marker = paths.bin_dir(backend) / "RELEASE"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    return None


def release_matches_pin(backend: str) -> bool:
    return installed_release(backend) == PRISM_RELEASE_TAG


def turboquant_installed(backend: str) -> bool:
    """True only after `suiban install turboquant` built + swapped the binary."""
    return (paths.bin_dir(backend) / "TURBOQUANT").is_file()
