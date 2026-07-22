"""Compute-backend detection: cuda > rocm > vulkan > cpu (metal on darwin).

Detection is deliberately cheap and shell-out free where possible; every probe is
individually fallible and falls through to the next.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable

BACKENDS = ("cuda", "rocm", "metal", "vulkan", "cpu")


def _nvml_probe() -> bool:
    try:
        import pynvml

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return True
    except Exception:
        return False


def detect_backend(
    *,
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    nvml_probe: Callable[[], bool] = _nvml_probe,
) -> str:
    """Best available backend for this machine (probes injectable for tests)."""
    plat = platform if platform is not None else sys.platform
    if plat == "darwin":
        return "metal"
    if which("nvidia-smi") is not None or nvml_probe():
        return "cuda"
    if which("rocm-smi") is not None or which("rocminfo") is not None:
        return "rocm"
    if which("vulkaninfo") is not None:
        return "vulkan"
    return "cpu"
