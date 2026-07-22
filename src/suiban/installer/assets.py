"""Fork release asset resolution for the pinned tag prism-b9596-9fcaed7.

Naming rules below were verified against the LIVE release asset list via the GitHub API
(2026-07-21) — the release is a mixed bag and each quirk here is real:

- linux CUDA:   llama-prism-<build>-<sha>-bin-linux-cuda-<12.4|12.8>-x64.tar.gz
- linux CPU:    llama-prism-<build>-<sha>-bin-ubuntu-<arch>.tar.gz        ("ubuntu", no
  backend token — only the CUDA builds say "linux")
- linux vulkan: llama-prism-<build>-<sha>-bin-ubuntu-vulkan-<arch>.tar.gz
- linux rocm:   llama-prism-<build>-<sha>-bin-ubuntu-rocm-7.2-x64.tar.gz  (x64 only)
- macos:        llama-prism-<build>-<sha>-bin-macos-<arch>.tar.gz         (Metal implicit;
  an arm64 "-kleidiai" variant exists — we install the plain one)
- windows:      mainline-style names WITHOUT the prism prefix: llama-bin-win-<cpu|vulkan>-
  <arch>.zip, llama-bin-win-hip-radeon-x64.zip; CUDA is the odd one out:
  llama-prism-b1-<sha>-bin-win-cuda-12.4-x64.zip ("b1", not the build number) plus the
  cudart-llama-bin-win-cuda-12.4-x64.zip runtime companion.
"""

from __future__ import annotations

from dataclasses import dataclass

from suiban.llama.binary import PRISM_BUILD, PRISM_RELEASE_TAG, PRISM_REPO, PRISM_SHA

VALID_OS = ("linux", "macos", "windows")
VALID_ARCH = ("x64", "arm64")
BACKENDS_BY_OS: dict[str, tuple[str, ...]] = {
    "linux": ("cuda", "rocm", "vulkan", "cpu"),
    "macos": ("metal", "cpu"),
    "windows": ("cuda", "rocm", "vulkan", "cpu"),
}

# Linux CUDA builds exist per toolkit line; 12.8 is smaller and fine for any recent
# driver, 12.4 is the compatibility fallback for older driver branches.
LINUX_CUDA_VERSIONS = ("12.8", "12.4")

_PRISM_STEM = f"llama-prism-{PRISM_BUILD}-{PRISM_SHA}-bin"


@dataclass(frozen=True)
class ReleaseAssets:
    """Asset filenames to download for one (os, backend, arch) combination.
    `primary` contains llama-server; `extras` are companions (win cudart)."""

    primary: str
    extras: tuple[str, ...] = ()

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.primary, *self.extras)


def asset_names(
    os_name: str, backend: str, arch: str, *, cuda_version: str | None = None
) -> ReleaseAssets:
    if os_name not in VALID_OS:
        raise ValueError(f"unknown os {os_name!r}; expected one of {VALID_OS}")
    if arch not in VALID_ARCH:
        raise ValueError(f"unknown arch {arch!r}; expected one of {VALID_ARCH}")
    if backend not in BACKENDS_BY_OS[os_name]:
        raise ValueError(
            f"backend {backend!r} has no {os_name} prebuilt; available: {BACKENDS_BY_OS[os_name]}"
        )

    if os_name == "windows":
        if backend == "cuda":
            # Real quirk: the win CUDA archive says "b1", not the build number.
            return ReleaseAssets(
                primary=f"llama-prism-b1-{PRISM_SHA}-bin-win-cuda-12.4-{arch}.zip",
                extras=(f"cudart-llama-bin-win-cuda-12.4-{arch}.zip",),
            )
        if backend == "rocm":
            return ReleaseAssets(primary="llama-bin-win-hip-radeon-x64.zip")
        return ReleaseAssets(primary=f"llama-bin-win-{backend}-{arch}.zip")

    if os_name == "macos":
        # Metal support is implicit in the macos builds; "metal" and "cpu" resolve alike.
        return ReleaseAssets(primary=f"{_PRISM_STEM}-macos-{arch}.tar.gz")

    # linux
    if backend == "cuda":
        version = cuda_version or LINUX_CUDA_VERSIONS[0]
        if version not in LINUX_CUDA_VERSIONS:
            raise ValueError(
                f"no linux CUDA {version} prebuilt at {PRISM_RELEASE_TAG}; "
                f"available: {LINUX_CUDA_VERSIONS}"
            )
        return ReleaseAssets(primary=f"{_PRISM_STEM}-linux-cuda-{version}-{arch}.tar.gz")
    if backend == "rocm":
        if arch != "x64":
            raise ValueError("the rocm prebuilt exists for x64 only")
        return ReleaseAssets(primary=f"{_PRISM_STEM}-ubuntu-rocm-7.2-x64.tar.gz")
    if backend == "vulkan":
        return ReleaseAssets(primary=f"{_PRISM_STEM}-ubuntu-vulkan-{arch}.tar.gz")
    return ReleaseAssets(primary=f"{_PRISM_STEM}-ubuntu-{arch}.tar.gz")


def release_url(asset_name: str) -> str:
    return f"https://github.com/{PRISM_REPO}/releases/download/{PRISM_RELEASE_TAG}/{asset_name}"
