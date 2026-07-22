#!/usr/bin/env python3
"""Clone the pinned PrismML llama.cpp fork, apply the TurboQuant patchset, build.

Stdlib-only (runs before `uv sync`). Idempotent: safe to re-run; already-applied
patches are detected and skipped, an existing clone at the right tag is reused,
and cmake reconfigure/rebuild are incremental.

Usage:
    python3 apply_patches.py                 # apply patches to an existing clone
    python3 apply_patches.py --clone         # clone first if needed, then apply
    python3 apply_patches.py --clone --cpu-only   # ... then configure+build (CPU)
    python3 apply_patches.py --clone --cuda       # ... configure+build (CUDA backend)

--cuda requires the NVIDIA CUDA toolkit (nvcc). It is searched on PATH, then in
CUDA_HOME/CUDA_PATH, then /usr/local/cuda; if not found the build is skipped
with an explanation and exit code 2 (`suiban install turboquant` surfaces it).
The CUDA build passes -DGGML_CUDA_FA_ALL_QUANTS=ON: our fattn patch already
allows the mixed K=q8_0 / V=tq4_0|tq3_0 pair in default builds, but the flag
additionally compiles every mixed K/V flash-attention pair so non-suiban cache
combinations work too.

The clone lives at suiban/vendor/llama.cpp (gitignored, never committed).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent
CLONE_DIR = VENDOR_DIR / "llama.cpp"
PATCHES_DIR = VENDOR_DIR / "patches"
BUILD_DIR_NAME = "build-cpu"
CUDA_BUILD_DIR_NAME = "build-cuda"

FORK_URL = "https://github.com/PrismML-Eng/llama.cpp"
FORK_TAG = "prism-b9596-9fcaed7"

# Build targets that exercise every patched file (ggml core, ggml-cpu, common,
# tests, llama-bench). Deliberately not a full build — keeps CI/bootstrap fast.
BUILD_TARGETS = ["test-quantize-fns", "llama-bench"]


def run(
    cmd: list[str], cwd: Path | None = None, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=capture)


def clone_fork() -> None:
    """Shallow-clone the pinned fork tag; reuse a clone that is already correct."""
    if (CLONE_DIR / ".git").is_dir():
        head = subprocess.run(
            ["git", "describe", "--tags", "--always"], cwd=CLONE_DIR, capture_output=True, text=True
        )
        desc = head.stdout.strip()
        if desc == FORK_TAG:
            print(f"[clone] existing clone at {FORK_TAG} — reusing")
            return
        print(f"[clone] existing clone is at '{desc}', expected '{FORK_TAG}'", file=sys.stderr)
        print(
            "[clone] remove vendor/llama.cpp (or fetch the tag manually) and re-run",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if CLONE_DIR.exists():
        raise SystemExit(f"[clone] {CLONE_DIR} exists but is not a git clone — remove it first")
    print(f"[clone] shallow-cloning {FORK_URL} @ {FORK_TAG}")
    run(["git", "clone", "--depth", "1", "--branch", FORK_TAG, FORK_URL, str(CLONE_DIR)])


def patch_is_applied(patch: Path) -> bool:
    """A patch counts as applied when it reverse-applies cleanly."""
    res = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=CLONE_DIR,
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def applied_stack_top(patches: list[Path]) -> int:
    """Index of the highest patch that is applied, -1 for a pristine clone.

    Per-patch reverse-apply detection is not enough on its own: when a later
    patch edits lines an earlier one introduced (0007 reshapes 0006's
    dequantize_V_tq), the earlier patch no longer reverse-applies even though
    it IS applied. The stack applies strictly in order, so the highest patch
    that reverse-applies cleanly marks everything at or below it as applied."""
    for i in range(len(patches) - 1, -1, -1):
        if patch_is_applied(patches[i]):
            return i
    return -1


def apply_patches() -> None:
    if not (CLONE_DIR / ".git").is_dir():
        raise SystemExit(f"[apply] no clone at {CLONE_DIR} — run with --clone first")
    patches = sorted(PATCHES_DIR.glob("[0-9][0-9][0-9][0-9]-*.patch"))
    if not patches:
        raise SystemExit(f"[apply] no patches found in {PATCHES_DIR}")
    top = applied_stack_top(patches)
    for i, patch in enumerate(patches):
        if i <= top:
            print(f"[apply] {patch.name}: already applied — skipping")
            continue
        check = subprocess.run(
            ["git", "apply", "--check", str(patch)], cwd=CLONE_DIR, capture_output=True, text=True
        )
        if check.returncode != 0:
            print(f"[apply] {patch.name}: does NOT apply cleanly:", file=sys.stderr)
            print(check.stderr, file=sys.stderr)
            print(
                "[apply] the clone may be dirty or at the wrong tag; "
                "`git -C vendor/llama.cpp checkout -- .` restores pristine sources",
                file=sys.stderr,
            )
            raise SystemExit(1)
        run(["git", "apply", str(patch)], cwd=CLONE_DIR)
        print(f"[apply] {patch.name}: applied")
    print("[apply] patchset complete")


def build_cpu() -> None:
    if shutil.which("cmake") is None:
        raise SystemExit("[build] cmake not found on PATH")
    build_dir = CLONE_DIR / BUILD_DIR_NAME
    jobs = str(os.cpu_count() or 4)
    print(f"[build] configuring (Release, CPU-only) in {build_dir}")
    run(
        [
            "cmake",
            "-B",
            BUILD_DIR_NAME,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=OFF",
            "-DBUILD_SHARED_LIBS=ON",
            "-DLLAMA_BUILD_TESTS=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
        ],
        cwd=CLONE_DIR,
    )
    print(f"[build] building targets {BUILD_TARGETS} with -j{jobs}")
    run(["cmake", "--build", BUILD_DIR_NAME, "--target", *BUILD_TARGETS, "-j", jobs], cwd=CLONE_DIR)
    print(f"[build] done — binaries in {build_dir / 'bin'}")


def find_nvcc() -> Path | None:
    """Locate nvcc: PATH, then CUDA_HOME/CUDA_PATH, then /usr/local/cuda."""
    on_path = shutil.which("nvcc")
    if on_path:
        return Path(on_path)
    candidates = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root:
            candidates.append(Path(root) / "bin" / "nvcc")
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def build_cuda(cuda_host_compiler: str | None) -> None:
    """Configure + build the CUDA backend. Exits 2 (with explanation) when nvcc is missing."""
    if shutil.which("cmake") is None:
        raise SystemExit("[build] cmake not found on PATH")
    nvcc = find_nvcc()
    if nvcc is None:
        print(
            "[cuda] nvcc not found (PATH, CUDA_HOME, CUDA_PATH, /usr/local/cuda) — "
            "skipping the CUDA TurboQuant build.\n"
            "[cuda] The CUDA kernels need the NVIDIA CUDA toolkit to compile; a GPU "
            "driver alone is not enough. Install the toolkit (e.g. from "
            "developer.nvidia.com/cuda-downloads) and re-run:\n"
            "[cuda]     python3 apply_patches.py --cuda\n"
            "[cuda] Until then llama-server runs with the CPU TurboQuant build or the "
            "prebuilt binary fallback (K=q8_0/V=q8_0 + notice).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    build_dir = CLONE_DIR / CUDA_BUILD_DIR_NAME
    jobs = str(os.cpu_count() or 4)
    print(f"[cuda] using nvcc: {nvcc}")
    print(f"[cuda] configuring (Release, GGML_CUDA=ON, GGML_CUDA_FA_ALL_QUANTS=ON) in {build_dir}")
    cmake_args = [
        "cmake",
        "-B",
        CUDA_BUILD_DIR_NAME,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_FA_ALL_QUANTS=ON",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_TESTS=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
    ]
    if cuda_host_compiler:
        cmake_args.append(f"-DCMAKE_CUDA_HOST_COMPILER={cuda_host_compiler}")
    run(cmake_args, cwd=CLONE_DIR)
    print(f"[cuda] building targets {BUILD_TARGETS} with -j{jobs}")
    run(
        ["cmake", "--build", CUDA_BUILD_DIR_NAME, "--target", *BUILD_TARGETS, "-j", jobs],
        cwd=CLONE_DIR,
    )
    print(f"[cuda] done — binaries in {build_dir / 'bin'}")


# Tool binaries promoted alongside llama-server: `suiban bench kv` runs
# llama-perplexity from the same directory (honest n/a fallback when absent).
INSTALL_BINARIES = ("llama-server", "llama-perplexity")


def install_into_home(backend: str) -> None:
    """Build llama-server (+ llama-perplexity for `suiban bench kv`) and promote
    them plus the fork's shared libs into ~/.bonsai/bin/<backend>/, writing the
    TURBOQUANT marker suiban keys on.

    This is the "swap the binary" step behind `suiban install turboquant`: after it,
    binary resolution prefers this TurboQuant-enabled build over the prebuilts and the
    KV cache stops falling back to q8_0/q8_0."""
    build_dir_name = CUDA_BUILD_DIR_NAME if backend == "cuda" else BUILD_DIR_NAME
    build_dir = CLONE_DIR / build_dir_name
    if not build_dir.is_dir():
        raise SystemExit(f"[install] no {build_dir_name} build dir — build the backend first")
    jobs = str(os.cpu_count() or 4)
    print(f"[install] building {', '.join(INSTALL_BINARIES)} ({backend})")
    run(
        ["cmake", "--build", build_dir_name, "--target", *INSTALL_BINARIES, "-j", jobs],
        cwd=CLONE_DIR,
    )
    server = build_dir / "bin" / "llama-server"
    if not server.is_file():
        raise SystemExit(f"[install] build finished but {server} is missing")
    home = Path(os.environ.get("SUIBAN_HOME", str(Path.home() / ".bonsai")))
    bin_dir = home / "bin" / backend
    bin_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in sorted((build_dir / "bin").iterdir()):
        if (
            item.name not in INSTALL_BINARIES
            and ".so" not in item.name
            and ".dylib" not in item.name
        ):
            continue
        target = bin_dir / item.name
        if item.is_symlink():
            link_target = os.readlink(item)
            target.unlink(missing_ok=True)
            os.symlink(link_target, target)
        else:
            target.unlink(missing_ok=True)  # never overwrite through a stale symlink
            shutil.copy2(item, target)
        copied += 1
    (bin_dir / "RELEASE").write_text(FORK_TAG + "\n", encoding="utf-8")
    patches = sorted(p.name for p in PATCHES_DIR.glob("[0-9][0-9][0-9][0-9]-*.patch"))
    (bin_dir / "TURBOQUANT").write_text("\n".join([FORK_TAG, *patches]) + "\n", encoding="utf-8")
    print(f"[install] {copied} files -> {bin_dir}")
    print("[install] TURBOQUANT marker written — suiban now serves with TQ4_0/TQ3_0 V-cache")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--clone", action="store_true", help=f"shallow-clone {FORK_URL} @ {FORK_TAG} if not present"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="cmake configure + build (Release, GGML_CUDA=OFF) after patching",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="cmake configure + build the CUDA backend (Release, GGML_CUDA=ON, "
        "GGML_CUDA_FA_ALL_QUANTS=ON) after patching; exits 2 if nvcc is missing",
    )
    parser.add_argument(
        "--cuda-host-compiler",
        metavar="PATH",
        default=None,
        help="host C++ compiler for nvcc (-DCMAKE_CUDA_HOST_COMPILER). Needed when the "
        "system compiler is newer than the CUDA toolkit supports (e.g. GCC 16 with "
        "CUDA 13.x: pass a GCC <= 15 path). The standard CMake env var CUDAHOSTCXX "
        "works too and is picked up automatically when this flag is not given",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        default=None,
        help="build the given backend after patching — the interface `suiban install "
        "turboquant` uses (equivalent to --cpu-only / --cuda)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="after building, build llama-server and promote it + the fork's shared "
        "libs into ~/.bonsai/bin/<backend>/ (SUIBAN_HOME honored), writing the "
        "TURBOQUANT marker so suiban serves with the TQ4_0/TQ3_0 V-cache",
    )
    args = parser.parse_args()

    backend = args.backend or ("cuda" if args.cuda else "cpu" if args.cpu_only else None)
    if args.install and backend is None:
        parser.error("--install requires a backend (--backend, --cuda, or --cpu-only)")

    if args.clone:
        clone_fork()
    apply_patches()
    if args.cpu_only or args.backend == "cpu":
        build_cpu()
    if args.cuda or args.backend == "cuda":
        build_cuda(args.cuda_host_compiler)
    if args.install:
        install_into_home(backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
