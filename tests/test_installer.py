"""Installer: release asset naming (incl. the Windows quirk), backend detection,
model download flow with the network fully mocked."""

from __future__ import annotations

from pathlib import Path

import pytest

from suiban.errors import BonsaiError
from suiban.installer import models as model_store
from suiban.installer.assets import asset_names, release_url
from suiban.installer.backend import detect_backend
from suiban.llama.binary import PRISM_RELEASE_TAG


# -- release asset names ------------------------------------------------------
# Expected names verified against the LIVE release asset listing (GitHub API,
# 2026-07-21). Every quirk asserted here exists in the real release.
@pytest.mark.parametrize(
    ("os_name", "backend", "arch", "expected"),
    [
        ("linux", "cuda", "x64", "llama-prism-b9596-9fcaed7-bin-linux-cuda-12.8-x64.tar.gz"),
        ("linux", "rocm", "x64", "llama-prism-b9596-9fcaed7-bin-ubuntu-rocm-7.2-x64.tar.gz"),
        ("linux", "vulkan", "x64", "llama-prism-b9596-9fcaed7-bin-ubuntu-vulkan-x64.tar.gz"),
        ("linux", "cpu", "arm64", "llama-prism-b9596-9fcaed7-bin-ubuntu-arm64.tar.gz"),
        ("macos", "metal", "arm64", "llama-prism-b9596-9fcaed7-bin-macos-arm64.tar.gz"),
        ("macos", "cpu", "x64", "llama-prism-b9596-9fcaed7-bin-macos-x64.tar.gz"),
    ],
)
def test_fork_scheme_asset_names(os_name: str, backend: str, arch: str, expected: str) -> None:
    assets = asset_names(os_name, backend, arch)
    assert assets.primary == expected
    assert assets.extras == ()


def test_linux_cuda_version_fallback() -> None:
    assets = asset_names("linux", "cuda", "x64", cuda_version="12.4")
    assert assets.primary == "llama-prism-b9596-9fcaed7-bin-linux-cuda-12.4-x64.tar.gz"
    with pytest.raises(ValueError):
        asset_names("linux", "cuda", "x64", cuda_version="11.8")


def test_windows_naming_quirk_cuda_needs_cudart() -> None:
    # Real quirks: win CUDA archive is prefixed "llama-prism-b1" (not the build number)
    # and needs the separate cudart runtime zip.
    assets = asset_names("windows", "cuda", "x64")
    assert assets.primary == "llama-prism-b1-9fcaed7-bin-win-cuda-12.4-x64.zip"
    assert assets.extras == ("cudart-llama-bin-win-cuda-12.4-x64.zip",)


def test_windows_non_cuda_has_no_cudart() -> None:
    assets = asset_names("windows", "vulkan", "x64")
    assert assets.primary == "llama-bin-win-vulkan-x64.zip"
    assert assets.extras == ()
    assert asset_names("windows", "rocm", "x64").primary == "llama-bin-win-hip-radeon-x64.zip"


@pytest.mark.parametrize(
    ("os_name", "backend", "arch"),
    [
        ("linux", "metal", "x64"),  # no Metal on linux
        ("macos", "cuda", "arm64"),  # no CUDA on macos
        ("linux", "rocm", "arm64"),  # rocm prebuilt is x64-only
        ("beos", "cpu", "x64"),
        ("linux", "cpu", "riscv"),
    ],
)
def test_invalid_combinations_rejected(os_name: str, backend: str, arch: str) -> None:
    with pytest.raises(ValueError):
        asset_names(os_name, backend, arch)


def test_release_url_uses_pinned_tag() -> None:
    url = release_url("llama-prism-b9596-9fcaed7-bin-linux-cuda-12.8-x64.tar.gz")
    assert PRISM_RELEASE_TAG == "prism-b9596-9fcaed7"  # the correct pin, per plan
    assert url == (
        "https://github.com/PrismML-Eng/llama.cpp/releases/download/"
        "prism-b9596-9fcaed7/llama-prism-b9596-9fcaed7-bin-linux-cuda-12.8-x64.tar.gz"
    )


# -- backend detection --------------------------------------------------------
def _no_which(_name: str) -> None:
    return None


def test_detect_darwin_is_metal() -> None:
    assert detect_backend(platform="darwin", which=_no_which, nvml_probe=lambda: False) == "metal"


def test_detect_cuda_via_nvidia_smi() -> None:
    which = lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None  # noqa: E731
    assert detect_backend(platform="linux", which=which, nvml_probe=lambda: False) == "cuda"


def test_detect_cuda_via_nvml() -> None:
    assert detect_backend(platform="linux", which=_no_which, nvml_probe=lambda: True) == "cuda"


def test_detect_rocm() -> None:
    which = lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None  # noqa: E731
    assert detect_backend(platform="linux", which=which, nvml_probe=lambda: False) == "rocm"


def test_detect_vulkan() -> None:
    which = lambda name: "/usr/bin/vulkaninfo" if name == "vulkaninfo" else None  # noqa: E731
    assert detect_backend(platform="linux", which=which, nvml_probe=lambda: False) == "vulkan"


def test_detect_cpu_fallback() -> None:
    assert detect_backend(platform="linux", which=_no_which, nvml_probe=lambda: False) == "cpu"


# -- model downloads (network mocked) ----------------------------------------
# Real file listings as verified live against the HF tree API (2026-07-21).
TERNARY_27B_FILES = [
    "README.md",
    "Ternary-Bonsai-27B-F16.gguf",
    "Ternary-Bonsai-27B-PQ2_0.gguf",  # future fork format — must never be picked
    "Ternary-Bonsai-27B-Q2_0.gguf",
    "Ternary-Bonsai-27B-Q2_g64.gguf",  # mainline format — must never be picked
    "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
    "Ternary-Bonsai-27B-dspark-bf16.gguf",
    "Ternary-Bonsai-27B-mmproj-BF16.gguf",
    "Ternary-Bonsai-27B-mmproj-Q8_0.gguf",
]
ONEBIT_8B_FILES = ["README.md", "Bonsai-8B-Q1_0.gguf", "Bonsai-8B.gguf"]


def test_pick_repo_file_by_family() -> None:
    assert model_store.pick_repo_file(TERNARY_27B_FILES, "ternary", "bonsai-27b") == (
        "Ternary-Bonsai-27B-Q2_0.gguf"
    )
    assert model_store.pick_repo_file(ONEBIT_8B_FILES, "1bit", "bonsai-8b") == (
        "Bonsai-8B-Q1_0.gguf"
    )
    assert model_store.pick_repo_file(TERNARY_27B_FILES, "ternary", "bonsai-27b-mmproj") == (
        "Ternary-Bonsai-27B-mmproj-Q8_0.gguf"
    )
    assert model_store.pick_repo_file(TERNARY_27B_FILES, "ternary", "bonsai-27b-dspark") == (
        "Ternary-Bonsai-27B-dspark-Q4_1.gguf"
    )


def test_pick_repo_file_missing_raises() -> None:
    with pytest.raises(BonsaiError):
        model_store.pick_repo_file(["README.md"], "ternary", "bonsai-27b")


def test_install_models_mocked(bonsai_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[str] = []
    payload = b"gguf-fake-weight"
    # The real pinned sizes are GiB; make the tripwire expect our tiny fake so the happy
    # path passes the (now hard-failing) size check without writing gigabytes.
    monkeypatch.setitem(
        model_store.EXPECTED_BYTES,
        "ternary",
        {logical: len(payload) for logical in model_store.HF_REPOS["ternary"]},
    )

    def fake_list_repo_files(repo: str) -> list[str]:
        stem = repo.split("/")[1].removesuffix("-gguf")  # e.g. Ternary-Bonsai-27B
        return [
            f"{stem}-Q2_0.gguf",
            f"{stem}-PQ2_0.gguf",
            f"{stem}-mmproj-Q8_0.gguf",
            f"{stem}-dspark-Q4_1.gguf",
        ]

    def fake_hf_hub_download(*, repo_id: str, filename: str, local_dir: str) -> str:
        downloads.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)

    reports = model_store.install_models(
        "ternary",
        progress=lambda _msg: None,
        list_repo_files=fake_list_repo_files,
        hf_hub_download=fake_hf_hub_download,
    )
    assert len(reports) == 5  # four sizes + mmproj (dspark skipped: opt-in)
    assert all("PQ2_0" not in r.filename for r in reports)
    assert not any(r.logical.endswith("-dspark") for r in reports)
    # sizes now match the (monkeypatched) tripwire — a passing install
    assert all(r.size_ok for r in reports)
    manifest = model_store.load_manifest("ternary")
    assert set(manifest) >= {"bonsai-27b", "bonsai-8b", "bonsai-4b", "bonsai-1.7b"}
    assert model_store.downloaded_families("bonsai-27b") == ["ternary"]
    # resolution works after install
    assert model_store.resolve_model_path("bonsai-27b", "ternary").is_file()
    # idempotent: second run skips everything
    reports2 = model_store.install_models(
        "ternary",
        progress=lambda _msg: None,
        list_repo_files=fake_list_repo_files,
        hf_hub_download=fake_hf_hub_download,
    )
    assert reports2 == []


def test_resolve_missing_model_gives_remediation(bonsai_home: Path) -> None:
    with pytest.raises(BonsaiError) as exc_info:
        model_store.resolve_model_path("bonsai-27b", "ternary")
    assert "suiban install models" in exc_info.value.message


def test_install_models_size_mismatch_hard_fails(bonsai_home: Path) -> None:
    """H7: a size that trips the ±2% tripwire is a HARD FAILURE, not a warning — a
    truncated or swapped weight must never be recorded in the manifest and loaded."""

    def fake_list_repo_files(repo: str) -> list[str]:
        stem = repo.split("/")[1].removesuffix("-gguf")
        return [f"{stem}-Q2_0.gguf", f"{stem}-mmproj-Q8_0.gguf"]

    written: list[Path] = []

    def fake_hf_hub_download(*, repo_id: str, filename: str, local_dir: str) -> str:
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny")  # nowhere near the GiB-scale expected bytes
        written.append(path)
        return str(path)

    with pytest.raises(BonsaiError) as exc_info:
        model_store.install_models(
            "ternary",
            progress=lambda _msg: None,
            list_repo_files=fake_list_repo_files,
            hf_hub_download=fake_hf_hub_download,
        )
    assert exc_info.value.code == "model_size_mismatch"
    # the mismatched file is deleted and never recorded
    assert written and not written[0].exists()
    assert model_store.load_manifest("ternary") == {}


def test_install_models_wraps_download_oserror(bonsai_home: Path) -> None:
    """H7/robustness: a disk-full / network OSError from hf_hub_download becomes a clean
    BonsaiError with a fix hint, not a raw traceback out of `suiban install models`."""

    def fake_list_repo_files(repo: str) -> list[str]:
        stem = repo.split("/")[1].removesuffix("-gguf")
        return [f"{stem}-Q2_0.gguf"]

    def boom(*, repo_id: str, filename: str, local_dir: str) -> str:
        raise OSError(28, "No space left on device")

    with pytest.raises(BonsaiError) as exc_info:
        model_store.install_models(
            "ternary",
            progress=lambda _msg: None,
            list_repo_files=fake_list_repo_files,
            hf_hub_download=boom,
        )
    assert exc_info.value.code == "model_download_failed"
    assert "disk" in exc_info.value.message.lower()


# -- binary download integrity (H7) -------------------------------------------
def _fake_release_tarball(path: Path, payload: bytes = b"\x7fELF fake llama-server") -> Path:
    """A minimal but real release tarball carrying a flattened llama-server."""
    import io
    import tarfile

    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("build/bin/llama-server")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return path


def test_asset_digest_manifest_covers_every_installable_asset() -> None:
    """Every asset our resolver can hand the installer must have a checked-in SHA-256, so
    no real install ever falls back to the un-verified (verify=false) path."""
    from suiban.installer.binaries import load_asset_digests

    digests = load_asset_digests()
    assert digests, "assets_sha256.json digests failed to load"
    combos = [
        ("linux", "cuda", "x64"),
        ("linux", "rocm", "x64"),
        ("linux", "vulkan", "x64"),
        ("linux", "vulkan", "arm64"),
        ("linux", "cpu", "x64"),
        ("linux", "cpu", "arm64"),
        ("macos", "metal", "arm64"),
        ("macos", "cpu", "x64"),
        ("windows", "cuda", "x64"),
        ("windows", "vulkan", "x64"),
        ("windows", "cpu", "x64"),
        ("windows", "cpu", "arm64"),
        ("windows", "rocm", "x64"),
    ]
    for os_name, backend, arch in combos:
        for name in asset_names(os_name, backend, arch).all_names:
            assert name in digests, f"no checked-in digest for installable asset {name}"
            assert len(digests[name]) == 64  # sha256 hex


def test_verify_asset_matching_digest_passes(tmp_path: Path) -> None:
    import hashlib

    from suiban.installer.binaries import verify_asset

    f = tmp_path / "asset.tar.gz"
    f.write_bytes(b"the real bytes")
    digest = hashlib.sha256(b"the real bytes").hexdigest()
    assert verify_asset(f, "asset.tar.gz", {"asset.tar.gz": digest}, progress=lambda _m: None)


def test_verify_asset_mismatch_hard_fails_and_deletes(tmp_path: Path) -> None:
    from suiban.installer.binaries import verify_asset

    f = tmp_path / "asset.tar.gz"
    f.write_bytes(b"tampered bytes")
    with pytest.raises(BonsaiError) as exc_info:
        verify_asset(f, "asset.tar.gz", {"asset.tar.gz": "0" * 64}, progress=lambda _m: None)
    assert exc_info.value.code == "asset_sha256_mismatch"
    assert not f.exists()  # a tampered download is removed, never left to be extracted


def test_verify_asset_no_digest_warns_returns_false(tmp_path: Path) -> None:
    from suiban.installer.binaries import verify_asset

    f = tmp_path / "asset.tar.gz"
    f.write_bytes(b"whatever")
    msgs: list[str] = []
    assert verify_asset(f, "asset.tar.gz", {}, progress=msgs.append) is False
    assert any("WITHOUT an integrity" in m for m in msgs)  # surfaced, never silent


def test_install_binaries_happy_path_verifies_and_extracts(
    bonsai_home: Path, tmp_path: Path
) -> None:
    from suiban.installer import binaries

    src = _fake_release_tarball(tmp_path / "src.tar.gz")
    digest = binaries._sha256_file(src)
    name = asset_names("linux", "cpu", "x64").primary

    def fake_fetch(url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return dest

    server = binaries.install_binaries(
        "cpu",
        os_name="linux",
        arch="x64",
        progress=lambda _m: None,
        fetch=fake_fetch,
        digests={name: digest},
    )
    assert server.is_file()
    assert server.name == "llama-server"
    assert server.stat().st_mode & 0o111  # marked executable
    assert (server.parent / "RELEASE").read_text().strip() == PRISM_RELEASE_TAG


def test_install_binaries_sha_mismatch_hard_fails(bonsai_home: Path, tmp_path: Path) -> None:
    from suiban.installer import binaries

    src = _fake_release_tarball(tmp_path / "src.tar.gz")
    name = asset_names("linux", "cpu", "x64").primary

    def fake_fetch(url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return dest

    with pytest.raises(BonsaiError) as exc_info:
        binaries.install_binaries(
            "cpu",
            os_name="linux",
            arch="x64",
            progress=lambda _m: None,
            fetch=fake_fetch,
            digests={name: "0" * 64},  # wrong digest → tamper/corruption
        )
    assert exc_info.value.code == "asset_sha256_mismatch"
    # no server binary was laid down from the rejected archive
    from suiban import paths

    assert not (paths.bin_dir("cpu") / "llama-server").exists()


# -- archive extraction -------------------------------------------------------
def test_extract_tarball_recreates_soname_symlinks(tmp_path: Path) -> None:
    """The linux/macos release tarballs ship versioned libs plus soname symlinks
    (libfoo.so.0 -> libfoo.so.0.13.1); llama-server fails to start without them.
    Regression test for the flattening extractor dropping tar symlink members."""
    import io
    import tarfile

    from suiban.installer.binaries import extract_binaries

    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        payload = b"\x7fELF fake lib"
        info = tarfile.TarInfo("build/bin/libggml-base.so.0.13.1")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("build/bin/libggml-base.so.0")
        link.type = tarfile.SYMTYPE
        link.linkname = "libggml-base.so.0.13.1"
        tf.addfile(link)
        server = tarfile.TarInfo("build/bin/llama-server")
        server.size = len(payload)
        tf.addfile(server, io.BytesIO(payload))

    bin_dir = tmp_path / "bin"
    extracted = extract_binaries(archive, bin_dir)
    names = {p.name for p in extracted}
    assert {"libggml-base.so.0.13.1", "libggml-base.so.0", "llama-server"} <= names
    symlink = bin_dir / "libggml-base.so.0"
    assert symlink.is_symlink()
    assert symlink.resolve().name == "libggml-base.so.0.13.1"
    assert (bin_dir / "llama-server").stat().st_mode & 0o111  # executable
