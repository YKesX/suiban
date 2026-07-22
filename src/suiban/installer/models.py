"""Bonsai model weight downloads (Hugging Face) and local resolution.

Layout: ~/.bonsai/models/<family>/ with a manifest.json mapping logical names to the
downloaded GGUF filenames. Both families coexist side by side; the family toggle never
deletes anything.

Integrity: huggingface_hub verifies each file's own etag/sha against the HF repo on
download — that is the transport integrity source. On top of it, our expected byte sizes
(the verified recon table, ±2%) are a tripwire: a size mismatch is a HARD FAILURE
(BonsaiError), never a silent or warning-only pass, because a truncated or swapped weight
is not safe to load.

Repo slugs and filenames verified live against the HF tree API (2026-07-21): each family
lives in its OWN repo line — 1-bit in prism-ml/Bonsai-<size>-gguf (file *-Q1_0.gguf),
ternary in prism-ml/Ternary-Bonsai-<size>-gguf (file *-Q2_0.gguf; the sibling PQ2_0 is a
future fork format and Q2_0_g64 is the mainline format — both must never be selected).
The 27B repos additionally ship mmproj (vision projector) and the DSpark drafter.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from suiban import paths
from suiban.errors import BonsaiError

GIB = 1024**3

FAMILIES = ("ternary", "1bit")

# Logical entries per family, with exact byte sizes from the HF tree API. mmproj (Q8_0)
# and the DSpark drafter ride with the 27B; each family dir is self-contained.
EXPECTED_BYTES: dict[str, dict[str, int]] = {
    "ternary": {
        "bonsai-27b": 7_165_121_600,
        "bonsai-8b": 2_182_184_672,
        "bonsai-4b": 1_074_969_344,
        "bonsai-1.7b": 463_290_464,
        "bonsai-27b-mmproj": 629_246_880,
        "bonsai-27b-dspark": 1_946_393_568,
    },
    "1bit": {
        "bonsai-27b": 3_803_452_480,
        "bonsai-8b": 1_158_654_496,
        "bonsai-4b": 572_270_624,
        "bonsai-1.7b": 248_302_272,
        "bonsai-27b-mmproj": 629_246_880,
        "bonsai-27b-dspark": 1_787_468_768,
    },
}

SIZE_TOLERANCE = 0.02  # ±2%

HF_REPOS: dict[str, dict[str, str]] = {
    "ternary": {
        "bonsai-27b": "prism-ml/Ternary-Bonsai-27B-gguf",
        "bonsai-8b": "prism-ml/Ternary-Bonsai-8B-gguf",
        "bonsai-4b": "prism-ml/Ternary-Bonsai-4B-gguf",
        "bonsai-1.7b": "prism-ml/Ternary-Bonsai-1.7B-gguf",
        "bonsai-27b-mmproj": "prism-ml/Ternary-Bonsai-27B-gguf",
        "bonsai-27b-dspark": "prism-ml/Ternary-Bonsai-27B-gguf",
    },
    "1bit": {
        "bonsai-27b": "prism-ml/Bonsai-27B-gguf",
        "bonsai-8b": "prism-ml/Bonsai-8B-gguf",
        "bonsai-4b": "prism-ml/Bonsai-4B-gguf",
        "bonsai-1.7b": "prism-ml/Bonsai-1.7B-gguf",
        "bonsai-27b-mmproj": "prism-ml/Bonsai-27B-gguf",
        "bonsai-27b-dspark": "prism-ml/Bonsai-27B-gguf",
    },
}

# Exact GGUF filename suffix per family variant. Suffix matching (not fragment) is
# load-bearing: the ternary repos also ship *-PQ2_0.gguf (future fork format, unsupported)
# and *-Q2_0_g64.gguf / *-Q2_g64.gguf (mainline format, incompatible with the fork).
FAMILY_QUANT_SUFFIX = {"ternary": "-q2_0.gguf", "1bit": "-q1_0.gguf"}


def manifest_path(family: str) -> Path:
    return paths.models_dir(family) / "manifest.json"


def load_manifest(family: str) -> dict[str, dict]:
    mf = manifest_path(family)
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def save_manifest(family: str, manifest: dict[str, dict]) -> None:
    mf = manifest_path(family)
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def downloaded_families(model: str) -> list[str]:
    """Which families have this model on disk (for /v1/models bonsai metadata)."""
    return [family for family in FAMILIES if model in load_manifest(family)]


def _resolve(logical: str, family: str) -> Path:
    entry = load_manifest(family).get(logical)
    if entry is None:
        raise BonsaiError(
            500,
            f"{logical} ({family}) is not downloaded. Run: suiban install models --family {family}",
            code="model_missing",
        )
    path = paths.models_dir(family) / entry["file"]
    if not path.is_file():
        raise BonsaiError(
            500,
            f"{logical} ({family}) manifest entry points at a missing file: {path}. "
            f"Re-run: suiban install models --family {family}",
            code="model_missing",
        )
    return path


def resolve_model_path(model: str, family: str) -> Path:
    return _resolve(model, family)


def resolve_mmproj_path(family: str) -> Path:
    return _resolve("bonsai-27b-mmproj", family)


def resolve_dspark_path(family: str) -> Path:
    # DSpark is opt-in (default off; ~1.8 GiB VRAM): install_models skips it unless
    # include_dspark=True (surfaced as `suiban install models --dspark`).
    return _resolve("bonsai-27b-dspark", family)


@dataclass
class DownloadReport:
    logical: str
    family: str
    filename: str
    bytes_on_disk: int
    expected_bytes: int

    @property
    def size_ok(self) -> bool:
        if self.expected_bytes <= 0:
            return True
        drift = abs(self.bytes_on_disk - self.expected_bytes) / self.expected_bytes
        return drift <= SIZE_TOLERANCE


def pick_repo_file(files: list[str], family: str, logical: str) -> str:
    """Choose the GGUF for a logical entry from a repo file listing."""
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if logical.endswith("-mmproj"):
        # Q8_0 projector preferred (629 MB) over BF16 (931 MB); both work.
        candidates = [f for f in ggufs if f.lower().endswith("mmproj-q8_0.gguf")] or [
            f for f in ggufs if "mmproj" in f.lower()
        ]
    elif logical.endswith("-dspark"):
        candidates = [f for f in ggufs if f.lower().endswith("dspark-q4_1.gguf")]
    else:
        suffix = FAMILY_QUANT_SUFFIX[family]
        candidates = [
            f
            for f in ggufs
            if f.lower().endswith(suffix)
            and "mmproj" not in f.lower()
            and "dspark" not in f.lower()
        ]
    if not candidates:
        raise BonsaiError(
            500,
            f"no GGUF matching {logical} ({family}) in repo listing: {files}",
            code="model_file_not_found",
        )
    return sorted(candidates)[0]


def install_models(
    family: str,
    *,
    include_dspark: bool = False,
    progress: Callable[[str], None] = print,
    list_repo_files: Callable[[str], list[str]] | None = None,
    hf_hub_download: Callable[..., str] | None = None,
) -> list[DownloadReport]:
    """Download one family's weights. Network functions are injectable so tests never
    touch the network. The DSpark drafter (~1.8 GiB, opt-in feature) is skipped unless
    include_dspark is set."""
    if family not in FAMILIES:
        raise BonsaiError(400, f"unknown family {family!r}", code="family_unknown")
    if list_repo_files is None or hf_hub_download is None:
        import huggingface_hub

        list_repo_files = list_repo_files or huggingface_hub.list_repo_files
        hf_hub_download = hf_hub_download or huggingface_hub.hf_hub_download

    dest = paths.models_dir(family)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(family)
    reports: list[DownloadReport] = []
    for logical, repo in HF_REPOS[family].items():
        if logical.endswith("-dspark") and not include_dspark:
            continue
        expected = EXPECTED_BYTES[family].get(logical, 0)
        if logical in manifest and (dest / manifest[logical]["file"]).is_file():
            progress(f"  {logical} ({family}): already present, skipping")
            continue
        progress(f"  {logical} ({family}): resolving {repo} ...")
        files = list_repo_files(repo)
        filename = pick_repo_file(files, family, logical)
        progress(f"  {logical}: downloading {filename} (~{expected / GIB:.2f} GiB)")
        try:
            local = Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest)))
        except OSError as exc:
            # disk-full / permission / network I/O — a raw traceback here is useless to a
            # user mid-install; give them the one fact that fixes it.
            raise BonsaiError(
                500,
                f"download of {logical} ({family}) from {repo} failed: {exc}. Check free "
                f"disk space and write permission for {dest}, and your network, then "
                f"re-run: suiban install models --family {family}",
                code="model_download_failed",
            ) from exc
        report = DownloadReport(
            logical=logical,
            family=family,
            filename=local.name,
            bytes_on_disk=local.stat().st_size,
            expected_bytes=expected,
        )
        if not report.size_ok:
            # huggingface_hub already verified the file's own etag/sha over the wire; a
            # size that still deviates >2% from the pinned recon table means we fetched a
            # different (truncated, swapped, or re-quantized) build than we shipped
            # against — not safe to load. Hard-fail; do not record it in the manifest.
            local.unlink(missing_ok=True)
            raise BonsaiError(
                500,
                f"{logical} ({family}) is {report.bytes_on_disk} bytes but the pinned "
                f"build is {expected} (>2% off): the download is truncated, corrupt, or a "
                f"different build. Deleted it — re-run: suiban install models "
                f"--family {family}",
                code="model_size_mismatch",
            )
        manifest[logical] = {"file": local.name, "bytes": report.bytes_on_disk}
        save_manifest(family, manifest)
        reports.append(report)
    return reports
