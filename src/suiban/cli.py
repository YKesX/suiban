"""suiban CLI: serve · doctor · install (models/binaries/turboquant) · bench."""

from __future__ import annotations

import errno
import socket
import subprocess
import sys
from pathlib import Path

import typer

from suiban import __version__
from suiban.config import ConfigManager
from suiban.errors import BonsaiError
from suiban.installer.backend import detect_backend
from suiban.llama import binary as binary_mod
from suiban.sched.telemetry import pick_provider

app = typer.Typer(
    name="suiban",
    help="bonsai inference & orchestration core",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
install_app = typer.Typer(help="Download/build runtime pieces into ~/.bonsai", no_args_is_help=True)
bench_app = typer.Typer(help="Benchmarks (real hardware)", no_args_is_help=True)
skills_app = typer.Typer(help="Manage agentskills.io skills", no_args_is_help=True)
app.add_typer(install_app, name="install")
app.add_typer(bench_app, name="bench")
app.add_typer(skills_app, name="skills")


@app.command()
def version() -> None:
    """Print the suiban version."""
    typer.echo(f"suiban {__version__}")


def probe_bind(host: str, port: int) -> str | None:
    """Pre-flight bind probe (audit 2026-07-22): try to bind (host, port) and release
    it immediately. Returns None when the port is free, or a remediation message when
    it is already in use — so `serve` prints a suiban fix instead of letting uvicorn
    raise a bare `[Errno 98] address already in use` traceback. Any other bind error
    (unresolvable host, etc.) returns None: uvicorn's own error is more accurate than
    a guess here."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return None
    family, socktype, proto, _canon, sockaddr = infos[0]
    with socket.socket(family, socktype, proto) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(sockaddr)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return (
                    f"port {port} is already in use — another suiban? change "
                    "server.port in ~/.bonsai/config.toml"
                )
            return None
    return None


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Override server.host from config"),
    port: int | None = typer.Option(None, help="Override server.port from config"),
) -> None:
    """Start the API server (http://127.0.0.1:8686 by default)."""
    import uvicorn

    from suiban.app import create_app

    try:
        settings = ConfigManager().load()
    except BonsaiError as exc:
        # A malformed ~/.bonsai/config.toml surfaces as a clean message + remedy, not a
        # raw traceback (config.py turns the parse error into this BonsaiError).
        typer.echo(f"error: {exc.message}", err=True)
        raise typer.Exit(1) from exc
    bind_host = host or settings.server.host
    bind_port = port or settings.server.port
    problem = probe_bind(bind_host, bind_port)
    if problem is not None:
        typer.echo(f"error: {problem}", err=True)
        raise typer.Exit(1)
    uvicorn.run(
        create_app(bind_host=bind_host),
        host=bind_host,
        port=bind_port,
        log_level="info",
    )


@app.command()
def doctor() -> None:
    """Check binary / models / config / telemetry; print exact remediation commands."""
    ok = True

    def check(label: str, passed: bool, detail: str, remedy: str | None = None) -> None:
        nonlocal ok
        mark = "ok " if passed else "FAIL"
        typer.echo(f"[{mark}] {label}: {detail}")
        if not passed:
            ok = False
            if remedy:
                typer.echo(f"       fix: {remedy}")

    # config
    try:
        config = ConfigManager()
        settings = config.load()
        check("config", True, f"{config.config_file} (family={settings.quant_family})")
    except BonsaiError as exc:
        check("config", False, exc.message, "fix or delete the file; suiban recreates defaults")
        settings = None

    # backend + binary
    backend = detect_backend()
    typer.echo(f"[info] compute backend: {backend}")
    try:
        path = binary_mod.resolve_server_binary(backend)
        pin_ok = binary_mod.release_matches_pin(backend)
        detail = f"{path} (release {binary_mod.installed_release(backend) or 'unknown'})"
        check(
            "binary",
            pin_ok,
            detail,
            None
            if pin_ok
            else "suiban install binaries  # re-pin to " + binary_mod.PRISM_RELEASE_TAG,
        )
        tq = binary_mod.turboquant_installed(backend)
        typer.echo(
            f"[info] turboquant kernels: {'installed' if tq else 'not installed'}"
            + ("" if tq else "  (KV falls back to q8_0/q8_0; suiban install turboquant)")
        )
    except BonsaiError as exc:
        check("binary", False, exc.message, "suiban install binaries")

    # models
    from suiban.installer import models as model_store

    family = settings.quant_family if settings else "ternary"
    manifest = model_store.load_manifest(family)
    sizes = ("bonsai-27b", "bonsai-8b", "bonsai-4b", "bonsai-1.7b")
    missing = [m for m in sizes if m not in manifest]
    check(
        f"models ({family})",
        not missing,
        "all four sizes downloaded" if not missing else f"missing: {', '.join(missing)}",
        f"suiban install models --family {family}" if missing else None,
    )

    # telemetry
    provider = pick_provider()
    snapshot = provider.snapshot()
    if snapshot.gpus:
        gpus = ", ".join(f"{g.name} ({g.vram_total_mb} MiB)" for g in snapshot.gpus)
        typer.echo(f"[ok ] telemetry: {snapshot.source} — {gpus}")
    else:
        typer.echo(
            f"[ok ] telemetry: {snapshot.source} — no GPU (CPU-only loadout, "
            f"{snapshot.ram_total_mb} MiB RAM)"
        )

    raise typer.Exit(0 if ok else 1)


@install_app.command("models")
def install_models_cmd(
    family: str = typer.Option("ternary", help="ternary | 1bit | both"),
    dspark: bool = typer.Option(
        False, "--dspark", help="also fetch the 27B DSpark speculative drafter (~1.8 GiB, opt-in)"
    ),
) -> None:
    """Download Bonsai GGUF weights from Hugging Face into ~/.bonsai/models/<family>/."""
    from suiban.installer import models as model_store

    families = ("ternary", "1bit") if family == "both" else (family,)
    for fam in families:
        typer.echo(f"installing {fam} family:")
        try:
            reports = model_store.install_models(fam, include_dspark=dspark, progress=typer.echo)
        except BonsaiError as exc:
            typer.echo(f"error: {exc.message}", err=True)
            raise typer.Exit(1) from exc
        for report in reports:
            flag = "" if report.size_ok else "  [SIZE MISMATCH]"
            typer.echo(f"  done: {report.filename} ({report.bytes_on_disk} bytes){flag}")


@install_app.command("binaries")
def install_binaries_cmd(
    backend: str | None = typer.Option(None, help="cuda|rocm|metal|vulkan|cpu (auto-detected)"),
) -> None:
    """Download the pinned PrismML fork prebuilts into ~/.bonsai/bin/<backend>/."""
    from suiban.installer.binaries import install_binaries

    resolved = backend or detect_backend()
    try:
        server = install_binaries(resolved, progress=typer.echo)
    except BonsaiError as exc:
        typer.echo(f"error: {exc.message}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"llama-server ready: {server}")
    typer.echo("note: prebuilts have NO TurboQuant kernels — KV runs q8_0/q8_0 until")
    typer.echo("      you run: suiban install turboquant  (CUDA/CPU backends only)")


@install_app.command("turboquant")
def install_turboquant_cmd(
    backend: str | None = typer.Option(None, help="cuda|cpu (others unsupported in v1)"),
    cuda_host_compiler: str | None = typer.Option(
        None,
        "--cuda-host-compiler",
        help="host C++ compiler for nvcc when the system compiler is too new "
        "(e.g. GCC 16 with CUDA 13.x: pass a GCC <= 15 path)",
    ),
) -> None:
    """Clone+patch+build the fork with the vendored TurboQuant patchset, then swap the
    binary. Delegates to vendor/apply_patches.py."""
    resolved = backend or detect_backend()
    if resolved not in ("cuda", "cpu"):
        typer.echo(
            f"warning: TurboQuant kernels are CUDA/CPU only in v1 — backend '{resolved}' "
            "is unsupported (ROCm/Vulkan/Metal are out of scope; KV stays q8_0/q8_0). Skipping.",
            err=True,
        )
        raise typer.Exit(1)
    script = _vendor_script("apply_patches.py")
    if script is None:
        typer.echo(
            "error: vendor/apply_patches.py not found — run from a source checkout "
            "(the TurboQuant build is a source-tree operation).",
            err=True,
        )
        raise typer.Exit(1)
    cmd = [sys.executable, str(script), "--clone", f"--backend={resolved}", "--install"]
    if cuda_host_compiler:
        cmd += ["--cuda-host-compiler", cuda_host_compiler]
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


@skills_app.command("import")
def skills_import_cmd(
    source: str = typer.Argument(
        ...,
        help="openclaw | hermes | a path to a skills directory (scanned recursively "
        "for <name>/SKILL.md)",
    ),
) -> None:
    """Import agentskills.io SKILL.md skills into ~/.bonsai/skills/. suiban skills use the
    same SKILL.md format, so skills are portable both ways."""
    from suiban import paths
    from suiban.memory.skill_import import SkillImportError, import_skills
    from suiban.memory.skills import SkillStore

    store = SkillStore(paths.skills_dir())
    store.ensure()
    # A bare keyword is a known ecosystem; anything else is an explicit directory path.
    if source in ("openclaw", "hermes"):
        src, path = source, None
    else:
        src, path = "path", source
    try:
        result = import_skills(store, src, path)
    except SkillImportError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for name in result.imported:
        typer.echo(f"imported: {name}")
    for skipped in result.skipped:
        typer.echo(f"skipped:  {skipped['name']} ({skipped['reason']})", err=True)
    typer.echo(f"done: {len(result.imported)} imported, {len(result.skipped)} skipped")


@bench_app.command("kv")
def bench_kv(
    haystack_tokens: int = typer.Option(
        16384,
        help="Largest needle haystack to attempt (the ladder runs 4k/8k/16k up to this; "
        "sizes the slot ctx cannot fit are reported as not run).",
    ),
) -> None:
    """KV V-cache benchmark on YOUR hardware: llama-perplexity + needle retrieval at
    4k/8k/16k + multi-turn replay for V in {tq4_0, tq3_0, q4_0, q8_0} at K=q8_0.
    Report: ~/.bonsai/reports/."""
    import asyncio

    from suiban import paths
    from suiban.bench import kv as kv_bench
    from suiban.llama.backend import mock_enabled

    settings = ConfigManager().load()
    backend = detect_backend()
    mock = mock_enabled()
    family = settings.quant_family

    if not mock:
        # Pre-flight: honest errors with remedies beat four identical failures.
        from suiban.installer import models as model_store

        try:
            binary_mod.resolve_server_binary(backend)
        except BonsaiError as exc:
            typer.echo(f"error: {exc.message}", err=True)
            raise typer.Exit(1) from exc
        # Bench the family that is actually on disk: same degradation the planner
        # applies (e.g. the 8 GB tier runs the 27B as 1-bit even when ternary is
        # configured), and say so instead of failing on the configured family.
        try:
            model_store.resolve_model_path("bonsai-27b", family)
        except BonsaiError:
            fallback = next(
                (f for f in model_store.downloaded_families("bonsai-27b") if f != family), None
            )
            if fallback is None:
                typer.echo(
                    f"error: bonsai-27b is not downloaded in any family. "
                    f"Run: suiban install models --family {family}",
                    err=True,
                )
                raise typer.Exit(1) from None
            typer.echo(
                f"note: configured family '{family}' is not on disk; benchmarking the "
                f"downloaded '{fallback}' 27B instead (matches the planner's degradation)."
            )
            family = fallback

    sizes = tuple(s for s in kv_bench.NEEDLE_SIZES if s <= haystack_tokens)
    if not sizes:
        sizes = (min(kv_bench.NEEDLE_SIZES),)
    provider = kv_bench.real_slot_provider(
        compute_backend=backend, family=family, use_mock=mock, haystack_sizes=sizes
    )
    results = asyncio.run(kv_bench.run_kv_bench(provider, haystack_sizes=sizes))

    snapshot = pick_provider().snapshot()
    if snapshot.gpus:
        hardware = ", ".join(f"{g.name} ({g.vram_total_mb} MiB)" for g in snapshot.gpus)
    else:
        hardware = f"CPU only ({snapshot.ram_total_mb} MiB RAM)"
    release = binary_mod.installed_release(backend) or "unknown release"
    machine_line = f"{hardware}; backend {backend}; binary {release}; model bonsai-27b ({family})"

    report = kv_bench.render_report(
        results, machine_line=machine_line, mock=mock, haystack_sizes=sizes
    )
    reports_dir = paths.reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = kv_bench.report_path(reports_dir)
    out_path.write_text(report, encoding="utf-8")
    typer.echo(report)
    typer.echo(f"report written: {out_path}")


def _vendor_script(name: str) -> Path | None:
    """Locate vendor/<name> in a source checkout (repo root = parents of src/)."""
    for base in Path(__file__).resolve().parents:
        candidate = base / "vendor" / name
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
