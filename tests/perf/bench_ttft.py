"""TTFT hot-path micro-benchmark (audit 2026-07-22, workstream P).

NOT collected by pytest (no ``test_`` prefix) and NOT part of CI — it is a
measurement harness. It profiles the pre-first-token work the chat router does in
``routers/chat.py::_prepare_loop`` that lands in the memory package:

* ``SkillStore.list()``  — cached (new) vs uncached (old, re-read every call)
* ``enforce_context_budget`` — single-pass (new) vs O(M^2) reference (old)

Both "old" behaviours are reimplemented here verbatim so before/after run against the
same data on the same machine. Wall-clock via ``perf_counter``; a cumulative view via
``cProfile``.

Run:  ``python tests/perf/bench_ttft.py``
"""

from __future__ import annotations

import cProfile
import json
import pstats
import tempfile
import time
from io import StringIO
from pathlib import Path

from suiban.memory import injection as inj
from suiban.memory.compression import estimate_tokens, keep_recent_messages
from suiban.memory.injection import (
    HARD_LIMIT_FRACTION,
    TRIM_ORDER,
    _injected_header,
    _split_injected,
)
from suiban.memory.skills import Skill, SkillStore

N_SKILLS = 50
N_CALLS = 100
N_MESSAGES = 400
SLOT_CTX = 8192


# -- reference "old" implementations (pre-audit, for before/after parity) ------
def old_list(store: SkillStore) -> list[Skill]:
    """The pre-audit list(): glob + re-read + re-parse every SKILL.md/meta.json."""
    store.ensure()
    out: list[Skill] = []
    for skill_file in sorted(store._dir.glob("*/SKILL.md")):  # noqa: SLF001
        skill = store.get(skill_file.parent.name)
        if skill is not None:
            out.append(skill)
    return out


def old_enforce(messages: list[dict], slot_ctx: int):
    """The pre-audit O(M^2) enforce_context_budget: re-estimate the whole list per
    popped block and per deleted message. Kept verbatim for equivalence testing."""
    limit = int(slot_ctx * HARD_LIMIT_FRACTION)
    tokens_before = estimate_tokens(messages)
    if slot_ctx <= 0 or tokens_before <= limit:
        return messages, None
    out = [dict(m) for m in messages]
    dropped_blocks = dropped_injections = dropped_messages = 0
    for header in TRIM_ORDER:
        index = next((i for i, m in enumerate(out) if _injected_header(m) == header), None)
        if index is None:
            continue
        head, blocks = _split_injected(out[index]["content"])
        while blocks and estimate_tokens(out) > limit:
            blocks.pop()
            dropped_blocks += 1
            if blocks:
                out[index]["content"] = head + "\n" + "\n".join(blocks)
            else:
                del out[index]
                dropped_injections += 1
                break
        if estimate_tokens(out) <= limit:
            break
    keep = keep_recent_messages(slot_ctx)
    while estimate_tokens(out) > limit:
        body_indexes = [i for i, m in enumerate(out) if m.get("role") != "system"]
        if len(body_indexes) <= keep:
            break
        del out[body_indexes[0]]
        dropped_messages += 1
    tokens_after = estimate_tokens(out)
    report = inj.TrimReport(
        dropped_blocks=dropped_blocks,
        dropped_injections=dropped_injections,
        dropped_messages=dropped_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        limit_tokens=limit,
        still_over=tokens_after > limit,
    )
    return out, report


# -- fixtures ------------------------------------------------------------------
def make_skills(skills_dir: Path, n: int) -> SkillStore:
    skills_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        name = f"skill-{i:03d}"
        d = skills_dir / name
        d.mkdir(parents=True, exist_ok=True)
        body = "\n".join(
            f"{j}. step {j} for {name} with keyword deploy release build" for j in range(20)
        )
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: procedure {i} covering deploy and release\n---\n\n"
            f"# {name}\n{body}\n",
            encoding="utf-8",
        )
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "source": "learned",
                    "updated_at": "2026-07-22T00:00:00Z",
                    "verified": i % 2 == 0,
                }
            ),
            encoding="utf-8",
        )
    return SkillStore(skills_dir)


def make_conversation(n: int) -> list[dict]:
    """One system head + one memory injection + one skill injection + n body turns."""
    mem_blocks = [
        f"<<<memory m{i} (archive) note>>>\n{'recall ' * 40}\n<<<end memory>>>" for i in range(6)
    ]
    skill_blocks = [
        f"<<<skill skill-{i:03d} v1>>>\n{'procedure ' * 40}\n<<<end skill>>>" for i in range(2)
    ]
    messages = [
        {"role": "system", "content": "mode prompt " * 20},
        {"role": "system", "content": inj.MEMORY_CONTEXT_HEADER + "\n" + "\n".join(mem_blocks)},
        {"role": "system", "content": inj.SKILL_CONTEXT_HEADER + "\n" + "\n".join(skill_blocks)},
    ]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        body = "lorem ipsum dolor sit amet " * 30
        messages.append({"role": role, "content": f"turn {i}: " + body})
    return messages


# -- timing --------------------------------------------------------------------
def time_calls(fn, calls: int) -> float:
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    return (time.perf_counter() - start) / calls * 1e6  # microseconds/call


def bench_skills() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = make_skills(Path(tmp) / "skills", N_SKILLS)
        # Old: uncached (re-read every call). New: warm the cache, then measure hits.
        old_us = time_calls(lambda: old_list(store), N_CALLS)
        store.list()  # cold populate
        new_us = time_calls(store.list, N_CALLS)
        # Correctness sanity: same skills, same order.
        assert [s.name for s in old_list(store)] == [s.name for s in store.list()]
        print(f"\n== SkillStore.list()  ({N_SKILLS} skills, mean over {N_CALLS} calls) ==")
        print(f"  old (uncached, re-read+parse): {old_us:9.1f} us/call")
        print(f"  new (mtime/stat-cached hit):   {new_us:9.1f} us/call")
        print(f"  speedup: {old_us / new_us:6.1f}x  (-{(1 - new_us / old_us) * 100:.1f}%)")


def bench_budget() -> None:
    messages = make_conversation(N_MESSAGES)
    old_us = time_calls(lambda: old_enforce(messages, SLOT_CTX), N_CALLS)
    new_us = time_calls(lambda: inj.enforce_context_budget(messages, SLOT_CTX), N_CALLS)
    old_out, old_rep = old_enforce(messages, SLOT_CTX)
    new_out, new_rep = inj.enforce_context_budget(messages, SLOT_CTX)
    assert [m["content"] for m in old_out] == [m["content"] for m in new_out]
    assert old_rep == new_rep, (old_rep, new_rep)
    print(f"\n== enforce_context_budget  ({N_MESSAGES} msgs, mean over {N_CALLS} calls) ==")
    print(f"  old (O(M^2) re-estimate):     {old_us:9.1f} us/call")
    print(f"  new (single-pass running sum):{new_us:9.1f} us/call")
    print(f"  speedup: {old_us / new_us:6.1f}x  (-{(1 - new_us / old_us) * 100:.1f}%)")
    print(
        f"  parity: kept {len(new_out)} msgs, dropped {new_rep.dropped_messages} tail / "
        f"{new_rep.dropped_blocks} blocks (identical old vs new)"
    )


def profile_hotpath() -> None:
    """cProfile the injection sequence _prepare_loop runs, old vs new."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_skills(Path(tmp) / "skills", N_SKILLS)
        store.list()  # warm new cache
        messages = make_conversation(N_MESSAGES)
        query = "please deploy the release build"

        def run_new() -> None:
            for _ in range(N_CALLS):
                skills = store.list()
                inj.build_skill_context(skills, query, SLOT_CTX)
                inj.enforce_context_budget(messages, SLOT_CTX)

        def run_old() -> None:
            for _ in range(N_CALLS):
                skills = old_list(store)
                inj.build_skill_context(skills, query, SLOT_CTX)
                old_enforce(messages, SLOT_CTX)

        for label, fn in (("OLD", run_old), ("NEW", run_new)):
            pr = cProfile.Profile()
            pr.enable()
            fn()
            pr.disable()
            buf = StringIO()
            st = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
            st.print_stats("memory/(injection|skills|compression)")
            print(f"\n== cProfile {label} ({N_CALLS} hot-path iterations) ==")
            total = pr.getstats()  # noqa: F841
            for line in buf.getvalue().splitlines():
                if any(
                    k in line
                    for k in (
                        "list",
                        "enforce_context_budget",
                        "build_skill_context",
                        "estimate_tokens",
                        "get",
                        "_signature",
                        "ncalls",
                    )
                ):
                    print("  " + line.strip())


if __name__ == "__main__":
    bench_skills()
    bench_budget()
    profile_hotpath()
