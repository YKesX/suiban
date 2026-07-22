#!/usr/bin/env bash
# Live RSS + VRAM soak for the suiban stack (audit 2026-07-22, workstream P).
#
# NOT run in CI and NOT part of the test suite — it needs a real `suiban serve` with a
# loaded model. It drives 200 chat turns on ONE session and samples resident memory
# (VmRSS of the server process tree) + GPU VRAM (nvidia-smi), then runs a
# repeated-Ultra spawn check, looking for growth that never plateaus (a leak) vs a
# curve that flattens (steady state). Prints two CSVs; eyeball them or plot them.
#
# Usage:
#   tests/perf/soak_live.sh                 # autodetect pid + 127.0.0.1:8686
#   SUIBAN_PID=12345 HOST=127.0.0.1 PORT=8686 TURNS=200 tests/perf/soak_live.sh
#   AUTH_TOKEN=... tests/perf/soak_live.sh  # only if bound to a non-loopback host
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8686}"
TURNS="${TURNS:-200}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
ULTRA_ROUNDS="${ULTRA_ROUNDS:-8}"
BASE="http://${HOST}:${PORT}"
SESSION="soak-$(date +%s)"
AUTH_HEADER=()
[[ -n "${AUTH_TOKEN:-}" ]] && AUTH_HEADER=(-H "Authorization: Bearer ${AUTH_TOKEN}")

# -- locate the server process tree ------------------------------------------
detect_pid() {
  if [[ -n "${SUIBAN_PID:-}" ]]; then echo "$SUIBAN_PID"; return; fi
  # The uvicorn/serve master. Fall back to any process whose cmdline names suiban serve.
  pgrep -f 'suiban serve' | head -1 || pgrep -f 'suiban' | head -1
}
PID="$(detect_pid || true)"
if [[ -z "${PID}" ]]; then
  echo "could not find the suiban server pid; pass SUIBAN_PID=<pid>" >&2; exit 1
fi

# All descendant pids of $1 (the server + its llama-server / worker subprocesses).
descendants() {
  local root="$1" kids
  echo "$root"
  kids="$(pgrep -P "$root" 2>/dev/null || true)"
  for k in $kids; do descendants "$k"; done
}

# Summed VmRSS (KiB) across the whole server tree.
tree_rss_kb() {
  local total=0 kb
  for p in $(descendants "$PID" | sort -u); do
    kb="$(awk '/^VmRSS:/{print $2}' "/proc/$p/status" 2>/dev/null || echo 0)"
    total=$(( total + ${kb:-0} ))
  done
  echo "$total"
}

# Count of llama-server subprocesses under the tree (orphan/leak detector).
llama_count() {
  local n=0
  for p in $(descendants "$PID" | sort -u); do
    grep -qa 'llama-server' "/proc/$p/cmdline" 2>/dev/null && n=$(( n + 1 )) || true
  done
  echo "$n"
}

# Used VRAM (MiB), summed across GPUs. 0 if no nvidia-smi.
vram_used_mb() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s+=$1} END{print s+0}'
}

chat_turn() {  # $1 = mode, $2 = user text
  curl -sS -m 300 "${AUTH_HEADER[@]}" -H 'Content-Type: application/json' \
    -X POST "${BASE}/v1/chat/completions" \
    -d "{\"model\":\"bonsai-auto\",\"mode\":\"$1\",\"effort\":\"low\",\
\"session_id\":\"${SESSION}\",\"stream\":false,\
\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$2" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')}]}" \
    >/dev/null
}

# -- wait for health ----------------------------------------------------------
until curl -sS -m 5 "${BASE}/v1/system/health" >/dev/null 2>&1; do sleep 1; done
echo "# server pid=${PID}, session=${SESSION}, base=${BASE}"

# -- 200-turn single-session RSS soak ----------------------------------------
echo "# turn,rss_tree_kb,vram_used_mb,llama_procs"
printf "0,%s,%s,%s\n" "$(tree_rss_kb)" "$(vram_used_mb)" "$(llama_count)"
for i in $(seq 1 "$TURNS"); do
  chat_turn "chat" "Soak turn ${i}. Briefly restate the running total of turns so far and add one fresh fact: fact-${i} is the number ${i} squared is $(( i * i ))."
  if (( i % SAMPLE_EVERY == 0 )); then
    printf "%s,%s,%s,%s\n" "$i" "$(tree_rss_kb)" "$(vram_used_mb)" "$(llama_count)"
  fi
done

# -- repeated-Ultra VRAM / process-leak check --------------------------------
# Each Ultra run may fan out worker slots; after it settles, VRAM + llama-server
# count must return to (roughly) the pre-Ultra baseline — no monotonic creep, no
# orphaned llama-server processes.
echo "# ultra_round,rss_tree_kb,vram_used_mb,llama_procs"
printf "baseline,%s,%s,%s\n" "$(tree_rss_kb)" "$(vram_used_mb)" "$(llama_count)"
for r in $(seq 1 "$ULTRA_ROUNDS"); do
  chat_turn "ultra" "Ultra round ${r}: decompose 'summarize the number ${r}' into two trivial subtasks and answer them."
  sleep 3   # let workers tear down (loadout is fixed per run; workers are transient)
  printf "%s,%s,%s,%s\n" "$r" "$(tree_rss_kb)" "$(vram_used_mb)" "$(llama_count)"
done

echo "# done. Leak signal = rss_tree_kb / vram_used_mb rising monotonically without a"
echo "# plateau across the 200 turns, or llama_procs / vram not returning to baseline"
echo "# after the Ultra rounds. A curve that flattens is steady state (healthy)."
