#!/usr/bin/env bash
# Personal timeline: a few weeks of ordinary statements, then the questions people
# actually ask their memory. Every model call is Remembero's own 2.3B writer running
# locally under llama.cpp; the answer is assembled by code with computed notes (dates
# resolved, distances stated, quantities totalled, every line quoting its sentence).
#
#   WRITER_GGUF=/path/to/r23-gemma4-e2b-Q8_0.gguf examples/personal-timeline/run.sh
#
# With a writer already serving on http://127.0.0.1:8081/v1 the variable is not needed.
# Any OpenAI-compatible endpoint works too: set LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
REMEMBERO="${REMEMBERO:-node $root/dist/cli.js}"
show="node $here/../lib/show.mjs"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8081/v1}"
export LLM_MODEL="${LLM_MODEL:-rembero-writer}"
export LLM_API_KEY="${LLM_API_KEY:-local}"
export REMBERO_HOME="$(mktemp -d "${TMPDIR:-/tmp}/rembero-timeline.XXXXXX")"
ns=personal
say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

health="${LLM_BASE_URL%/v1}/health"
if ! curl -sf "$health" >/dev/null 2>&1; then
  if [ -n "${WRITER_GGUF:-}" ] && command -v llama-server >/dev/null; then
    say "Starting the local writer from $WRITER_GGUF"
    port="${LLM_BASE_URL##*:}"; port="${port%%/*}"
    llama-server -m "$WRITER_GGUF" --port "$port" --alias "$LLM_MODEL" -c 16384 -np 2 -ngl 99 \
      --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}' >"$REMBERO_HOME/llama-server.log" 2>&1 &
    trap 'kill %1 2>/dev/null || true' EXIT
    for _ in $(seq 1 120); do curl -sf "$health" >/dev/null 2>&1 && break; sleep 1; done
  fi
  curl -sf "$health" >/dev/null 2>&1 || { echo "no model at $LLM_BASE_URL; set WRITER_GGUF or LLM_BASE_URL" >&2; exit 1; }
fi

say "Fresh memory root: $REMBERO_HOME   (writer: $LLM_MODEL at $LLM_BASE_URL)"
say "Remembering nine statements"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  printf '\n\033[2m> %s\033[0m\n' "$line"
  $REMEMBERO remember "$line" -n $ns 2>&1 | $show || true
done < "$here/statements.txt"

say "What the memory now holds"
$REMEMBERO list -n $ns | $show

say "Asking"
while IFS= read -r q; do
  [ -z "$q" ] && continue
  printf '\n\033[1m? %s\033[0m\n' "$q"
  $REMEMBERO recall "$q" -n $ns --answer-mode evidence --related 2>&1 | $show || true
done < "$here/questions.txt"

printf '\n\033[2mMemory for this run is in %s (safe to delete).\033[0m\n' "$REMBERO_HOME"
