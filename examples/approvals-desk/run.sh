#!/usr/bin/env bash
# Approvals desk: a year of "may X approve Y on this date" at a community foundation,
# decided from the delegations schedule, a Slack message and an amendment, with quotes.
# The only model is Remembero's own writer, at ingest time; decisions are deterministic.
#
#   WRITER_GGUF=/path/to/r23-gemma4-e2b-Q8_0.gguf examples/approvals-desk/run.sh
#
# The writer is started on demand by the workspace when nothing answers on the port.
# REMEMBRO_EXTRACTOR=rules runs the rule-based extractor instead (no model; it reads
# the simpler fixture format only, so most steps of this script will not decide).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root/remembro-eval"
if [ ! -x .venv/bin/python ]; then
  echo "creating remembro-eval/.venv"; python3 -m venv .venv && .venv/bin/pip -q install -e . pyyaml
fi
export REMEMBRO_EXTRACTOR="${REMEMBRO_EXTRACTOR:-llm}"
export REMEMBRO_MODEL="${REMEMBRO_MODEL:-rembero-writer}"
export REMEMBRO_BASE_URL="${REMEMBRO_BASE_URL:-http://127.0.0.1:8081/v1}"
export REMEMBRO_API_KEY="${REMEMBRO_API_KEY:-local}"
[ -n "${WRITER_GGUF:-}" ] && export REMEMBRO_LOCAL_GGUF="$WRITER_GGUF"
PYTHONPATH=src exec .venv/bin/python -m remembro.exercise "$here/approvals-desk.yaml" --extractor "$REMEMBRO_EXTRACTOR" "$@"
