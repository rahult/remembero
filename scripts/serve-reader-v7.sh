#!/bin/bash
# Serve reader v7 (the thinking-step reader) locally on port 8084.
# Harness flags to match: --reader-model rembero-reader-v7 --reading notes --reader-max-tokens 1024
set -e
MODEL=${1:-/Volumes/Atlas/models/rembero/reader-v7-gemma4-e4b-Q8_0.gguf}
exec llama-server -m "$MODEL" --port 8084 -c 12288 -np 1 -ngl 99 \
  --alias rembero-reader-v7 --reasoning-budget 0 \
  --chat-template-kwargs '{"enable_thinking":false}'
