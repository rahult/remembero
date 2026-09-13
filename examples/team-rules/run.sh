#!/usr/bin/env bash
# Team rules: a project memory that answers with proofs, refuses inconsistent writes,
# and previews a change before it is made. No model, no network; runs in about a second.
#
#   examples/team-rules/run.sh            # from the repository root, after `npm run build:core`
#   REMEMBERO="npx -y remembero" examples/team-rules/run.sh   # against the published package
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
REMEMBERO="${REMEMBERO:-node $root/dist/cli.js}"
show="node $here/../lib/show.mjs"
export REMBERO_HOME="$(mktemp -d "${TMPDIR:-/tmp}/rembero-team-rules.XXXXXX")"
ns=team
say() { printf '\n\033[1m%s\033[0m\n\033[2m$ %s\033[0m\n' "$1" "$2"; }
run() { say "$1" "$2"; shift; eval "$@" | $show; }

say "Fresh memory root" "export REMBERO_HOME=$REMBERO_HOME"
run "Load the team's facts, rules and constraint" "$REMEMBERO import $ns $here/knowledge.dl"

run "Which projects are at risk?" "$REMEMBERO query 'at_risk(Project)' -n $ns"
run "Who owes whom an update, and why? (a proof per row)" "$REMEMBERO explain 'needs_follow_up(From, To, Project)' -n $ns"
run "Why does Maya not owe Sam an update on Orion?" "$REMEMBERO why-not 'needs_follow_up(maya, sam, orion)' -n $ns"
run "Who is available this week? (closed-world negation over on_leave)" "$REMEMBERO query 'available(Person)' -n $ns"
run "Integrity check: clean" "$REMEMBERO check -n $ns"

say "Try to mark Nova blocked without saying what blocks it" "$REMEMBERO assert 'status(nova, blocked).' -n $ns"
if $REMEMBERO assert 'status(nova, blocked).' -n $ns 2>&1 | $show; then echo "unexpected: the write was accepted"; exit 1; else echo "(rejected: the constraint held; nothing was written)"; fi

run "What if Orion became blocked on a design review? (simulation, nothing written)" \
  "$REMEMBERO what-if 'needs_follow_up(From, To, Project)' --assume 'status(orion, blocked).' --assume 'blocked_on(orion, design_review).' --without 'status(orion, active).' -n $ns"

run "Atlas unblocks: supersede its status" "$REMEMBERO supersede --pattern 'status(atlas, _)' 'status(atlas, active).' -n $ns"
run "The follow-up is gone because the fact behind it is" "$REMEMBERO query 'needs_follow_up(From, To, Project)' -n $ns"
run "The life story of Atlas's status" "$REMEMBERO history 'status(atlas, _)' -n $ns"
run "How is Rahul connected to Sam?" "$REMEMBERO connect rahul sam -n $ns"

printf '\n\033[2mMemory for this run is in %s (safe to delete).\033[0m\n' "$REMBERO_HOME"
