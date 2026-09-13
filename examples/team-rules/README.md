# Team rules: a project memory that shows its work

A four-person product team on a Monday morning. Who owns what, who promised whom an
update, which project is blocked and on what, who is on leave. Seven rules turn those facts
into the questions a lead actually asks, and one constraint keeps the memory honest.

No model and no network. The whole run takes about a second.

```sh
npm run build:core          # once
examples/team-rules/run.sh
```

Or against the published package: `REMEMBERO="npx -y remembero" examples/team-rules/run.sh`.

## What you will see

1. **Which projects are at risk** is a one-line query over a rule, not a search.
2. **Who owes whom an update, and why**: every row carries its proof, down to the facts
   and the day each was recorded.
3. **Why Maya does not owe Sam an update**: `why-not` names the missing fact instead of
   returning an empty list.
4. **Who is available**: closed-world negation over `on_leave`.
5. **A write the store refuses.** Marking Nova blocked without saying what blocks it
   violates the constraint, so nothing is written and the violation comes back with its
   proof. This is the default integrity mode, not an opt-in.
6. **What if Orion became blocked?** A simulation that reports the rows it would add and
   whether any constraint would break, without writing anything.
7. **Atlas unblocks.** `supersede` ends the old status and keeps it as history; the
   follow-up disappears because the fact behind it did; `history` tells the whole story.
8. **How Rahul is connected to Sam** through the explicit graph.

`knowledge.dl` is the entire memory: readable facts, rules and one constraint. Change it
and run again. The captured output of one run is in `transcript.md`.
