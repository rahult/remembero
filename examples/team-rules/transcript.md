# Transcript: examples/team-rules/run.sh

One run, captured 2026-09-14. Paths and timestamps are from that run.

```

Fresh memory root
$ export REMBERO_HOME=$TMP/rembero-team-rules

Load the team's facts, rules and constraint
$ remembero import team examples/team-rules/knowledge.dl
imported 27 clause(s), 0 duplicate(s) skipped

Which projects are at risk?
$ remembero query 'at_risk(Project)' -n team
Project = atlas

Who owes whom an update, and why? (a proof per row)
$ remembero explain 'needs_follow_up(From, To, Project)' -n team
rows:
  From = rahul, Project = atlas, To = maya
    because
      needs_follow_up(rahul, maya, atlas)  <- rule 5
        promised_update(rahul, maya, atlas)  [2026-09-13]
        status(atlas, blocked)  [2026-09-13]

Why does Maya not owe Sam an update on Orion?
$ remembero why-not 'needs_follow_up(maya, sam, orion)' -n team
status: blocked
query: needs_follow_up(maya, sam, orion)
summary: No stored result matches needs_follow_up(maya, sam, orion). Required fact status(orion, blocked) is missing.
explanation:
  rows: (none)
failures:
  rules_blocked: needs_follow_up(maya, sam, orion)
    nearby:
      -
        fact: needs_follow_up(rahul, maya, atlas).
        explanation:
          rows:
            yes
              because
                needs_follow_up(rahul, maya, atlas)  <- rule 5
                  promised_update(rahul, maya, atlas)  [2026-09-13]
                  status(atlas, blocked)  [2026-09-13]

Who is available this week? (closed-world negation over on_leave)
$ remembero query 'available(Person)' -n team
Person = rahul
Person = maya
Person = sam

Integrity check: clean
$ remembero check -n team
status: consistent
constraintCount: 1
violationCount: 0
checks:
  -
    clause: :- status(Project, blocked), \+ blocked_on(Project, _).
    query: status(Project, blocked), \+ blocked_on(Project, _)
    sources: 1
    rows: (none)

Try to mark Nova blocked without saying what blocks it
$ remembero assert 'status(nova, blocked).' -n team
error: integrity_violation
message: integrity enforcement rejected 1 new violation(s)
mode: no_new_violations
baselineViolationCount: 0
blockingViolationCount: 1
blockingViolations:
  Project = nova
introducedViolationCount: 1
introducedViolations:
  Project = nova
candidate:
  status: violations
  constraintCount: 1
  violationCount: 1
  checks:
    -
      clause: :- status(Project, blocked), \+ blocked_on(Project, _).
      query: status(Project, blocked), \+ blocked_on(Project, _)
      sources: 1
      rows:
        Project = nova
          because
            status(nova, blocked)  [2026-09-13]
          because
            not blocked_on(nova, _)
(rejected: the constraint held; nothing was written)

What if Orion became blocked on a design review? (simulation, nothing written)
$ remembero what-if 'needs_follow_up(From, To, Project)' --assume 'status(orion, blocked).' --assume 'blocked_on(orion, design_review).' --without 'status(orion, active).' -n team
changed: true
application:
  namespace: team
  namespaces: team
  assumed: status(orion, blocked).; blocked_on(orion, design_review).
  duplicateAssumptions: (none)
  retracted: status(orion, active).
  unmatchedRetractions: (none)
  assumedRules: (none)
  duplicateRuleAssumptions: (none)
  retractedRules: (none)
  unmatchedRuleRetractions: (none)
baseline:
  rows:
    From = rahul, Project = atlas, To = maya
      because
        needs_follow_up(rahul, maya, atlas)  <- rule 5
          promised_update(rahul, maya, atlas)  [2026-09-13]
          status(atlas, blocked)  [2026-09-13]
candidate:
  rows:
    From = rahul, Project = atlas, To = maya
      because
        needs_follow_up(rahul, maya, atlas)  <- rule 5
          promised_update(rahul, maya, atlas)  [2026-09-13]
          status(atlas, blocked)  [2026-09-13]
    From = maya, Project = orion, To = sam
      because
        needs_follow_up(maya, sam, orion)  <- rule 5
          promised_update(maya, sam, orion)  [2026-09-13]
          status(orion, blocked)  [assumed]
resultDelta:
  added:
    From = maya, Project = orion, To = sam
      because
        needs_follow_up(maya, sam, orion)  <- rule 5
          promised_update(maya, sam, orion)  [2026-09-13]
          status(orion, blocked)  [assumed]
  removed: (none)
  evidenceChanged: (none)
  unchangedCount: 1
integrityDelta:
  baseline:
    status: consistent
    constraintCount: 1
    violationCount: 0
    checks:
      -
        clause: :- status(Project, blocked), \+ blocked_on(Project, _).
        query: status(Project, blocked), \+ blocked_on(Project, _)
        sources: 1
        rows: (none)
  candidate:
    status: consistent
    constraintCount: 1
    violationCount: 0
    checks:
      -
        clause: :- status(Project, blocked), \+ blocked_on(Project, _).
        query: status(Project, blocked), \+ blocked_on(Project, _)
        sources: 1
        rows: (none)
  introduced: (none)
  resolved: (none)

Atlas unblocks: supersede its status
$ remembero supersede --pattern 'status(atlas, _)' 'status(atlas, active).' -n team
added: status(atlas, active).
duplicates: 0
retracted: 1
archived: status_until(atlas, blocked, '2026-09-13T14:21:22.449Z').

The follow-up is gone because the fact behind it is
$ remembero query 'needs_follow_up(From, To, Project)' -n team
(none)

The life story of Atlas's status
$ remembero history 'status(atlas, _)' -n team
1.11 2026-09-13T14:21:21.581Z team asserted: status(atlas, blocked).
2.0 2026-09-13T14:21:22.449Z team superseded: status(atlas, blocked). -> status_until(atlas, blocked, '2026-09-13T14:21:22.449Z').
2.2 2026-09-13T14:21:22.449Z team asserted [current]: status(atlas, active).

How is Rahul connected to Sam?
$ remembero connect rahul sam -n team
status: connected
shortestHops: 2
searchComplete: true
paths:
  -
    hops: 2
    entities: rahul; maya; sam
skippedNonGroundFacts: 0

Memory for this run is in $TMP/rembero-team-rules (safe to delete).
```
