# Deterministic integrity constraints

Remembero 0.9 adds explicit, headless Datalog constraints to the same portable `.dl`
authority as facts and rules. A constraint is a body with no head:

```prolog
:- works_at(Person, Left), works_at(Person, Right), Left < Right.
```

The body describes a forbidden current state. If it has any solution, the knowledge view
violates the constraint. Remembero never guesses that `works_at/2`, `owner/2`, or any other
predicate is single-valued; integrity exists only where a user explicitly declares it.

Declare policy through the raw, local surfaces:

```bash
remembero assert ':- works_at(Person, Left), works_at(Person, Right), Left < Right.'
remembero check
```

The equivalent MCP operations are `assert_facts` and `check_integrity`. Natural-language
`remember` and ambient auto-capture cannot create, modify, or retract constraints.

## Examples

One current employer per person:

```prolog
:- works_at(Person, Left), works_at(Person, Right), Left < Right.
```

Mutually exclusive status:

```prolog
:- status(Person, active), status(Person, terminated).
```

A required relationship, using closed-world absence explicitly:

```prolog
:- employee(Person), \+ manager(Person, _).
```

The ordering comparison in the first example is deliberate: it reports one ordered pair
instead of the symmetric `(Left, Right)` and `(Right, Left)` duplicates. Constraints use
the same range-safety rules as queries. Variables in comparisons and negated literals
must be bound by an earlier positive relation.

## Inspection result

`remembero check` returns JSON with:

- `status`: `unconstrained`, `consistent`, or `violations`;
- the number of distinct alpha-equivalent constraints checked;
- the complete violation count within the configured bound;
- one check per constraint, including its stable content-derived ID, stored clause,
  equivalent query, declaration sources, violating rows, deterministic proofs, rule
  table, and query-scoped graph.

`checks` is the complete policy audit, not only the findings list. A satisfied policy has
one or more `rows`; a clean policy remains visible with an empty `rows` array and empty
evidence graph.

The command exits `0` for `unconstrained` or `consistent`, and `2` after printing a
complete `violations` result. Invalid input and bounded failures exit `1`.

Use `--proof-limit <n>` to inspect alternative derivations of a violation and
`--max-violations <n>` to set the complete-result cap. MCP uses `proofLimit` and
`maxViolations`. If another proof or violation exists beyond the requested bound,
inspection fails rather than presenting an incomplete audit as complete.

Version 0.18 adds `remembero conflicts [focus]` and MCP `conflict_views` when a person- or
entity-oriented projection is more useful than the policy-oriented `checks` array. It
groups complete rows by the first alpha-stable binding authored in each constraint and
combines their source/proof graphs without inferring new policy. See
[the conflict-view contract](CONFLICT-VIEWS.md).

Version 0.20 lets policy inspect reusable aggregate-derived facts. A constraint such as
`:- team_size(Team, Count), Count > 12.` retains the exact grouped contributor proof and
is enforced atomically like every other explicit forbidden state. See
[the aggregate-rule contract](RULE-AGGREGATION.md).

Tentative claims do not participate in default audit or enforcement. Version 0.21 permits
an explicit `include_tentative` audit/conflict view before promotion; accepting a claim
then runs the ordinary atomic write guard. See
[the knowledge-trust contract](TRUSTED-KNOWLEDGE.md).

## Deterministic and temporal boundary

- Constraints are stored, diffed, exported, imported, journaled, sourced, and
  alpha-deduplicated like other clauses.
- They never derive facts, affect stratification, change rule numbers, or alter ordinary
  query and recall results. Checking is an explicit, read-only operation in 0.9.
- The selected namespace set forms one knowledge view. Constraints from any selected
  namespace inspect facts from the whole selected union. Requested namespace order still
  controls deterministic source witnesses; `*` uses sorted namespace order.
- Store files contain only current clauses. A superseded fact archived as `_until` does
  not violate a constraint over the base predicate. A policy can inspect history only by
  naming the `_until` predicate explicitly.
- Duplicate alpha-equivalent declarations are checked once, while all active declaration
  sources remain visible.
- No conflict table, materialized graph, policy sidecar, or probabilistic score is
  persisted. Reports are deterministic projections from the selected `.dl` files and
  their append-only provenance journal.
- Integrity constraints are portable-engine only in 0.9. The experimental SQLite
  extension rejects headless programs rather than silently ignoring policy.

## v0.9 audit and v0.10 enforcement

Version 0.9 is audit-first and never rejects writes. Version 0.10 preserves that default
and adds opt-in atomic enforcement across supported portable-store mutation paths. It
reuses this exact constraint and evidence contract under a global writer lock; it still
does not infer or automatically repair conflicts. See
[INTEGRITY-ENFORCEMENT.md](INTEGRITY-ENFORCEMENT.md).

## Functional dependencies (single-valued predicates)

Declare that the first `k` arguments of a predicate determine the rest:

```prolog
rembero_functional(works_at, 1).      % one employer per person
rembero_functional(started_at, 2).    % one start date per (person, company)
```

When natural-language `remember` extracts `works_at(mira, initech).` and the store holds
`works_at(mira, acme).`, the pipeline retracts the old value itself (or archives it under
`archive_until`) whether or not the model emitted a `retract` line. Restating the same value
is a duplicate, not a supersession. Predicate aliases (`rembero_predicate_alias(from, to).`)
are applied before the check, so `employed_by(mira, initech).` supersedes `works_at` too.
