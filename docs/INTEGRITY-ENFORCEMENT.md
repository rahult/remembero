# Atomic integrity enforcement

Remembero 0.10 can turn the explicit constraints introduced in 0.9 into an opt-in
write boundary. Audit remains the default: upgrading does not silently reject writes.

```bash
remembero assert 'suspended(mira).' --integrity-mode strict
remembero remember 'Mira is now suspended' --integrity-mode strict
```

Every enforced write evaluates the complete post-mutation candidate before changing a
`.dl` file, mutation journal, or cache. A rejection exits the CLI with status `3` and
returns structured JSON containing the candidate integrity audit, proofs, durable and
proposed sources, and query-scoped graph evidence.

## Modes

- `strict` rejects any write whose resulting selected knowledge view has a violation.
  Use it after `remembero check` is clean.
- `no_new_violations` is the migration mode. Existing violation identities may remain or
  be removed, but the write cannot introduce another identity. Identity is the stable
  alpha-normalized constraint ID plus its ordered bound values; renaming variables does
  not make a legacy violation look new.
- `no_new_violations` is the default since 0.57; `off` is an explicit opt-out that retains the 0.9 audit-only behavior. `remembero init` registers the MCP server with `strict`.

Configure a CLI or MCP server process with:

```bash
export REMBERO_INTEGRITY_MODE=strict
export REMBERO_INTEGRITY_NAMESPACES='*'
remembero serve
```

`REMBERO_INTEGRITY_NAMESPACES` accepts `*` or a comma-separated ordered list. When it is
omitted, enforcement checks only the namespace being written. An explicit scope must
include the target namespace. This prevents a misleading check that omits the candidate
write itself.

For a staged migration:

```bash
remembero check --namespaces '*'
REMBERO_INTEGRITY_MODE=no_new_violations \
REMBERO_INTEGRITY_NAMESPACES='*' remembero assert 'project(atlas).'
# Repair every reported row, then move the process default to strict.
```

`--proof-limit` and `--max-violations` bound write-rejection evidence when an integrity
mode is active. If complete checking or output cannot fit its bound, the write still
fails closed.

## Library and MCP

Library mutations accept an `integrity` policy in `MutationContext`:

```ts
store.assert('personal', 'suspended(mira).', {
  integrity: { mode: 'strict', namespaces: ['policy', 'personal'] },
});
```

`rememberText` and `PipelineDeps` accept `integrityEnforcement`. Rejections throw
`IntegrityViolationError`; its `toJSON()` payload is stable and machine-readable.

The MCP `remember`, `assert_facts`, and `forget` tools accept `integrityMode`,
`integrityNamespaces`, `proofLimit`, and `maxViolations`. MCP deliberately does not
expose an `off` override: the server operator owns any configured default. A tool call
may strengthen an MCP server default but cannot weaken a strict default. `check_integrity` remains the
read-only audit surface.

## Atomicity and concurrency

All supported `.dl` mutation paths participate in one cross-process mutation lock,
including non-enforcing writers. Candidate validation and commit therefore observe one
stable namespace snapshot, even when a policy joins facts stored in different
namespaces. The target namespace file and journal entry are written only after the
candidate passes.

Delete-style natural-language updates now use `MemoryStore.replace`: all requested fact
retractions and additions form one candidate and one journal transition. Archive-style
updates continue to use `supersede`. History replays delete replacements as a
`retracted` event followed by asserted replacements; archived supersessions remain
`superseded` events.

Auto-capture operational reservation/failure events remain journaled even when its fact
write is rejected; no rejected fact reaches the `.dl` authority. Integrity-enforced
review pruning accepts one namespace per operation so its preflight cannot imply a
multi-file atomic commit that the portable store does not provide.

## Trust boundary

- Constraints remain explicit raw Datalog policy. Natural-language remember and ambient
  capture still cannot create, modify, or retract them.
- Direct hand edits to `.dl` files and writers from older Remembero versions do not
  participate in the 0.10 lock. Stop writers before editing, then run `remembero check`.
- Separate `REMBERO_HOME` roots are separate authorities. Namespaces are organization,
  not access control.
- The experimental SQLite extension remains audit/enforcement-independent in 0.10.
- No conflict index, graph sidecar, repair agent, or inferred predicate semantics is
  introduced. The readable `.dl` program remains the only knowledge and policy source.

Version 0.11 optionally evaluates both baseline and candidate through the same explicit
position-scoped entity identity view. Set `entityIdentity: 'canonical'` (or the matching
CLI/MCP option) only when aliases should denote one policy subject. The mutation and
journal remain literal even when the candidate projection is canonical.
