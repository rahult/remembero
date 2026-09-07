# Transitive closure predicates

Any binary predicate `p` also answers `p_plus(X, Y)`: Y is reachable from X in one or
more `p` hops. Nothing is declared; the engine synthesizes the two closure rules on demand
the first time a program or query references `p_plus/2` without defining it:

```prolog
p_plus(X, Y) :- p(X, Y).
p_plus(X, Y) :- p(X, Z), p_plus(Z, Y).
```

So a chain question is one flat literal, and never a hand-written recursion:

```prolog
above(M) :- reports_to_plus(maya, M).          % everyone above maya
under(P) :- reports_to_plus(P, dana).          % everyone below dana
root(R)  :- waits_on_plus(atlas, R), \+ waits_on(R, _).   % end of the dependency chain
?- waits_on_plus(atlas, procurement_freeze).   % yes/no: one empty binding or none
```

## Why

The agent-boundary benchmark ([findings](research/AGENT-BOUNDARY-FINDINGS.md)) showed
that authoring recursive rules is the one Datalog skill no small local model has: every
multi-hop miss was a recursion error (wrong direction, redefining the base predicate,
fixed-depth unrolling). Moving the recursion into the engine turns multi-hop into the
direct-lookup shape those models already handle, and shrinks the dialect a fine-tuned
model has to learn to predicate choice, argument order, and negation.

## Rules

- **Binary only.** `review_slot_plus(P, D, W)` is an ordinary unknown predicate.
- **One or more hops.** `p_plus(a, a)` holds only when `a` lies on a cycle. There is no
  reflexive `p_star`.
- **Authored rules win.** A program that defines `p_plus/2` itself gets exactly its own
  rules; nothing is synthesized.
- **No doubled suffix.** `a_plus_plus` is refused (yields no rows) so a model typo cannot
  silently work.
- **Missing base.** `p_plus` over a predicate with no facts yields no rows, the same as
  any unknown predicate. In the SQLite bridge the base table must exist, so a typo'd base
  fails with the ordinary missing-relation error.

## Proofs and explanation

Synthesized rules are ordinary rules to the evaluator. They appear in proof ladders as
rule steps numbered after the authored rules, and `explain_query` lists them in its rule
catalog so every proof step resolves to clause text.

## SQLite bridge

The native extension knows nothing about closures. Any program or query referencing a
synthesized `p_plus` is routed to the portable engine (`sqliteDatalogExecutionMode`
returns `'portable'`), where the base predicate resolves to its SQLite relation.
`datalogSql` refuses such programs because a recursive closure cannot be compiled to one
SELECT.
