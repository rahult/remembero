# Query-dialect training data generator

Status: design, 2026-09-08. Follows the closure-predicate engine change
([CLOSURE-PREDICATES.md](../../CLOSURE-PREDICATES.md)) and the agent-boundary findings
([AGENT-BOUNDARY-FINDINGS.md](../../research/AGENT-BOUNDARY-FINDINGS.md)).

## Goal

Produce a few thousand execution-verified (question → flat Datalog program) examples so a
small open model can be fine-tuned on Thinking Machines' Tinker to author Remembero queries.
The dialect is deliberately narrow: single or few rules, `p_plus` for chains, `\+` for
absence, `X = const` for yes/no, comparisons, and `count(*)` aggregates. Recursion never
appears in a target.

The spike (2026-09-08, scratchpad) showed the residual small-model failures after closure
predicates are systematic mappings, not reasoning: argument direction, yes/no phrasing,
and reading rows back. Direction and yes/no are what this data set teaches. Row reading is
out of scope: a decision engine renders rows plus proof deterministically.

## Non-goals

- No training of the answer/prose leg.
- No reuse of the agent-boundary schema or its predicate names in training data.
- No RL in this cut; the executor makes a verifiable reward trivial later.

## Architecture

New directory `src/training/`, pure TypeScript, deterministic under a seed. One npm script
drives it end to end; every stage writes a file so stages can be rerun independently.

```
worlds.ts      seed → World { relations, facts, entities }
templates.ts   World → Candidate { category, direction, question, program }
verify.ts      Candidate → Example | rejection (executes program on the world)
paraphrase.ts  Example → Example[] (Luna via OpenRouter, cached, entity-preserving)
export.ts      Example[] → conversations.jsonl + manifest.json + heldout.jsonl
run.ts         CLI: npm run train:data -- --examples 3000 --worlds 60 --seed 7 ...
```

### worlds.ts

A World is a randomly drawn schema plus facts, seeded by a small xorshift RNG so the same
seed yields byte-identical worlds.

Relation kinds and the vocabulary pools they draw from (names never overlap the benchmark's
`works_on/reports_to/status/blocker/waits_on/prefers_meeting/review_slot/promised_update`):

| kind | shape | examples of names | facts |
| --- | --- | --- | --- |
| hierarchy | binary, forest, depth 3–5 | manages, part_of, located_in, parent_of, contains | chains of ≥3 |
| dependency | binary DAG, chains 3–5 | depends_on, blocked_by, imports, feeds, precedes | chains of ≥3, ≥1 root |
| attribute | binary entity→enum value | prefers, tier, mode, language, region | 60–80% of entities have one |
| membership | binary entity→group | member_of, assigned_to, belongs_to, tagged | many-to-many |
| schedule | ternary entity, slot, window | slot_for, shift, booking, review_at | sparse |

Every world has ≥1 hierarchy or dependency relation, ≥1 attribute relation with gaps
(for absence), and 12–30 entities drawn from themed name pools (people, services, places,
components) so entity strings differ across worlds. Predicate argument names are recorded
(`depends_on(Service, Upstream)`) because they appear in the system prompt.

### templates.ts

Each template is a function `(world, rng) → Candidate[]`. Categories mirror the benchmark
plus the failure classes the spike exposed. Every chain template is emitted in **both
directions** with the anchor in argument position 1 and position 2, and the direction is
recorded so the manifest can prove balance.

| category | template | program shape |
| --- | --- | --- |
| direct | value of attribute for X; who has value V | `q(V) :- attr(x, V).` |
| join | entities with attribute V in group G; pairs sharing a group | two positive literals, `A != B` |
| multihop-up | everything above/upstream of X | `q(Y) :- edge_plus(x, Y).` |
| multihop-down | everything below/downstream of X | `q(Y) :- edge_plus(Y, x).` |
| root | end of chain from X | `q(R) :- edge_plus(x, R), \+ edge(R, _).` |
| leaves | things nothing depends on under X | `q(L) :- edge_plus(L, x), \+ edge(_, L).` |
| chain-filter | above X with attribute V | `q(Y) :- edge_plus(x, Y), attr(Y, v).` |
| yes/no | is Y reachable from X; does X have V | `q(Y) :- edge_plus(x, Y), Y = y.` balanced true/false |
| absence | entities with no attribute; groups with no schedule | `q(E) :- member_of(E, _), \+ attr(E, _).` |
| count | how many above X; how many in G | `count(*) as N where edge_plus(x, Y)` |

Question phrasings are drawn from several surface patterns per template ("everyone above
X", "X's chain of command", "what does X ultimately depend on") so templated data already
varies before paraphrasing. Direction words are varied deliberately: "reports to", "manages",
"under", "above", "depends on", "is depended on by".

### verify.ts

Each Candidate is executed with `evaluateQuerySpec` over the world's facts through the
engine (closure synthesis included). A Candidate becomes an Example only if:

1. the program parses and every rule is non-recursive (no head predicate in its own body,
   directly or via another authored rule);
2. the answer set is non-empty, except yes/no-false and absence templates where the
   emptiness or the specific rows are the point;
3. for `_plus` templates, the closure answer differs from the one-hop answer, so the example
   actually requires the closure (otherwise a model could learn `edge` ≈ `edge_plus`);
4. the answer set has at most 30 rows.

Rejections are counted per template in the manifest. A template with >50% rejection fails
the run loudly: that is a template bug, not data to skip.

### paraphrase.ts

Luna (`openai/gpt-5.6-luna`) through the existing `OpenRouterClient`, `LLM_API_KEY` from
`.env`. One call per Example asks for N paraphrases as a JSON array. A paraphrase is kept only
if every entity and value constant that appears in the original question appears verbatim
(case-insensitive) in the paraphrase, so the gold program still fits. Results are cached in
`data/training/cache/paraphrase-<sha256>.json`; reruns make no calls. Concurrency is bounded
(8) and failures degrade to the templated question, never to a dropped Example.

### export.ts

Tinker's `FromConversationFileBuilder` reads JSONL where each line is
`{"messages":[{"role","content"},...]}` and trains on assistant messages
(tinker-cookbook `1f962ed`, 2026-09-03). We emit exactly that:

- **system**: the world's predicate listing with argument names, followed by the fixed
  dialect card: rule shape, `q` head, variables vs constants, `_`, `\+`, comparisons,
  "any binary predicate p also answers p_plus(X, Y): Y reachable from X in one or more
  hops; never write recursive rules", and the yes/no pattern.
- **user**: the question.
- **assistant**: the program, one canonical serialization (`serializeClause`), no prose.

Also written: `manifest.json` (seed, counts by category × direction × world, rejection
counts, paraphrase model and cache hits, held-out world ids), and `heldout.jsonl` (all
Examples from the 10% of worlds held out by seed, same format, never trained on).

### run.ts and scripts

`npm run train:data -- --examples 3000 --worlds 60 --paraphrases 3 --seed 7 --out data/training`
plus `--no-paraphrase` for a zero-cost dry run. `data/training/` is gitignored except the
manifest, which is committed as the record of a run.

## Training and evaluation (outside the TypeScript package)

- `benchmarks/tinker/sl_query_dialect.py`: a `tinker_cookbook` recipe using
  `FromConversationFileBuilder` on the exported JSONL, LoRA rank 32, one epoch, lr 2e-4,
  `TrainOnWhat.ALL_ASSISTANT_MESSAGES`. Reads `TINKER_API_KEY` from `.env`. Never run by
  npm scripts or CI; it costs money and requires explicit confirmation to launch.
- Base model: Tinker's list (2026-09-03) has no Qwen 3.5 2B. Smallest options are
  `Llama-3.2-1B`, `Llama-3.2-3B`, `Qwen3-4B-Instruct-2507`, `Qwen3.5-4B`. Recommendation:
  train `Llama-3.2-3B` first because it sits in the published matrix (multi-hop 0/6, total
  10/31) so before/after lands in the same table, then `Qwen3.5-4B` as the stronger target.
- Evaluation: `benchmarks/tinker/ollama_proxy.py` exposes Tinker's `SamplingClient` behind
  an Ollama-compatible `/api/chat` so the existing agent-boundary harness runs unchanged
  with `OLLAMA_URL` pointed at it and `--model` naming the checkpoint. The harness's Datalog
  prompt gains the `_plus` line and flat few-shots as a new `remembero-closure` condition
  (the published `remembero` condition is untouched). The 31 benchmark questions never
  appear in training; the worlds share no predicate names with the benchmark.

## Testing

- Worlds: same seed → identical JSON; every world satisfies the structural invariants;
  vocabulary never includes a benchmark predicate name.
- Templates: every template's program executes on 20 seeded worlds; no target contains a
  recursive rule; multihop templates emit both directions in equal counts.
- Verify: closure-requirement check rejects a `_plus` example whose one-hop answer is equal;
  rejection-rate guard throws.
- Paraphrase: entity-preservation validator rejects drift (fake client); cache hit makes no
  client call.
- Export: each line parses, has `messages` with system/user/assistant, assistant is a
  single canonical program; manifest counts equal line counts.

## Open questions resolved

- Size: ~3,000 verified examples before paraphrasing, ×(1+N) after. (User, 2026-09-08.)
- Paraphrase model: Luna. (User, 2026-09-08.)
- Base model: recommended Llama-3.2-3B then Qwen3.5-4B, pending user confirmation.
