# A small reader trained on structured evidence

Status: design, 2026-09-11. Serves moonshot read items 1, 4 and 5
([../../research/MOONSHOT.md](../../research/MOONSHOT.md)).

## Problem

Every LongMemEval number we have uses a frontier reader. No model under 8B has been shown to
read here; the 3B answer leg on the agent-boundary benchmark was noise, Gemma 4 31B
over-abstained (347/500) and Nemotron 3 Super over-answered (389/500). A reader we train has to
beat Luna (416) and then tie GLM 5.3 Flash (432 ± 7). The reader's failures with frontier
models cluster where the engine is exact: counts across sessions, date arithmetic, and the
latest value of a changed fact.

## Design

Two parts, the second depending on the first.

### 1. Structured evidence (a formation the reader sees, model-independent)

For a question, after retrieval, the reader receives in this order:

1. **Fact slice.** Facts from the whole remembered history whose predicate or constants
   overlap the question terms (the `expandByEntities` seed step, no hop), deduplicated,
   ordered by session date, each with its date and its `_until` interval when superseded:
   `2023-03-04  bought(user, model_kit, zero)` / `2023-01-10 → 2023-05-02  lives_in(user, austin)`.
   Bounded to 40 lines.
2. **Engine result** when the writer's program returned rows (`--engine-recall`): the
   program and its rows, a count with its counted items.
3. **Derived arithmetic.** For temporal questions, the interval between the question date
   and each dated fact in the slice, precomputed: `bought(...) — 63 days (9 weeks) before today`.
   The reader copies a number, it never subtracts.
4. **The raw sessions** as today, bounded.

Measured first with GLM 5.3 Flash reading, on the 500 from the cache, against 432 ± 7, so the
rendering is validated before a small model sees it. The fact slice is only as good as the
writer's recall; the r19+ real-session rounds are the coupling.

### 2. Reader training data without a teacher

Questions and gold answers are generated deterministically from the labelled real sessions
(GLM 5.3 Flash capped labels, `data/real/labels-glmflash8.jsonl`), so no frontier model is
in the loop and the answer is exact by construction:

| type                  | construction                                                                             | gold                                  |
| --------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------- |
| single-session-user   | one session, one fact `p(user, v)` → a templated question for `p`                        | `v`                                   |
| multi-session (count) | k sessions (3–8) from different haystacks sharing predicate `p` → "How many … have I …?" | k, plus the list                      |
| multi-session (list)  | same → "Which … have I …?"                                                               | the values                            |
| knowledge-update      | two sessions with `p(user, v1)`, later `p(user, v2)` → "What is my current …?"           | `v2`, with `v1` as the earlier value  |
| temporal-reasoning    | one dated fact and a question date → "How many weeks ago did I …?"                       | computed interval, one-unit tolerance |
| abstention            | a question about a predicate absent from the assembled history                           | "the history does not say"            |

Each example's context is assembled the way part 1 renders it: the fact slice from the chosen
sessions plus distractor sessions (same user, other predicates), the derived arithmetic, and
the raw session text. The question templates are the moonshot's second synthetic generator,
built like `extraction-data.ts` (kinds, pools, verification: the gold must be derivable from
the rendered evidence by the engine itself, or the example is rejected).

Held-out: whole haystacks, as the query generator holds out worlds. Abstention share 20%.

### 3. Training and measurement

- Base: Gemma 4 E4B (reader needs more capacity than the writer; the writer stays E2B).
- LoRA r32, one epoch, same Modal recipe; a third served alias.
- Measure on the 500 with `--reader-model` pointed at the adapter, structured evidence on.
  Milestones: > 416 (Luna), ≥ 425 (inside GLM Flash's band), then 450.
- Noise: two runs per configuration until a deterministic local reader makes one enough.

## Results so far (2026-09-12)

- v1 (2,976 deterministic examples): 235/500 at the training context budget; single-session
  types within ten of GLM, aggregation types reproduce answer shapes without the reading.
- v2 (8,885 deterministic examples with distinct counts, between-dates, 2–12 distractors):
  231/500; held-out loss on its own distribution 0.015 but temporal was the only type to rise.
  Learning the generator better did not transfer. Deterministic gold teaches the form of an
  answer, not how to find it across fifteen raw sessions.
- v3 (in progress): the design's second path. GLM 5.3 Flash writes a question of a drawn type
  over 5–15 assembled real sessions and answers it through the evaluation's exact reader
  prompt; the pair is kept when the answer matches the type (abstention questions must get
  abstentions, others must not). 3,000 + 200 examples, seven types including preference.

## Risks

- The fact slice can mislead when the writer misreads (precision 53%); the "supplementary"
  framing and the min-score threshold from the reserved-facts experiment apply.
- Synthetic questions may not transfer to LongMemEval's phrasing; paraphrase a share with the
  subscription model, as the query generator does.
- E4B needs an A100 to serve; cost is per recall, acceptable for evaluation, to be revisited
  for production.

## Order

Part 1 first (one week; a formation flag and a doc section with the number). Part 2's
generator second (one week). Training and the milestone runs third.
