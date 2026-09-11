# Moonshot: one small trained model for facts, recall and reading

Set 2026-09-11. The bet: a memory system whose every runtime model call goes to a model of at
most about 4B parameters that we trained, and which reads, recalls and writes facts at least
as well as the frontier readers we measure against. The engine, not the model, carries the
reasoning: Datalog answers the counting, the chaining and the time arithmetic; the graph of
facts carries the structure; the model only translates language into that structure and
back. Everything below is measured on benchmarks that already exist in this repo, so the
distance to each target is known today.

## The three numbers

| leg    | what it measures                                                      | today (small model)                | today (frontier)        | moonshot                                            |
| ------ | --------------------------------------------------------------------- | ---------------------------------- | ----------------------- | --------------------------------------------------- |
| facts  | schema-conditioned extraction, 103 cases, closed vocabulary           | 85–86 (Gemma 4 E2B r16)            | 96 (Luna)               | **100**                                             |
|        | fact recall on real transcripts vs frontier labels, at precision      | 29% at 53% (r17); 10% at 42% (r16) | –                       | **60% at 85%**                                      |
| recall | query-correct on the agent-boundary benchmark, closure condition, /31 | 27–28                              | 30 (GLM 5.3), 29 (Luna) | **31 on three seeds**, then ≥95% on a harder v3 set |
| read   | LongMemEval-S, 500 questions, GPT-4o judge, all retrieval ours        | never measured with a small reader | 432 (GLM 5.3 Flash)     | **450 with a reader we trained**                    |
|        | retrieval recall at the per-type k                                    | 92.5%                              | same                    | **96%**                                             |
|        | abstention accuracy on the 30 unanswerable questions                  | 0.93 (GLM reader)                  | –                       | **0.97**                                            |

And the constraint that makes it a moonshot rather than a shopping list: no external model
at runtime. Extraction, query authoring, reading, phrasing, embeddings and time ranges all run
on our own weights, at under one dollar of GPU time per 500 LongMemEval questions.

Milestones on the read leg, in order: our reader beats Luna (416); ties GLM Flash (432 within
the ten-answer noise); reaches 450. The first is the hard one, because no reader under 8B has
been shown to work here yet.

## Why the numbers are reachable

- **Reading is mostly not reading.** With GLM, 92% of the questions whose evidence was
  retrieved were answered. The misses cluster in three places: counting across sessions
  (multi-session), date arithmetic (temporal) and choosing the latest value (knowledge
  update). All three are things the Datalog engine does exactly when the facts are there.
  A small reader that receives "you bought 4 model kits: 2023-03-04 (Zero), 2023-05-19
  (Spitfire), …" does not need to count; one that receives "28 days before today" does not
  need to subtract dates. The reader's job shrinks to reading, which small models do.
- **Facts are the coupling.** The engine can only count what the writer extracted. Today the
  writer captures 29% of what a frontier labeller finds in a real session. Raising that is a
  data problem, and the two data rounds that targeted it (r14, r17) each moved recall by 10 to
  20 points. Sixty percent is three or four rounds away if the label quality is fixed first
  (capped, atomic facts from the subscription labeller, not Luna's verbose ones).
- **Recall is nearly there.** 27–28 of 31 with the frontier at 30. The misses are argument
  order and one prompt-and-grader interaction; the engine now tells the model exactly which
  argument to swap, and r18 is the first adapter trained to listen. The 31-question benchmark
  is saturating and needs a larger, harder successor before the number means much.
- **The retrieval levers are known.** Per-type k, the time range and the semantic route each
  bought answers; entity hops and fact-augmented keys did not. The remaining recall gap is on
  multi-session and temporal questions where the fact graph, once dense enough, can name the
  sessions directly.

## Levers, by leg

### Read (432 → 450 with our own reader)

1. **Structured evidence instead of transcripts.** Retrieve sessions as now, but hand the
   reader a rendered slice of the fact graph first: facts about the question's entities,
   dated, deduplicated, ordered by time, with the `_until` intervals for superseded values,
   then the minimal raw turns that support them. Measure on the 43 fully-evidenced misses.
2. **Engine-side aggregation.** For multi-session questions, run the count or list as a
   Datalog aggregate over the extracted facts and give the reader the result with its
   evidence rows. Gate it on coverage: only when every retrieved session yielded facts of
   the predicate in question.
3. **Deterministic time arithmetic.** The time-range extractor becomes the writer's job
   (roadmap item 4), and the answer to "how long ago" is computed from the session date and
   the fact's date, offered to the reader as a candidate, never guessed by it.
4. **A distilled reader.** Generate questions of the six LongMemEval types over the labelled
   real sessions, answer them with GLM 5.3 Flash on the subscription, and fine-tune Gemma 4
   E4B as a reader on (structured evidence, question) → answer, with the abstention split
   over-represented. Score each round on the 500 from the extraction cache.
5. **Abstention as a first-class label.** The reader learns "the history does not say" from
   examples where the evidence slice is genuinely silent, so that it neither over-abstains
   (Gemma 31B's 84 refusals) nor over-answers (Nemotron's 0.77).

### Facts (86 → 100; 29% → 60% real recall)

1. **Atomic, capped labels.** Re-label the real sessions with GLM 5.3 Flash on the
   subscription, at most eight facts per session, atomic (one relation, canonical atoms),
   and measure r17's recall and precision against the new labels before training on them.
2. **Bi-temporal stamping.** Every extracted fact carries the session date as `valid_from`;
   supersession archives the old value with its interval. Default on.
3. **Entity canonicalisation on the write path.** The alias table and the write-side guard
   (`canonicalizeAtoms`) already exist; extend them with a per-namespace entity index so the
   same person or project resolves to one atom across sessions. This is what makes the graph
   dense enough to aggregate over.
4. **Self-check.** After extraction, the engine checks each new fact for integrity violations
   and for constants absent from the source text (already a guard); add a cheap
   re-extraction vote on the facts the guard flags.

### Recall (28 → 31; then a v3 benchmark)

1. **Repair turns** (r18, training now): the model learns to take the engine's swap
   suggestion and to repeat a correct query that is legitimately empty.
2. **A harder query benchmark**: 100+ questions over larger seeded worlds, with multi-hop,
   negation, aggregation and time filters in proportion to real usage; three seeds; published
   with the frontier rows so the gap is visible.
3. **Closure and aggregation coverage in the training data** in the proportions the v3
   benchmark uses.

### Retrieval (92.5% → 96%)

1. **Local embeddings**: nomic-embed-text ties the hosted model under today's code and runs
   on any GPU; it replaces the hosted call.
2. **Graph-named sessions**: once facts are dense, a multi-session question about model kits
   retrieves the sessions that hold `model_kit` facts directly, not by lexical overlap. This
   is the entity hop measured negative earlier, re-tried when the fact store is dense enough
   to make it precise.
3. **Per-type retrieval policies** stay: k 15 multi-session, k 10 temporal with a range.

## Order of work

1. r18 evaluation (repair turns) and the local embedding switch. Days.
2. Re-label real sessions with the subscription labeller, capped and atomic; retrain the
   writer (r19); measure real recall and the 500. Two weeks.
3. Structured evidence for the reader, measured with GLM Flash first so the rendering is
   validated before a small reader sees it. Two weeks.
4. Engine-side aggregation and time arithmetic behind coverage gates. Two weeks.
5. The distilled E4B reader on the structured evidence. Then iterate until it passes Luna,
   then GLM Flash. This is the long pole.
6. The v3 query benchmark and the writer as time-range extractor, alongside.

Every step lands with its number in [RUN-MATRIX.md](RUN-MATRIX.md) and its model choice in
[MODEL-COMPARISON.md](MODEL-COMPARISON.md); the dependency map is in
[SELF-HOSTED-ROADMAP.md](SELF-HOSTED-ROADMAP.md).
