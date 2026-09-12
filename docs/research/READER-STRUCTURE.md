# Structure before reading: what a small reader gets wrong, and doing it deterministically

Status: 2026-09-12, in progress. Question: how to improve our own reader (Gemma 4 E4B reader v4,
328/500 on LongMemEval-S against 432 for GLM 5.3 Flash) using the pattern that worked for the
writer on the Remembro V0 branch: let the model read English, let deterministic code do the
structure, and train the model on what the structure gives it.

## Where the reader loses

Per question type, with full evidence in context (both readers, same extractor r19, same
retrieval), from the two 500-question runs in [RUN-MATRIX.md](RUN-MATRIX.md):

| type | reader v4 | GLM 5.3 Flash |
| --- | --- | --- |
| single-session-user | 60/70 (full-evidence 0.88) | 63/70 (0.93) |
| single-session-assistant | 50/56 (0.98) | 49/56 (0.96) |
| single-session-preference | 22/30 (0.83) | 29/30 (0.96) |
| knowledge-update | 60/78 (0.81) | 68/78 (0.94) |
| **multi-session** | **68/133 (0.50)** | 107/133 (0.85) |
| **temporal-reasoning** | **68/133 (0.51)** | 110/133 (0.91) |

Single-session reading is level. The gap is multi-session and temporal, and it is not
retrieval: with every evidence session in context, v4 answers half of them. Of the 119
questions v4 misses and GLM gets, 91 are those two types, and 81 of those had full evidence.

Reading those 81 side by side with GLM shows one shape of failure. The reader finds the
right sentences and computes wrong:

- counts off by one across sessions (3 festivals for 4, 15 fish for 17, 6 courses for 5);
- sums missing an item (two of three road-trip legs; 50 lb of feed for 70);
- differences wrong (4:22 minus 4:10 answered 17 minutes; "Feb 14 to Mar 15" answered 1 day;
  "Mar 4 to Mar 18" answered 2 days);
- relative dates unanchored ("just got back" said on July 15, asked on Aug 5, answered without
  the 3 weeks);
- percentages not computed (2 of 5 pairs of shoes).

This is exactly the failure the V0 writer track met: a small model can read a sentence and
cannot be trusted to do arithmetic, resolve "two weeks ago" against a timestamp, or keep a
count across five sessions. The prototypes said the same. `prototype2` showed a deterministic
structural reader is perfect on the constructions its rules cover and does not transfer to chat
text on its own; V0 showed the working division of labour is model-written claims checked and
computed by code, not code-parsed English.

## The lever: computed notes

`src/evals/computed-notes.ts`, behind `--computed-notes` on the answer harness. After the
retrieved chats, before the question, a deterministic block:

- **Dated events.** Every temporal expression in the user's turns, resolved against the date of
  the session it was said in: `yesterday`, `last night`, `two weeks ago`, `a couple of months
  ago` (marked approximate), `last Saturday`, `on Monday` (previous occurrence), `in 3 days`,
  `March 7th` and `2/15` (year assumed from the session and said so), `Feb 14, 2023`,
  `2022-01-15`. Each line quotes its sentence, names the session date and the expression, and
  states the distance to the question date, months first when long.
- **Order and gaps.** For order questions, the dated events earliest first, with a note that an
  event the question names but the history does not date may mean the history does not say.
  Gaps between dated events in days and weeks, the pair whose sentences best match the
  question first, at most four.
- **Quantities.** Numbers with units in sentences that belong to the question (the unit named
  in the question, or two shared content words), durations folded to minutes, listed with their
  sentences; a sum only when the question asks for a total and every figure is tightly matched;
  the difference and the ratio when there are exactly two.

Assistant turns are excluded, paragraph by paragraph: the assistant's hypotheticals ("homes in
the $250,000–$350,000 range") are not the user's facts. No model is involved; the block is
identical across runs.

## Results

Paired runs, same retrieval (recall@k identical in each pair), raw formation, the 266
multi-session and temporal questions, gpt-4o judge.

| reader | arm | multi-session | temporal | total | fixed / broke |
| --- | --- | --- | --- | --- | --- |
| v4 | baseline | 73/133 | 66/133 | 139/266 | |
| v4 | computed notes, first version | 77/133 | 76/133 | 153/266 | 45 / 31 |
| v4 | **computed notes, second version** | **82/133** | **85/133** | **167/266** | 50 / 22 |
| v4 | computed notes, third version (noun-modifier dates skipped, coverage line) | 82/133 | 84/133 | 166/266 | 48 / 21 |
| GLM 5.3 Flash | baseline | pending | | 202/266 | |
| GLM 5.3 Flash | computed notes | pending | | | |

On the dev half alone the first version went 73 → 80 (temporal 34 → 41, multi 39 → 39).

**What the breaks taught, first version.** About six of the 31 were judge noise (answers such
as "your parents" and "Emma" judged wrong). The rest were the block misleading a reader that
copies: gaps were listed in date order and the reader took the first gap rather than the pair
the question named; an assistant turn's later paragraphs had lost their role prefix and their
figures were summed as the user's; "2/15" without a year was not a date, so the reader invented
60 days for a 14-day gap; a sum was offered for every question and the reader copied it even
when a figure did not belong; the order line tempted the reader to answer when one named event
was absent from the history. The second version fixes each. The third version, which skips
dates that name a thing ("the March 15th issue") and adds a coverage line for order questions
whose one side the history never dates, is level with the second (166 against 167, one fewer
break): the rules have found what they can, and the reader's own noise (about ±4 on 266 with
identical prompts) dominates what remains.

## Where this goes

1. If GLM gains from the block too, the block belongs in the distillation teacher's context:
   `run-real-sessions distill --computed-notes` regenerates the reader's training questions
   with the block present, and a reader v5 learns to read it (and to ignore its stray lines)
   instead of meeting it cold. That is the "train the model on what the structure gives it"
   half of the V0 recipe.
2. The block is English-shaped and grows the same way the writer's boundary did: every break
   that is not noise becomes a rule with a test (`tests/computed-notes.test.ts`).
3. Counting across sessions is the remaining class the block does not touch: "how many
   festivals" needs the items enumerated, which is the engine-recall block's job over the fact
   store, not a regex over text.
