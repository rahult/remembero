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
| v4 | third version plus a dated roster of subject sentences for counting questions | 77/133 | 80/133 | 157/266 | 40 / 22 |
| GLM 5.3 Flash | baseline | 93/133 | 109/133 | 202/266 | |
| GLM 5.3 Flash | computed notes, third version | 98/133 | 109/133 | 207/266 | 18 / 13 |

On the dev half alone the first version went 73 → 80 (temporal 34 → 41, multi 39 → 39).

**All 500, raw formation, reader v4** (paired, retrieval identical, third version of the block):

| type | baseline | computed notes |
| --- | --- | --- |
| single-session-user | 60/70 | 59/70 |
| single-session-assistant | 48/56 | 52/56 |
| single-session-preference | 17/30 | 18/30 |
| knowledge-update | 60/78 | 62/78 |
| multi-session | 66/133 | 81/133 |
| temporal-reasoning | 67/133 | 87/133 |
| **total** | **318/500** | **359/500** |

Forty-one questions, no type hurt, no model call added, the block identical on every run. For
scale: the recorded reader v4 number in the richer hybrid formation was 328/500, so the block
in raw formation already beats the previous best configuration by 31, and GLM 5.3 Flash in
hybrid formation is 426.

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

## Reader v5: distilled with the block present

`run-real-sessions distill --computed-notes` ran overnight on 2026-09-12/13 with the v4
recipe (GLM 5.3 Flash teacher over seeded real haystacks, weights multi-session 40 /
temporal 35 / knowledge-update 15 / abstention 10, seed 7, target 6,000). It kept **4,713**
examples (temporal 1,965, multi-session 1,991, abstention 524, knowledge-update 233) from
12,400 attempts: 526 rejected by the checks, the rest errors, most of them the teacher's
reply truncated at its token budget, and the last several hundred `402 Insufficient credits`
when the OpenRouter account ran dry at about 4,700 kept. So v5 trains on 4,713 examples
against v4's 6,000, all of them with the block in the prompt. Training started 06:37 on Modal
(Gemma 4 E4B, max length 8192). The chain's evaluation step needs the gpt-4o judge and the
embedding route through OpenRouter, so it will fail; the v5 comparison runs instead with a
cheaper judge, DeepSeek (`deepseek-chat`, direct API), and local embeddings (Ollama
`nomic-embed-text`), and reader v4 is re-measured under the same judge and embeddings so the
pair stays fair. GLM 5.3 Flash through Ollama was tried as judge and rejected: a reasoning
model does not reliably answer the judge prompt with the single word it requires.

**Under the cheaper judge.** The same 266, reader v4, lexical retrieval only (no embedding
model), DeepSeek `deepseek-chat` as judge instead of gpt-4o:

| arm | multi-session | temporal | total |
| --- | --- | --- | --- |
| v4 baseline | 87/133 | 80/133 | 167/266 |
| v4 computed notes | 86/133 | 97/133 | 183/266 |

DeepSeek is the more lenient judge (167 against gpt-4o's 139 on the same baseline), and the
block's gain holds under it. This is the pair reader v5 is measured against.

## Running the reader locally

When Modal credit ran out on 2026-09-13 (training halted at step 54 of 74, checkpoint-50
kept on the volume), the reader endpoint went down with it. The volume still serves storage,
so reader v4's merged text-only checkpoint (14 GB) was pulled and converted on the Mac:

```sh
modal volume get rembero-finetune runs/reader-v4-gemma4-e4b/merged-text/ /Volumes/Atlas/models/rembero/reader-v4-merged-text/
git clone --depth 1 https://github.com/ggml-org/llama.cpp /Volumes/Atlas/models/llama.cpp
cd /Volumes/Atlas/models/llama.cpp && python3 -m venv .venv && .venv/bin/pip install -r requirements/requirements-convert_hf_to_gguf.txt "transformers>=5"
.venv/bin/python convert_hf_to_gguf.py /Volumes/Atlas/models/rembero/reader-v4-merged-text/merged-text --outtype f16 --outfile /Volumes/Atlas/models/rembero/reader-v4-gemma4-e4b-f16.gguf
llama-quantize /Volumes/Atlas/models/rembero/reader-v4-gemma4-e4b-f16.gguf /Volumes/Atlas/models/rembero/reader-v4-gemma4-e4b-Q8_0.gguf Q8_0
llama-server -m /Volumes/Atlas/models/rembero/reader-v4-gemma4-e4b-Q8_0.gguf --port 8082 -c 12288 -np 1 -ngl 99 --alias rembero-reader \
  --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'
```

Two things bit. The converter's pinned transformers 4 cannot read the tokenizer config that
transformers 5 wrote (upgrade it, as the Modal exporter does). And converting straight to
`--outtype q8_0` produced a model that emitted noise from the first token; converting to f16
and quantizing with `llama-quantize`, the exporter's path, gives a working 7.5 GiB Q8_0 that
answers "21 days ago" to the smoke question. On a 16 GB M-series Mac it reads a 24 KB prompt in
about 12 seconds and answers in about 30, one question at a time. The harness points at it with
`--reader-model rembero-reader --reader-base-url http://127.0.0.1:8082/v1`. On 40 questions
drawn from the DeepSeek-judged notes run, the local Q8 reader agrees with the served bf16 reader
on 37 (29 correct against 32), within the reader's own noise, at a median 22 seconds a question.

## Decisions (2026-09-13, with the user)

- Done means our own reader matches GLM Flash on the full 500 under the same judge and
  formation. DeepSeek `deepseek-chat` is the judge for everything from here; every stored run
  is re-judged with it (`node dist/evals/rejudge-longmemeval.js`, sidecar files next to each
  run, originals untouched) so tables share one judge; old gpt-4o numbers stay as history.
- Computed notes ship in the product now, `REMBERO_COMPUTED_NOTES` default on, at both attach
  points: evidence mode gains a Computed section under each row's sources, every line quoting
  its sentence; natural mode gets the block in the phrasing prompt.
- Iteration runs on a 100-question stratified subset (27 multi-session, 27 temporal, 16
  knowledge-update, 14 single-user, 10 single-assistant, 6 preference; seed 100), the 266 only
  to confirm. The lost local arms are not rerun.
- The next class is structured evidence from the writer's own facts (see below). Training
  resumes only when two consecutive structural changes each land inside the noise band; writer
  rounds then run locally in Unsloth Studio (MLX), reader rounds on RunPod H100 while Modal
  credit is out, v5 from its step-50 checkpoint when it returns.

## Structured evidence

`src/knowledge/structured-evidence.ts`, `--structured-evidence` on the harness. The writer
extracted facts at write time; before the reader sees them, code dates each fact (a temporal
expression inside it resolved against its session, else the session date), grounds it against
the session text it came from (a fact whose content words never appear there is dropped),
deduplicates, and where one claim has several values marks the latest current and the earlier
ones superseded, with their dates. The block goes before the chats: dated claims to compose
from first, the wording to check second. It is measured on the 100-question subset in hybrid
formation with the local r23 writer filling the extraction cache and the local reader v4
answering, four arms: baseline, computed notes, structured evidence, both.

## Where this goes

1. GLM gains a little from the block (+5, all multi-session) and loses nothing, so the block
   belongs in the distillation teacher's context:
   `run-real-sessions distill --computed-notes` regenerates the reader's training questions
   with the block present, and a reader v5 learns to read it (and to ignore its stray lines)
   instead of meeting it cold. That is the "train the model on what the structure gives it"
   half of the V0 recipe.
2. The block is English-shaped and grows the same way the writer's boundary did: every break
   that is not noise becomes a rule with a test (`tests/computed-notes.test.ts`).
3. Counting across sessions is the remaining class the block does not touch. A dated roster of
   every user sentence naming the question's subject was tried and reverted: 157 against 166,
   worse on both types. A long list displaces the dated events and gives a small reader more
   to misread, not less. Enumeration needs the items themselves, which is the engine-recall
   block's job over the fact store, not a regex over text.
