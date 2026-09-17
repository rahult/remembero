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
| v5 (distilled with the block present, 4,713 examples), retrieval depth 5/5, not paired | 73/133 | 92/133 | 165/266 |
| **v5, paired** (retrieval 4/15/10, recall@k 0.906 = v4's) | 81/133 | 100/133 | **181/266** |

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

**Run A, 2026-09-14 evening: finished on RunPod Serverless.** The step-50 checkpoint and the
data came off the Modal volume without compute credit (`modal volume get`), went onto a RunPod
network volume, and a serverless H100 worker resumed steps 51 to 74 with the recipe the
checkpoint's own SFTConfig recorded (batch 4 x accum 16, max length 8192, no
`chat_template_kwargs`, no `group_by_length`), Liger fused cross-entropy on. 62 minutes wall,
$4.96, train loss 0.194; the worker merged, exported text-only, converted to f16 and
quantized to Q8_0 (7.97 GB) on its own disk and left the GGUF on the volume. Two datacenter
moves on the way: EU-RO-1 had no 80 GB GPUs in stock and EU-NL-1 has no S3 endpoint; US-GA-2
has both. At 135 s a step, a fresh 74-step run costs about $13 on serverless, so v6 waits for
either a pod at $1.99/h or Modal credit. Runbook: `benchmarks/runpod/README.md`; contract and
core: `src/evals/reader-contract.ts`, `benchmarks/train/reader_lora.py`.

**v5 measured (2026-09-15, 02:00), paired.** Same 266, same flags as the v4 row, retrieval
depth passed explicitly (`--top-k 4 --multi-session-top-k 15 --temporal-top-k 10`), recall@k
0.906 for both: **v5 181 against v4 183**, multi-session 81 against 86, temporal 100 against
97; paired by question v5 gains 19 and loses 21. Inside the noise band. Training with the
computed-notes block present in every prompt, on a fresh 4,713-example distillation, produced
a reader level with v4; the block itself is what carries the gain (v4 raw 354 → 383 on the
500), and the reader does not read it any better for having been trained with it. Two
confounds remain and are recorded in the run matrix: v5 saw 4,713 examples against v4's
6,000, and its last 24 steps ran on a different stack (TRL 1.13, Liger). A first attempt at
this measurement, kept as `...-topk5.json`, used the harness's default depth of 5 sessions
and scored 165; it is not a pair and supports no verdict. The lesson went into the runbook:
the reader contract pins the prompt, not the retrieval depth. Decision: v4 stays the served
reader (no reason to swap for a level model), v5 is kept as an equal alternative, and
`data/training-reader-v6` (v4's 6,000 examples re-rendered with the block, contract
`dd+notes@24576`, zero teacher calls) is ready for the next paid run, which tests the data
size confound directly.

## One judge: every stored run re-judged with DeepSeek

All 78 stored runs (20,246 answers) re-judged with `deepseek-chat`; sidecars sit next to each
run, `rejudge-deepseek-chat-summary.json` collects them. DeepSeek is the more lenient judge
overall (0.766 against gpt-4o's 0.745 across every answer) and the ordering is unchanged:

| run (500 unless stated) | gpt-4o | DeepSeek |
| --- | --- | --- |
| GLM 5.3 Flash, hybrid, r19 extractor | 426 | **440** |
| reader v4, hybrid (previous best) | 328 | 350 |
| reader v4, raw baseline | 318 | 354 |
| reader v4, raw + computed notes | 359 | **383** |
| reader v3, hybrid | 320 | 348 |
| reader v4, 266 multi+temporal, baseline → notes v3 | 139 → 166 | 155 → 179 |
| GLM Flash, 266, baseline → notes | 202 → 207 | 213 → 215 |

The gap our reader has to close for parity is 57 on the 500 under the judge we now use.

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

## Reader v5 on the M4 Pro (handoff, 2026-09-13)

Modal halted v5 at step 54 of 74 with checkpoint-50 on the volume and no credit to finish it.
The same run goes locally instead: a 24 GB M4 Pro with Unsloth Studio 2026.8.19, whose
training backend on Apple Silicon is MLX (torch-free; `mlx_lm` 0.31 ships `gemma4_text`, so
Gemma 4 E4B loads). The recipe is committed as `benchmarks/unsloth/reader-v5-mlx.yaml`; this
section is everything the clone on that machine does not carry.

**Same recipe, one machine.** The Modal `train()` settings and their MLX equivalents:

| setting | Modal H100 (v4, v5) | M4 Pro MLX |
|---|---|---|
| base | google/gemma-4-E4B-it, bf16 | same, base loaded 4-bit (bf16 E4B plus 6k-token activations does not fit 24 GB) |
| LoRA | rank 32, alpha 64, q/k/v/o/gate/up/down, no dropout | same; vision and audio towers skipped |
| optimiser | lr 2e-4 linear, 1 epoch, weight decay 0 | same, warmup 5 steps |
| batch | 8 x accum 8 = 64 | 1 x accum 64 = 64, so 74 optimizer steps over 4,713 rows |
| loss | completion only, no packing, max length 8192 | same (`train_on_completions`, `packing: false`) |
| checkpoints | every 25 steps to the volume | every 10 steps to `outputs/reader-v5-mlx`; Ctrl+C and relaunch resumes |

The rows measured with the Gemma 4 tokenizer (300-row sample): mean 5,926 tokens, p95 6,685,
max 7,370, completion mean 126 tokens. Nothing truncates at 8192; if memory forces it,
7424 still covers the longest row.

**What the clone lacks (all gitignored).** Copy from the 16 GB Mac (`/Volumes/Atlas/Code/projects/rembero`):

```sh
# training data: 4,713 rows, 117 MB, plus its per-row meta and the manifest that is tracked
rsync -av data/training-reader-v5/ m4pro:~/rembero/data/training-reader-v5/
# the harness's environment: LLM_BASE_URL, LLM_API_KEY (DeepSeek direct API for the judge and
# the temporal-range model), HF_TOKEN (Gemma 4 is gated); MODAL_* and TINKER_* are unused here
rsync -av .env m4pro:~/rembero/.env
# reader v4 Q8_0 (7.5 GiB), the pair v5 is measured against, if that machine will run both
rsync -av /Volumes/Atlas/models/rembero/reader-v4-gemma4-e4b-Q8_0.gguf m4pro:~/models/
```

The LongMemEval dataset is fetched by `npm run bench:longmemeval:download` into
`.cache/longmemeval/`. The harness judge and temporal-range model are `deepseek-chat` through
`LLM_BASE_URL`; nothing else external is needed for the raw formation (no embeddings, no
extraction cache, no Ollama).

**Run order on the M4 Pro.**

```sh
git clone <repo> ~/rembero && cd ~/rembero && git checkout remembro-v0
npm ci && npm run build:core && npm run bench:longmemeval:download
unsloth --version                       # 2026.8.19 is what the config was dry-run against
unsloth train --config benchmarks/unsloth/reader-v5-mlx.yaml --dry-run
# smoke: downloads the model, proves 4-bit E4B at 8192 fits, prints seconds per step (x74)
unsloth train --config benchmarks/unsloth/reader-v5-mlx.yaml --max-steps 2 --output-dir outputs/reader-v5-smoke
caffeinate -i unsloth train --config benchmarks/unsloth/reader-v5-mlx.yaml 2>&1 | tee runs/local/reader-v5-mlx.log
unsloth list-checkpoints --outputs-dir outputs
unsloth export outputs/reader-v5-mlx/<final> exports/reader-v5 --format gguf --quantization q8_0 --max-seq-length 8192
```

Per-step speed on Apple Silicon is unmeasured; the smoke test decides. Above about 1,500 s a
step the epoch passes 30 hours and RunPod H100 (the fallback in the decisions above) is the
cheaper path, with `benchmarks/modal/train_lora.py`'s `train()` as the recipe to port.

**Serving and the smoke question.** Same as reader v4 (section "Running the reader locally"):

```sh
llama-server -m exports/reader-v5/<file>.gguf --port 8082 -c 12288 -np 1 -ngl 99 --alias rembero-reader \
  --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'
```

If the export emits noise from the first token, it is the direct-to-q8 converter bug noted in
that section: export f16 and quantize with `llama-quantize` instead.

**Measurement.** The pair is reader v4 on the 266 multi-session and temporal questions,
DeepSeek judge, raw formation, lexical retrieval, computed notes on: baseline 167, notes 183
(table above, "Under the cheaper judge"). v5 runs the notes arm with the settings those v4 runs
recorded (`readerMaxTokens 300`, `temporalRangeModel deepseek-chat`, `dateDistances`,
`computedNotes`, top-k 4 / 15 / 10, context 24,576 bytes, all defaults except the flags below):

```sh
node dist/evals/run-longmemeval-answer.js --question-types multi-session,temporal-reasoning \
  --computed-notes --date-distances --concurrency 1 \
  --reader-model rembero-reader --reader-base-url http://127.0.0.1:8082/v1 --reader-max-tokens 300 \
  --temporal-range-model deepseek-chat --judge-model deepseek-chat \
  --output docs/research/results/longmemeval-raw-reader-v5-local-notes-mt-all266.json
```

The v4 local runs went one type at a time (133 each, about 22 s a question on a 16 GB Mac, so
budget roughly 2 hours per type) and one of them logged 31 reader errors out of 133: rows the
summary counts as wrong. Check `summary.errors` in the output and rerun those with `--cases`
before reading the number. Noise on the 266 with identical prompts is about ±4, so v5 has to
clear 187 to count as a gain over v4 with notes; then it goes on the full 500 against GLM's 440.
Add the row to the run matrix (`docs/research/run-matrix/training-runs.json`, platform
"M4 Pro MLX", the smoke test's minutes and $0) and the result to the table above.

**When Modal credit returns.** checkpoint-50 of the Modal v5 run is still on the
`rembero-finetune` volume (`modal volume ls rembero-finetune runs/`); `train()` resumes from the
latest checkpoint in a run directory on relaunch. Two v5 readers trained from the same data on
different hardware are a free reproducibility check, not a conflict.

## Context tiers (2026-09-15): null result

Plan: `docs/superpowers/plans/2026-09-15-reader-tiers-runpod.md`, Task 4. The question was
whether a small reader does better when only the top-ranked retrieved sessions reach it in full
and the rest arrive as short code-built abstracts (`--full-sessions N --abstract-bytes 480`),
the "retrieve wide, rerank narrow, read full" shape from OpenViking. Every arm ran on one H100
pod (RunPod, US-GA-2) serving the merged reader v4, whose fingerprint matched the Modal
checkpoint of record, against the 266 multi-session and temporal questions with the flags of
the v4 183 row (`--top-k 4 --multi-session-top-k 15 --temporal-top-k 10 --context-bytes 24576
--date-distances --computed-notes`, DeepSeek judge). Arms differ only in the tier flags.

| Arm | Correct / 266 | Multi-session / 133 | Temporal / 133 | Gained vs untiered | Lost vs untiered |
|---|---|---|---|---|---|
| Untiered (`dd+notes@24576`) | 183 | 84 | 99 | | |
| Top 5 full | 178 | 85 | 93 | 14 | 19 |
| Top 3 full | 172 | 80 | 92 | 13 | 24 |
| Top 2 full | 162 | 75 | 87 | 9 | 30 |

The untiered pod run landed exactly on the stored 183, so the pod serves the reader of record.
Context recall is unchanged across arms (0.91 / 0.90): tiering never removes a session, it
shortens it.

Why it loses. Classify each lost question by where its evidence sessions sat: with 2 full, 25 of
the 30 losses had evidence ranked below the full tier, so the abstract held it; with 3 full, 17
of 24. With 5 full the abstract-tier losses fall to 7 and the remaining flips (11 lost, 9 gained
with all evidence in full text) look like run-to-run churn. Lexical rank puts evidence deep:
108 of the 266 questions have an evidence session at rank 4 or lower, 63 at rank 6 or lower.
A 480-byte content-word abstract keeps the topic and drops the detail a question turns on
(the date, the count, the second item), and temporal questions pay for it most (-6 even at 5).

Decision (plan rule "no arm clears the band: record the null and stop for review"): the reader
contract stays `dd+notes@24576`. No paired 500 is run, no tiered distillation (Task 5's
rank-ordered haystacks) is built, and `data/training-reader-v6` is already rendered under the
winning contract. Tiers would need a better ranker (evidence in the top few) or abstracts written
for the question, not a smaller budget, before they are worth another arm. Cost: about $1.50 of
H100 time for the sweep, plus the pod's prepare.

## Teacher gate for the thinking step (2026-09-16)

Plan: `docs/superpowers/plans/2026-09-16-reader-v7-v8-thinking.md`, Task 7. Before a student is
trained to write a thinking step (the dated items and the arithmetic, then a final `Answer:`
line that alone is judged), the teacher has to be at least as right when it writes one. GLM 5.3
Flash read the 266 under the contract flags of record (`--date-distances --computed-notes
--context-bytes 24576`, depth 4/15/10, DeepSeek judge), direct and with `--reading notes`.

| Arm | Correct / 266 | Multi-session / 133 | Temporal / 133 |
|---|---|---|---|
| Direct | 217 | 100 | 117 |
| Thinking step | 214 (1 judge error) | 99 | 115 |

Paired flips: multi-session 3 gained, 4 lost; temporal 3 gained, 5 lost. The difference is
inside noise and neither type falls by more than 5, so every aggregation type keeps the
thinking step in the v7 data. As with Luna, writing the items out does not make a large reader
more accurate; the bet for the student is different, that a small model which cannot hold
fifteen sessions' worth of items in one step can when it writes them down first. The teacher's
completions run long (p50 571 tokens including its hidden reasoning), so the length audit's
768-token completion cap will drop the longest thinking rows before training.

## Reader v7: the thinking step, trained (2026-09-16)

Plan: `docs/superpowers/plans/2026-09-16-reader-v7-v8-thinking.md`. Under the thinking contract
`dd+notes+think@24576` the reader writes the dated items it relies on and the arithmetic under
`Notes:`, then one `Answer:` line, and only that line is judged; the three aggregation types and
abstention questions get it, single-session questions keep the direct prompt. Data: v6's 6,000
rows re-answered by GLM 5.3 Flash under the thinking prompt and kept only where the judge said
the new final line matched the stored answer (4,484 kept, 569 disagreed, 278 without an
`Answer:` line), plus 578 rows built from the 210 questions reader v4 got wrong out of 644
freshly distilled ones (each miss about three times). The length audit dropped 26 rows the
trainer would have truncated: 5,754 train, 248 held out. Training: one H100 NVL pod (the
serverless endpoint had no GPU capacity in the volume's datacenter), recipe of record, 90 steps,
4.5 hours, train loss 0.518, held-out loss 0.504, about $14.

Paired against v4 on one pod, both arms at depth 4/15/10 with the DeepSeek judge:

| Questions | v4 (direct) | v7 (thinking step) | Difference |
|---|---|---|---|
| 500 (verdict) | 381 | 391 | +10 |
| 266 multi-session + temporal | 182 | 203 | +21 |

Per type on the 500: multi-session 85 → 93, temporal 99 → 104, preference 19 → 20,
single-session-user 66 → 66, single-session-assistant 52 → 50, knowledge-update 60 → 58.
Paired flips: multi-session +21/-13, temporal +17/-12, knowledge-update +9/-11.

**v7 is the best reader measured.** The gain is where the reader had to combine sessions, and it
comes from writing the items down before answering: a temporal answer now reads "7 days. The MoMA
visit was on or just before 2023-01-08 and the Met visit was on 2023-01-15, so the gap is 7 days."
The cost is length: completions run a median 231 tokens against v4's few words, so serving needs
`--reading notes --reader-max-tokens 1024` and `--max-model-len 12288`.

Knowledge-update is the one type that fell (60 → 58, 9 gained against 11 lost). Writing every
dated mention out invites the reader to compose from the superseded value as well as the current
one. That is the first thing the v8 data review reweights.

Two measurement notes for the runbook. A first paired attempt was thrown away: the local
distillation job and the harness's time-range model share one Ollama Cloud quota, and the
resulting rate limits errored 89 and 78 questions per arm. No local teacher job may run while an
arm runs. And the serverless training endpoint reported GPU stock it could not place, flapping
between running and throttled for twenty minutes without starting the job; training moved to a
pod (`pod.py create-train`), which starts only on `--cloud SECURE` when community capacity is out.

## Reader v8: miss-driven data, a negative result (2026-09-17)

The data review of v7 (`docs/research/results/reader-v7-review.json`) ran v7 over 2,210 freshly
distilled questions it had never trained on: 466 misses, 348 of them multi-session, and 230 of
those are counts ("how many …"). Counting across sessions is what the computed-notes block never
touched and what the thinking step does not fix by itself. `compose` built
`data/training-reader-v8` from the review's recommendation: 5,748 rows, 4,316 of v7's own rows
and 1,438 built from those misses (1,071 multi-session), each unique miss about three times,
held-out set and contract unchanged. Training: same pod recipe, 90 steps, 4.45 hours, train loss
0.501, held-out loss 0.507 (v7: 0.518 and 0.504).

Paired against v7 on one pod, both readers under the thinking contract:

| Questions | v7 | v8 | Difference |
|---|---|---|---|
| 500 | 391 | 384 | -7 |
| 266 multi-session + temporal | 204 | 194 | -10 |

Per type on the 500: multi-session 93 → 87, temporal 105 → 101, knowledge-update 59 → 57,
preference 18 → 21, single-session-user 66 → 67, single-session-assistant 50 → 51. On the 266 the
flips are multi-session +8/-17 and temporal +9/-10.

**The type the data targeted is the type that fell.** Mining keeps a row when the judge says the
student's answer differs from the teacher's, which assumes the teacher is right. GLM 5.3 Flash is
itself about 75% on multi-session (100/133 on the 266), so roughly a quarter of the mined
"misses" teach the teacher's own error, and duplicating each one three times multiplies that
noise. Worse, misses are by construction the questions where the teacher is least reliable: the
harder the question, the likelier the label is wrong. A miss-driven round needs the teacher's
answer verified before it becomes training data — a second teacher, self-consistency across
samples, or a code check against the computed notes — and that verification, not the mining,
is the next thing to build.

**Reader v7 remains the reader of record**: 391/500 and 203/266, the best measured. Its weights
are on the Mac (`/Volumes/Atlas/models/rembero/reader-v7-gemma4-e4b-Q8_0.gguf` and
`reader-v7-adapter/`). v8's weights stay on the network volume; nothing depends on them.

**Runbook, paid the hard way.** The v8 training pod finished at 04:10 and billed until 10:50,
about $20, because the start command parks on completion and the local watcher that should have
stopped it was killed by the Mac's memory pressure. Pods now stop themselves from the inside:
`self_stop` writes `SELF_STOP:` to the run's log on the volume and calls `runpodctl stop pod`,
training pods self-stop on `TRAIN_DONE` and `TRAIN_FAILED`, and serving pods have
`--idle-minutes` (45) and `--max-hours` (4) watchdogs. No local process is load-bearing for
billing any more.
