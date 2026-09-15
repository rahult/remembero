# Reader v7 and v8: a trained thinking step, data rebuilt after every run

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** train reader v7 and reader v8 on Gemma 4 E4B, each writing a short thinking step before its answer on the aggregation types, with v8's data rebuilt from a measured review of v7's misses.

**Architecture:** the reader contract gains a reading mode. Under `think`, multi-session, temporal and knowledge-update prompts use the harness's existing notes reading: the reader lists the dated items and the arithmetic under "Notes:", then a final "Answer:" line, and only that line is judged. Training data comes from three code paths added here: thinking regeneration of the v6 rows (teacher answers again under the thinking prompt, kept when its final line agrees with the stored answer), miss mining (fresh distilled questions answered by the current best reader, kept where it is wrong), and a composer that mixes them by weights a data review computes from the last run. A length audit drops rows the trainer would truncate. Pods serve two readers side by side so every verdict is paired on one machine.

**Tech Stack:** TypeScript (Node 24, vitest), Python 3.12 (transformers 5.5, TRL, vLLM 0.29 on RunPod), GLM 5.3 Flash through Ollama Cloud as teacher, DeepSeek `deepseek-chat` as judge.

**Spec:** the user's instruction of 2026-09-15 23:49 ("do improvements the training at least v7 and v8 just make sure we are optimising our training data after each run to better train next including thinking step") on top of the decisions in `docs/superpowers/plans/2026-09-15-reader-tiers-runpod.md` (target 440 on the 500 under DeepSeek, base Gemma 4 E4B, $30 a week, review after every two trained readers, verdicts from the paired 500 clearing about 8, the 266 for iteration, recipe of record without sweeps) and the tier null result in `docs/research/READER-STRUCTURE.md`. Glossary: `CONTEXT.md` (thinking step, miss mining, data review).

## Global Constraints

- Contamination: LongMemEval questions, answers and haystacks never enter training data. Mining, regeneration and the held-out review use the distillation session pool only. LongMemEval results inform the data review through counts per type and failure class, never through question text.
- Retrieval depth on every LongMemEval arm: `--top-k 4 --multi-session-top-k 15 --temporal-top-k 10 --split all --local-only --date-distances --computed-notes --context-bytes 24576 --temporal-range-model glm-5.3-flash:cloud --temporal-range-base-url http://127.0.0.1:11434/v1 --temporal-range-api-key ollama --judge-model deepseek-chat --concurrency 8`. Direct arms `--reader-max-tokens 300`; thinking arms add `--reading notes --reader-max-tokens 1024`.
- The runner caps `--concurrency` at 8.
- vLLM serve flags: bf16, `--max-model-len 12288` (thinking completions on 7k-token prompts overflow 8192), `--max-num-seqs 32`, `--reasoning-parser gemma4`, `--default-chat-template-kwargs '{"enable_thinking":false}'`, `--api-key`. With two readers on one GPU each takes `--gpu-memory-utilization 0.44`.
- A paired verdict needs both readers served by the same pod in the same session.
- Teacher `glm-5.3-flash:cloud` at `http://127.0.0.1:11434/v1`, key `ollama`, `maxTokens` 4000. Judge `deepseek-chat` at `https://api.deepseek.com/v1`, key read from `~/.config/fish/secrets.fish` and never printed or committed.
- No credentials on pods except the vLLM key. Pods in US-GA-2 (the network volume), H100 NVL community first ($2.59/h on 2026-09-15), H100 SXM secure as fallback. Stop pods whenever nothing is running on them.
- Training: RunPod Serverless endpoint of record, recipe of record (rank 32, alpha 64, lr 2e-4, one epoch, batch 4 x accum 16, Liger), `--max-length` from the length audit. No hyperparameter sweeps.
- Budget: RunPod balance $13.03 at plan start. Before each paid step compare the balance with the step's estimate in the task; stop and ask for a top-up when it is short. Weekly cap $30.
- Commit trailers: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JKn8pk6RZdP6dV42QfA5rA`.

---

### Task 1: The thinking step in the reader contract and the distiller

**Files:** Modify `src/evals/reader-contract.ts`, `src/training/reader-distill.ts`, `src/training/run-real-sessions.ts` (distill uses the final answer line for acceptance). Test: `tests/reader-contract.test.ts`, `tests/reader-thinking.test.ts` (new).

**Interfaces:**
- Produces: `ReaderContract.thinking: boolean` (false in `READER_CONTRACT_V5`); `contractFromFlags` sets it from `--reading notes` (any other `--reading` value leaves it false; `two-call` with a contract is an error); `contractId` appends `+think` before the tiers suffix, so `dd+notes+think@24576`; `contractRunnerFlags` appends `--reading notes` when thinking; `THINKING_TYPES: ReadonlySet<string>` = multi-session, temporal-reasoning, knowledge-update (the harness default `notesQuestionTypes`); `readingFor(contract, questionType): 'direct' | 'notes'`; `readerMessages` passes `readingFor(contract, instanceType)` as the builder's `reading` argument, where `instanceType` is the LongMemEval type it already maps (abstention reads as single-session-user, so direct); `acceptDistilled(type, answer, thinking = false)`: when thinking and the type is a thinking type, the reply must contain an `Answer:` line with text after it, and the abstention test runs on `finalAnswerLine(answer)`; `completionAnswer(reply, thinking, type)` returns the judged text (final line for thinking types, whole reply otherwise).

- [ ] Write failing tests: contract id and runner flags for `--date-distances --computed-notes --reading notes --context-bytes 24576` are `dd+notes+think@24576` and `[--date-distances, --computed-notes, --reading, notes, --context-bytes, 24576]`; V5 id unchanged; `--reading two-call` with `contractFromFlags` throws; `readerMessages` under a thinking contract gives a system prompt starting "Answer only from the supplied history. Work in two steps." for a temporal question and the direct system prompt for single-session-user and abstention; `acceptDistilled('multi-session', 'Notes:\n- x\nAnswer: 3', true)` true, without the Answer line false, `acceptDistilled('abstention', 'Notes:\n- nothing\nAnswer: I do not know', true)` true.
- [ ] Run `npx vitest run tests/reader-contract.test.ts tests/reader-thinking.test.ts`, expect failures.
- [ ] Implement; the distill command passes `contract.thinking` to `acceptDistilled` and stores the whole reply as the assistant turn.
- [ ] `npm run build && npx vitest run`, all green; commit.

### Task 2: Length audit

**Files:** Create `benchmarks/train/audit_lengths.py`, `benchmarks/train/test_audit_lengths.py`.

**Interfaces:**
- Produces: `python -m benchmarks.train.audit_lengths <data_dir> --max-length N [--max-completion M] [--write]` prints JSON `{rows, overLength, overCompletion, promptTokens: {p50,p95,max}, completionTokens: {p50,p95,max}}` per file (`conversations.jsonl`, `heldout.jsonl`), counting with the Gemma 4 E4B tokenizer through the same `apply_chat_template` path `reader_lora.to_prompt_completion` uses. `--write` rewrites the files without the rows over either limit (and their `.meta.jsonl` twins by line index) and records the counts in `manifest.json` under `lengthAudit`.
- Consumes: `reader_lora.to_prompt_completion`.

- [ ] Failing test with a tiny fake tokenizer (whitespace split) injected through a `tokenizer` parameter: a row whose prompt plus completion exceeds the limit is counted and removed with its meta twin; others keep order.
- [ ] Implement; run `.venv/bin/python -m pytest benchmarks/train/test_audit_lengths.py`.
- [ ] Run it on `data/training-reader-v6 --max-length 6912` and record whether reader v5 trained on truncated rows (a line in READER-STRUCTURE.md).
- [ ] Commit.

### Task 3: Thinking regeneration of existing rows

**Files:** Create `src/training/reader-think.ts`; modify `src/training/run-real-sessions.ts` (command `think`). Test: `tests/reader-think.test.ts`.

**Interfaces:**
- Consumes: `haystackFromMeta`, `readerMessages`, `completionAnswer`, `acceptDistilled` (Task 1).
- Produces: `node dist/training/run-real-sessions.js think --from data/training-reader-v6 --out <dir> --date-distances --computed-notes --reading notes --context-bytes 24576 [--concurrency 8]` with teacher through `LLM_*` and judge through `--judge-base-url --judge-model --judge-key-env DEEPSEEK_API_KEY`. For each meta row of a thinking type: rebuild the haystack, render under the contract, teacher answers, accept, then the judge answers yes/no to "do these two answers to the question state the same result?" (`agreementPrompt(question, storedAnswer, finalLine)`, exported and unit-tested). Agreeing rows are written with the thinking reply as the assistant turn; disagreeing rows go to `disagreements.jsonl` with both answers. Rows of other types are re-rendered unchanged through `rerenderRow`. Resumable by row index (`progress.jsonl`). Manifest: contract, counts per type (kept, rejected format, disagreed, errors), teacher, judge.

- [ ] Failing tests with stub teacher and judge clients: thinking-type row kept on agreement, moved to disagreements on "no", format rejection counted; single-session row re-rendered byte-identical to `rerenderRow`; resume skips done indices.
- [ ] Implement, build, test, commit.

### Task 4: Miss mining

**Files:** Create `src/training/reader-mine.ts`; modify `src/training/run-real-sessions.ts` (command `mine`). Test: `tests/reader-mine.test.ts`.

**Interfaces:**
- Consumes: a distilled directory made by `distill` under the teacher's contract; `readerMessages`, `completionAnswer`, `haystackFromMeta`.
- Produces: `mine --from <distilled dir> --out <dir> --judge-base-url <u> --judge-model <m> --judge-key-env DEEPSEEK_API_KEY --student-model <m> --student-base-url <u> --student-key-env READER_API_KEY --student-max-tokens <n>` plus the student's contract flags under a `--student-` prefix (`--student-reading notes` etc.; the contract for the student is parsed from those, never from the teacher's). For each row: render for the student under the student's contract, student answers, the judge compares the student's judged text with the teacher's final line using the LongMemEval judge prompt for the row's type (`buildLongMemEvalJudgePrompt` on a synthetic instance, abstention rows through its abstention branch). Writes `misses.jsonl` + meta (the teacher's row, unchanged), `results.jsonl` (id, type, correct, student text), `summary.json` (per type: asked, correct, misses); failure classes are computed later by `review` (Task 5) from `results.jsonl`. Resumable.

- [ ] Failing tests with stub student and judge: a wrong student answer lands in misses with the teacher's messages as rendered for the teacher; the student prompt is rendered with the student contract (direct system prompt when the student is v4); resume.
- [ ] Implement, build, test, commit.

### Task 5: Data review and composer

**Files:** Create `src/training/reader-review.ts`, `src/training/reader-compose.ts`; modify `src/training/run-real-sessions.ts` (commands `review`, `compose`). Test: `tests/reader-review.test.ts`, `tests/reader-compose.test.ts`.

**Interfaces:**
- Produces: `classifyMiss({type, question, expected, hypothesis, reply}): 'format' | 'false-abstain' | 'missed-abstain' | 'count' | 'date-arithmetic' | 'other'` (format: thinking type without an Answer line or cut at the token limit; false-abstain: hypothesis abstains, expected does not; missed-abstain: the reverse; count: both contain a number and the question asks how many; date-arithmetic: temporal type and both contain a number or a date; else other). `review --run <longmemeval result json> [--baseline <json>] --mined <mine dir> --out <review.json>` writes per type accuracy and deltas, failure-class counts from LongMemEval observations (counts only) and from mining, and `recommend: {typeWeights, missShare, dropTypesFromThinking}`: weights proportional to max(misses per type, 5% floor) from mining; miss share 0.25, raised to 0.35 when mined miss rate is above 40%; a type goes to `dropTypesFromThinking` when its paired LongMemEval accuracy under thinking fell by more than 5 questions.
- `compose --base <dir> --misses <dir> --out <dir> --miss-share 0.25 --rows 6000 --seed <n> [--heldout <dir>]`: base rows sampled or kept to fill `rows * (1 - share)`, misses duplicated round-robin to fill the rest, shuffled with the seed; heldout copied from `--heldout`; manifest names every source, the share, counts per type and the contract id (every source must carry the same contract id or compose refuses).

- [ ] Failing tests for each class, for the weight arithmetic (floor, share raise), for compose refusing mixed contracts and for the exact row counts.
- [ ] Implement, build, test, commit.

### Task 6: Two readers on one pod

**Files:** Modify `benchmarks/runpod/pod.py`, `benchmarks/runpod/test_pod.py`, `benchmarks/runpod/README.md`.

**Interfaces:**
- Produces: `create-serve --run a --served-name n [--run b --served-name m] [--gpu-type ...] [--cloud COMMUNITY]`. One run keeps today's command. Two runs prepare both, then start one vLLM per run on ports 8000 and 8001 with `--gpu-memory-utilization 0.44`, both with the same key and `--max-model-len 12288`; the pod exposes both ports; `.env` gets `RUNPOD_SERVE_URL` (first) and `RUNPOD_SERVE_URL_2` (second). Prepare failure of either still parks the pod with `PREPARE_FAILED`.

- [ ] Failing tests on the built start command and request body for two runs; the one-run command is unchanged except `--max-model-len 12288`.
- [ ] Implement, `.venv/bin/python -m pytest benchmarks/runpod`, commit, upload the changed code to the volume.

### Task 7: Teacher gate (no pod)

- [ ] GLM 5.3 Flash as reader on the 266 under the Global Constraints flags, `--reader-model glm-5.3-flash:cloud --reader-base-url http://127.0.0.1:11434/v1 --reader-api-key ollama --reader-max-tokens 4000`, direct and `--reading notes`, paired.
- [ ] A thinking type whose notes arm falls more than 5 below direct keeps direct answers in v7 data (the composer's `dropTypesFromThinking` seeded by hand); record the table in READER-STRUCTURE.md.

### Task 8: Build v7 data (pod about $3)

- [ ] `think` over v6 into `data/training-reader-v7-think`.
- [ ] `distill --reading notes ... --examples 6000 --heldout-examples 300 --type-weights multi-session=40,temporal-reasoning=35,knowledge-update=15,abstention=10 --seed 17` into `data/distill-v7-fresh`.
- [ ] Pod: `create-serve --run reader-v4-gemma4-e4b --served-name rembero-reader-v4`; smoke; `mine` with v4 as student (direct contract, max tokens 300) into `data/mined-v4`; stop the pod.
- [ ] `compose --base data/training-reader-v7-think --misses data/mined-v4 --out data/training-reader-v7 --miss-share 0.25 --rows 6000 --seed 7 --heldout data/distill-v7-fresh`; length audit with `--max-length 8192 --max-completion 768 --write`; commit manifests (not rows).

### Task 9: Train and measure v7 (about $6 training, $3 measurement)

- [ ] Upload `data/training-reader-v7` to the volume; `submit.py reader-v7-gemma4-e4b --max-length 8192`; fetch metrics.
- [ ] Pod with v4 and v7; paired 266 (v4 direct, v7 thinking); paired 500 when the 266 gap is inside noise or better; stop the pod between arms that wait on a local step.
- [ ] `mine` with v7 as student over the fresh held-out distilled set and a second fresh distill (`--seed 19`, 4000 examples) into `data/mined-v7`, on the same pod session; stop the pod.
- [ ] `review --run <v7 500> --baseline <v4 500> --mined data/mined-v7`; record the table, the failure classes and the recommendation in READER-STRUCTURE.md; commit.

### Task 10: Build, train and measure v8 (needs a top-up; stop and ask when short)

- [ ] `compose` v8 from `data/training-reader-v7` minus rows of any `dropTypesFromThinking` type (replaced by their v6 direct rows re-rendered), plus `data/mined-v7` at the recommended share and weights; audit.
- [ ] Train `reader-v8-gemma4-e4b`; pod with v7 and v8; paired 500; review against v7; record; commit.
- [ ] Stop for the two-reader review with the user.
