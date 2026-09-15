# Context tiers for the reader, measured and trained on RunPod

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the reader's even per-session budget with code-built context tiers (short abstracts for every retrieved session, full text for the few ranked highest), find the best tier size on the 266 paired against v4, confirm on the 500, then mine v4's misses and train v6 on the winning contract.

**Architecture:** Tiers are a property of the reader contract, rendered by the one context builder that distillation, training and evaluation share. Reader inference runs on a RunPod community pod (vLLM serving v4's merged weights rebuilt from its adapter on the pod), driven by the existing harness from the Mac; training later runs on an H100 pod in the same datacenter against the same network volume.

**Tech Stack:** TypeScript (vitest), Python (transformers 5, PEFT, vLLM), RunPod REST API and S3 volume, DeepSeek judge, GLM 5.3 Flash via Ollama Cloud (time range, teacher).

**Spec:** the decisions of the 2026-09-15 grilling session, recorded here: target 440 within the noise band on the 500 under the DeepSeek judge; base stays Gemma 4 E4B; $30 a week, review after every two trained readers; verdicts from the paired 500 and must clear about 8 points, the 266 for iteration, the 100 as smoke; context budget tiered by code first; full-text sessions chosen by the existing lexical rank; tier sizes swept at 2, 3 and 5; training resumes only after the tiering verdict on data re-rendered under the winning contract; data recipe is miss-driven distillation folded into the 6,000; write-time counts deferred; training on a community pod with no hyperparameter sweeps. Glossary: `CONTEXT.md` (reader contract, lane, context tier, paired run).

## Global Constraints

- Every arm passes retrieval depth explicitly: `--top-k 4 --multi-session-top-k 15 --temporal-top-k 10`, plus `--split all --local-only --context-bytes 24576 --date-distances --computed-notes --reader-max-tokens 300 --temporal-range-model glm-5.3-flash:cloud --temporal-range-base-url http://127.0.0.1:11434/v1 --temporal-range-api-key ollama --judge-model deepseek-chat`. These are the flags of the v4 183 row; arms differ only in the tier flags.
- A paired verdict needs both arms served by the same pod and model; the stored 183 is a reference, not a pair.
- RunPod: inference in EUR-NO-1 (network volumes and an S3 endpoint both present; RTX 5090 32 GB community at $0.69/h), network volume mounted at `/workspace` on pods. Training later in US-GA-2 or US-CA-2 (H100 stock, volumes, S3) on its own volume, since no datacenter had volumes, S3 and a cheap 48 GB card on 2026-09-15 (US-KS-2 has S3 and A6000 stock but no network volumes). Stop pods whenever no arm is running.
- vLLM serving reproduces the Modal serve of record: bf16, `--max-model-len 8192`, `--max-num-seqs 32`, `--gpu-memory-utilization 0.92`, `--reasoning-parser gemma4`, `--default-chat-template-kwargs '{"enable_thinking":false}'`, an `--api-key` (the proxy URL is public).
- Gemma 4 merged checkpoints need three steps before vLLM loads them: merge the adapter, export text-only, restore the KV-sharing tensors transformers drops (see `restore_dropped_weights` in `benchmarks/modal/train_lora.py`).
- No credentials leave the Mac except the RunPod pod's own vLLM key; model weights reach the pod as the public base from Hugging Face plus the adapter uploaded to the volume.
- Budget: RunPod balance $4.99 at plan start covers Tasks 1-4 (inference, about $3); Task 6 (training) needs a top-up and stops to ask.
- Commit trailers: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JKn8pk6RZdP6dV42QfA5rA`.

---

### Task 1: Context tiers in the builder and the contract

**Files:** Modify `src/evals/longmemeval-answer.ts` (`buildLongMemEvalAnswerContext`, the per-question call sites), `src/evals/reader-contract.ts`, `src/evals/run-longmemeval-answer.ts` (flags), `src/training/reader-distill.ts` (`readerMessages` passes the contract's tiers). Test: `tests/reader-tiers.test.ts`, additions to `tests/reader-contract.test.ts`.

**Interfaces produced:**
- `buildLongMemEvalAnswerContext(..., structuredEvidence = false, tiers?: { fullSessions: number; abstractBytes: number })` (new trailing optional argument).
- `ReaderContract` gains `fullSessions: number | null` (null = even split, today's behaviour) and `abstractBytes: number` (default 320). `READER_CONTRACT_V5` gets `fullSessions: null, abstractBytes: 320` so its id and rendering are unchanged.
- `contractId` appends `+full<n>` when `fullSessions` is set (abstract bytes appended as `a<bytes>` only when not 320); `contractFromFlags` reads `--full-sessions <n>` and `--abstract-bytes <n>` (positive integers; `--abstract-bytes` between 120 and 2048; throw otherwise); `contractRunnerFlags` emits them when set; `contractBuilderArgs` returns the tiers object as a fifth element.
- Runner flags `--full-sessions <n>`, `--abstract-bytes <n>`; refuse combining `--full-sessions` with `--focused-budget` (two budget policies in one arm are not a pair).

**Rendering rules (tiered mode):**
1. `usable` keeps its retrieval rank order. The first `fullSessions` of them are full; the rest are abstracts. If `usable.length <= fullSessions`, every session is full and the output must be byte-identical to the even-split output for the same inputs.
2. An abstract section is `### Retrieved session <rank> (abstract: lines matching the question)\n<date line>\n<facts line if any>` followed by the session's `USER:` sentences that contain at least one question content word (the same `recallWords(question)` words of length ≥ 4 that the focused budget uses), in their original order, joined by a space, cut at a sentence boundary so the whole section is at most `abstractBytes` bytes; if no sentence matches, the first user sentence cut to fit.
3. Full sections use `sourceWindow` with budget `floor((contextBytes - sum of abstract section bytes) / fullCount)` each, minus the header, exactly as the even split computes its body today.
4. Sections are ordered in the prompt as today (by session date, then rank). The computed-notes block and structured evidence are built from the full text of every selected session, unchanged, so they are byte-identical between even-split and tiered modes.
5. Total history bytes never exceed `contextBytes` plus the headers' slack the even split already allows.

**Tests (write first):** tiered with 3 full of 8 sessions renders 3 full and 5 abstract sections, each abstract ≤ `abstractBytes`; abstract picks only matching user sentences and never assistant text; all-full case byte-identical to even split; computed-notes block byte-identical across modes; contract id and flags round-trip (`dd+notes+full3@24576`, `--full-sessions 3`); `--full-sessions` with `--focused-budget` refused; `readerMessages` with a tiered contract renders the same messages as the builder.

- [ ] RED, GREEN, `npm run build:core`, full `npx vitest run` once, commit.

---

### Task 2: RunPod reader-serving tooling

**Files:** Create `benchmarks/runpod/pod.py`, `benchmarks/runpod/prepare_reader.py`, `benchmarks/runpod/test_pod.py`; modify `benchmarks/train/reader_lora.py` (add `restore_dropped_weights(run_dir: Path, base_model: str) -> int`, ported from the Modal function, and `merge_adapter(run_dir: Path, base_model: str) -> Path` writing `run_dir/merged`), `benchmarks/runpod/README.md` (a "Reader serving pod" section).

**`prepare_reader.py --root /workspace --run <run> [--base-model google/gemma-4-E4B-it]`:** if `<root>/runs/<run>/merged-text/config.json` exists and a `.prepared` marker is present, exit 0 (idempotent). Otherwise: `merge_adapter` from `<root>/runs/<run>/adapter`, `export_text_only`, `restore_dropped_weights`, copy tokenizer and processor files, write the marker, delete `merged/` to save volume space. `HF_HOME=<root>/hf`.

**`pod.py` subcommands (REST `https://rest.runpod.io/v1`, key from `RUNPOD_API_KEY`):**
- `create-serve --run <run> --served-name <name> [--gpu "NVIDIA GeForce RTX 5090"] [--datacenter EUR-NO-1] [--volume <id>]`: POST `/pods` with `cloudType COMMUNITY`, `gpuTypeIds` [the given GPU], `gpuTypePriority availability`, `dataCenterIds` [datacenter], `networkVolumeId`, `volumeMountPath /workspace`, `containerDiskInGb 40`, `ports ["8000/http", "22/tcp"]`, `env {VLLM_API_KEY, HF_HOME: /workspace/hf, PYTHONPATH: /workspace/code, PUBLIC_KEY: <contents of ~/.ssh/id_ed25519.pub if present>}`, image `vllm/vllm-openai:<pinned tag that supports Gemma 4; record it>` with `dockerEntrypoint ["bash","-lc"]` and a start command that runs `pip install -q "peft>=0.17" && python -m benchmarks.runpod.prepare_reader --root /workspace --run <run> && exec vllm serve /workspace/runs/<run>/merged-text <the Global Constraints serve flags> --served-model-name <name> --host 0.0.0.0 --port 8000`. Prints the pod id and `https://<id>-8000.proxy.runpod.net/v1`, and writes both to `.env` keys `RUNPOD_SERVE_POD_ID`, `RUNPOD_SERVE_URL`.
- `wait --pod <id>`: poll `GET /v1/models` on the proxy URL with the key every 20 s until it lists the served name or 30 minutes pass; print elapsed time.
- `stop --pod <id>`, `start --pod <id>`, `terminate --pod <id>`, `status`.
- A `VLLM_API_KEY` is generated once (`secrets.token_urlsafe(32)`) and stored in `.env` if absent.

**Tests:** `build_pod_payload(...)` and `serve_command(...)` are pure functions; unit tests assert the payload fields, that the serve command carries every Global Constraints flag, and that no Hugging Face or Modal credential appears in the payload. No network in tests.

- [ ] Tests first, implement, `.venv/bin/python -m pytest benchmarks/runpod -q`, commit.

---

### Task 3: Volume, weights and the serving pod (controller operations)

- [ ] Create a 60 GB network volume in EUR-NO-1 (`POST /v1/networkvolumes`), write `RUNPOD_VOLUME_ID`/`RUNPOD_DATACENTER` to `.env` (keep the old values under `_PREV` names).
- [ ] `modal volume get rembero-finetune runs/reader-v4-gemma4-e4b/adapter` to the Mac; `volume.py put` it to `runs/reader-v4-gemma4-e4b/adapter`; put `benchmarks/__init__.py`, `benchmarks/train/*`, `benchmarks/runpod/*` under `code/`.
- [ ] `pod.py create-serve --run reader-v4-gemma4-e4b --served-name rembero-reader-v4`; `pod.py wait`; smoke question through the proxy (expect the dated bike answer, not noise).

### Task 4: The tier sweep on the 266, then the 500

- [ ] Four arms on the 266 against the pod, concurrency 8, same pod: even split (baseline), `--full-sessions 2`, `3`, `5`, each tier arm with `--abstract-bytes 480` (the header and date line take ~165 B, so 320 leaves one sentence). Outputs `docs/research/results/longmemeval-raw-reader-v4-pod-<arm>-mt-all266.json`. Stop the pod after.
- [ ] Verdict on the 266: the best tier arm against the pod baseline, per type and paired flips. The baseline must land within noise of the stored 183 (a serving sanity check).
- [ ] If the best arm clears about 7 on the 266, run it and the baseline on the full 500, paired; the verdict needs about 8 on the 500. Record the table and the decision in `docs/research/READER-STRUCTURE.md` and the winning contract id.
- [ ] If no arm clears the band, record the null and stop the plan for review before any training money.

### Task 5: Miss-driven distillation (after a positive Task 4 verdict)

- [ ] First make distillation rank-ordered under a tiered contract: order each haystack's sessions by the harness's own lexical retrieval score for the question (the same search the evaluation uses), so the full-text sessions in training are chosen as in evaluation; `readerMessages` refuses tiered contracts until this lands.
- [ ] Generate fresh distillation questions with the existing `distill` command under the winning contract, teacher `glm-5.3-flash:cloud` through Ollama, at least 6,000 attempts, type weights multi-session 40 / temporal 35 / knowledge-update 15 / abstention 10.
- [ ] Answer each with v4 on the pod under the same contract; judge v4's answer against the teacher's with DeepSeek; keep the rows v4 got wrong (the misses).
- [ ] Build `data/training-reader-v7`: `data/training-reader-v6` re-rendered under the winning contract plus the misses, misses weighted by duplication to a quarter of the rows; manifest records the contract, the miss count and the weighting.

### Task 6: Train and measure (needs a RunPod top-up; stop and ask first)

- [ ] H100 NVL community pod in US-KS-2 on the same volume, the bootstrap training command, recipe of record (rank 32, alpha 64, lr 2e-4, batch 4 x accum 16, Liger, max length 6912), one run, export, serve on the inference pod, paired 500 against v4 under the winning contract.
