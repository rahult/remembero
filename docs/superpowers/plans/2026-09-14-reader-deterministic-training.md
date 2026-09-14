# Reader training with a deterministic contract, on RunPod for under $10

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the reader so that every prompt it learns from is byte-identical to the prompt it will be evaluated and served with, and finish reader v5 plus one optional v6 for under $10 of RunPod credit.

**Architecture:** One `ReaderContract` object names the deterministic structure the reader is given before reading (date distances, computed notes, focused budget, structured evidence, context bytes). The distiller, the evaluation harness and the training manifest all derive from it, and a test proves the distilled prompt equals the harness prompt. Training moves off Modal into a plain Python core shared by the Modal app and a RunPod runbook; v5 resumes from the step-50 checkpoint still on the Modal volume, which costs about $2 instead of $5.

**Tech Stack:** TypeScript (vitest), Python 3.12 with transformers 5 / TRL / PEFT, llama.cpp, RunPod community H100 at $1.99/h, Modal volume (read-only, no compute credit needed), Ollama Cloud GLM 5.3 Flash as the free teacher, DeepSeek as judge.

**Spec:** `docs/research/READER-STRUCTURE.md` sections "Reader v5", "Decisions (2026-09-13, with the user)" and "Reader v5 on the M4 Pro"; the design conversation of 2026-09-14 (this plan's header records its conclusions).

## Global Constraints

- Base model `google/gemma-4-E4B-it`; it is not gated (checked 2026-09-14), no Hugging Face token needed.
- Recipe unchanged from v4/v5: rank-32 LoRA, alpha 64, lr 2e-4 linear, 1 epoch, **batch 4 × grad-accum 16** (effective 64; the v5 checkpoint's saved SFTConfig records 4 × 16, not the Modal function's 8 × 8 defaults), max length 8192, completion-only loss, no packing, gradient checkpointing, seed 42. The saved config also shows `chat_template_kwargs` and `group_by_length` absent (dropped by TRL 1.x and transformers 5 on Modal too), so the core keeps that behaviour and never reintroduces them.
- A structure block enters the contract only after a paired run on the 266 (DeepSeek judge) shows a gain outside the ±4 noise band. Today that is date distances and computed notes (raw 500: 354 → 383). Focused budget and structured evidence stay flags until they earn their place (subset-100 hybrid: baseline 74, evidence 71, both 75, inside noise).
- Budget: $10 on RunPod **Serverless** (the user's choice, 2026-09-14 16:04): H100 80GB flex workers at $4.18-4.79/h, billed per second from worker start to stop, nothing while idle. Run A (finish v5 from step 50) ≈ $2.50. Run B (v6, full run) ≈ $6, only if Task 7's gate passes and the balance allows. No pods.
- Base model stays Gemma 4 E4B (research 2026-09-14: the 2-3B alternatives save under $1 a run and lose ~16 RULER@128k points; Qwen3.5-4B has no reader evidence). Cost levers instead: Liger fused linear cross-entropy (`use_liger_kernel`; removes the 34 GB logits tensor, ~+20% throughput) on every GPU run, with automatic fallback to the plain path if the architecture is unsupported; `max_length 6912` (covers the p95 row, 6,685 tokens) for fresh runs. Run A keeps 8192 because it resumes a checkpoint trained at 8192.
- Serverless facts that bind the worker: default execution timeout 600 s (set `policy.executionTimeout` to 10,800,000 ms per job); `/run` payload 10 MB (data, checkpoint and GGUF travel by network volume mounted at `/runpod-volume`, loaded and fetched over RunPod's S3-compatible API); the image is built for linux/amd64 and pushed to Docker Hub; progress via `runpod.serverless.progress_update`.
- The teacher for any new distillation is GLM 5.3 Flash through Ollama Cloud (`glm-5.3-flash:cloud` at `http://127.0.0.1:11434/v1`), which is on the subscription, not OpenRouter.
- Judge for every measurement: `deepseek-chat`, official-compatible protocol, as every stored run.

---

### Task 1: The reader contract

**Files:**
- Create: `src/evals/reader-contract.ts`
- Modify: `src/training/reader-distill.ts:247-283` (`readerMessages`)
- Test: `tests/reader-contract.test.ts`

**Interfaces:**
- Produces: `interface ReaderContract { dateDistances: boolean; computedNotes: boolean; focusedBudget: boolean; structuredEvidence: boolean; contextBytes: number }`, `const READER_CONTRACT_V5: ReaderContract`, `function contractId(c: ReaderContract): string` (e.g. `dd+notes@24576`), `function contractFromFlags(argv: string[]): ReaderContract`, `function contractRunnerFlags(c: ReaderContract): string[]` (the exact `run-longmemeval-answer` flags), `function contractBuilderArgs(c)` returning the trailing positional arguments of `buildLongMemEvalAnswerContext`.
- `readerMessages(haystack, question, type, contract = contractFromEnv())` keeps its old behaviour when called without a contract.

- [ ] **Step 1: Write the failing test**

```ts
// tests/reader-contract.test.ts
import { describe, expect, it } from 'vitest';
import {
  READER_CONTRACT_V5,
  contractFromFlags,
  contractId,
  contractRunnerFlags,
} from '../src/evals/reader-contract.js';
import { buildLongMemEvalAnswerContext } from '../src/evals/longmemeval-answer.js';
import { readerMessages, type Haystack } from '../src/training/reader-distill.js';

const haystack: Haystack = {
  questionDate: '2023-07-10',
  sessions: [
    { id: 's1', date: '2023-05-15', facts: ['likes(user, stand_up_comedy).'], transcript: 'USER: I saw John Mulaney on 3 May, it was 56 days before my exam.\nASSISTANT: Nice.' },
    { id: 's2', date: '2023-06-01', facts: [], transcript: 'USER: I ran 12 km on 26 May.\nASSISTANT: Great run.' },
  ],
};

describe('reader contract', () => {
  it('names itself stably', () => {
    expect(contractId(READER_CONTRACT_V5)).toBe('dd+notes@24576');
    expect(contractId({ ...READER_CONTRACT_V5, structuredEvidence: true, focusedBudget: true })).toBe('dd+notes+focus+evidence@24576');
  });

  it('reads the same flags the evaluation runner takes', () => {
    const c = contractFromFlags(['--date-distances', '--computed-notes', '--structured-evidence']);
    expect(c).toEqual({ ...READER_CONTRACT_V5, structuredEvidence: true });
    expect(contractRunnerFlags(c)).toEqual(['--date-distances', '--computed-notes', '--structured-evidence']);
  });

  it('renders the distillation prompt byte-for-byte as the harness renders it', () => {
    const contract = { ...READER_CONTRACT_V5, structuredEvidence: true };
    const distilled = readerMessages(haystack, 'How long ago did I see John Mulaney?', 'temporal-reasoning', contract);
    const instance = {
      question_id: 'distill-temporal-reasoning', question_type: 'temporal-reasoning',
      question: 'How long ago did I see John Mulaney?', question_date: '2023/07/10 (Sat) 09:00',
      answer: '', haystack_session_ids: [], haystack_dates: [], haystack_sessions: [], answer_session_ids: [],
    } as never;
    const harness = buildLongMemEvalAnswerContext(
      instance,
      haystack.sessions.map((s) => ({ opId: s.id, ts: `${s.date}T09:00:00.000Z`, text: s.transcript, facts: s.facts })),
      contract.contextBytes, [], 'direct', undefined,
      contract.dateDistances, contract.computedNotes, contract.focusedBudget, contract.structuredEvidence,
    );
    expect(distilled).toEqual(harness.messages);
    expect(distilled[1]!.content).toContain('Computed');
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run tests/reader-contract.test.ts`
Expected: FAIL, `Cannot find module '../src/evals/reader-contract.js'`.

- [ ] **Step 3: Write the contract module**

```ts
// src/evals/reader-contract.ts
/**
 * The reader contract: the deterministic structure a reader is handed before it reads.
 * The distiller renders training prompts with it, the evaluation runner takes the same
 * flags, and the training manifest records its id, so a trained reader is always
 * evaluated and served with exactly the prompt it learned from.
 */
export interface ReaderContract {
  /** Session headers state the distance to the question date. */
  dateDistances: boolean;
  /** The computed-notes block (dates resolved, gaps, quantities), every line quoting its sentence. */
  computedNotes: boolean;
  /** Context share per session weighted by the question's words it contains. */
  focusedBudget: boolean;
  /** The writer's own facts, dated, grounded and deduplicated, before the chats. */
  structuredEvidence: boolean;
  contextBytes: number;
}

export const DEFAULT_READER_CONTEXT_BYTES = 24 * 1024;

/** What reader v5 was distilled with: the two blocks with a measured gain. */
export const READER_CONTRACT_V5: ReaderContract = {
  dateDistances: true,
  computedNotes: true,
  focusedBudget: false,
  structuredEvidence: false,
  contextBytes: DEFAULT_READER_CONTEXT_BYTES,
};

const FLAGS: Array<[keyof Omit<ReaderContract, 'contextBytes'>, string, string]> = [
  ['dateDistances', '--date-distances', 'dd'],
  ['computedNotes', '--computed-notes', 'notes'],
  ['focusedBudget', '--focused-budget', 'focus'],
  ['structuredEvidence', '--structured-evidence', 'evidence'],
];

export function contractId(contract: ReaderContract): string {
  const parts = FLAGS.filter(([key]) => contract[key]).map(([, , short]) => short);
  return `${parts.length === 0 ? 'plain' : parts.join('+')}@${contract.contextBytes}`;
}

export function contractFromFlags(argv: readonly string[], base: ReaderContract = READER_CONTRACT_V5): ReaderContract {
  const contract = { ...base, dateDistances: false, computedNotes: false, focusedBudget: false, structuredEvidence: false };
  for (const [key, flag] of FLAGS) if (argv.includes(flag)) contract[key] = true;
  const bytes = argv.indexOf('--context-bytes');
  if (bytes >= 0 && argv[bytes + 1] !== undefined) contract.contextBytes = Number(argv[bytes + 1]);
  return contract;
}

/** The environment the distiller has honoured so far, kept so old commands still work. */
export function contractFromEnv(env: NodeJS.ProcessEnv = process.env): ReaderContract {
  return {
    ...READER_CONTRACT_V5,
    computedNotes: env.REMEMBERO_READER_COMPUTED_NOTES === '1',
    focusedBudget: env.REMEMBERO_READER_FOCUSED_BUDGET === '1',
    structuredEvidence: env.REMEMBERO_READER_STRUCTURED_EVIDENCE === '1',
  };
}

/** The flags that make run-longmemeval-answer build this exact prompt. */
export function contractRunnerFlags(contract: ReaderContract): string[] {
  const flags = FLAGS.filter(([key]) => contract[key]).map(([, flag]) => flag);
  if (contract.contextBytes !== DEFAULT_READER_CONTEXT_BYTES) flags.push('--context-bytes', String(contract.contextBytes));
  return flags;
}

/** The trailing positional arguments of buildLongMemEvalAnswerContext, in its order. */
export function contractBuilderArgs(contract: ReaderContract): [boolean, boolean, boolean, boolean] {
  return [contract.dateDistances, contract.computedNotes, contract.focusedBudget, contract.structuredEvidence];
}
```

- [ ] **Step 4: Make `readerMessages` take the contract**

In `src/training/reader-distill.ts` add the import and change the function:

```ts
import { contractBuilderArgs, contractFromEnv, type ReaderContract } from '../evals/reader-contract.js';

/** The exact reader prompt the evaluation builds, over the haystack, for the question. */
export function readerMessages(
  haystack: Haystack,
  question: string,
  type: DistillType,
  contract: ReaderContract = contractFromEnv(),
): Conversation['messages'] {
  const instance = { /* unchanged */ } as unknown as LongMemEvalInstance;
  const context = buildLongMemEvalAnswerContext(
    instance,
    haystack.sessions.map((s) => ({ opId: s.id, ts: `${s.date}T09:00:00.000Z`, text: s.transcript, facts: s.facts })),
    contract.contextBytes,
    [],
    'direct',
    undefined,
    ...contractBuilderArgs(contract),
  );
  return context.messages as Conversation['messages'];
}
```

Delete the old positional `true, process.env.REMEMBERO_READER_COMPUTED_NOTES === '1'` arguments; nothing else in the function changes.

- [ ] **Step 5: Run the tests**

Run: `npx vitest run tests/reader-contract.test.ts tests/reader-distill.test.ts`
Expected: PASS (both files).

- [ ] **Step 6: Commit**

```bash
git add src/evals/reader-contract.ts src/training/reader-distill.ts tests/reader-contract.test.ts
git commit -m "Reader contract: one object names the structure the reader reads; distiller renders it exactly as the harness"
```

---

### Task 2: The distill command takes the contract and records it

**Files:**
- Modify: `src/training/run-real-sessions.ts:583-600` (flag parsing), `:665` (the `readerMessages` call), the manifest object near `:718`
- Test: `tests/reader-contract.test.ts` (one more case)

**Interfaces:**
- Consumes: `contractFromFlags`, `contractId`, `contractRunnerFlags` from Task 1.
- Produces: `manifest.json` gains `"contract": {"id": "dd+notes@24576", "runnerFlags": [...], ...contract}`. `--computed-notes` keeps working; `--date-distances`, `--focused-budget`, `--structured-evidence`, `--context-bytes <n>` are new.

- [ ] **Step 1: Write the failing test**

Append to `tests/reader-contract.test.ts`:

```ts
import { distillManifestContract } from '../src/training/run-real-sessions.js';

it('the distill manifest records the contract it rendered with', () => {
  const entry = distillManifestContract(['--computed-notes', '--date-distances']);
  expect(entry.id).toBe('dd+notes@24576');
  expect(entry.runnerFlags).toEqual(['--date-distances', '--computed-notes']);
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run tests/reader-contract.test.ts`
Expected: FAIL, `distillManifestContract` is not exported.

- [ ] **Step 3: Implement**

In `src/training/run-real-sessions.ts`:

```ts
import { contractFromFlags, contractId, contractRunnerFlags, type ReaderContract } from '../evals/reader-contract.js';

export function distillManifestContract(argv: readonly string[]): ReaderContract & { id: string; runnerFlags: string[] } {
  const contract = contractFromFlags(argv);
  return { ...contract, id: contractId(contract), runnerFlags: contractRunnerFlags(contract) };
}
```

In the distill command: replace the line `if (process.argv.includes('--computed-notes')) process.env.REMEMBERO_READER_COMPUTED_NOTES = '1';` with `const contract = contractFromFlags(process.argv);`. Change the call at line 665 to `readerMessages(haystack, parsed.question, type, contract)`. In the manifest object add `contract: distillManifestContract(process.argv),`. Note: `contractFromFlags` sets `dateDistances` only when `--date-distances` is passed; v5 was rendered with date distances on, so every new distill command passes it explicitly (see Task 6).

- [ ] **Step 4: Run the tests and the build**

Run: `npx vitest run tests/reader-contract.test.ts && npm run build:core`
Expected: PASS, build clean.

- [ ] **Step 5: Commit**

```bash
git add src/training/run-real-sessions.ts tests/reader-contract.test.ts
git commit -m "Distill: the contract comes from flags and is written into the manifest"
```

---

### Task 3: A training core that runs anywhere

**Files:**
- Create: `benchmarks/train/reader_lora.py`, `benchmarks/train/__init__.py` (empty)
- Modify: `benchmarks/modal/train_lora.py:380-500` (`train`), `:200-225` (`export_text_only`)
- Test: `benchmarks/train/test_reader_lora.py` (pytest, CPU smoke on a 135M model)

**Interfaces:**
- Produces: `train_lora(data_dir: Path, run_dir: Path, base_model: str, *, epochs=1, lr=2e-4, lora_rank=32, batch_size=8, grad_accum=8, max_length=8192, merge=True, save_steps=25, on_save=None, resume=True, liger=False) -> dict` whose dict carries `resumed_from`, `liger` (whether the fused kernel was actually used) and `max_length` and `export_text_only(run_dir: Path) -> Path` and `to_prompt_completion(path) -> list[dict]`, all free of Modal. The Modal `train()` becomes a thin wrapper that mounts the volume paths and passes `on_save=volume.commit`.

- [ ] **Step 1: Write the failing smoke test**

```python
# benchmarks/train/test_reader_lora.py
import json
from pathlib import Path

import pytest

from benchmarks.train.reader_lora import export_text_only, to_prompt_completion, train_lora

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


def write_rows(path: Path, n: int = 6) -> None:
    rows = [
        {"messages": [
            {"role": "system", "content": "Answer only from the supplied history."},
            {"role": "user", "content": f"History chats:\n\n### Retrieved session 1\nUSER: my number is {i}.\n\nQuestion: what is my number?"},
            {"role": "assistant", "content": f"Your number is {i}."},
        ]}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_to_prompt_completion_puts_loss_on_the_assistant_turn(tmp_path):
    write_rows(tmp_path / "conversations.jsonl", 2)
    rows = to_prompt_completion(str(tmp_path / "conversations.jsonl"))
    assert rows[0]["completion"] == [{"role": "assistant", "content": "Your number is 0."}]
    assert [m["role"] for m in rows[0]["prompt"]] == ["system", "user"]


@pytest.mark.slow
def test_train_lora_runs_on_cpu_and_resumes(tmp_path):
    data = tmp_path / "data"; data.mkdir(); write_rows(data / "conversations.jsonl")
    run = tmp_path / "run"
    saves: list[int] = []
    metrics = train_lora(data, run, TINY, epochs=1, batch_size=1, grad_accum=1, max_length=128,
                         merge=False, on_save=lambda: saves.append(1), save_steps=2)
    assert metrics["train_loss"] is not None and (run / "adapter" / "adapter_config.json").exists()
    assert saves, "the save callback never fired"
    # a second call resumes from the last checkpoint and finishes immediately
    again = train_lora(data, run, TINY, epochs=1, batch_size=1, grad_accum=1, max_length=128, merge=False, save_steps=2)
    assert again["resumed_from"] is not None
    assert again["liger"] is False  # CPU smoke never asks for Liger
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest benchmarks/train/test_reader_lora.py -q`
Expected: FAIL, `No module named 'benchmarks.train.reader_lora'`. (If `trl`/`peft` are missing in `.venv`, `.venv/bin/pip install "transformers>=5,<6" "trl>=0.24" "peft>=0.17" "datasets>=3" "accelerate>=1" pytest` first.)

- [ ] **Step 3: Write the core, moved verbatim out of the Modal file**

```python
# benchmarks/train/reader_lora.py
"""LoRA SFT of a small chat model on conversations.jsonl, the recipe Remembero's writer and
reader runs use, with nothing Modal-specific: paths are arguments and persistence is a
callback. benchmarks/modal/train_lora.py and benchmarks/runpod/run.sh both call this."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Callable

LORA_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def to_prompt_completion(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            if messages[-1]["role"] != "assistant":
                raise ValueError("last message must be the assistant turn")
            rows.append({"prompt": messages[:-1], "completion": [messages[-1]]})
    return rows


def load_base_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = dict(dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, attn_implementation="sdpa")
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def lora_targets(model) -> list[str]:
    import torch.nn as nn

    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(tower in name for tower in ("vision", "audio", "embed_vision", "embed_audio")):
            continue
        parts = name.split(".")
        leaf, parent = parts[-1], (parts[-2] if len(parts) > 1 else "")
        if leaf in LORA_PROJECTIONS or (leaf == "linear" and parent in LORA_PROJECTIONS):
            names.append(name)
    if not names:
        raise RuntimeError("no LoRA target modules found; unexpected model layout")
    return names


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((run_dir / "trainer").glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1]))
    return checkpoints[-1] if checkpoints else None


def train_lora(
    data_dir: Path,
    run_dir: Path,
    base_model: str,
    *,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 32,
    batch_size: int = 8,
    grad_accum: int = 8,
    max_length: int = 8192,
    merge: bool = True,
    save_steps: int = 25,
    liger: bool = False,
    on_save: Callable[[], None] | None = None,
    resume: bool = True,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model(base_model)
    train_rows = to_prompt_completion(str(data_dir / "conversations.jsonl"))
    heldout = data_dir / "heldout.jsonl"
    held_rows = to_prompt_completion(str(heldout)) if heldout.exists() else []
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(held_rows) if held_rows else None
    peft_config = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, bias="none",
                             task_type="CAUSAL_LM", target_modules=lora_targets(model))
    wanted = dict(
        output_dir=str(run_dir / "trainer"), num_train_epochs=epochs, learning_rate=lr,
        lr_scheduler_type="linear", per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, per_device_eval_batch_size=batch_size,
        bf16=torch.cuda.is_available(), max_length=max_length, logging_steps=10,
        save_strategy="steps", save_steps=save_steps, save_total_limit=1,
        eval_strategy="epoch" if eval_ds is not None else "no", report_to=[],
        gradient_checkpointing=True, group_by_length=True, packing=False,
        completion_only_loss=True, chat_template_kwargs={"enable_thinking": False},
        # Liger's fused linear cross-entropy never materialises the logits tensor (34 GB at
        # batch 8 x 8k over Gemma's 262k vocabulary); it is what lets the run fit and go faster.
        use_liger_kernel=liger,
    )
    accepted = inspect.signature(SFTConfig.__init__).parameters
    config = SFTConfig(**{k: v for k, v in wanted.items() if k in accepted})

    class Persist(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if on_save is not None:
                on_save()

    try:
        trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                             processing_class=tokenizer, peft_config=peft_config, callbacks=[Persist()])
    except Exception as error:  # Liger has no patch for this architecture: fall back, say so
        if not liger:
            raise
        print(f"liger unavailable for {base_model} ({str(error)[:120]}); training without it")
        config = SFTConfig(**{k: v for k, v in {**wanted, "use_liger_kernel": False}.items() if k in accepted})
        model = load_base_model(base_model)
        trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                             processing_class=tokenizer, peft_config=peft_config, callbacks=[Persist()])
    metrics_liger = bool(getattr(trainer.args, "use_liger_kernel", False))
    checkpoint = latest_checkpoint(run_dir) if resume else None
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    metrics: dict = {"base_model": base_model, "resumed_from": str(checkpoint) if checkpoint else None, "liger": metrics_liger, "max_length": max_length}
    if eval_ds is not None:
        metrics["heldout_loss"] = trainer.evaluate().get("eval_loss")
    metrics["train_loss"] = next((h["train_loss"] for h in reversed(trainer.state.log_history) if "train_loss" in h), None)
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if merge:
        merged_dir = run_dir / "merged"
        base = load_base_model(base_model)
        PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        metrics["merged_dir"] = str(merged_dir)
    metrics["wall_seconds"] = round(time.time() - started, 1)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if on_save is not None:
        on_save()
    return metrics


def export_text_only(run_dir: Path) -> Path:
    """Re-save a merged multimodal checkpoint (Gemma 4) as its text-only causal LM, the layout
    llama.cpp's converter reads."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    import transformers

    merged_dir, text_dir = run_dir / "merged", run_dir / "merged-text"
    config = AutoConfig.from_pretrained(str(merged_dir))
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return merged_dir
    full = AutoModelForImageTextToText.from_pretrained(str(merged_dir), dtype=torch.bfloat16)
    language_model = getattr(full.model, "language_model", None) or getattr(full, "language_model")
    text_cls = getattr(transformers, text_config.architectures[0] if getattr(text_config, "architectures", None) else "Gemma4ForCausalLM")
    text_model = text_cls(text_config)
    text_model.model.load_state_dict(language_model.state_dict(), strict=False)
    if hasattr(full, "lm_head") and hasattr(text_model, "lm_head"):
        text_model.lm_head.load_state_dict(full.lm_head.state_dict())
    text_model.to(torch.bfloat16).save_pretrained(str(text_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(merged_dir)).save_pretrained(str(text_dir))
    return text_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--base-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--max-length", type=int, default=8192); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8); ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--liger", action="store_true")
    a = ap.parse_args()
    print(json.dumps(train_lora(Path(a.data), Path(a.run), a.base_model, batch_size=a.batch_size,
                                grad_accum=a.grad_accum, max_length=a.max_length, merge=not a.no_merge, liger=a.liger), indent=2))
```

- [ ] **Step 4: Point the Modal app at the core**

In `benchmarks/modal/train_lora.py`: add `.add_local_python_source("benchmarks")` to `train_image` (after `.env(...)`), replace the body of `train()` with:

```python
    from benchmarks.train.reader_lora import train_lora

    metrics = train_lora(Path(VOL) / "data" / run, Path(VOL) / "runs" / run, base_model,
                         epochs=epochs, lr=lr, lora_rank=lora_rank, batch_size=batch_size,
                         grad_accum=grad_accum, max_length=max_length, merge=merge, on_save=volume.commit)
    metrics.update(run=run, gpu=TRAIN_GPU)
    if merge:
        save_processor_files(base_model, Path(VOL) / "runs" / run / "merged")
    (Path(VOL) / "runs" / "latest").write_text(run)
    volume.commit()
    return metrics
```

and the body of `export_text_only()` with `from benchmarks.train.reader_lora import export_text_only as core; path = core(Path(VOL) / "runs" / run); volume.commit(); return str(path)`. Delete the moved helpers (`_to_prompt_completion`, `load_base_model`, `lora_targets`) and make `check_data` import `to_prompt_completion` from the core. Modal cannot be exercised now (no credit); `python -c "import ast,sys; ast.parse(open('benchmarks/modal/train_lora.py').read())"` and the CPU smoke test are the verification.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest benchmarks/train/test_reader_lora.py -q -m "not slow" && .venv/bin/python -m pytest benchmarks/train/test_reader_lora.py -q -m slow`
Expected: PASS; the slow test downloads the 135M model once and trains for well under two minutes on CPU.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/train benchmarks/modal/train_lora.py
git commit -m "Training core shared by Modal and any rented GPU; resume and persistence are arguments"
```

---

### Task 4: The serverless worker

**Files:**
- Create: `benchmarks/runpod/Dockerfile`, `benchmarks/runpod/handler.py`, `benchmarks/runpod/submit.py`, `benchmarks/runpod/volume.py`, `benchmarks/runpod/README.md`
- Test: `benchmarks/runpod/test_handler.py` (pytest; the handler's stage plan and input validation, no GPU), `docker build` of the image.

**Interfaces:**
- Consumes: `train_lora`, `export_text_only` from `benchmarks/train/reader_lora.py` (Task 3).
- Produces: a Docker image `<dockerhub-user>/rembero-reader-train:v1` whose handler takes `{"run": str, "base_model": str, "max_length": int, "liger": bool, "batch_size": int, "grad_accum": int, "quant": "Q8_0"}` and, with the network volume at `/runpod-volume`, reads `data/<run>/conversations.jsonl`, resumes from `runs/<run>/trainer/checkpoint-*` if present, and leaves `runs/<run>/<run>-<quant>.gguf` plus `runs/<run>/metrics.json` on the volume; `submit.py` submits and follows a job; `volume.py put|get` moves files over the S3-compatible API.

- [ ] **Step 1: Write the failing handler test**

```python
# benchmarks/runpod/test_handler.py
import pytest

from benchmarks.runpod.handler import job_config, STAGES


def test_job_config_fills_defaults_and_validates():
    cfg = job_config({"run": "reader-v5-gemma4-e4b"})
    assert cfg["base_model"] == "google/gemma-4-E4B-it" and cfg["max_length"] == 8192
    assert cfg["liger"] is True and cfg["quant"] == "Q8_0" and cfg["batch_size"] == 4 and cfg["grad_accum"] == 16
    assert cfg["data_dir"] == "/runpod-volume/data/reader-v5-gemma4-e4b"
    assert cfg["run_dir"] == "/runpod-volume/runs/reader-v5-gemma4-e4b"


def test_job_config_rejects_a_run_name_that_escapes_the_volume():
    with pytest.raises(ValueError):
        job_config({"run": "../etc"})


def test_stage_order_is_train_export_convert_quantize():
    assert STAGES == ("train", "export-text", "convert-f16", "quantize")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest benchmarks/runpod/test_handler.py -q`
Expected: FAIL, `No module named 'benchmarks.runpod.handler'`. (`pip install runpod` into `.venv` first if `import runpod` fails; the handler imports it lazily so the test does not need a worker.)

- [ ] **Step 3: Write the handler**

```python
# benchmarks/runpod/handler.py
"""RunPod Serverless worker: train (or resume) one reader run from the network volume,
export it text-only, convert to GGUF and quantize, all on the volume. One job = one run.
Progress is reported per stage so `submit.py` can print it; the job result is metrics.json."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

VOLUME = Path("/runpod-volume")
STAGES = ("train", "export-text", "convert-f16", "quantize")
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
DEFAULTS = {"base_model": "google/gemma-4-E4B-it", "max_length": 8192, "liger": True,
            "batch_size": 4, "grad_accum": 16, "quant": "Q8_0", "merge": True}


def job_config(inp: dict) -> dict:
    run = str(inp.get("run", ""))
    if not RUN_NAME.match(run):
        raise ValueError(f"run must match {RUN_NAME.pattern}, got {run!r}")
    cfg = {**DEFAULTS, **{k: v for k, v in inp.items() if k in DEFAULTS}, "run": run}
    cfg["data_dir"] = str(VOLUME / "data" / run)
    cfg["run_dir"] = str(VOLUME / "runs" / run)
    return cfg


def run_job(cfg: dict, progress=lambda msg: None) -> dict:
    from benchmarks.train.reader_lora import export_text_only, train_lora

    data_dir, run_dir = Path(cfg["data_dir"]), Path(cfg["run_dir"])
    if not (data_dir / "conversations.jsonl").exists():
        raise FileNotFoundError(f"{data_dir}/conversations.jsonl is not on the volume")
    progress("train: starting")
    metrics = train_lora(data_dir, run_dir, cfg["base_model"], batch_size=cfg["batch_size"],
                         grad_accum=cfg["grad_accum"], max_length=cfg["max_length"],
                         merge=cfg["merge"], liger=cfg["liger"], on_save=lambda: progress("train: checkpoint saved"))
    progress("export-text")
    text_dir = export_text_only(run_dir)
    f16 = run_dir / f"{cfg['run']}-f16.gguf"
    quantized = run_dir / f"{cfg['run']}-{cfg['quant']}.gguf"
    progress("convert-f16")
    subprocess.run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(text_dir), "--outtype", "f16", "--outfile", str(f16)], check=True)
    progress("quantize")
    subprocess.run(["/opt/llama.cpp/build/bin/llama-quantize", str(f16), str(quantized), cfg["quant"]], check=True)
    f16.unlink()
    metrics["gguf"] = str(quantized)
    metrics["gguf_bytes"] = quantized.stat().st_size
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def handler(job: dict) -> dict:
    import runpod

    cfg = job_config(job.get("input") or {})
    return run_job(cfg, progress=lambda msg: runpod.serverless.progress_update(job, msg))


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
```

- [ ] **Step 4: Write the Dockerfile**

```dockerfile
# benchmarks/runpod/Dockerfile — build from the repository root:
#   docker build --platform linux/amd64 -f benchmarks/runpod/Dockerfile -t <dockerhub-user>/rembero-reader-train:v1 .
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
ENV HF_HOME=/runpod-volume/hf TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN pip install --no-cache-dir "transformers>=5.0,<6" "trl>=0.24" "peft>=0.17" "datasets>=3.0" "accelerate>=1.0" \
    "liger-kernel>=0.8.2" sentencepiece protobuf runpod
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential && rm -rf /var/lib/apt/lists/* \
 && git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
 && cmake --build /opt/llama.cpp/build --target llama-quantize -j 8 \
 && pip install --no-cache-dir -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt "transformers>=5.0,<6"
WORKDIR /app
COPY benchmarks/train /app/benchmarks/train
COPY benchmarks/runpod/handler.py /app/benchmarks/runpod/handler.py
RUN touch /app/benchmarks/__init__.py /app/benchmarks/runpod/__init__.py
ENV PYTHONPATH=/app
CMD ["python", "-u", "/app/benchmarks/runpod/handler.py"]
```

The tag above was confirmed on Docker Hub on 2026-09-14; if it is gone when you build, list `https://hub.docker.com/r/runpod/pytorch/tags`, pick the newest `2.x-py3.11-cuda12.x-cudnn-devel` tag, and record the one used in the README. Liger's `apply_liger_kernel_to_gemma4` (RMSNorm, GeGLU, fused linear cross-entropy) exists from the pinned version on, so the fallback in `train_lora` is a safety net, not the expected path. `HF_HOME` on the volume means the 16 GB base model downloads once and is reused by every later job.

- [ ] **Step 5: Write `volume.py` (files in and out over the S3-compatible API)**

```python
# benchmarks/runpod/volume.py
"""Copy files to and from a RunPod network volume through its S3-compatible API.
  python benchmarks/runpod/volume.py put <local-path> <volume-path>   (file or directory)
  python benchmarks/runpod/volume.py get <volume-path> <local-path>
Env: RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY, RUNPOD_VOLUME_ID, RUNPOD_DATACENTER (e.g. EU-RO-1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3


def client():
    dc = os.environ["RUNPOD_DATACENTER"]
    return boto3.client("s3", endpoint_url=f"https://s3api-{dc.lower()}.runpod.io/", region_name=dc,
                        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY"], aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_KEY"])


def put(local: Path, remote: str) -> None:
    s3, bucket = client(), os.environ["RUNPOD_VOLUME_ID"]
    files = [local] if local.is_file() else sorted(p for p in local.rglob("*") if p.is_file())
    for f in files:
        key = remote if local.is_file() else f"{remote.rstrip('/')}/{f.relative_to(local)}"
        print(f"put {f} -> {key} ({f.stat().st_size / 2**20:.1f} MiB)")
        s3.upload_file(str(f), bucket, key)


def get(remote: str, local: Path) -> None:
    s3, bucket = client(), os.environ["RUNPOD_VOLUME_ID"]
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"get {remote} -> {local}")
    s3.download_file(bucket, remote, str(local))


if __name__ == "__main__":
    op, a, b = sys.argv[1:4]
    put(Path(a), b) if op == "put" else get(a, Path(b))
```

- [ ] **Step 6: Write `submit.py`**

```python
# benchmarks/runpod/submit.py
"""Submit one training job to the serverless endpoint and follow it to the end.
  RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... python benchmarks/runpod/submit.py reader-v5-gemma4-e4b [--max-length 6912] [--no-liger]
Sets the execution timeout to three hours (the endpoint default of 600 s would kill the job)."""

from __future__ import annotations

import argparse
import json
import os
import time

import runpod

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--max-length", type=int, default=8192)
ap.add_argument("--no-liger", action="store_true"); ap.add_argument("--quant", default="Q8_0")
ap.add_argument("--batch-size", type=int, default=4); ap.add_argument("--grad-accum", type=int, default=16)
a = ap.parse_args()
runpod.api_key = os.environ["RUNPOD_API_KEY"]
endpoint = runpod.Endpoint(os.environ["RUNPOD_ENDPOINT_ID"])
job = endpoint.run({"input": {"run": a.run, "max_length": a.max_length, "liger": not a.no_liger, "quant": a.quant,
                              "batch_size": a.batch_size, "grad_accum": a.grad_accum},
                    "policy": {"executionTimeout": 3 * 60 * 60 * 1000, "ttl": 24 * 60 * 60 * 1000}})
print("job", job.job_id)
last = None
started = time.time()
while True:
    status = job.status()
    detail = job._fetch_job() if hasattr(job, "_fetch_job") else {}
    line = f"{status} {detail.get('output') if isinstance(detail.get('output'), str) else ''}".strip()
    if line != last:
        print(f"[{(time.time() - started) / 60:5.1f} min] {line}"); last = line
    if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        break
    time.sleep(30)
print(json.dumps(job.output(), indent=2) if status == "COMPLETED" else f"job ended {status}")
```

- [ ] **Step 7: Write the README**

`benchmarks/runpod/README.md`, in this order, every command literal: (1) the cost table (H100 flex $4.18-4.79/h; Run A ≈ 26 min at ~65 s/step with batch 4 × accum 16 plus a few minutes of cold start ≈ $2.50; a fresh 4,700-row run at 6912 ≈ 80 min ≈ $6; network volume 40 GB ≈ $2.80/month, delete it when done); (2) one-time setup: Docker Hub login, `docker build --platform linux/amd64 ... && docker push`; in the RunPod console create a network volume (40 GB, note its datacenter), an S3 API key, and a Serverless endpoint (H100 80GB, max workers 1, container disk 40 GB, the network volume attached, the image from Docker Hub, execution timeout 10800 s), and an API key; export `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `RUNPOD_VOLUME_ID`, `RUNPOD_DATACENTER`, `RUNPOD_S3_ACCESS_KEY`, `RUNPOD_S3_SECRET_KEY` in `.env` (already gitignored); `.venv/bin/pip install runpod boto3`; (3) per run: `volume.py put` the data directory and, for a resume, the checkpoint directory, then `submit.py`, then `volume.py get` the GGUF; (4) how to see logs (endpoint → Requests → the job) and what a Liger fallback looks like in them; (5) the local serve line from READER-STRUCTURE.md.

- [ ] **Step 8: Run the handler tests, build the image, verify the handler starts**

Run: `.venv/bin/python -m pytest benchmarks/runpod/test_handler.py -q && docker build --platform linux/amd64 -f benchmarks/runpod/Dockerfile -t rembero-reader-train:local . && docker run --rm --platform linux/amd64 -e RUNPOD_REALTIME_PORT= rembero-reader-train:local python -c "import benchmarks.runpod.handler as h, liger_kernel, trl; print('handler ok', h.STAGES)"`
Expected: 3 tests pass; image builds (15 GB is normal); the run prints `handler ok ('train', 'export-text', 'convert-f16', 'quantize')`. The image is not pushed in this task; Task 5 pushes it under the user's Docker Hub name.

- [ ] **Step 9: Commit**

```bash
git add benchmarks/runpod
git commit -m "RunPod Serverless worker: train, resume, export and quantize a reader on a network volume"
```

---

### Task 5: Run A, finish reader v5 from step 50 on Serverless (≈ $2.50)

**Files:**
- Modify: `docs/research/run-matrix/training-runs.json` (one row), `docs/research/READER-STRUCTURE.md` (a paragraph under "Reader v5 on the M4 Pro")

**Interfaces:**
- Consumes: Task 4's image, `volume.py`, `submit.py`; the Modal volume paths `runs/reader-v5-gemma4-e4b/trainer/checkpoint-50` and `data/reader-v5-gemma4-e4b/conversations.jsonl`.
- Produces: `/Volumes/Atlas/models/rembero/reader-v5-gemma4-e4b-Q8_0.gguf`.

This task spends money and needs the user's RunPod account: stop and ask before Step 4 if `RUNPOD_API_KEY` is not in `.env`.

- [ ] **Step 1: Prove the local data is the data the checkpoint was trained on**

Run:
```bash
.venv/bin/modal volume get rembero-finetune data/reader-v5-gemma4-e4b/conversations.jsonl /tmp/v5-volume.jsonl
shasum -a 256 /tmp/v5-volume.jsonl data/training-reader-v5/conversations.jsonl
```
Expected: identical digests. If they differ, upload the volume's file; resume requires the same rows in the same order.

- [ ] **Step 2: Fetch the checkpoint from the Modal volume (no compute credit needed)**

Run: `.venv/bin/modal volume get rembero-finetune runs/reader-v5-gemma4-e4b/trainer/checkpoint-50 /Volumes/Atlas/models/rembero/reader-v5-checkpoint-50/ && python3 -c "import json; s=json.load(open('/Volumes/Atlas/models/rembero/reader-v5-checkpoint-50/trainer_state.json')); print(s['global_step'], '/', s['max_steps'])"`
Expected: the directory with `adapter_model.safetensors`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin`; prints `50 / 74`.

- [ ] **Step 3: Push the image and create the endpoint** (README setup section)

`docker tag rembero-reader-train:local <dockerhub-user>/rembero-reader-train:v1 && docker push <dockerhub-user>/rembero-reader-train:v1`; create the volume, S3 key, endpoint and API key in the console; fill `.env`.

- [ ] **Step 4: Load the volume**

```bash
set -a; source .env; set +a
.venv/bin/python benchmarks/runpod/volume.py put data/training-reader-v5/conversations.jsonl data/reader-v5-gemma4-e4b/conversations.jsonl
.venv/bin/python benchmarks/runpod/volume.py put /Volumes/Atlas/models/rembero/reader-v5-checkpoint-50 runs/reader-v5-gemma4-e4b/trainer/checkpoint-50
```
Expected: 117 MB and a few hundred MB uploaded, listed file by file.

- [ ] **Step 5: Submit and follow**

Run: `.venv/bin/python benchmarks/runpod/submit.py reader-v5-gemma4-e4b --max-length 8192 --batch-size 4 --grad-accum 16 2>&1 | tee runs/local/reader-v5-serverless.log`
Expected: progress lines `train: starting`, several `train: checkpoint saved`, `export-text`, `convert-f16`, `quantize`, then COMPLETED with metrics showing `resumed_from: .../checkpoint-50`, `liger: true` (or a logged fallback), and `gguf_bytes` near 8.0e9. Wall time 30 to 40 minutes including the cold start and the base-model download.

- [ ] **Step 6: Fetch and verify locally**

```bash
.venv/bin/python benchmarks/runpod/volume.py get runs/reader-v5-gemma4-e4b/reader-v5-gemma4-e4b-Q8_0.gguf /Volumes/Atlas/models/rembero/reader-v5-gemma4-e4b-Q8_0.gguf
llama-server -m /Volumes/Atlas/models/rembero/reader-v5-gemma4-e4b-Q8_0.gguf --port 8083 -c 12288 -np 1 -ngl 99 --alias rembero-reader-v5 --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}' &
sleep 60; curl -s http://127.0.0.1:8083/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"rembero-reader-v5","messages":[{"role":"user","content":"History chats:\n\n### Retrieved session 1\nSession date: 2023-05-15\nUSER: I bought my bike on 3 May.\n\nQuestion (asked 2023-05-24): how many days ago did I buy my bike?"}],"max_tokens":60}' | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```
Expected: a sentence with "21 days", not noise. Noise means the conversion took the direct q8 path; the handler converts f16 then quantizes, so noise here is a real defect to report.

- [ ] **Step 7: Record the run and the bill**

Read the charge from the RunPod billing page. Append to `docs/research/run-matrix/training-runs.json`: `{"run": "reader-v5 E4B", "date": "2026-09-14", "base": "Gemma 4 E4B", "platform": "Modal H100 (steps 1-50) + RunPod Serverless H100 (51-74)", "data": "4,713 GLM-distilled examples with computed notes and date distances (contract dd+notes@24576), max_length 8192", "tasks": "reader", "minutes": <measured>, "costUsd": <measured>, "heldoutLoss": null, "notes": "resumed from checkpoint-50 off the Modal volume; Liger <used|fell back>"}`. Add the paragraph to READER-STRUCTURE.md. Commit. Leave the endpoint (idle costs nothing); delete the network volume only after Task 7 decides whether Run B happens.

---

### Task 6: Measure v5 against v4 under the contract it was trained with

**Files:**
- Create: `docs/research/results/longmemeval-raw-reader-v5-local-notes-subset100.json` (+ sidecar), then the 266.
- Modify: `docs/research/READER-STRUCTURE.md` (the v5 table row)

**Interfaces:**
- Consumes: the served v5 on port 8083; `contractRunnerFlags(READER_CONTRACT_V5)` = `--date-distances --computed-notes`.

- [ ] **Step 1: The subset, raw formation, the contract's flags and nothing else**

Run (this Mac, nothing else on the GPU):
```bash
node dist/evals/run-longmemeval-answer.js --formation raw --cases "$(cat .cache/longmemeval/subset-100.txt)" \
  --date-distances --computed-notes --concurrency 1 \
  --reader-model rembero-reader-v5 --reader-base-url http://127.0.0.1:8083/v1 --reader-max-tokens 300 \
  --judge-model deepseek-chat \
  --output docs/research/results/longmemeval-raw-reader-v5-local-notes-subset100.json
```
Expected: 100 judged, `summary.errors` 0; compare with the stored v4 raw+notes subset run (make one with the same command against port 8082 if none is stored). A difference inside ±4 is noise.

- [ ] **Step 2: If the subset is level or better, the 266**

Same command with `--question-types multi-session,temporal-reasoning` in place of `--cases`, output `longmemeval-raw-reader-v5-local-notes-mt-all266.json`. v4 under the same judge and flags: 183. Record per type.

- [ ] **Step 3: Write the row into READER-STRUCTURE.md and commit**

Add the v5 row to the "Under the cheaper judge" table with its 266 numbers and the subset numbers, and one sentence on whether training with the block present moved the multi-session or temporal columns beyond noise.

---

### Task 7: The v6 gate, and v6 only if it passes (≈ $6, optional)

**Files:**
- Create: `data/training-reader-v6/` (only if the gate passes)
- Modify: `docs/research/READER-STRUCTURE.md`

**Interfaces:**
- Consumes: Task 2's distill flags; the served v5.

- [ ] **Step 1: Paired runs on the 266 with the two unproven blocks, reader v5, raw formation**

Two runs with the same command as Task 6 Step 2 plus `--focused-budget` (run 1) and `--structured-evidence` (run 2; hybrid formation with `--extraction-cache .cache/longmemeval/extraction-r23 --extraction-model rembero-writer --extraction-base-url http://127.0.0.1:8081/v1`, since structured evidence needs the writer's facts; start the r23 writer on 8081 first, and stop the v5 reader if memory is short, which on this 16 GB Mac it will be, so run these on the 24 GB MacBook or sequentially).

- [ ] **Step 2: Decide**

Gate: a block joins the contract only if its paired run beats the v5 raw+notes 266 number by more than 4. Write the two numbers and the decision into READER-STRUCTURE.md either way. If neither passes, stop here: v5 is the reader, and about $8 of the credit remains.

- [ ] **Step 3: Distill v6 with the winning contract, free teacher**

```bash
node dist/training/run-real-sessions.js distill --out data/training-reader-v6 \
  --date-distances --computed-notes <winning flags> \
  --base-url http://127.0.0.1:11434/v1 --model glm-5.3-flash:cloud --api-key ollama --max-tokens 2048 \
  --examples 4000 --train-count 3000 --seed 7 \
  --type-weights multi-session=40,temporal-reasoning=35,knowledge-update=15,abstention=10 --concurrency 4
```
Expected: `manifest.json` with `contract.id` naming the winning flags; about 4,000 kept rows. Check that `.venv` is not needed: this is the Node pipeline. Ollama Cloud rate limits show as errors in `stats`; rerun with `--concurrency 2` if they exceed a few percent.

- [ ] **Step 4: Train v6 on RunPod**

`volume.py put data/training-reader-v6 data/reader-v6-gemma4-e4b`, then `submit.py reader-v6-gemma4-e4b --max-length 6912`, then `volume.py get` the GGUF. ≈ 80 minutes ≈ $6 at the serverless rate; check the balance first and, if it cannot cover it, say so and stop.

- [ ] **Step 5: Measure v6 exactly as Task 6, with `contractRunnerFlags` of the v6 manifest**

The runner flags come from the manifest: `python3 -c "import json; print(' '.join(json.load(open('data/training-reader-v6/manifest.json'))['contract']['runnerFlags']))"`. Any other flag set is a different prompt from the one trained on and the run is not the v6 measurement. Record the row and the run-matrix entry; commit.

---

## Self-review

- Spec coverage: the deterministic flow (Tasks 1, 2, 5's contract id, 6 and 7's flag derivation), the cost ceiling (Tasks 4, 5, 7 with figures), the resume from checkpoint-50 (Task 5), the gate for new structure (global constraints, Task 7), the teacher off OpenRouter (Task 7 Step 3). Nothing in the discussion is left without a task.
- Placeholders: `<ip>`, `<port>`, `<measured>` and `<winning flags>` are the only ones and each is a value the executor obtains in the step before.
- Type consistency: `readerMessages(haystack, question, type, contract)` in Tasks 1 and 2; `train_lora(data_dir, run_dir, base_model, ..., liger)` in Tasks 3 and 4; `contractRunnerFlags` in Tasks 1, 2, 6, 7; `export_text_only(run_dir)` in Tasks 3 and 4; `job_config`, `STAGES`, `run_job` in Task 4 only.
- Revision 2026-09-14 16:15: Tasks 4 and 5 rewritten for RunPod Serverless after the user's direction; Liger and the 6912 max length folded into Task 3 and the constraints after the model research.
