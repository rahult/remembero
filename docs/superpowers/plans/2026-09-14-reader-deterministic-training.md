# Reader training with a deterministic contract, on RunPod for under $10

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the reader so that every prompt it learns from is byte-identical to the prompt it will be evaluated and served with, and finish reader v5 plus one optional v6 for under $10 of RunPod credit.

**Architecture:** One `ReaderContract` object names the deterministic structure the reader is given before reading (date distances, computed notes, focused budget, structured evidence, context bytes). The distiller, the evaluation harness and the training manifest all derive from it, and a test proves the distilled prompt equals the harness prompt. Training moves off Modal into a plain Python core shared by the Modal app and a RunPod runbook; v5 resumes from the step-50 checkpoint still on the Modal volume, which costs about $2 instead of $5.

**Tech Stack:** TypeScript (vitest), Python 3.12 with transformers 5 / TRL / PEFT, llama.cpp, RunPod community H100 at $1.99/h, Modal volume (read-only, no compute credit needed), Ollama Cloud GLM 5.3 Flash as the free teacher, DeepSeek as judge.

**Spec:** `docs/research/READER-STRUCTURE.md` sections "Reader v5", "Decisions (2026-09-13, with the user)" and "Reader v5 on the M4 Pro"; the design conversation of 2026-09-14 (this plan's header records its conclusions).

## Global Constraints

- Base model `google/gemma-4-E4B-it`; it is not gated (checked 2026-09-14), no Hugging Face token needed.
- Recipe unchanged from v4/v5: rank-32 LoRA, alpha 64, lr 2e-4 linear, 1 epoch, batch 8 × grad-accum 8, max length 8192, completion-only loss, no packing, gradient checkpointing, seed 42 (TRL default).
- A structure block enters the contract only after a paired run on the 266 (DeepSeek judge) shows a gain outside the ±4 noise band. Today that is date distances and computed notes (raw 500: 354 → 383). Focused budget and structured evidence stay flags until they earn their place (subset-100 hybrid: baseline 74, evidence 71, both 75, inside noise).
- Budget: $10 on RunPod. Run A (finish v5) ≈ $2. Run B (v6) ≈ $4, only if Task 7's gate passes. Never leave a pod running idle; every runbook step ends with the stop command.
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
- Produces: `train_lora(data_dir: Path, run_dir: Path, base_model: str, *, epochs=1, lr=2e-4, lora_rank=32, batch_size=8, grad_accum=8, max_length=8192, merge=True, on_save=None, resume=True) -> dict` and `export_text_only(run_dir: Path) -> Path` and `to_prompt_completion(path) -> list[dict]`, all free of Modal. The Modal `train()` becomes a thin wrapper that mounts the volume paths and passes `on_save=volume.commit`.

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
    )
    accepted = inspect.signature(SFTConfig.__init__).parameters
    config = SFTConfig(**{k: v for k, v in wanted.items() if k in accepted})

    class Persist(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if on_save is not None:
                on_save()

    trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tokenizer, peft_config=peft_config, callbacks=[Persist()])
    checkpoint = latest_checkpoint(run_dir) if resume else None
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    metrics: dict = {"base_model": base_model, "resumed_from": str(checkpoint) if checkpoint else None}
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
    a = ap.parse_args()
    print(json.dumps(train_lora(Path(a.data), Path(a.run), a.base_model, batch_size=a.batch_size,
                                grad_accum=a.grad_accum, max_length=a.max_length, merge=not a.no_merge), indent=2))
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

### Task 4: The RunPod runbook and scripts

**Files:**
- Create: `benchmarks/runpod/README.md`, `benchmarks/runpod/pod-setup.sh`, `benchmarks/runpod/run.sh`
- Test: `bash -n` on both scripts; a dry run of `run.sh --dry-run` prints the commands without executing.

**Interfaces:**
- Consumes: `benchmarks/train/reader_lora.py` (Task 3).
- Produces: on the pod, `/workspace/runs/<run>/gguf/<run>-Q8_0.gguf` and a printed `scp` line to fetch it.

- [ ] **Step 1: Write `pod-setup.sh`**

```bash
#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (PyTorch template, CUDA 12.x). Idempotent.
set -euo pipefail
cd /workspace
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install -q "transformers>=5.0,<6" "trl>=0.24" "peft>=0.17" "datasets>=3.0" "accelerate>=1.0" sentencepiece protobuf
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
  cmake --build llama.cpp/build --target llama-quantize -j "$(nproc)"
  pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt "transformers>=5.0,<6"
fi
mkdir -p data runs
echo "pod ready: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
```

- [ ] **Step 2: Write `run.sh`**

```bash
#!/usr/bin/env bash
# Train (or resume) one run, export text-only, convert to GGUF, quantize, print the fetch line.
#   run.sh <run-name> [--max-length 8192] [--dry-run]
# Data at /workspace/data/<run>/conversations.jsonl; an existing
# /workspace/runs/<run>/trainer/checkpoint-N resumes from step N.
set -euo pipefail
run="$1"; shift
dry=0; args=()
for a in "$@"; do [ "$a" = "--dry-run" ] && dry=1 || args+=("$a"); done
cd /workspace/rembero
cmds=(
  "python -m benchmarks.train.reader_lora --data /workspace/data/$run --run /workspace/runs/$run ${args[*]:-}"
  "python -c 'from pathlib import Path; from benchmarks.train.reader_lora import export_text_only; print(export_text_only(Path(\"/workspace/runs/$run\")))'"
  "python /workspace/llama.cpp/convert_hf_to_gguf.py /workspace/runs/$run/merged-text --outtype f16 --outfile /workspace/runs/$run/$run-f16.gguf"
  "/workspace/llama.cpp/build/bin/llama-quantize /workspace/runs/$run/$run-f16.gguf /workspace/runs/$run/$run-Q8_0.gguf Q8_0"
  "rm -f /workspace/runs/$run/$run-f16.gguf"
)
for c in "${cmds[@]}"; do echo "+ $c"; [ "$dry" = 1 ] || eval "$c"; done
echo "fetch with: scp -P <port> root@<pod-ip>:/workspace/runs/$run/$run-Q8_0.gguf /Volumes/Atlas/models/rembero/"
echo "then STOP THE POD."
```

- [ ] **Step 3: Write the README**

`benchmarks/runpod/README.md` holds, in this order: (1) cost table (community H100 $1.99/h; reader step ≈ 80 s at batch 64 × 8k; v5 finish 24 steps ≈ 35 min ≈ $1.20 plus ~15 min setup ≈ $1.70 total; a full 4,700-row run ≈ 100 min ≈ $3.50; download of the base model 16 GB ≈ 3 min on RunPod); (2) create the pod: RunPod console → Pods → Deploy → filter H100 80GB, Community Cloud, the current "RunPod PyTorch" template, 80 GB container disk, 0 GB volume, enable SSH over exposed TCP, deploy; copy the SSH command; (3) push the repo's `benchmarks/` and the data: `rsync -avz -e "ssh -p <port>" benchmarks root@<ip>:/workspace/rembero/benchmarks/` and `rsync -avz -e "ssh -p <port>" data/training-reader-v5/ root@<ip>:/workspace/data/reader-v5-gemma4-e4b/`; (4) `bash /workspace/rembero/benchmarks/runpod/pod-setup.sh`; (5) resume: fetch the checkpoint from Modal first (Task 5); (6) `bash benchmarks/runpod/run.sh reader-v5-gemma4-e4b`; (7) fetch the GGUF, **stop the pod**, then terminate it once the file is verified locally; (8) the local serve line from READER-STRUCTURE.md and the evaluation command from Task 6. Every command literal, no prose placeholders except `<ip>` and `<port>`.

- [ ] **Step 4: Verify the scripts parse and the dry run prints five commands**

Run: `bash -n benchmarks/runpod/pod-setup.sh benchmarks/runpod/run.sh && (cd /tmp && mkdir -p workspace/rembero && cd workspace/rembero && bash /Volumes/Atlas/Code/projects/rembero/benchmarks/runpod/run.sh reader-v5-gemma4-e4b --dry-run 2>&1 | grep -c '^+ ')`
Expected: `5`. (The dry run only echoes; `cd /workspace/rembero` fails outside the pod, so run it from a throwaway directory named the same or accept the cd error as the only output difference.)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/runpod
git commit -m "RunPod runbook: train, resume, export and fetch a reader for a few dollars"
```

---

### Task 5: Run A, finish reader v5 from step 50 (≈ $2)

**Files:**
- Modify: `docs/research/run-matrix/training-runs.json` (one row), `docs/research/READER-STRUCTURE.md` (a paragraph under "Reader v5 on the M4 Pro")

**Interfaces:**
- Consumes: Task 4's scripts; the Modal volume paths `runs/reader-v5-gemma4-e4b/trainer/checkpoint-50` (adapter, optimizer, scheduler, RNG, trainer_state) and `data/reader-v5-gemma4-e4b/conversations.jsonl`.
- Produces: `/Volumes/Atlas/models/rembero/reader-v5-gemma4-e4b-Q8_0.gguf`.

- [ ] **Step 1: Prove the local data is the data the checkpoint was trained on**

Run:
```bash
.venv/bin/modal volume get rembero-finetune data/reader-v5-gemma4-e4b/conversations.jsonl /tmp/v5-volume.jsonl
shasum -a 256 /tmp/v5-volume.jsonl data/training-reader-v5/conversations.jsonl
```
Expected: identical digests. If they differ, use the volume's file on the pod; resume requires the same rows in the same order.

- [ ] **Step 2: Fetch the checkpoint from the volume (no compute credit needed)**

Run: `.venv/bin/modal volume get rembero-finetune runs/reader-v5-gemma4-e4b/trainer/checkpoint-50 /Volumes/Atlas/models/rembero/reader-v5-checkpoint-50/ && du -sh /Volumes/Atlas/models/rembero/reader-v5-checkpoint-50`
Expected: a directory with `adapter_model.safetensors`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin`; a few hundred MB.

- [ ] **Step 3: Confirm the resume point**

Run: `python3 -c "import json; s=json.load(open('/Volumes/Atlas/models/rembero/reader-v5-checkpoint-50/trainer_state.json')); print(s['global_step'], '/', s['max_steps'])"`
Expected: `50 / 74`.

- [ ] **Step 4: Create the pod and push data, checkpoint and code** (README steps 2 to 4)

Push the checkpoint to `/workspace/runs/reader-v5-gemma4-e4b/trainer/checkpoint-50/`. Run `pod-setup.sh`. Note the clock: the pod meter starts at deploy.

- [ ] **Step 5: Resume and export**

Run on the pod: `bash benchmarks/runpod/run.sh reader-v5-gemma4-e4b 2>&1 | tee /workspace/runs/reader-v5.log`
Expected: `resumed_from: .../checkpoint-50`, 24 more steps, `train_loss` printed, then a 7.5 GiB `reader-v5-gemma4-e4b-Q8_0.gguf`.

- [ ] **Step 6: Fetch, stop the pod, verify locally**

Run the printed `scp` line, then stop the pod in the console. Then locally:
```bash
llama-server -m /Volumes/Atlas/models/rembero/reader-v5-gemma4-e4b-Q8_0.gguf --port 8083 -c 12288 -np 1 -ngl 99 --alias rembero-reader-v5 --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}' &
sleep 60; curl -s http://127.0.0.1:8083/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"rembero-reader-v5","messages":[{"role":"user","content":"History chats:\n\n### Retrieved session 1\nSession date: 2023-05-15\nUSER: I bought my bike on 3 May.\n\nQuestion (asked 2023-05-24): how many days ago did I buy my bike?"}],"max_tokens":60}' | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```
Expected: a sentence with "21 days", not noise. Noise means the conversion took the direct q8 path; re-run the f16 then quantize steps.

- [ ] **Step 7: Record the run**

Append to `docs/research/run-matrix/training-runs.json` a row `{"run": "reader-v5 E4B", "date": "2026-09-14", "base": "Gemma 4 E4B", "platform": "Modal H100 (steps 1-50) + RunPod H100 (51-74)", "data": "4,713 GLM-distilled examples with computed notes and date distances (contract dd+notes@24576), max_length 8192", "tasks": "reader", "minutes": <measured>, "costUsd": <measured>, "heldoutLoss": null, "notes": "resumed from checkpoint-50 off the Modal volume"}` with the measured minutes and the RunPod charge. Commit with the doc paragraph.

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

### Task 7: The v6 gate, and v6 only if it passes (≈ $4, optional)

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

Push `data/training-reader-v6/` to `/workspace/data/reader-v6-gemma4-e4b/`, run `bash benchmarks/runpod/run.sh reader-v6-gemma4-e4b`, fetch, stop the pod. ≈ 100 minutes ≈ $3.50.

- [ ] **Step 5: Measure v6 exactly as Task 6, with `contractRunnerFlags` of the v6 manifest**

The runner flags come from the manifest: `python3 -c "import json; print(' '.join(json.load(open('data/training-reader-v6/manifest.json'))['contract']['runnerFlags']))"`. Any other flag set is a different prompt from the one trained on and the run is not the v6 measurement. Record the row and the run-matrix entry; commit.

---

## Self-review

- Spec coverage: the deterministic flow (Tasks 1, 2, 5's contract id, 6 and 7's flag derivation), the cost ceiling (Tasks 4, 5, 7 with figures), the resume from checkpoint-50 (Task 5), the gate for new structure (global constraints, Task 7), the teacher off OpenRouter (Task 7 Step 3). Nothing in the discussion is left without a task.
- Placeholders: `<ip>`, `<port>`, `<measured>` and `<winning flags>` are the only ones and each is a value the executor obtains in the step before.
- Type consistency: `readerMessages(haystack, question, type, contract)` in Tasks 1 and 2; `train_lora(data_dir, run_dir, base_model, ...)` in Tasks 3 and 4; `contractRunnerFlags` in Tasks 1, 2, 6, 7; `export_text_only(run_dir)` in Tasks 3 and 4.
