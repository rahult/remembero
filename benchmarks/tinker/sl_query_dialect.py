"""Supervised fine-tune of a small model on Remembero's flat query dialect.

Manual and paid. Confirm before running.

    python benchmarks/tinker/sl_query_dialect.py \
        --data data/training/conversations.jsonl \
        --model Llama-3.2-3B \
        --log-path runs/tinker/llama-3.2-3b-dialect

Reads TINKER_API_KEY from .env at the repository root. Field names follow
tinker_cookbook.supervised.train.Config at cookbook commit 1f962ed. Held-out
loss ("heldout/nll") is computed on heldout.jsonl — whole worlds never seen in
training — not on a slice of the training file.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Smallest models Tinker lists (model_info.py at 1f962ed). No Qwen 3.5 2B exists there.
ORG = {
    "Llama-3.2-1B": "meta-llama",
    "Llama-3.2-3B": "meta-llama",
    "Qwen3-4B-Instruct-2507": "Qwen",
    "Qwen3.5-4B": "Qwen",
}


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY missing (put it in .env at the repo root)")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="conversations.jsonl from npm run train:data")
    parser.add_argument("--model", default="Llama-3.2-3B", choices=sorted(ORG))
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--heldout",
        default=None,
        help="heldout.jsonl (whole held-out worlds); defaults to the file next to --data",
    )
    args = parser.parse_args()
    heldout_path = Path(args.heldout) if args.heldout else Path(args.data).with_name("heldout.jsonl")
    if not heldout_path.exists():
        raise SystemExit(f"held-out file not found: {heldout_path}")

    model_name = f"{ORG[args.model]}/{args.model}"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=2048,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    # test_size=0: never slice the training file for evaluation. Held-out loss
    # comes from heldout.jsonl, whose worlds are absent from training.
    dataset = FromConversationFileBuilder(common_config=common, file_path=args.data, test_size=0)
    heldout_builder = FromConversationFileBuilder(
        common_config=common, file_path=str(heldout_path), test_size=0
    )

    def heldout_evaluator() -> NLLEvaluator:
        heldout_dataset, _ = heldout_builder()
        return NLLEvaluator.from_dataset(
            heldout_dataset, name="heldout", tokenizer=get_tokenizer(model_name)
        )
    config = (
        chz.Blueprint(train.Config)
        .apply(
            {
                "log_path": args.log_path,
                "model_name": model_name,
                "recipe_name": "remembero_query_dialect",
                "renderer_name": renderer_name,
                "dataset_builder": dataset,
                "learning_rate": args.lr,
                "lr_schedule": "linear",
                "num_epochs": args.epochs,
                "lora_rank": args.lora_rank,
                "evaluator_builders": [heldout_evaluator],
                "eval_every": 20,
                "save_every": 50,
            }
        )
        .make()
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
