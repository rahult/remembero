#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "fastembed==0.8.0",
#   "llama-index-core==0.14.23",
#   "llama-index-embeddings-fastembed==0.6.0",
# ]
# ///
"""rembero.memory-systems.v1 bridge for LlamaIndex VectorMemory over SimpleVectorStore.

One process for the whole run; a fresh VectorMemory per question. No LLM."""

import json
import os
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import VectorMemory
from llama_index.embeddings.fastembed import FastEmbedEmbedding

PROTOCOL_VERSION = "rembero.memory-systems.v1"
MODEL_ID = "BAAI/bge-small-en-v1.5"
ZERO_USAGE = {"modelCalls": 0, "inputTokens": 0, "outputTokens": 0, "costUsd": 0.0}


def session_text(session: dict[str, Any]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session["turns"])


def answer(request: dict[str, Any], embedding: FastEmbedEmbedding) -> dict[str, Any]:
    top_k_value = request.get("topK", 5)
    top_k = top_k_value if isinstance(top_k_value, int) and top_k_value > 0 else 5
    ingest_started = perf_counter()
    memory = VectorMemory.from_defaults(
        embed_model=embedding, retriever_kwargs={"similarity_top_k": top_k}
    )
    for session in request["sessions"]:
        memory.put(
            ChatMessage(
                role=MessageRole.USER,
                content=session_text(session),
                additional_kwargs={"session_id": str(session["id"])},
            )
        )
    ingest_ms = (perf_counter() - ingest_started) * 1000
    search_started = perf_counter()
    matches = memory.get(input=str(request["question"]))
    search_ms = (perf_counter() - search_started) * 1000
    session_ids: list[str] = []
    for message in matches:
        session_id = message.additional_kwargs.get("session_id")
        if session_id is not None and str(session_id) not in session_ids:
            session_ids.append(str(session_id))
    return {
        "questionId": str(request["questionId"]),
        "retrieved": [
            {"sessionId": session_id, "rank": index + 1}
            for index, session_id in enumerate(session_ids[:top_k])
        ],
        "memories": [],
        "unsupported": ["memories"],
        "usage": dict(ZERO_USAGE),
        "wallMs": {"ingest": ingest_ms, "search": search_ms},
    }


def main() -> None:
    embedding = FastEmbedEmbedding(
        model_name=MODEL_ID, providers=["CPUExecutionProvider"], doc_embed_type="passage"
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        question_id = ""
        response: dict[str, Any]
        try:
            request = json.loads(line)
            question_id = str(request.get("questionId", ""))
            if request.get("protocolVersion") != PROTOCOL_VERSION:
                raise ValueError(
                    f"unsupported protocol version {request.get('protocolVersion')!r}"
                )
            response = answer(request, embedding)
        except Exception as error:  # a malformed line or one bad question must not end the run
            response = {
                "questionId": question_id,
                "error": f"{type(error).__name__}: {error}"[:300],
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
