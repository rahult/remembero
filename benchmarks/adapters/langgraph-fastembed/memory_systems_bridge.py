#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "fastembed==0.8.0",
#   "langchain-core==1.6.3",
#   "langgraph==1.2.10",
# ]
# ///
"""rembero.memory-systems.v1 bridge for LangGraph's InMemoryStore semantic search.

One process for the whole run, one JSON line per question on stdin, one on stdout. A fresh
namespace per question is a fresh store: nothing a question ingests is visible to the next.
No LLM: the memory is the session text, indexed by local FastEmbed bge-small."""

import json
import os
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore

PROTOCOL_VERSION = "rembero.memory-systems.v1"
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
ZERO_USAGE = {"modelCalls": 0, "inputTokens": 0, "outputTokens": 0, "costUsd": 0.0}


class FastEmbedEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.model = TextEmbedding(model_name=MODEL_ID, providers=["CPUExecutionProvider"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.query_embed([text])).tolist()


def session_text(session: dict[str, Any]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session["turns"])


def answer(request: dict[str, Any], embeddings: FastEmbedEmbeddings) -> dict[str, Any]:
    ingest_started = perf_counter()
    store = InMemoryStore(
        index={"dims": EMBEDDING_DIMENSIONS, "embed": embeddings, "fields": ["text"]}
    )
    namespace = ("rembero-memory-systems", str(request["questionId"]))
    for session in request["sessions"]:
        # Only "text" is indexed and only the key is returned, so the session date is not
        # stored: nothing here reads it, and an unread field reads as a signal that it does.
        store.put(namespace, str(session["id"]), {"text": session_text(session)})
    ingest_ms = (perf_counter() - ingest_started) * 1000
    search_started = perf_counter()
    top_k_value = request.get("topK", 5)
    top_k = top_k_value if isinstance(top_k_value, int) and top_k_value > 0 else 5
    matches = store.search(namespace, query=str(request["question"]), limit=top_k)
    search_ms = (perf_counter() - search_started) * 1000
    return {
        "questionId": str(request["questionId"]),
        "retrieved": [
            {
                "sessionId": str(item.key),
                "rank": index + 1,
                "score": float(getattr(item, "score", 0.0) or 0.0),
            }
            for index, item in enumerate(matches)
        ],
        "memories": [],
        "unsupported": ["memories"],
        "usage": dict(ZERO_USAGE),
        "wallMs": {"ingest": ingest_ms, "search": search_ms},
    }


def main() -> None:
    embeddings = FastEmbedEmbeddings()
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
            response = answer(request, embeddings)
        except Exception as error:  # a malformed line or one bad question must not end the run
            response = {
                "questionId": question_id,
                "error": f"{type(error).__name__}: {error}"[:300],
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
