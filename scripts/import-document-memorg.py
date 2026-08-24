#!/usr/bin/env python3
"""Import a Remembero document-memory artifact into Memorg 0.1.2.

The deterministic JSON manifest remains authoritative. Memorg generates runtime UUIDs and
timestamps; stable Remembero keys are copied into metadata for correlation and verification.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


FORMAT = "remembero-memorg-import"
VERSION = 1
TARGET_VERSION = "0.1.2"


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("artifact must be a regular file")
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("artifact exceeds 16 MiB")
    artifact = json.loads(raw)
    if artifact.get("format") != FORMAT or artifact.get("version") != VERSION:
        raise ValueError("unsupported artifact format or version")
    target = artifact.get("target")
    if target != {
        "package": "memorg",
        "version": TARGET_VERSION,
        "method": "MemorgSystem.create_memory_item",
    }:
        raise ValueError("unsupported Memorg target")
    items = artifact.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("artifact must contain memory items")
    body = {
        "format": artifact["format"],
        "version": artifact["version"],
        "target": target,
        "items": items,
    }
    encoded = json.dumps(
        body, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    if artifact.get("sha256") != expected:
        raise ValueError("artifact failed SHA-256 validation")
    seen: set[str] = set()
    for index, item in enumerate(items):
        key = item.get("key")
        parent_key = item.get("parent_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"item {index} has an invalid key")
        if key in seen:
            raise ValueError(f"duplicate item key: {key}")
        if index == 0:
            if parent_key is not None:
                raise ValueError("root item must not have a parent")
        elif not isinstance(parent_key, str) or parent_key not in seen:
            raise ValueError(f"item {key} does not follow its parent")
        seen.add(key)
    return artifact


def initialize_generic_collection(db_path: Path) -> None:
    """Create the generic collection omitted by Memorg 0.1.2's SQLite initializer."""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS custom_memory (
                id TEXT PRIMARY KEY,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS custom_memory_fts
            USING fts5(id, content, data)
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS custom_memory_ai AFTER INSERT ON custom_memory
            BEGIN
                INSERT INTO custom_memory_fts(id, content, data)
                VALUES (new.id, json_extract(new.data, '$.content'), new.data);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS custom_memory_ad AFTER DELETE ON custom_memory
            BEGIN
                DELETE FROM custom_memory_fts WHERE id = old.id;
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS custom_memory_au AFTER UPDATE ON custom_memory
            BEGIN
                UPDATE custom_memory_fts
                SET content = json_extract(new.data, '$.content'), data = new.data
                WHERE id = old.id;
            END
            """
        )


class NoopVectorStore:
    async def add_vector(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def search_nearest(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    async def delete_vector(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_stats(self) -> dict[str, int]:
        return {"vector_count": 0, "index_size": 0}


async def import_artifact(artifact: dict[str, Any], db_path: Path) -> dict[str, Any]:
    try:
        from memorg import MemorgSystem  # type: ignore[import-not-found]
        from memorg.storage.sqlite_storage import SQLiteStorageAdapter  # type: ignore[import-not-found]
    except ImportError:
        # PyPI 0.1.2 publishes the same API under the historical `app` package.
        from app.main import MemorgSystem  # type: ignore[import-not-found]
        from app.storage.sqlite_storage import SQLiteStorageAdapter  # type: ignore[import-not-found]

    storage = SQLiteStorageAdapter(str(db_path))
    initialize_generic_collection(db_path)
    system = MemorgSystem(
        storage=storage,
        vector_store=NoopVectorStore(),
        openai_client=None,
    )
    runtime_ids: dict[str, str] = {}
    for item in artifact["items"]:
        parent_key = item["parent_key"]
        metadata = {
            **item["metadata"],
            "remembero_memorg_key": item["key"],
            "remembero_parent_key": parent_key,
            "remembero_export_sha256": artifact["sha256"],
        }
        created = await system.create_memory_item(
            content=item["content"],
            item_type=item["item_type"],
            parent_id=None if parent_key is None else runtime_ids[parent_key],
            metadata=metadata,
            tags=item["tags"],
        )
        runtime_ids[item["key"]] = created.id
    with sqlite3.connect(db_path) as connection:
        stored = connection.execute("SELECT COUNT(*) FROM custom_memory").fetchone()[0]
        indexed = connection.execute("SELECT COUNT(*) FROM custom_memory_fts").fetchone()[0]
    if stored != len(artifact["items"]) or indexed != stored:
        raise RuntimeError("Memorg import count verification failed")
    return {
        "status": "imported",
        "database": str(db_path),
        "items": stored,
        "indexed_items": indexed,
        "export_sha256": artifact["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--database", type=Path, default=Path("document-intelligence.memorg.db"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifact = load_artifact(args.artifact)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "items": len(artifact["items"]),
                    "documents": sum(
                        item["item_type"] == "document" for item in artifact["items"]
                    ),
                    "sha256": artifact["sha256"],
                },
                sort_keys=True,
            )
        )
        return
    db_path = args.database.resolve()
    if db_path.exists():
        raise ValueError("refusing to overwrite an existing Memorg database")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(import_artifact(artifact, db_path))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
