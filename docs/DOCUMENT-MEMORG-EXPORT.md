# Memorg document-memory export

Remembero exports the real-PDF document-intelligence corpus as a deterministic import plan for
[Memorg 0.1.2](https://pypi.org/project/memorg/), whose generic memory API accepts `content`,
`item_type`, `parent_id`, `metadata`, and `tags`.

## Generate and verify

```bash
npm run export:documents:memorg
```

Installed-package users can use the single Remembero CLI:

```bash
remembero document-memorg > document-intelligence.memorg.json
remembero verify-document-memorg document-intelligence.memorg.json
```

This writes the authoritative `docs/research/results/document-intelligence.memorg.json` plus a
hash-identical browser download under `web/public/documents/`, then verifies hierarchy, authority
states, resource bounds, and SHA-256 digest.

The artifact contains 66 parent-first items:

| Memory kind | Count | Authority |
| --- | ---: | --- |
| Collection root | 1 | Collection metadata |
| Real PDF documents | 4 | Source identity and hashes |
| Selected page regions | 20 | Evidence only |
| Accepted claims | 17 | Proof-bearing |
| Proposed claims | 4 | Review required; never proof-bearing |
| Reviewed rules | 8 | Proof-bearing |
| Guided question contracts | 12 | Evaluation contract |

The digest covers every field except `sha256`. Identical source fixtures produce identical JSON
and the same digest. Memorg's runtime database uses generated UUIDs and timestamps, so the JSON
manifest—not a generated SQLite file—is the portable authority.

## Import into Memorg

Install Memorg in a separate Python environment, then run:

```bash
python scripts/import-document-memorg.py \
  docs/research/results/document-intelligence.memorg.json \
  --database document-intelligence.memorg.db
```

To validate the artifact without installing Memorg or writing a database:

```bash
python scripts/import-document-memorg.py \
  docs/research/results/document-intelligence.memorg.json \
  --dry-run
```

The importer refuses to overwrite an existing database. Stable Remembero keys, parent keys, and
the export digest are copied into every Memorg item's metadata. Memorg 0.1.2 omits initialization
for its own `custom_memory` collection; the importer creates only that missing official-shaped
SQLite table, FTS index, and maintenance triggers before calling
`MemorgSystem.create_memory_item`.

The importer was verified in a clean temporary Python 3.12 environment with Memorg 0.1.2:
66 items were stored, all 66 were present in Memorg's FTS index, and the authority counts matched
the manifest exactly. The frozen result is
`docs/research/results/document-intelligence-memorg-import-v1-summary.json`.

## Boundary

- Source-region text is searchable context, not accepted truth.
- Proposed claims retain `authority: proposed_only` and a `review-required` tag.
- Accepted claims and reviewed rules retain `proof-bearing` tags.
- PDF and rendered-page SHA-256 values remain attached to their source and region memories.
- The export does not include embeddings, credentials, private files, or the generated Memorg
  UUIDs.
