# Backup, restore, and multi-machine sync

Remembero's authority is plain files under `$REMBERO_HOME` (default `~/.rembero`):
human-editable `.dl` namespace files, the append-only `journal.log` plus rotated
checkpoint segments, and the derived `semantic.sqlite` / `.semantic-embeddings/` caches
(safe to delete; they rebuild).

## Backup and restore

```bash
remembero backup memory-backup.json     # verified, content-addressed whole-store bundle
remembero restore memory-backup.json    # restore into a fresh or empty store
```

`backup` writes one SHA-256-addressed bundle of every namespace (clauses plus durable
sources) and re-verifies the written file. `restore` verifies the digest first and then
fails closed: each bundle namespace must be empty in the target store, or already contain
exactly the bundle's clauses (so retries are idempotent). A tampered file never restores.

Boundary: restored clauses record one `Restored from backup <digest>` source; the
original per-clause provenance stays readable inside the backup file itself, and exact
journal history does not transfer — recorded-time queries start fresh in the target
store.

For point-in-time archival without a separate file, `remembero checkpoint` rotates the
journal into immutable verified segments; that is history management, not a backup.

## Moving to a second machine

1. Install Remembero and run `remembero init` on the new machine.
2. Copy a fresh `remembero backup` file across and run `remembero restore` there.

## Continuous sync (two machines, one memory)

There is deliberately no sync service. The store is small plain text, so version it:

```bash
cd ~/.rembero && git init && git add memory/*.dl memory/journal.log && git commit -m 'memory'
```

Push to a private remote and pull before starting work on the other machine. Two cautions:

- The cross-process mutation lock serializes writers on **one** machine only. Do not run
  sessions against the same namespaces on two machines between syncs; last merge wins and
  a diverged `journal.log` will fail recorded-time replay reconciliation (current facts
  still load — history queries refuse until you keep one journal lineage).
- Do not sync `semantic.sqlite`, `.semantic-embeddings/`, or lock files; they are
  machine-local. A `.gitignore` with `semantic.sqlite`, `.semantic-embeddings/`, and
  `*.lock` covers it.

File-sync tools (Syncthing, Dropbox) work under the same single-writer caution, but git
gives you conflict visibility instead of silent clobbering.
